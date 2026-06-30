"""Free-running, timer-driven MuJoCo simulator node.

This is the asynchronous counterpart to the lockstep ``sync_simulator_node`` (where
one ``/commands`` message triggers exactly one control step). Here the
simulation advances on a fixed-rate timer instead:

- Commands arrive asynchronously; the latest one is *held* and reused every
  tick until replaced (latest-command-wins).
- Each timer tick applies the held command, advances ``steps_per_tick`` physics
  steps, and publishes the resulting state.

Because the sim free-runs regardless of whether commands arrive, there is no
start-up deadlock and no seed publish is needed -- the first tick already
publishes the reset state. The ROS timer fires in wall time, so it also paces
the sim to real time (no manual render_sleep): the real-time factor is
``control_hz * steps_per_tick * timestep``. This node publishes ``/clock`` and
must therefore run with ``use_sim_time:=false`` (the default).
"""
from __future__ import annotations

import numpy as np

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState

from goat.utils import ros_bridge
from goat.utils.mujoco_sim import MujocoSim, SimConfig


class AsyncSimulatorNode(Node):
    def __init__(self) -> None:
        super().__init__('async_simulator_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('timestep', 0.0)
        self.declare_parameter('control_hz', 100.0)
        self.declare_parameter('steps_per_tick', 1)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('home_keyframe', '')
        self.declare_parameter('joint_order', [], ParameterDescriptor(dynamic_typing=True))

        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise RuntimeError('Parameter "model_path" must be set.')
        timestep = float(self.get_parameter('timestep').value)
        if timestep <= 0.0:
            raise RuntimeError('Parameter "timestep" must be set (> 0) from yaml/launch.')
        self._control_hz = float(self.get_parameter('control_hz').value)
        if self._control_hz <= 0.0:
            raise RuntimeError('Parameter "control_hz" must be > 0.')
        self._steps_per_tick = max(1, int(self.get_parameter('steps_per_tick').value))
        self._use_viewer = bool(self.get_parameter('use_viewer').value)
        home_keyframe = self.get_parameter('home_keyframe').value or None
        joint_order = list(self.get_parameter('joint_order').value) or None

        self.sim = MujocoSim(SimConfig(
            model_path=model_path,
            use_viewer=self._use_viewer,
            home_keyframe=home_keyframe,
            timestep=timestep,
            joint_order=joint_order,
        ))
        self.sim.reset()  # MujocoSim logs the model summary at load time
        if self._use_viewer:
            self.sim.open_viewer()

        self._shutting_down = False

        # Held command: the latest ctrl applied on every tick until a new
        # command replaces it. Zero at reset so the sim free-runs from rest.
        # The command callback and the timer callback run in the same executor
        # thread, so no lock is needed around this.
        self._held_ctrl = np.zeros(self.sim.nu, dtype=float)

        # RELIABLE so commands are not dropped on a busy link; latest-wins means
        # only the most recent matters, but a reliable depth tolerates bursts.
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_sub = self.create_subscription(JointState, 'commands', self._on_cmd, cmd_qos)
        self.joint_pub = self.create_publisher(JointState, 'sim_joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'sim_imu', 10)
        self.clock_pub = self.create_publisher(Clock, 'clock', 10)

        # The loop: a wall-clock timer drives every step. It always fires, so
        # quit / viewer-close are detected here too -- no housekeeping timer.
        self._timer = self.create_timer(1.0 / self._control_hz, self._on_tick)

        rtf = self._control_hz * self._steps_per_tick * self.sim.timestep
        self.get_logger().info(
            f'async_simulator_node up: model={model_path}, '
            f'control_hz={self._control_hz}, steps_per_tick={self._steps_per_tick}, '
            f'use_viewer={self._use_viewer}, real_time_factor={rtf:.3f}'
        )

    # ------------------------------------------------------------------ #
    # Async command intake: just hold the latest ctrl (no stepping here)
    # ------------------------------------------------------------------ #
    def _on_cmd(self, msg: JointState) -> None:
        self._held_ctrl = ros_bridge.cmd_to_ctrl(msg, self.sim)

    # ------------------------------------------------------------------ #
    # Main loop: one timer tick == one control step
    # ------------------------------------------------------------------ #
    def _on_tick(self) -> None:
        if self.sim.is_quit_requested:
            self._shutdown()
            return
        if self._use_viewer and not self.sim.is_viewer_running:
            self._shutdown()
            return

        if self.sim.consume_reset_request():
            self.sim.reset()
            self._held_ctrl = np.zeros(self.sim.nu, dtype=float)

        self.sim.set_ctrl(self._held_ctrl)
        if not self.sim.is_paused:
            self.sim.step(self._steps_per_tick)

        self._publish_state()

        if self._use_viewer:
            self.sim.sync()

    def _publish_state(self) -> None:
        """Publish clock + joint/imu state for the current MjData."""
        stamp = ros_bridge.sim_time_to_msg(self.sim.sim_time)
        self.clock_pub.publish(Clock(clock=stamp))
        self.joint_pub.publish(ros_bridge.joint_state_msg(self.sim, stamp))
        self.imu_pub.publish(ros_bridge.imu_msg(self.sim, stamp))

    @property
    def shutdown_requested(self) -> bool:
        return self._shutting_down

    def _shutdown(self) -> None:
        """Request shutdown: close the viewer and flag the spin loop to exit.

        This only flips state. The actual context teardown (destroy_node then
        rclpy.shutdown, in that order) is owned by main(), so we never tear the
        context down from inside a callback the context is still driving.
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info('shutting down async_simulator_node')
        self.sim.close_viewer()

    def destroy_node(self) -> bool:
        self.sim.close_viewer()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AsyncSimulatorNode()
    try:
        # Manual spin so a viewer-close / quit (which only sets a flag) breaks
        # the loop; teardown below then runs in the correct order.
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
