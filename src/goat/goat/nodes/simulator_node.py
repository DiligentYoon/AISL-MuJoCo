"""Command-driven lockstep MuJoCo simulator node.

There is NO step timer. The simulation advances purely in simulation time and
ONLY when a control command arrives: each ``/cmd`` (sensor_msgs/JointState)
triggers one control step (``steps_per_cmd`` physics steps) and publishes the
resulting state. A low-rate housekeeping timer only watches for quit / viewer
close so the node can exit even while no commands flow -- it never steps.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState

from goat.utils import ros_bridge
from goat.utils.mujoco_sim import MujocoSim, SimConfig


class SimulatorNode(Node):
    def __init__(self) -> None:
        super().__init__('simulator_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('steps_per_cmd', 1)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('render_sleep', True)
        self.declare_parameter('home_keyframe', '')

        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise RuntimeError('Parameter "model_path" must be set.')
        self._steps_per_cmd = max(1, int(self.get_parameter('steps_per_cmd').value))
        self._use_viewer = bool(self.get_parameter('use_viewer').value)
        self._render_sleep = bool(self.get_parameter('render_sleep').value)
        home_keyframe = self.get_parameter('home_keyframe').value or None

        self.sim = MujocoSim(SimConfig(
            model_path=model_path,
            use_viewer=self._use_viewer,
            home_keyframe=home_keyframe,
        ))
        self.sim.reset()
        self.sim.inspect()
        if self._use_viewer:
            self.sim.open_viewer()

        # Wall-clock target for optional render pacing.
        self._control_period = self._steps_per_cmd * self.sim.timestep
        self._shutting_down = False

        # RELIABLE so commands are not dropped: in this synchronous closed loop
        # a dropped /cmd stalls the external controller (deadlock), not data.
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_sub = self.create_subscription(
            JointState, 'cmd', self._on_cmd, cmd_qos)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu', 10)
        self.clock_pub = self.create_publisher(Clock, 'clock', 10)

        # Housekeeping only: watch quit/viewer-close. Never advances the sim.
        self._housekeep_timer = self.create_timer(0.1, self._on_housekeep)

        self.get_logger().info(
            f'simulator_node up: model={model_path}, '
            f'steps_per_cmd={self._steps_per_cmd}, use_viewer={self._use_viewer}'
        )

    # ------------------------------------------------------------------ #
    # Main loop: one /cmd == one control step
    # ------------------------------------------------------------------ #
    def _on_cmd(self, msg: JointState) -> None:
        if self.sim.is_quit_requested:
            self._shutdown()
            return

        if self.sim.consume_reset_request():
            self.sim.reset()

        self.sim.set_ctrl(ros_bridge.cmd_to_ctrl(msg, self.sim))

        wall_start = time.monotonic()
        if not self.sim.is_paused:
            self.sim.step(self._steps_per_cmd)

        stamp = ros_bridge.sim_time_to_msg(self.sim.sim_time)
        self.clock_pub.publish(Clock(clock=stamp))
        self.joint_pub.publish(ros_bridge.joint_state_msg(self.sim, stamp))
        self.imu_pub.publish(ros_bridge.imu_msg(self.sim, stamp))

        if self._use_viewer:
            if not self.sim.is_viewer_running:
                self._shutdown()
                return
            self.sim.sync()

        if self._render_sleep and self._use_viewer:
            remaining = self._control_period - (time.monotonic() - wall_start)
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------ #
    # Housekeeping: quit / viewer close (no stepping)
    # ------------------------------------------------------------------ #
    def _on_housekeep(self) -> None:
        if self.sim.is_quit_requested:
            self._shutdown()
        elif self._use_viewer and not self.sim.is_viewer_running:
            self._shutdown()

    def _shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info('shutting down simulator_node')
        self.sim.close_viewer()
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        self.sim.close_viewer()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
