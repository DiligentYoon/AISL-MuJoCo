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
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState

from goat.utils import ros_bridge
from goat.utils.mujoco_sim import MujocoSim, SimConfig


class SyncSimulatorNode(Node):
    def __init__(self) -> None:
        super().__init__('sync_simulator_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('timestep', 0.0)
        self.declare_parameter('steps_per_cmd', 1)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('render_sleep', True)
        self.declare_parameter('home_keyframe', '')
        self.declare_parameter(
            'joint_order', [],
            ParameterDescriptor(dynamic_typing=True),
        )

        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise RuntimeError('Parameter "model_path" must be set.')
        timestep = float(self.get_parameter('timestep').value)
        if timestep <= 0.0:
            raise RuntimeError('Parameter "timestep" must be set (> 0) from yaml/launch.')
        self._steps_per_cmd = max(1, int(self.get_parameter('steps_per_cmd').value))
        self._use_viewer = bool(self.get_parameter('use_viewer').value)
        self._render_sleep = bool(self.get_parameter('render_sleep').value)
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

        # Wall-clock target for optional render pacing.
        self._control_period = self._steps_per_cmd * self.sim.timestep
        self._shutting_down = False

        # RELIABLE so commands are not dropped: in this synchronous closed loop
        # a dropped /cmd stalls the external controller (deadlock), not data.
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_sub = self.create_subscription(JointState, 'commands', self._on_cmd, cmd_qos)

        # Latched (TRANSIENT_LOCAL, depth 1) so a controller that subscribes
        # AFTER this node has published the initial seed still receives the
        # latest state -- this is what breaks the start-up deadlock: the sim
        # seeds the loop once, the controller wakes on that state and sends the
        # first /cmd. depth 1 keeps only the newest sample (lockstep => the
        # controller never wants stale states).
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.joint_pub = self.create_publisher(JointState, 'sim_joint_states', state_qos)
        self.imu_pub = self.create_publisher(Imu, 'sim_imu', state_qos)
        self.clock_pub = self.create_publisher(Clock, 'clock', state_qos)

        # Seed the loop: publish the post-reset state once WITHOUT stepping so
        # the controller has something to act on. Without this no /cmd ever
        # arrives and the sim never steps (deadlock).
        self._publish_state()

        # Housekeeping only: watch quit/viewer-close. Never advances the sim.
        self._housekeep_timer = self.create_timer(0.1, self._on_housekeep)

        self.get_logger().info(
            f'sync_simulator_node up: model={model_path}, '
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

        self._publish_state()

        if self._use_viewer:
            if not self.sim.is_viewer_running:
                self._shutdown()
                return
            self.sim.sync()

        if self._render_sleep and self._use_viewer:
            remaining = self._control_period - (time.monotonic() - wall_start)
            if remaining > 0:
                time.sleep(remaining)

    def _publish_state(self) -> None:
        """Publish clock + joint/imu state for the current MjData.

        Called both after the initial reset (the loop seed) and after every
        control step, so the wire format is identical in both cases.
        """
        stamp = ros_bridge.sim_time_to_msg(self.sim.sim_time)
        self.clock_pub.publish(Clock(clock=stamp))
        self.joint_pub.publish(ros_bridge.joint_state_msg(self.sim, stamp))
        self.imu_pub.publish(ros_bridge.imu_msg(self.sim, stamp))

    # ------------------------------------------------------------------ #
    # Housekeeping: quit / viewer close (no stepping)
    # ------------------------------------------------------------------ #
    def _on_housekeep(self) -> None:
        if self.sim.is_quit_requested:
            self._shutdown()
        elif self._use_viewer and not self.sim.is_viewer_running:
            self._shutdown()

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
        self.get_logger().info('shutting down sync_simulator_node')
        self.sim.close_viewer()

    def destroy_node(self) -> bool:
        self.sim.close_viewer()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyncSimulatorNode()
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
