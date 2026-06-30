"""Minimal lockstep test commander for the sync simulator.

The sync simulator is lockstep: it advances ONE control step per ``/commands``
message and seeds the loop with a single latched ``/sim_joint_states``. This
node closes that loop so the simulation actually runs without a real
controller: it subscribes to ``/sim_joint_states`` and, for EACH state
received, publishes exactly one ``/commands`` -- a strict 1:1 cadence that keeps
the lockstep contract (one command -> one step -> one state -> one command).

It is a stand-in for a real controller, useful for eyeballing the viewer and
for verifying the end-to-end pipeline. Two open-loop torque profiles:

- ``sine`` (default): torque = amplitude * sin(2*pi*freq*sim_time) per actuator,
  so the pendulum visibly swings.
- ``const``: a fixed torque vector.

QoS mirrors the simulator exactly so discovery matches and the latched seed is
received even though this node starts after the simulator publishes it.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


class TestCommanderNode(Node):
    def __init__(self) -> None:
        super().__init__('test_commander_node')

        # Actuator names must match the model's <actuator> entries so the
        # simulator maps effort -> the right ctrl index (see ros_bridge).
        self.declare_parameter('actuators', ['shoulder_motor', 'elbow_motor'])
        self.declare_parameter('mode', 'sine')          # 'sine' | 'const'
        self.declare_parameter('amplitude', [1.0, 0.6])  # per-actuator torque
        self.declare_parameter('frequency', 0.5)         # Hz, sine only

        self._actuators = list(self.get_parameter('actuators').value)
        self._mode = str(self.get_parameter('mode').value)
        self._amplitude = list(self.get_parameter('amplitude').value)
        self._frequency = float(self.get_parameter('frequency').value)
        if len(self._amplitude) < len(self._actuators):
            # pad so every actuator has an amplitude
            self._amplitude += [0.0] * (len(self._actuators) - len(self._amplitude))

        # Match the simulator's publishers/subscribers exactly.
        cmd_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(JointState, 'commands', cmd_qos)

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_sub = self.create_subscription(
            JointState, 'sim_joint_states', self._on_state, state_qos
        )

        self.get_logger().info(
            f'test_commander up: mode={self._mode}, actuators={self._actuators}, '
            f'amplitude={self._amplitude}, frequency={self._frequency}Hz '
            f'(1 command per received state)'
        )

    def _on_state(self, msg: JointState) -> None:
        sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self._mode == 'const':
            effort = list(self._amplitude[:len(self._actuators)])
        else:  # sine
            w = 2.0 * math.pi * self._frequency
            effort = [a * math.sin(w * sim_time) for a in self._amplitude[:len(self._actuators)]]

        cmd = JointState()
        cmd.header.stamp = msg.header.stamp  # echo sim-time for traceability
        cmd.name = self._actuators
        cmd.effort = effort
        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TestCommanderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
