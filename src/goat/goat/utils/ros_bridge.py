"""Conversions between MuJoCo state and ROS2 messages.

This is the only place allowed to depend on ROS message types, keeping
``mujoco_sim`` ROS-free.
"""
from __future__ import annotations

import logging
import mujoco
import numpy as np

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

from goat.utils.mujoco_sim import EFFORT, POSITION, VELOCITY, MujocoSim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# stdlib logging -> ROS (rosout) bridge
# ---------------------------------------------------------------------- #
class _RosLogBridge(logging.Handler):
    """Forward stdlib logging records to a rclpy node's logger (rosout)."""

    def __init__(self, node) -> None:
        super().__init__()
        self._logger = node.get_logger()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        level = record.levelno
        if level >= logging.ERROR:
            self._logger.error(msg)
        elif level >= logging.WARNING:
            self._logger.warn(msg)
        elif level >= logging.INFO:
            self._logger.info(msg)
        else:
            self._logger.debug(msg)


def install_ros_logging_bridge(node, logger_name: str = "goat",
                               level: int = logging.INFO) -> None:
    """Route stdlib logging under ``logger_name`` into the node's rosout logger.

    ``mujoco_sim`` / ``ros_bridge`` log via the stdlib logging module so they
    stay ROS-free. Under a ROS launch those records are otherwise dropped: INFO
    is below the last-resort handler's WARNING threshold and nothing reaches
    rosout. Call this once early in a node's ``__init__`` (before the sim is
    built) so their INFO/WARNING -- e.g. the load-time model inspection and the
    actuator-interface resolver warnings -- show up in the ROS log.
    """
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    log.propagate = False
    if any(isinstance(h, _RosLogBridge) for h in log.handlers):
        return  # idempotent: don't stack a bridge on re-init
    log.addHandler(_RosLogBridge(node))


# Which JointState field supplies the ctrl scalar for each actuator interface.
# Keyed by the interface constants MujocoSim resolves at load time.
_FIELD_BY_INTERFACE = {
    EFFORT: "effort",
    POSITION: "position",
    VELOCITY: "velocity",
}


# ---------------------------------------------------------------------- #
# sim -> ROS
# ---------------------------------------------------------------------- #
def sim_time_to_msg(sim_time: float) -> Time:
    sec = int(sim_time)
    nanosec = int(round((sim_time - sec) * 1e9))
    # guard against rounding to exactly 1e9; keep nanosec an int (Time.nanosec
    # rejects floats) by subtracting an integer, not the float 1e9.
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Time(sec=sec, nanosec=nanosec)


def joint_state_msg(sim: MujocoSim, stamp: Time) -> JointState:
    """Build a JointState from current MjData.

    Joints are emitted in ``sim.publish_joint_ids`` order (the controller
    convention; see SimConfig.joint_order), so state output matches the
    name-based command path. Reads each joint's qpos/qvel via its address, which
    is correct for hinge/slide (1-DoF) joints; free/ball joints span multiple
    DoFs and would need per-type handling -- not addressed by these models.
    """
    model = sim.model
    data = sim.data
    msg = JointState()
    msg.header.stamp = stamp

    names = []
    positions = []
    velocities = []
    for jid in sim.publish_joint_ids:
        names.append(sim_joint_name(sim, jid))
        positions.append(float(data.qpos[model.jnt_qposadr[jid]]))
        velocities.append(float(data.qvel[model.jnt_dofadr[jid]]))
    msg.name = names
    msg.position = positions
    msg.velocity = velocities
    return msg


def sim_joint_name(sim: MujocoSim, jid: int) -> str:
    return mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jid)


def _first_sensor_of_type(sim: MujocoSim, sensor_type):
    """Return the sensordata slice of the first sensor of the given type, or None."""
    model = sim.model
    for sid in range(model.nsensor):
        if model.sensor_type[sid] == int(sensor_type):
            adr = model.sensor_adr[sid]
            dim = model.sensor_dim[sid]
            return sim.data.sensordata[adr:adr + dim]
    return None


def imu_msg(sim: MujocoSim, stamp: Time, frame_id: str = "imu_link") -> Imu:
    """Build a sensor_msgs/Imu from MuJoCo IMU sensors.

    Reads the first ``framequat`` (orientation, MuJoCo order w,x,y,z), ``gyro``
    (angular velocity), and ``accelerometer`` (linear acceleration) sensors
    found in the model. Missing sensors fall back to identity / zero. Define
    these sensors on an IMU site in the MJCF to get meaningful values.
    Covariance left at 0 (unknown) per REP-145 convention.
    """

    msg = Imu()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.orientation.w = 1.0  # identity default

    quat = _first_sensor_of_type(sim, mujoco.mjtSensor.mjSENS_FRAMEQUAT)
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)  # MuJoCo: w,x,y,z -> ROS: x,y,z,w
        msg.orientation.x = x
        msg.orientation.y = y
        msg.orientation.z = z
        msg.orientation.w = w

    gyro = _first_sensor_of_type(sim, mujoco.mjtSensor.mjSENS_GYRO)
    if gyro is not None:
        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])

    acc = _first_sensor_of_type(sim, mujoco.mjtSensor.mjSENS_ACCELEROMETER)
    if acc is not None:
        msg.linear_acceleration.x = float(acc[0])
        msg.linear_acceleration.y = float(acc[1])
        msg.linear_acceleration.z = float(acc[2])

    return msg

def odom_msg(sim: MujocoSim, stamp: Time, frame_id: str = "odom", child_frame_id: str = "base_link") -> Odometry:
    """Build a nav_msgs/Odometry from MuJoCo state.

    Uses the first free joint (base) for pose and velocity. If no free joint is
    present, returns an Odometry message with zero pose and velocity.
    """
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id

    # Find the first free joint (base)
    base_jid = None
    for jid in range(sim.model.njnt):
        if sim.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            base_jid = jid
            break

    if base_jid is not None:
        pos_adr = sim.model.jnt_qposadr[base_jid]
        vel_adr = sim.model.jnt_dofadr[base_jid]

        # Position and orientation
        msg.pose.pose.position.x = float(sim.data.qpos[pos_adr])
        msg.pose.pose.position.y = float(sim.data.qpos[pos_adr + 1])
        msg.pose.pose.position.z = float(sim.data.qpos[pos_adr + 2])

        quat = sim.data.qpos[pos_adr + 3:pos_adr + 7]  # w, x, y, z
        msg.pose.pose.orientation.x = float(quat[1])
        msg.pose.pose.orientation.y = float(quat[2])
        msg.pose.pose.orientation.z = float(quat[3])
        msg.pose.pose.orientation.w = float(quat[0])

        # Linear and angular velocity
        msg.twist.twist.linear.x = float(sim.data.qvel[vel_adr])
        msg.twist.twist.linear.y = float(sim.data.qvel[vel_adr + 1])
        msg.twist.twist.linear.z = float(sim.data.qvel[vel_adr + 2])

        msg.twist.twist.angular.x = float(sim.data.qvel[vel_adr + 3])
        msg.twist.twist.angular.y = float(sim.data.qvel[vel_adr + 4])
        msg.twist.twist.angular.z = float(sim.data.qvel[vel_adr + 5])

    return msg

# ---------------------------------------------------------------------- #
# ROS -> sim   (command interface = sensor_msgs/JointState)
# ---------------------------------------------------------------------- #
def cmd_to_ctrl(msg: JointState, sim: MujocoSim) -> np.ndarray:
    """Map a JointState command to a full-length ctrl vector.

    Each actuator reads its scalar from the single JointState field its
    interface expects (``sim.actuator_interfaces``, resolved from the model):
    EFFORT -> ``msg.effort``, POSITION -> ``msg.position``, VELOCITY ->
    ``msg.velocity``. So a motor gets a torque, a position actuator a target
    angle, a velocity actuator a target speed -- all written raw into ctrl.

    Two addressing modes:
    - named (``msg.name`` set): each name -> actuator id, value read from that
      actuator's field at the *name index* ``i`` (name[i] pairs with field[i]).
    - unnamed: applied in actuator order; actuator ``aid`` reads its field at
      *index aid* (arrays are treated sparse per actuator index for mixed
      models -- fill each actuator's slot in the array its interface uses).

    Unknown names are warned and ignored; actuators whose expected field has no
    value at the needed index keep ctrl = 0 and are summarized in one warning.
    """
    ctrl = np.zeros(sim.nu, dtype=float)
    interfaces = sim.actuator_interfaces
    fields = {
        "effort": list(msg.effort),
        "position": list(msg.position),
        "velocity": list(msg.velocity),
    }
    missing = []  # aggregated so a short/empty cmd warns once, not per actuator

    if not msg.name:
        for aid in range(sim.nu):
            field = _FIELD_BY_INTERFACE[interfaces[aid]]
            arr = fields[field]
            if aid < len(arr):
                ctrl[aid] = arr[aid]
            else:
                missing.append((aid, field))
    else:
        for i, name in enumerate(msg.name):
            aid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                logger.warning("unknown actuator '%s' in cmd; ignored", name)
                continue
            field = _FIELD_BY_INTERFACE[interfaces[aid]]
            arr = fields[field]
            if i < len(arr):
                ctrl[aid] = arr[i]
            else:
                missing.append((name, field))

    if missing:
        logger.warning("cmd missing values for %d actuator(s): %s; those ctrl stay 0",
                       len(missing), missing)
    return ctrl
