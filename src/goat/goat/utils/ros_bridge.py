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

from goat.utils.mujoco_sim import MujocoSim

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------- #
# ROS -> sim   (command interface = sensor_msgs/JointState)
# ---------------------------------------------------------------------- #
def cmd_to_ctrl(msg: JointState, sim: MujocoSim) -> np.ndarray:
    """Map a JointState command to a full-length ctrl vector.

    Uses ``msg.effort`` (torque) for the double_pendulum's motor actuators.
    Names in ``msg.name`` are matched to actuator names; if empty, fields are
    applied in actuator order. Unknown names / length mismatches are warned
    and ignored. Actuators not addressed keep ctrl = 0.
    """

    ctrl = np.zeros(sim.nu, dtype=float)
    effort = list(msg.effort)

    if not msg.name:
        n = min(len(effort), sim.nu)
        if len(effort) != sim.nu:
            logger.warning("cmd effort len %d != nu %d; applying first %d",
                           len(effort), sim.nu, n)
        ctrl[:n] = effort[:n]
        return ctrl

    if len(effort) < len(msg.name):
        logger.warning("cmd has %d names but %d effort values; missing -> 0",
                       len(msg.name), len(effort))

    for i, name in enumerate(msg.name):
        aid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            logger.warning("unknown actuator '%s' in cmd; ignored", name)
            continue
        if i < len(effort):
            ctrl[aid] = effort[i]
    return ctrl
