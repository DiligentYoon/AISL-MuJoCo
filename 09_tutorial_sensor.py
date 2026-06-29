# 09_tutorial_sensor.py
# MuJoCo sensor tutorial
# - jointpos / jointvel sensor
# - gyro / accelerometer / velocimeter sensor attached to sites
# - framepos / framequat sensor
# - data.sensordata layout inspection and named sensor access
#
# This example continues the double-pendulum style tutorial.

import time
import numpy as np

import mujoco
import mujoco.viewer


xml = """
<mujoco model="double_pendulum_sensors">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <worldbody>
    <light name="top_light" pos="0 0 3"/>

    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

    <geom name="pivot_marker"
          type="sphere"
          pos="0 0 1.2"
          size="0.04"
          rgba="1.0 0.2 0.2 1"
          contype="0"
          conaffinity="0"/>

    <body name="link1" pos="0 0 1.2">
      <joint name="shoulder_hinge"
             type="hinge"
             axis="0 1 0"
             pos="0 0 0"
             damping="0.05"
             armature="0.01"
             limited="true"
             range="-2.8 2.8"/>

      <!-- Site used as a simple IMU-like sensor frame on link1. -->
      <site name="link1_imu_site"
            pos="0 0 -0.10"
            size="0.035"
            rgba="1.0 0.0 1.0 1"/>

      <geom name="link1_rod"
            type="capsule"
            fromto="0 0 0 0 0 -0.55"
            size="0.025"
            density="500"
            rgba="0.2 0.4 0.8 1"/>

      <geom name="link1_bob"
            type="sphere"
            pos="0 0 -0.55"
            size="0.055"
            mass="0.4"
            rgba="0.1 0.1 0.8 1"/>

      <geom name="elbow_marker"
            type="sphere"
            pos="0 0 -0.55"
            size="0.035"
            rgba="1.0 0.6 0.1 1"
            contype="0"
            conaffinity="0"/>

      <body name="link2" pos="0 0 -0.55">
        <joint name="elbow_hinge"
               type="hinge"
               axis="0 1 0"
               pos="0 0 0"
               damping="0.03"
               armature="0.005"
               limited="true"
               range="-2.8 2.8"/>

        <!-- Site used as a simple IMU-like sensor frame on link2. -->
        <site name="link2_imu_site"
              pos="0 0 -0.10"
              size="0.030"
              rgba="0.0 1.0 1.0 1"/>

        <geom name="link2_rod"
              type="capsule"
              fromto="0 0 0 0 0 -0.50"
              size="0.022"
              density="450"
              rgba="0.2 0.8 0.4 1"/>

        <geom name="link2_bob"
              type="sphere"
              pos="0 0 -0.50"
              size="0.05"
              mass="0.3"
              rgba="0.1 0.6 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder_motor"
           joint="shoulder_hinge"
           gear="1"
           ctrllimited="true"
           ctrlrange="-2.0 2.0"/>

    <motor name="elbow_motor"
           joint="elbow_hinge"
           gear="1"
           ctrllimited="true"
           ctrlrange="-1.5 1.5"/>
  </actuator>

  <sensor>
    <!-- Scalar joint state sensors. These copy from qpos/qvel for scalar joints. -->
    <jointpos name="shoulder_pos_sensor" joint="shoulder_hinge"/>
    <jointvel name="shoulder_vel_sensor" joint="shoulder_hinge"/>
    <jointpos name="elbow_pos_sensor" joint="elbow_hinge"/>
    <jointvel name="elbow_vel_sensor" joint="elbow_hinge"/>

    <!-- IMU-like site sensors. Values are expressed in the local site frame. -->
    <gyro name="link1_gyro" site="link1_imu_site"/>
    <accelerometer name="link1_acc" site="link1_imu_site"/>
    <velocimeter name="link1_vel" site="link1_imu_site"/>

    <gyro name="link2_gyro" site="link2_imu_site"/>
    <accelerometer name="link2_acc" site="link2_imu_site"/>
    <velocimeter name="link2_vel" site="link2_imu_site"/>

    <!-- Frame sensors: global pose of a site. Useful for inspecting sensor frames. -->
    <framepos name="link1_imu_pos" objtype="site" objname="link1_imu_site"/>
    <framequat name="link1_imu_quat" objtype="site" objname="link1_imu_site"/>
    <framepos name="link2_imu_pos" objtype="site" objname="link2_imu_site"/>
    <framequat name="link2_imu_quat" objtype="site" objname="link2_imu_site"/>
  </sensor>

  <keyframe>
    <key name="hanging"
         qpos="0.0 0.0"
         qvel="0.0 0.0"
         ctrl="0.0 0.0"/>

    <key name="wide_open"
         qpos="0.8 -0.8"
         qvel="0.0 0.0"
         ctrl="0.0 0.0"/>

    <key name="folded"
         qpos="1.2 1.0"
         qvel="0.0 0.0"
         ctrl="0.0 0.0"/>

    <key name="moving_test"
         qpos="0.5 -0.5"
         qvel="1.0 -1.0"
         ctrl="0.0 0.0"/>
  </keyframe>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

JOINT_NAMES = ["shoulder_hinge", "elbow_hinge"]
ACTUATOR_NAMES = ["shoulder_motor", "elbow_motor"]
KEYFRAME_NAMES = ["hanging", "wide_open", "folded", "moving_test"]
SENSOR_NAMES = [
    "shoulder_pos_sensor",
    "shoulder_vel_sensor",
    "elbow_pos_sensor",
    "elbow_vel_sensor",
    "link1_gyro",
    "link1_acc",
    "link1_vel",
    "link2_gyro",
    "link2_acc",
    "link2_vel",
    "link1_imu_pos",
    "link1_imu_quat",
    "link2_imu_pos",
    "link2_imu_quat",
]

joint_qposadr = {name: model.joint(name).qposadr[0] for name in JOINT_NAMES}
joint_dofadr = {name: model.joint(name).dofadr[0] for name in JOINT_NAMES}
ctrlrange = {name: model.actuator(name).ctrlrange.copy() for name in ACTUATOR_NAMES}
keyframe_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    for name in KEYFRAME_NAMES
}


def reset_torque_cmd():
    for actuator_name in ACTUATOR_NAMES:
        state["torque_cmd"][actuator_name] = 0.0


def reset_to_keyframe(keyframe_name):
    key_id = keyframe_ids[keyframe_name]
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    reset_torque_cmd()
    mujoco.mj_forward(model, data)
    print(f"reset to keyframe: {keyframe_name}")


def print_torque_cmd():
    print(
        "torque_cmd:",
        f"shoulder={state['torque_cmd']['shoulder_motor']:.3f},",
        f"elbow={state['torque_cmd']['elbow_motor']:.3f}",
    )


def print_basic_model_info():
    print("\n========== Basic Model Info ==========")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("nbody:", model.nbody)
    print("njnt:", model.njnt)
    print("ngeom:", model.ngeom)
    print("nsite:", model.nsite)
    print("nsensor:", model.nsensor)
    print("nsensordata:", model.nsensordata)
    print("timestep:", model.opt.timestep)


def print_sensor_layout():
    # Make sure sensor outputs are up-to-date.
    mujoco.mj_forward(model, data)

    print("\n========== Sensor Layout ==========")
    print("nsensor:", model.nsensor)
    print("nsensordata:", model.nsensordata)

    for sensor_id in range(model.nsensor):
        sensor_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
        )
        if sensor_name is None:
            sensor_name = f"unnamed_sensor_{sensor_id}"

        adr = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        sensor_type = int(model.sensor_type[sensor_id])
        sensor_data = data.sensordata[adr:adr + dim]

        print(
            f"[sensor {sensor_id}] "
            f"name={sensor_name}, "
            f"type={sensor_type}, "
            f"adr={adr}, "
            f"dim={dim}, "
            f"data={sensor_data.copy()}"
        )


def print_sensor_values():
    mujoco.mj_forward(model, data)

    print("\n========== Sensor Values ==========")
    for name in SENSOR_NAMES:
        print(f"{name:22s}: {data.sensor(name).data.copy()}")


def print_direct_vs_sensor_comparison():
    mujoco.mj_forward(model, data)

    print("\n========== Direct State vs Sensor ==========")
    q_shoulder_direct = data.qpos[joint_qposadr["shoulder_hinge"]]
    dq_shoulder_direct = data.qvel[joint_dofadr["shoulder_hinge"]]
    q_elbow_direct = data.qpos[joint_qposadr["elbow_hinge"]]
    dq_elbow_direct = data.qvel[joint_dofadr["elbow_hinge"]]

    print(
        "shoulder q  direct/sensor:",
        q_shoulder_direct,
        data.sensor("shoulder_pos_sensor").data[0],
    )
    print(
        "shoulder dq direct/sensor:",
        dq_shoulder_direct,
        data.sensor("shoulder_vel_sensor").data[0],
    )
    print(
        "elbow q     direct/sensor:",
        q_elbow_direct,
        data.sensor("elbow_pos_sensor").data[0],
    )
    print(
        "elbow dq    direct/sensor:",
        dq_elbow_direct,
        data.sensor("elbow_vel_sensor").data[0],
    )

    print("link1 gyro local:", data.sensor("link1_gyro").data.copy())
    print("link1 acc  local:", data.sensor("link1_acc").data.copy())
    print("link1 vel  local:", data.sensor("link1_vel").data.copy())
    print("link1 imu global pos :", data.sensor("link1_imu_pos").data.copy())
    print("link1 imu global quat:", data.sensor("link1_imu_quat").data.copy())


def print_keyframe_info():
    print("\n========== Keyframe Info ==========")
    print("nkey:", model.nkey)
    for name in KEYFRAME_NAMES:
        key_id = keyframe_ids[name]
        print(
            f"[{key_id}] {name}: "
            f"qpos={model.key_qpos[key_id].copy()}, "
            f"qvel={model.key_qvel[key_id].copy()}, "
            f"ctrl={model.key_ctrl[key_id].copy()}"
        )


def inspect_all():
    print_basic_model_info()
    print_sensor_layout()
    print_keyframe_info()
    print_direct_vs_sensor_comparison()


state = {
    "paused": False,
    "show_joint": False,
    "show_contact": False,
    "quit_requested": False,
    "inspect_requested": False,
    "sensor_print_requested": False,
    "compare_requested": False,
    "keyframe_reset_requested": "wide_open",
    "torque_cmd": {
        "shoulder_motor": 0.0,
        "elbow_motor": 0.0,
    },
}


def key_callback(keycode):
    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    if key == " ":
        state["paused"] = not state["paused"]
        print("paused:", state["paused"])

    elif key == "r":
        state["keyframe_reset_requested"] = "wide_open"
        print("reset requested: wide_open")

    elif key == "1":
        state["keyframe_reset_requested"] = "hanging"
        print("keyframe reset requested: hanging")

    elif key == "2":
        state["keyframe_reset_requested"] = "wide_open"
        print("keyframe reset requested: wide_open")

    elif key == "3":
        state["keyframe_reset_requested"] = "folded"
        print("keyframe reset requested: folded")

    elif key == "4":
        state["keyframe_reset_requested"] = "moving_test"
        print("keyframe reset requested: moving_test")

    elif key == "j":
        state["show_joint"] = not state["show_joint"]
        print("show_joint:", state["show_joint"])

    elif key == "c":
        state["show_contact"] = not state["show_contact"]
        print("show_contact:", state["show_contact"])

    elif key == "a":
        state["torque_cmd"]["shoulder_motor"] -= 0.2
        print_torque_cmd()

    elif key == "d":
        state["torque_cmd"]["shoulder_motor"] += 0.2
        print_torque_cmd()

    elif key == "z":
        state["torque_cmd"]["elbow_motor"] -= 0.2
        print_torque_cmd()

    elif key == "x":
        state["torque_cmd"]["elbow_motor"] += 0.2
        print_torque_cmd()

    elif key == "s":
        reset_torque_cmd()
        print_torque_cmd()

    elif key == "i":
        state["inspect_requested"] = True
        print("inspection requested")

    elif key == "p":
        state["sensor_print_requested"] = True
        print("sensor print requested")

    elif key == "o":
        state["compare_requested"] = True
        print("direct vs sensor comparison requested")

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")


print_interval = 1.0
print_every = max(1, int(print_interval / model.opt.timestep))
step_count = 0

# Initialize from keyframe before printing.
reset_to_keyframe("wide_open")
inspect_all()


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

    while viewer.is_running():
        step_start = time.time()

        if state["quit_requested"]:
            break

        if state["keyframe_reset_requested"] is not None:
            reset_to_keyframe(state["keyframe_reset_requested"])
            state["keyframe_reset_requested"] = None
            step_count = 0

        if state["inspect_requested"]:
            inspect_all()
            state["inspect_requested"] = False

        # torque command clipping + data.ctrl input
        for actuator_name in ACTUATOR_NAMES:
            ctrl_min, ctrl_max = ctrlrange[actuator_name]
            clipped_cmd = float(np.clip(
                state["torque_cmd"][actuator_name], ctrl_min, ctrl_max
            ))
            state["torque_cmd"][actuator_name] = clipped_cmd
            data.actuator(actuator_name).ctrl[0] = clipped_cmd

        with viewer.lock():
            if hasattr(mujoco.mjtVisFlag, "mjVIS_CONTACTPOINT"):
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                    state["show_contact"]
                )
            if hasattr(mujoco.mjtVisFlag, "mjVIS_JOINT"):
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = int(
                    state["show_joint"]
                )

        if not state["paused"]:
            mujoco.mj_step(model, data)
            step_count += 1
        else:
            # Keep derived quantities and sensor outputs current while paused.
            mujoco.mj_forward(model, data)

        if state["sensor_print_requested"]:
            print_sensor_values()
            state["sensor_print_requested"] = False

        if state["compare_requested"]:
            print_direct_vs_sensor_comparison()
            state["compare_requested"] = False

        if step_count > 0 and step_count % print_every == 0:
            q1 = data.sensor("shoulder_pos_sensor").data[0]
            dq1 = data.sensor("shoulder_vel_sensor").data[0]
            q2 = data.sensor("elbow_pos_sensor").data[0]
            dq2 = data.sensor("elbow_vel_sensor").data[0]
            gyro1 = data.sensor("link1_gyro").data.copy()
            acc1 = data.sensor("link1_acc").data.copy()
            gyro2 = data.sensor("link2_gyro").data.copy()

            print(
                f"time={data.time:.3f}, "
                f"q1={q1:.4f}, dq1={dq1:.4f}, "
                f"q2={q2:.4f}, dq2={dq2:.4f}, "
                f"gyro1={gyro1}, acc1={acc1}, gyro2={gyro2}"
            )

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("simulation finished")
