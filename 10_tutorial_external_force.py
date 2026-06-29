# 10_tutorial_external_force.py
# MuJoCo external force tutorial.
# 기존 double pendulum + keyframe + sensor 구조를 유지하면서,
# data.xfrc_applied를 이용해 특정 body에 외력/외부 토크를 인가하는 예제.
#
# Key bindings:
#   Space : pause / resume
#   R     : reset to wide_open keyframe
#   1~4   : reset to keyframes
#   A/D   : shoulder motor torque decrease/increase
#   Z/X   : elbow motor torque decrease/increase
#   S     : reset all motor commands
#   J     : joint visualization on/off
#   C     : contact point visualization on/off
#   I     : print model/body/joint/actuator/keyframe info
#   P     : print sensor values
#   O     : compare direct qpos/qvel with joint sensors
#   F/G   : apply +X/-X push force to link2 for a short duration
#   T/Y   : apply +Y/-Y push force to link2 for a short duration
#   B     : apply external torque about Y axis to link2 for a short duration
#   N     : clear external force immediately
#   Q     : quit

import time
import numpy as np

import mujoco
import mujoco.viewer


xml = """
<mujoco model="double_pendulum_external_force">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <worldbody>
    <light name="top_light" pos="0 0 3"/>

    <geom name="ground"
          type="plane"
          size="5 5 0.1"
          rgba="0.8 0.8 0.8 1"/>

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

        <site name="link2_imu_site"
              pos="0 0 -0.10"
              size="0.035"
              rgba="1.0 0.0 1.0 1"/>

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
    <jointpos name="shoulder_pos_sensor" joint="shoulder_hinge"/>
    <jointvel name="shoulder_vel_sensor" joint="shoulder_hinge"/>
    <jointpos name="elbow_pos_sensor" joint="elbow_hinge"/>
    <jointvel name="elbow_vel_sensor" joint="elbow_hinge"/>

    <gyro name="link1_gyro" site="link1_imu_site"/>
    <accelerometer name="link1_acc" site="link1_imu_site"/>
    <velocimeter name="link1_vel" site="link1_imu_site"/>

    <gyro name="link2_gyro" site="link2_imu_site"/>
    <accelerometer name="link2_acc" site="link2_imu_site"/>
    <velocimeter name="link2_vel" site="link2_imu_site"/>

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
BODY_NAMES_FOR_PUSH = ["link1", "link2"]
KEYFRAME_NAMES = ["hanging", "wide_open", "folded", "moving_test"]

joint_dofadr = {name: model.joint(name).dofadr[0] for name in JOINT_NAMES}
ctrlrange = {name: model.actuator(name).ctrlrange.copy() for name in ACTUATOR_NAMES}
body_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in BODY_NAMES_FOR_PUSH
}
keyframe_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    for name in KEYFRAME_NAMES
}


def _name(obj_type, obj_id):
    out = mujoco.mj_id2name(model, obj_type, obj_id)
    return out if out is not None else f"unnamed_{obj_type}_{obj_id}"


def reset_torque_cmd():
    for actuator_name in ACTUATOR_NAMES:
        state["torque_cmd"][actuator_name] = 0.0


def clear_external_force():
    data.xfrc_applied[:, :] = 0.0
    state["push_steps_remaining"] = 0
    state["active_push_body"] = None


def reset_to_keyframe(keyframe_name):
    key_id = keyframe_ids[keyframe_name]
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    reset_torque_cmd()
    clear_external_force()
    mujoco.mj_forward(model, data)
    print(f"reset to keyframe: {keyframe_name}")


def print_basic_info():
    print("\n========== Basic Model Info ==========")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("nbody:", model.nbody)
    print("njnt:", model.njnt)
    print("nsite:", model.nsite)
    print("nsensor:", model.nsensor)
    print("nkey:", model.nkey)
    print("timestep:", model.opt.timestep)

    print("\n========== Body IDs ==========")
    for name, body_id in body_ids.items():
        print(f"{name}: body_id={body_id}")

    print("\n========== Actuator Info ==========")
    for actuator_id in range(model.nu):
        name = _name(mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        print(
            f"[actuator {actuator_id}] name={name}, "
            f"ctrlrange={model.actuator_ctrlrange[actuator_id].copy()}, "
            f"ctrl={data.ctrl[actuator_id]}"
        )

    print("\n========== xfrc_applied ==========")
    for name, body_id in body_ids.items():
        print(f"{name}: {data.xfrc_applied[body_id].copy()}")


def print_sensor_values():
    mujoco.mj_forward(model, data)
    print("\n========== Sensor Values ==========")
    for sensor_id in range(model.nsensor):
        sensor_name = _name(mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        adr = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        print(f"{sensor_name}: {data.sensordata[adr:adr + dim].copy()}")


def print_direct_vs_sensor():
    mujoco.mj_forward(model, data)
    print("\n========== Direct State vs Sensor ==========")
    print(
        "shoulder q  direct/sensor:",
        data.joint("shoulder_hinge").qpos[0],
        data.sensor("shoulder_pos_sensor").data[0],
    )
    print(
        "shoulder dq direct/sensor:",
        data.joint("shoulder_hinge").qvel[0],
        data.sensor("shoulder_vel_sensor").data[0],
    )
    print(
        "elbow q     direct/sensor:",
        data.joint("elbow_hinge").qpos[0],
        data.sensor("elbow_pos_sensor").data[0],
    )
    print(
        "elbow dq    direct/sensor:",
        data.joint("elbow_hinge").qvel[0],
        data.sensor("elbow_vel_sensor").data[0],
    )


def print_torque_cmd():
    print(
        "torque_cmd:",
        f"shoulder={state['torque_cmd']['shoulder_motor']:.3f},",
        f"elbow={state['torque_cmd']['elbow_motor']:.3f}",
    )


def request_push(body_name, force=None, torque=None, duration_s=0.15):
    force = np.zeros(3) if force is None else np.asarray(force, dtype=float)
    torque = np.zeros(3) if torque is None else np.asarray(torque, dtype=float)

    push_steps = max(1, int(duration_s / model.opt.timestep))
    state["push_body_name"] = body_name
    state["push_force"] = force
    state["push_torque"] = torque
    state["push_steps_remaining"] = push_steps
    state["active_push_body"] = body_name

    print(
        f"push requested: body={body_name}, "
        f"force={force}, torque={torque}, duration={duration_s:.3f}s, "
        f"steps={push_steps}"
    )


def apply_external_force_if_requested():
    # xfrc_applied is persistent in mjData, so clear it every loop first.
    data.xfrc_applied[:, :] = 0.0

    if state["push_steps_remaining"] <= 0:
        state["active_push_body"] = None
        return

    body_name = state["push_body_name"]
    body_id = body_ids[body_name]

    data.xfrc_applied[body_id, 0:3] = state["push_force"]
    data.xfrc_applied[body_id, 3:6] = state["push_torque"]
    state["push_steps_remaining"] -= 1


state = {
    "paused": False,
    "show_contact": False,
    "show_joint": False,
    "quit_requested": False,
    "inspect_requested": False,
    "sensor_print_requested": False,
    "compare_requested": False,
    "keyframe_reset_requested": None,
    "torque_cmd": {
        "shoulder_motor": 0.0,
        "elbow_motor": 0.0,
    },
    "push_body_name": "link2",
    "push_force": np.zeros(3),
    "push_torque": np.zeros(3),
    "push_steps_remaining": 0,
    "active_push_body": None,
}

# Initialize model state.
reset_to_keyframe("wide_open")
print_basic_info()


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
        print("reset requested: hanging")

    elif key == "2":
        state["keyframe_reset_requested"] = "wide_open"
        print("reset requested: wide_open")

    elif key == "3":
        state["keyframe_reset_requested"] = "folded"
        print("reset requested: folded")

    elif key == "4":
        state["keyframe_reset_requested"] = "moving_test"
        print("reset requested: moving_test")

    elif key == "c":
        state["show_contact"] = not state["show_contact"]
        print("show_contact:", state["show_contact"])

    elif key == "j":
        state["show_joint"] = not state["show_joint"]
        print("show_joint:", state["show_joint"])

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

    # Push force examples. Forces/torques are in world coordinates.
    elif key == "f":
        request_push("link2", force=[8.0, 0.0, 0.0], duration_s=0.15)

    elif key == "g":
        request_push("link2", force=[-8.0, 0.0, 0.0], duration_s=0.15)

    elif key == "t":
        request_push("link2", force=[0.0, 8.0, 0.0], duration_s=0.15)

    elif key == "y":
        request_push("link2", force=[0.0, -8.0, 0.0], duration_s=0.15)

    elif key == "b":
        request_push("link2", torque=[0.0, 2.0, 0.0], duration_s=0.15)

    elif key == "n":
        clear_external_force()
        print("external force cleared")

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")


print_interval = 0.5
print_every = max(1, int(print_interval / model.opt.timestep))
step_count = 0

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
            print_basic_info()
            state["inspect_requested"] = False

        if state["sensor_print_requested"]:
            print_sensor_values()
            state["sensor_print_requested"] = False

        if state["compare_requested"]:
            print_direct_vs_sensor()
            state["compare_requested"] = False

        # Apply actuator commands.
        for actuator_name in ACTUATOR_NAMES:
            ctrl_min, ctrl_max = ctrlrange[actuator_name]
            clipped_cmd = float(
                np.clip(state["torque_cmd"][actuator_name], ctrl_min, ctrl_max)
            )
            state["torque_cmd"][actuator_name] = clipped_cmd
            data.actuator(actuator_name).ctrl[0] = clipped_cmd

        # Apply external force/torque just before stepping.
        apply_external_force_if_requested()

        with viewer.lock():
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

        if step_count > 0 and step_count % print_every == 0:
            q1 = data.joint("shoulder_hinge").qpos[0]
            dq1 = data.joint("shoulder_hinge").qvel[0]
            q2 = data.joint("elbow_hinge").qpos[0]
            dq2 = data.joint("elbow_hinge").qvel[0]

            u1 = data.actuator("shoulder_motor").ctrl[0]
            u2 = data.actuator("elbow_motor").ctrl[0]

            tau1 = data.qfrc_actuator[joint_dofadr["shoulder_hinge"]]
            tau2 = data.qfrc_actuator[joint_dofadr["elbow_hinge"]]

            link2_id = body_ids["link2"]
            xfrc_link2 = data.xfrc_applied[link2_id].copy()

            # print(
            #     f"time={data.time:.3f}, "
            #     f"q1={q1:.4f}, dq1={dq1:.4f}, ctrl1={u1:.3f}, tau1={tau1:.3f}, "
            #     f"q2={q2:.4f}, dq2={dq2:.4f}, ctrl2={u2:.3f}, tau2={tau2:.3f}, "
            #     f"active_push={state['active_push_body']}, "
            #     f"xfrc_link2={xfrc_link2}"
            # )

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("simulation finished")
