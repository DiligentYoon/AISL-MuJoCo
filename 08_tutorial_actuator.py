# 08_tutorial_actuator.py
# MuJoCo actuator type 비교 튜토리얼.
# motor / position / velocity actuator에서 data.ctrl의 의미가 어떻게 달라지는지 확인한다.

import time
import numpy as np

import mujoco
import mujoco.viewer


xml = """
<mujoco model="actuator_type_comparison">
  <compiler angle="radian"/>

  <!-- kv가 포함된 position/velocity actuator를 비교하므로 implicitfast integrator를 사용한다. -->
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>

  <worldbody>
    <light name="top_light" pos="0 0 3"/>

    <geom name="ground"
          type="plane"
          size="5 5 0.1"
          rgba="0.85 0.85 0.85 1"
          contype="0"
          conaffinity="0"/>

    <!-- 시각적 기준점 -->
    <geom name="motor_pivot_marker"
          type="sphere"
          pos="-1.0 0 1.2"
          size="0.04"
          rgba="1.0 0.2 0.2 1"
          contype="0"
          conaffinity="0"/>

    <geom name="position_pivot_marker"
          type="sphere"
          pos="0.0 0 1.2"
          size="0.04"
          rgba="1.0 0.2 0.2 1"
          contype="0"
          conaffinity="0"/>

    <geom name="velocity_pivot_marker"
          type="sphere"
          pos="1.0 0 1.2"
          size="0.04"
          rgba="1.0 0.2 0.2 1"
          contype="0"
          conaffinity="0"/>

    <!-- 1) motor actuator: data.ctrl이 torque-like command -->
    <body name="motor_pendulum" pos="-1.0 0 1.2">
      <joint name="motor_hinge"
             type="hinge"
             axis="0 1 0"
             damping="0.02"
             armature="0.01"
             limited="true"
             range="-2.8 2.8"/>

      <geom name="motor_rod"
            type="capsule"
            fromto="0 0 0 0 0 -0.6"
            size="0.025"
            density="500"
            rgba="0.2 0.4 1.0 1"/>

      <geom name="motor_bob"
            type="sphere"
            pos="0 0 -0.6"
            size="0.06"
            mass="0.5"
            rgba="0.1 0.2 0.9 1"/>
    </body>

    <!-- 2) position actuator: data.ctrl이 target joint position -->
    <body name="position_pendulum" pos="0.0 0 1.2">
      <joint name="position_hinge"
             type="hinge"
             axis="0 1 0"
             damping="0.02"
             armature="0.01"
             limited="true"
             range="-2.8 2.8"/>

      <geom name="position_rod"
            type="capsule"
            fromto="0 0 0 0 0 -0.6"
            size="0.025"
            density="500"
            rgba="0.2 0.8 0.4 1"/>

      <geom name="position_bob"
            type="sphere"
            pos="0 0 -0.6"
            size="0.06"
            mass="0.5"
            rgba="0.1 0.6 0.2 1"/>
    </body>

    <!-- 3) velocity actuator: data.ctrl이 target joint velocity -->
    <body name="velocity_pendulum" pos="1.0 0 1.2">
      <joint name="velocity_hinge"
             type="hinge"
             axis="0 1 0"
             damping="0.02"
             armature="0.01"
             limited="true"
             range="-2.8 2.8"/>

      <geom name="velocity_rod"
            type="capsule"
            fromto="0 0 0 0 0 -0.6"
            size="0.025"
            density="500"
            rgba="0.9 0.6 0.2 1"/>

      <geom name="velocity_bob"
            type="sphere"
            pos="0 0 -0.6"
            size="0.06"
            mass="0.5"
            rgba="0.8 0.4 0.1 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor_actuator"
           joint="motor_hinge"
           gear="1"
           ctrllimited="true"
           ctrlrange="-2.0 2.0"/>

    <position name="position_actuator"
              joint="position_hinge"
              kp="20"
              kv="2"
              ctrllimited="true"
              ctrlrange="-1.5 1.5"
              forcelimited="true"
              forcerange="-8.0 8.0"/>

    <velocity name="velocity_actuator"
              joint="velocity_hinge"
              kv="5"
              ctrllimited="true"
              ctrlrange="-4.0 4.0"
              forcelimited="true"
              forcerange="-8.0 8.0"/>
  </actuator>

  <keyframe>
    <key name="initial"
         qpos="0.8 0.8 0.8"
         qvel="0.0 0.0 0.0"
         ctrl="0.0 0.0 0.0"/>
  </keyframe>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

JOINT_NAMES = ["motor_hinge", "position_hinge", "velocity_hinge"]
ACTUATOR_NAMES = ["motor_actuator", "position_actuator", "velocity_actuator"]

joint_dofadr = {name: model.joint(name).dofadr[0] for name in JOINT_NAMES}
ctrlrange = {name: model.actuator(name).ctrlrange.copy() for name in ACTUATOR_NAMES}
key_initial_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "initial")


def reset_simulation():
    mujoco.mj_resetDataKeyframe(model, data, key_initial_id)
    mujoco.mj_forward(model, data)


def print_model_info():
    print("\n========== Basic Model Info ==========")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("njnt:", model.njnt)
    print("nbody:", model.nbody)
    print("ngeom:", model.ngeom)
    print("timestep:", model.opt.timestep)
    print("integrator:", model.opt.integrator)

    print("\n========== Joint Mapping ==========")
    for name in JOINT_NAMES:
        print(
            f"{name}: "
            f"qposadr={model.joint(name).qposadr.copy()}, "
            f"dofadr={model.joint(name).dofadr.copy()}, "
            f"q={data.joint(name).qpos.copy()}, "
            f"dq={data.joint(name).qvel.copy()}"
        )

    print("\n========== Actuator Mapping ==========")
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        print(
            f"[{actuator_id}] {name}: "
            f"ctrlrange={model.actuator(name).ctrlrange.copy()}, "
            f"forcerange={model.actuator_forcerange[actuator_id].copy()}, "
            f"gainprm={model.actuator_gainprm[actuator_id].copy()}, "
            f"biasprm={model.actuator_biasprm[actuator_id].copy()}, "
            f"ctrl={data.actuator(name).ctrl.copy()}"
        )


reset_simulation()
print_model_info()

state = {
    "paused": False,
    "reset_requested": False,
    "show_joint": False,
    "quit_requested": False,
    "inspect_requested": False,

    # actuator별 ctrl 의미가 모두 다르다.
    "motor_ctrl": 0.0,          # torque command
    "position_target": 0.0,     # target joint angle [rad]
    "velocity_target": 0.0,     # target joint velocity [rad/s]
}


def clip_commands():
    state["motor_ctrl"] = float(np.clip(
        state["motor_ctrl"],
        ctrlrange["motor_actuator"][0],
        ctrlrange["motor_actuator"][1],
    ))
    state["position_target"] = float(np.clip(
        state["position_target"],
        ctrlrange["position_actuator"][0],
        ctrlrange["position_actuator"][1],
    ))
    state["velocity_target"] = float(np.clip(
        state["velocity_target"],
        ctrlrange["velocity_actuator"][0],
        ctrlrange["velocity_actuator"][1],
    ))


def apply_commands():
    clip_commands()
    data.actuator("motor_actuator").ctrl[0] = state["motor_ctrl"]
    data.actuator("position_actuator").ctrl[0] = state["position_target"]
    data.actuator("velocity_actuator").ctrl[0] = state["velocity_target"]


def reset_commands():
    state["motor_ctrl"] = 0.0
    state["position_target"] = 0.0
    state["velocity_target"] = 0.0


def print_commands():
    print(
        "commands:",
        f"motor_tau_like={state['motor_ctrl']:.3f},",
        f"pos_target={state['position_target']:.3f},",
        f"vel_target={state['velocity_target']:.3f}"
    )


def key_callback(keycode):
    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    if key == " ":
        state["paused"] = not state["paused"]
        print("paused:", state["paused"])

    elif key == "r":
        state["reset_requested"] = True
        print("reset requested")

    elif key == "j":
        state["show_joint"] = not state["show_joint"]
        print("show_joint:", state["show_joint"])

    elif key == "i":
        state["inspect_requested"] = True
        print("inspection requested")

    # motor actuator: torque-like ctrl
    elif key == "a":
        state["motor_ctrl"] -= 0.2
        print_commands()

    elif key == "d":
        state["motor_ctrl"] += 0.2
        print_commands()

    # position actuator: target q ctrl
    elif key == "z":
        state["position_target"] -= 0.1
        print_commands()

    elif key == "x":
        state["position_target"] += 0.1
        print_commands()

    # velocity actuator: target dq ctrl
    elif key == "n":
        state["velocity_target"] -= 0.5
        print_commands()

    elif key == "m":
        state["velocity_target"] += 0.5
        print_commands()

    elif key == "s":
        reset_commands()
        print_commands()

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")


print_interval = 0.5
print_every = max(1, int(print_interval / model.opt.timestep))
step_count = 0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
        viewer.cam.distance = 3.3
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

    while viewer.is_running():
        step_start = time.time()

        if state["quit_requested"]:
            break

        if state["reset_requested"]:
            reset_simulation()
            reset_commands()
            state["reset_requested"] = False
            step_count = 0
            print("simulation reset")

        if state["inspect_requested"]:
            mujoco.mj_forward(model, data)
            print_model_info()
            state["inspect_requested"] = False

        apply_commands()

        with viewer.lock():
            if hasattr(mujoco.mjtVisFlag, "mjVIS_JOINT"):
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = int(state["show_joint"])

        if not state["paused"]:
            mujoco.mj_step(model, data)
            step_count += 1

        if step_count > 0 and step_count % print_every == 0:
            q_motor = data.joint("motor_hinge").qpos[0]
            dq_motor = data.joint("motor_hinge").qvel[0]
            tau_motor = data.qfrc_actuator[joint_dofadr["motor_hinge"]]

            q_pos = data.joint("position_hinge").qpos[0]
            dq_pos = data.joint("position_hinge").qvel[0]
            tau_pos = data.qfrc_actuator[joint_dofadr["position_hinge"]]

            q_vel = data.joint("velocity_hinge").qpos[0]
            dq_vel = data.joint("velocity_hinge").qvel[0]
            tau_vel = data.qfrc_actuator[joint_dofadr["velocity_hinge"]]

            # print(
            #     f"time={data.time:.3f} | "
            #     f"MOTOR q={q_motor:+.3f}, dq={dq_motor:+.3f}, ctrl={data.actuator('motor_actuator').ctrl[0]:+.3f}, tau={tau_motor:+.3f} | "
            #     f"POSITION q={q_pos:+.3f}, dq={dq_pos:+.3f}, target={data.actuator('position_actuator').ctrl[0]:+.3f}, tau={tau_pos:+.3f} | "
            #     f"VELOCITY q={q_vel:+.3f}, dq={dq_vel:+.3f}, target={data.actuator('velocity_actuator').ctrl[0]:+.3f}, tau={tau_vel:+.3f}"
            # )

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("simulation finished")
