# 03_tutorial_hinge_joint.py
# 해당 예제는 hinge joint가 포함된 1-DOF pendulum을 시뮬레이션하기 위함.
# qpos, qvel이 joint coordinate로 어떻게 표현되는지 확인하고,
# motor actuator를 통해 data.ctrl로 torque command를 인가해본다.

import time
import numpy as np

import mujoco
import mujoco.viewer

# Hinge Pendulum
xml = """
<mujoco model="hinge_pendulum">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <worldbody>
    <light name="top_light" pos="0 0 3"/>

    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- 고정된 pivot 위치 표시용 구 -->
    <geom name="pivot_marker"
          type="sphere"
          pos="0 0 1.0"
          size="0.04"
          rgba="1.0 0.2 0.2 1"
          contype="0"
          conaffinity="0"/>

    <!-- pendulum body: 이 body가 hinge joint를 기준으로 회전함 -->
    <body name="pendulum" pos="0 0 1.0">
      <joint name="hinge_joint"
             type="hinge"
             axis="0 1 0"
             pos="0 0 0"
             damping="0.05"
             armature="0.01"
             limited="true"
             range="-2.1 2.1"/>

      <geom name="rod"
            type="capsule"
            fromto="0 0 0 0 0 -0.7"
            size="0.025"
            density="500"
            rgba="0.2 0.4 0.8 1"/>

      <geom name="bob"
            type="sphere"
            pos="0 0 -0.7"
            size="0.06"
            mass="0.5"
            rgba="0.1 0.1 0.8 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="hinge_motor"
           joint="hinge_joint"
           gear="1"
           ctrllimited="true"
           ctrlrange="-2.0 2.0"/>
  </actuator>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("timestep:", model.opt.timestep)

# 이름 기반으로 joint / actuator id 가져오기
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_joint")
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_motor")

# joint가 qpos/qvel 배열에서 어디에 위치하는지 확인
joint_qposadr = model.jnt_qposadr[joint_id]
joint_dofadr = model.jnt_dofadr[joint_id]

print("joint_id:", joint_id)
print("actuator_id:", actuator_id)
print("joint_qposadr:", joint_qposadr)
print("joint_dofadr:", joint_dofadr)
print("actuator ctrlrange:", model.actuator_ctrlrange[actuator_id])

# 초기 pendulum angle
initial_q = 0.7  # rad
initial_dq = 0.0


def reset_simulation():
    mujoco.mj_resetData(model, data)

    data.qpos[joint_qposadr] = initial_q
    data.qvel[joint_dofadr] = initial_dq
    data.ctrl[actuator_id] = 0.0

    mujoco.mj_forward(model, data)


reset_simulation()


state = {
    "paused": False,
    "reset_requested": False,
    "show_contact": False,
    "show_joint": False,
    "quit_requested": False,

    # actuator torque command
    "torque_cmd": 0.0,
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
        state["reset_requested"] = True
        print("reset requested")

    elif key == "c":
        state["show_contact"] = not state["show_contact"]
        print("show_contact:", state["show_contact"])
    
    # ============ Joint 관련 키 바인딩 ============= #
    elif key == "j":
        state["show_joint"] = not state["show_joint"]
        print("show_joint:", state["show_joint"])

    elif key == "a":
        state["torque_cmd"] -= 0.2
        print("torque_cmd:", state["torque_cmd"])

    elif key == "d":
        state["torque_cmd"] += 0.2
        print("torque_cmd:", state["torque_cmd"])

    elif key == "s":
        state["torque_cmd"] = 0.0
        print("torque_cmd reset:", state["torque_cmd"])

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")


print_interval = 0.5
print_every = max(1, int(print_interval / model.opt.timestep))
step_count = 0

ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

    while viewer.is_running():
        step_start = time.time()

        # "q" : 시뮬레이션 종료 처리
        if state["quit_requested"]:
            viewer.close()
            break

        # "r" : 시뮬레이션 리셋 처리
        if state["reset_requested"]:
            reset_simulation()
            state["reset_requested"] = False
            step_count = 0
            print("simulation reset")

        # torque command clipping
        torque_cmd = np.clip(state["torque_cmd"], ctrl_min, ctrl_max)
        state["torque_cmd"] = torque_cmd

        # [NOTE] motor actuator에 torque command 입력
        data.ctrl[actuator_id] = torque_cmd

        # GUI visualization option 처리
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                state["show_contact"]
            )

            # joint visualization flag
            if hasattr(mujoco.mjtVisFlag, "mjVIS_JOINT"):
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = int(
                    state["show_joint"]
                )

        # "Space" : 시뮬레이션 pause 처리
        if not state["paused"]:
            mujoco.mj_step(model, data)
            step_count += 1

        if step_count > 0 and step_count % print_every == 0:
            q = data.qpos[joint_qposadr]
            dq = data.qvel[joint_dofadr]
            tau_cmd = data.ctrl[actuator_id]
            tau_applied = data.qfrc_actuator[joint_dofadr]

            print(
                f"time={data.time:.3f}, "
                f"q={q:.4f} rad, "
                f"dq={dq:.4f} rad/s, "
                f"ctrl={tau_cmd:.3f}, "
                f"qfrc_actuator={tau_applied:.3f}"
            )

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("simulation finished")