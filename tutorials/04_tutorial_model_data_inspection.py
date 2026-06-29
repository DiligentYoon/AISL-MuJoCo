# 04_tutorial_double_pendulum_inspection.py
# 해당 예제는 물리 객체를 double pendulum으로 확장한 버전
# nested body 구조, 2개의 hinge joint, 2개의 motor actuator를 inspection하고,
# 각 joint/actuator의 qpos, qvel, ctrl mapping을 확인하는 전체적인 예제

import time
import numpy as np

import mujoco
import mujoco.viewer


xml = """
<mujoco model="double_pendulum">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <worldbody>
    <light name="top_light" pos="0 0 3"/>

    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- 고정된 첫 번째 pivot 위치 표시용 구 -->
    <geom name="pivot_marker"
          type="sphere"
          pos="0 0 1.2"
          size="0.04"
          rgba="1.0 0.2 0.2 1"
          contype="0"
          conaffinity="0"/>

    <!-- link1 body: world에 대해 shoulder_hinge를 기준으로 회전 -->
    <body name="link1" pos="0 0 1.2">
      <joint name="shoulder_hinge"
             type="hinge"
             axis="0 1 0"
             pos="0 0 0"
             damping="0.05"
             armature="0.01"
             limited="true"
             range="-2.8 2.8"/>

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

      <!-- 두 번째 pivot 위치 표시용 구: link1 끝단에 고정 -->
      <geom name="elbow_marker"
            type="sphere"
            pos="0 0 -0.55"
            size="0.035"
            rgba="1.0 0.6 0.1 1"
            contype="0"
            conaffinity="0"/>

      <!-- link2 body: link1 끝단에 붙어서 elbow_hinge를 기준으로 상대 회전 -->
      <body name="link2" pos="0 0 -0.55">
        <joint name="elbow_hinge"
               type="hinge"
               axis="0 1 0"
               pos="0 0 0"
               damping="0.03"
               armature="0.005"
               limited="true"
               range="-2.8 2.8"/>

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
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Model내에 포함된 components들의 name
JOINT_NAMES = ["shoulder_hinge", "elbow_hinge"]
ACTUATOR_NAMES = ["shoulder_motor", "elbow_motor"]
GEOM_NAMES_FOR_EXAMPLE = ["link1_bob", "link2_bob"]
BODY_NAMES_FOR_EXAMPLE = ["link1", "link2"]


def _model_name(model):
    try:
        return model.names.decode(errors="ignore").split("\x00")[0]
    except AttributeError:
        return str(model.names).split("\x00")[0]


def _joint_dims(model, joint_id):
    joint_type = int(model.jnt_type[joint_id])

    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4, 3
    if joint_type in [int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)]:
        return 1, 1

    raise ValueError(f"Unsupported joint type: {joint_type}")


def print_basic_model_info(model):
    print("\n========== Basic Model Info ==========")
    print("model name:", _model_name(model))
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("nbody:", model.nbody)
    print("njnt:", model.njnt)
    print("ngeom:", model.ngeom)
    print("nsite:", model.nsite)
    print("nsensor:", model.nsensor)
    print("timestep:", model.opt.timestep)


def print_body_info(model, data):
    print("\n========== Body Info ==========")
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = f"unnamed_body_{body_id}"

        parent_id = model.body_parentid[body_id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        if parent_name is None:
            parent_name = "None"

        mass = model.body_mass[body_id]

        print(
            f"[body {body_id}] "
            f"name={body_name}, "
            f"parent_id={parent_id}({parent_name}), "
            f"mass={mass:.4f}, "
            f"xpos={data.xpos[body_id].copy()}"
        )


def print_joint_info(model, data):
    print("\n========== Joint Info ==========")
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            joint_name = f"unnamed_joint_{joint_id}"

        joint_type = model.jnt_type[joint_id]
        qposadr = model.jnt_qposadr[joint_id]
        dofadr = model.jnt_dofadr[joint_id]
        qpos_dim, dof_dim = _joint_dims(model, joint_id)
        joint_axis = model.jnt_axis[joint_id]
        joint_range = model.jnt_range[joint_id]

        qpos_slice = data.qpos[qposadr:qposadr + qpos_dim]
        qvel_slice = data.qvel[dofadr:dofadr + dof_dim]

        print(
            f"[joint {joint_id}] "
            f"name={joint_name}, "
            f"type={int(joint_type)}, "
            f"qposadr={qposadr}, "
            f"dofadr={dofadr}, "
            f"qpos_dim={qpos_dim}, "
            f"dof_dim={dof_dim}, "
            f"axis={joint_axis.copy()}, "
            f"range={joint_range.copy()}"
        )
        print(f"    qpos value = {qpos_slice.copy()}")
        print(f"    qvel value = {qvel_slice.copy()}")


def print_geom_info(model, data):
    print("\n========== Geom Info ==========")
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is None:
            geom_name = f"unnamed_geom_{geom_id}"

        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_type = model.geom_type[geom_id]
        rgba = model.geom_rgba[geom_id]

        print(
            f"[geom {geom_id}] "
            f"name={geom_name}, "
            f"type={int(geom_type)}, "
            f"body={body_name}, "
            f"xpos={data.geom_xpos[geom_id].copy()}, "
            f"rgba={rgba.copy()}"
        )


def print_actuator_info(model, data):
    print("\n========== Actuator Info ==========")
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if actuator_name is None:
            actuator_name = f"unnamed_actuator_{actuator_id}"

        ctrlrange = model.actuator_ctrlrange[actuator_id]
        gear = model.actuator_gear[actuator_id]

        print(
            f"[actuator {actuator_id}] "
            f"name={actuator_name}, "
            f"ctrl_index={actuator_id}, "
            f"ctrlrange={ctrlrange.copy()}, "
            f"gear={gear.copy()}, "
            f"ctrl_value={data.ctrl[actuator_id]}"
        )


def print_named_access_examples(model, data):
    print("\n========== Named Access Examples ==========")

    for joint_name in JOINT_NAMES:
        print(f"model.joint('{joint_name}').qposadr:", model.joint(joint_name).qposadr)
        print(f"model.joint('{joint_name}').dofadr:", model.joint(joint_name).dofadr)
        print(f"data.joint('{joint_name}').qpos:", data.joint(joint_name).qpos)
        print(f"data.joint('{joint_name}').qvel:", data.joint(joint_name).qvel)

    for actuator_name in ACTUATOR_NAMES:
        print(f"model.actuator('{actuator_name}').ctrlrange:", model.actuator(actuator_name).ctrlrange)
        print(f"data.actuator('{actuator_name}').ctrl:", data.actuator(actuator_name).ctrl)

    for geom_name in GEOM_NAMES_FOR_EXAMPLE:
        print(f"data.geom('{geom_name}').xpos:", data.geom(geom_name).xpos)

    for body_name in BODY_NAMES_FOR_EXAMPLE:
        print(f"data.body('{body_name}').xpos:", data.body(body_name).xpos)


def inspect_model_and_data(model, data):
    # xpos, geom_xpos 같은 값은 mj_forward 이후에 확실히 갱신된다.
    mujoco.mj_forward(model, data)

    print_basic_model_info(model)
    print_body_info(model, data)
    print_joint_info(model, data)
    print_geom_info(model, data)
    print_actuator_info(model, data)
    print_named_access_examples(model, data)


# 초기 상태 세팅
joint_qposadr = {name: model.joint(name).qposadr[0] for name in JOINT_NAMES}
joint_dofadr = {name: model.joint(name).dofadr[0] for name in JOINT_NAMES}

initial_q = {
    "shoulder_hinge": 0.8,
    "elbow_hinge": -0.6,
}
initial_dq = {
    "shoulder_hinge": 0.0,
    "elbow_hinge": 0.0,
}


def reset_simulation():
    mujoco.mj_resetData(model, data)

    for joint_name in JOINT_NAMES:
        data.qpos[joint_qposadr[joint_name]] = initial_q[joint_name]
        data.qvel[joint_dofadr[joint_name]] = initial_dq[joint_name]

    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


reset_simulation()
inspect_model_and_data(model, data)


state = {
    "paused": False,
    "reset_requested": False,
    "show_contact": False,
    "show_joint": False,
    "quit_requested": False,
    "inspect_requested": False,

    # actuator torque command
    "torque_cmd": {
        "shoulder_motor": 0.0,
        "elbow_motor": 0.0,
    },
}


def print_torque_cmd():
    print(
        "torque_cmd:",
        f"shoulder={state['torque_cmd']['shoulder_motor']:.3f},",
        f"elbow={state['torque_cmd']['elbow_motor']:.3f}"
    )


# 시뮬레이션 루프 제어를 위한 키 바인딩
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

    elif key == "j":
        state["show_joint"] = not state["show_joint"]
        print("show_joint:", state["show_joint"])

    # shoulder motor torque command
    elif key == "a":
        state["torque_cmd"]["shoulder_motor"] -= 0.2
        print_torque_cmd()

    elif key == "d":
        state["torque_cmd"]["shoulder_motor"] += 0.2
        print_torque_cmd()

    # elbow motor torque command
    elif key == "z":
        state["torque_cmd"]["elbow_motor"] -= 0.2
        print_torque_cmd()

    elif key == "x":
        state["torque_cmd"]["elbow_motor"] += 0.2
        print_torque_cmd()

    # 모든 torque command 0으로 reset
    elif key == "s":
        for actuator_name in ACTUATOR_NAMES:
            state["torque_cmd"][actuator_name] = 0.0
        print_torque_cmd()

    # Model description inspection 기능
    elif key == "i":
        state["inspect_requested"] = True
        print("inspection requested")

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")


print_interval = 1.0
print_every = max(1, int(print_interval / model.opt.timestep))
step_count = 0

ctrlrange = {
    name: model.actuator(name).ctrlrange.copy()
    for name in ACTUATOR_NAMES
}


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

        if state["reset_requested"]:
            reset_simulation()
            state["reset_requested"] = False
            step_count = 0
            print("simulation reset")

        if state["inspect_requested"]:
            inspect_model_and_data(model, data)
            state["inspect_requested"] = False

        # torque command clipping + data.ctrl 입력
        for actuator_name in ACTUATOR_NAMES:
            ctrl_min, ctrl_max = ctrlrange[actuator_name]
            clipped_cmd = float(np.clip(
                state["torque_cmd"][actuator_name],
                ctrl_min,
                ctrl_max,
            ))
            state["torque_cmd"][actuator_name] = clipped_cmd
            data.actuator(actuator_name).ctrl[0] = clipped_cmd

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

            print(
                f"time={data.time:.3f}, "
                f"q1={q1:.4f}, dq1={dq1:.4f}, ctrl1={u1:.3f}, tau1={tau1:.3f}, "
                f"q2={q2:.4f}, dq2={dq2:.4f}, ctrl2={u2:.3f}, tau2={tau2:.3f}"
            )

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("simulation finished")
