# 07_tutorial_friction.py
# MuJoCo friction tutorial.
# Goal:
#   - Compare sliding behavior under different contact friction coefficients.
#   - Inspect contact pairs and contact forces.
#   - Keep the same GUI/key-binding style used in the previous tutorials.
#
# Keys:
#   Space : pause / resume
#   R     : reset all sliders
#   C     : contact point visualization on/off
#   V     : contact force visualization on/off
#   F     : print current contact info
#   1     : set all pair sliding friction to LOW
#   2     : set all pair sliding friction to MEDIUM
#   3     : set all pair sliding friction to HIGH
#   Q     : quit

import time
import numpy as np

import mujoco
import mujoco.viewer


xml = """
<mujoco model="friction_sliding_blocks">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <default>
    <geom condim="3" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>

  <worldbody>
    <light name="top_light" pos="0 0 4"/>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.85 0.85 0.85 1"/>

    <!-- Three identical inclined ramps. Friction is controlled explicitly by <contact><pair>. -->
    <geom name="low_ramp"
          type="box"
          pos="0 -0.8 0.35"
          euler="0 -15 0"
          size="1.2 0.18 0.04"
          rgba="0.7 0.7 0.7 1"/>

    <geom name="mid_ramp"
          type="box"
          pos="0 0.0 0.35"
          euler="0 -15 0"
          size="1.2 0.18 0.04"
          rgba="0.7 0.7 0.7 1"/>

    <geom name="high_ramp"
          type="box"
          pos="0 0.8 0.35"
          euler="0 -15 0"
          size="1.2 0.18 0.04"
          rgba="0.7 0.7 0.7 1"/>

    <!-- Free bodies placed above each ramp. -->
    <body name="low_slider" pos="-0.55 -0.8 0.62">
      <freejoint name="low_freejoint"/>
      <geom name="low_slider_geom"
            type="box"
            size="0.08 0.08 0.08"
            mass="0.5"
            rgba="0.2 0.4 1.0 1"/>
    </body>

    <body name="mid_slider" pos="-0.55 0.0 0.62">
      <freejoint name="mid_freejoint"/>
      <geom name="mid_slider_geom"
            type="box"
            size="0.08 0.08 0.08"
            mass="0.5"
            rgba="0.2 0.8 0.4 1"/>
    </body>

    <body name="high_slider" pos="-0.55 0.8 0.62">
      <freejoint name="high_freejoint"/>
      <geom name="high_slider_geom"
            type="box"
            size="0.08 0.08 0.08"
            mass="0.5"
            rgba="1.0 0.3 0.2 1"/>
    </body>
  </worldbody>

  <!-- Explicit contact pairs make friction interpretation clear. -->
<contact>
  <pair name="low_pair"
        geom1="low_ramp"
        geom2="low_slider_geom"
        condim="3"
        friction="0.05 0.05 0.005 0.0001 0.0001"/>

  <pair name="mid_pair"
        geom1="mid_ramp"
        geom2="mid_slider_geom"
        condim="3"
        friction="0.2 0.2 0.005 0.0001 0.0001"/>

  <pair name="high_pair"
        geom1="high_ramp"
        geom2="high_slider_geom"
        condim="3"
        friction="1.5 1.5 0.005 0.0001 0.0001"/>
</contact>

  <keyframe>
    <key name="initial"
         qpos="-0.55 -0.8 0.62 1 0 0 0   -0.55 0.0 0.62 1 0 0 0   -0.55 0.8 0.62 1 0 0 0"
         qvel="0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

BODY_NAMES = ["low_slider", "mid_slider", "high_slider"]
GEOM_NAMES = ["low_slider_geom", "mid_slider_geom", "high_slider_geom"]
PAIR_NAMES = ["low_pair", "mid_pair", "high_pair"]

key_id_initial = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "initial")
pair_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
    for name in PAIR_NAMES
}

FRICTION_PRESETS = {
    "low": 0.05,
    "medium": 0.40,
    "high": 1.20,
}


def reset_simulation():
    mujoco.mj_resetDataKeyframe(model, data, key_id_initial)
    mujoco.mj_forward(model, data)
    print("simulation reset")


def set_all_pair_sliding_friction(mu):
    for pair_id in pair_ids.values():
        # pair_friction = [tangent1, tangent2, torsional, rolling1, rolling2]
        model.pair_friction[pair_id, 0] = mu
        model.pair_friction[pair_id, 1] = mu
        model.pair_friction[pair_id, 2] = 0.005
        model.pair_friction[pair_id, 3] = 0.0001
        model.pair_friction[pair_id, 4] = 0.0001

    print(f"set all pair tangential friction to mu={mu:.3f}")


def print_model_info():
    print("\n========== Model Info ==========")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("nbody:", model.nbody)
    print("ngeom:", model.ngeom)
    print("npair:", model.npair)
    for name in PAIR_NAMES:
        pair_id = pair_ids[name]
        print(
            f"pair {pair_id}: {name}, friction={model.pair_friction[pair_id].copy()}"
        )


def geom_name(geom_id):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name is not None else f"unnamed_geom_{geom_id}"


def print_contact_info(max_contacts_to_print=20):
    print("\n========== Contact Info ==========")
    print("time:", f"{data.time:.3f}", "ncon:", data.ncon)

    for contact_id in range(min(data.ncon, max_contacts_to_print)):
        contact = data.contact[contact_id]
        force_torque = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, force_torque)

        frame = contact.frame.reshape(3, 3).copy()
        force_contact = force_torque[:3]
        force_world = frame.T @ force_contact

        print(
            f"[contact {contact_id}] "
            f"{geom_name(contact.geom1)} <-> {geom_name(contact.geom2)}, "
            f"dist={contact.dist:.6f}, pos={contact.pos.copy()}"
        )
        print("    force_contact:", force_contact)
        print("    force_world  :", force_world)


def print_slider_state():
    parts = []
    for body_name in BODY_NAMES:
        body = data.body(body_name)
        xpos = body.xpos.copy()
        cvel = body.cvel.copy()
        # cvel convention is spatial velocity; for this tutorial, xpos is the main signal.
        parts.append(
            f"{body_name}: x={xpos[0]:+.3f}, y={xpos[1]:+.3f}, z={xpos[2]:+.3f}"
        )
    print(f"time={data.time:.3f}, ncon={data.ncon}, " + " | ".join(parts))


state = {
    "paused": False,
    "reset_requested": False,
    "quit_requested": False,
    "show_contact": False,
    "show_contact_force": False,
    "contact_print_requested": False,
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

    elif key == "v":
        state["show_contact_force"] = not state["show_contact_force"]
        print("show_contact_force:", state["show_contact_force"])

    elif key == "f":
        state["contact_print_requested"] = True
        print("contact print requested")

    elif key == "1":
        set_all_pair_sliding_friction(FRICTION_PRESETS["low"])

    elif key == "2":
        set_all_pair_sliding_friction(FRICTION_PRESETS["medium"])

    elif key == "3":
        set_all_pair_sliding_friction(FRICTION_PRESETS["high"])

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")


reset_simulation()
print_model_info()

print_interval = 0.5
print_every = max(1, int(print_interval / model.opt.timestep))
step_count = 0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25

    while viewer.is_running():
        step_start = time.time()

        if state["quit_requested"]:
            break

        if state["reset_requested"]:
            reset_simulation()
            state["reset_requested"] = False
            step_count = 0

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                state["show_contact"]
            )
            if hasattr(mujoco.mjtVisFlag, "mjVIS_CONTACTFORCE"):
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(
                    state["show_contact_force"]
                )

        if not state["paused"]:
            mujoco.mj_step(model, data)
            step_count += 1

        if state["contact_print_requested"]:
            print_contact_info()
            state["contact_print_requested"] = False

        if step_count > 0 and step_count % print_every == 0:
            print_slider_state()

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("simulation finished")
