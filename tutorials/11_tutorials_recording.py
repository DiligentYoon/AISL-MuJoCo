# 11_tutorial_recording.py
# 해당 예제는 MuJoCo viewer camera 제어와 observer-view recording을 설명하기 위함.
# 범위:
#   13-A. viewer free/tracking camera 제어
#   13-B. viewer camera와 유사한 observer view를 mujoco.Renderer로 녹화
# 카메라 센서/robot-mounted camera/depth/segmentation은 다루지 않는다.

import os
import time
from datetime import datetime

import numpy as np

import mujoco
import mujoco.viewer


xml = """
<mujoco model="double_pendulum_recording">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

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
TRACK_BODY_NAMES = ["link1", "link2"]

joint_dofadr = {name: model.joint(name).dofadr[0] for name in JOINT_NAMES}
ctrlrange = {name: model.actuator(name).ctrlrange.copy() for name in ACTUATOR_NAMES}
keyframe_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    for name in KEYFRAME_NAMES
}
body_ids = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in TRACK_BODY_NAMES
}


# -----------------------------
# State / utilities
# -----------------------------
state = {
    "paused": False,
    "reset_requested": "wide_open",
    "quit_requested": False,
    "show_joint": False,
    "show_contact": False,

    # Camera state
    "camera_mode": "free",  # free, track_link1, track_link2

    # Recording state
    "recording": False,
    "save_requested": False,
    "clear_recording_requested": False,
    "frames": [],
    "record_fps": 30,
    "record_width": 960,
    "record_height": 540,
    "record_dir": "recordings",

    # Actuator torque command
    "torque_cmd": {
        "shoulder_motor": 0.0,
        "elbow_motor": 0.0,
    },
}

record_stride = max(1, int(round(1.0 / (state["record_fps"] * model.opt.timestep))))


def reset_torque_cmd():
    for actuator_name in ACTUATOR_NAMES:
        state["torque_cmd"][actuator_name] = 0.0


def print_torque_cmd():
    print(
        "torque_cmd:",
        f"shoulder={state['torque_cmd']['shoulder_motor']:.3f},",
        f"elbow={state['torque_cmd']['elbow_motor']:.3f}",
    )


def reset_to_keyframe(keyframe_name):
    key_id = keyframe_ids[keyframe_name]
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    reset_torque_cmd()
    mujoco.mj_forward(model, data)
    print(f"reset to keyframe: {keyframe_name}")


def print_camera_info(viewer):
    with viewer.lock():
        print("\n========== Viewer Camera Info ==========")
        print("camera_mode:", state["camera_mode"])
        print("cam.type:", int(viewer.cam.type))
        print("cam.trackbodyid:", int(viewer.cam.trackbodyid))
        print("cam.lookat:", viewer.cam.lookat.copy())
        print("cam.distance:", float(viewer.cam.distance))
        print("cam.azimuth:", float(viewer.cam.azimuth))
        print("cam.elevation:", float(viewer.cam.elevation))


def apply_camera_mode(viewer):
    """Apply selected observer camera mode to the GUI viewer."""
    with viewer.lock():
        if state["camera_mode"] == "free":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0.0, 0.0, 0.55]
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -20

        elif state["camera_mode"] == "track_link1":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = body_ids["link1"]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20

        elif state["camera_mode"] == "track_link2":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = body_ids["link2"]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20

        else:
            raise ValueError(f"Unknown camera mode: {state['camera_mode']}")

    print("camera_mode:", state["camera_mode"])


def copy_viewer_camera(viewer, target_camera):
    """Copy GUI viewer camera state into a separate MjvCamera for offscreen rendering."""
    with viewer.lock():
        target_camera.type = viewer.cam.type
        target_camera.fixedcamid = viewer.cam.fixedcamid
        target_camera.trackbodyid = viewer.cam.trackbodyid
        target_camera.lookat[:] = viewer.cam.lookat[:]
        target_camera.distance = viewer.cam.distance
        target_camera.azimuth = viewer.cam.azimuth
        target_camera.elevation = viewer.cam.elevation


def capture_frame(renderer, viewer, render_camera):
    """Render one RGB frame using a camera copied from the GUI viewer."""
    copy_viewer_camera(viewer, render_camera)

    # MuJoCo Renderer accepts an MjvCamera in recent Python bindings.
    # Fallback to default camera if the local binding does not accept it.
    try:
        renderer.update_scene(data, camera=render_camera)
    except TypeError:
        renderer.update_scene(data)

    pixels = renderer.render()
    return pixels.copy()


def save_recording():
    if not state["frames"]:
        print("no frames to save")
        return

    os.makedirs(state["record_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(state["record_dir"], f"mujoco_recording_{timestamp}.mp4")

    try:
        import imageio.v2 as imageio
    except ImportError:
        fallback_path = os.path.join(state["record_dir"], f"mujoco_recording_{timestamp}.npz")
        np.savez_compressed(fallback_path, frames=np.asarray(state["frames"], dtype=np.uint8))
        print("imageio is not installed, saved compressed frames instead:", fallback_path)
        print("install with: pip install imageio imageio-ffmpeg")
        return

    imageio.mimsave(path, state["frames"], fps=state["record_fps"])
    print(f"saved recording: {path}  ({len(state['frames'])} frames, {state['record_fps']} fps)")


def print_key_help():
    print("\n========== Key Bindings ==========")
    print("Space : pause / resume")
    print("R     : reset to wide_open keyframe")
    print("1~4   : reset to hanging / wide_open / folded / moving_test")
    print("A/D   : shoulder motor torque - / +")
    print("Z/X   : elbow motor torque - / +")
    print("S     : zero all motor commands")
    print("J     : joint visualization on/off")
    print("C     : contact point visualization on/off")
    print("F     : free observer camera")
    print("T     : tracking camera on link2")
    print("G     : tracking camera on link1")
    print("K     : print current camera info")
    print("V     : start/stop recording")
    print("L     : clear recorded frames")
    print("Q     : quit")


# -----------------------------
# Keyboard callback
# -----------------------------
def key_callback(keycode):
    try:
        key = chr(keycode).lower()
    except ValueError:
        return

    if key == " ":
        state["paused"] = not state["paused"]
        print("paused:", state["paused"])

    elif key == "r":
        state["reset_requested"] = "wide_open"
        print("reset requested: wide_open")

    elif key == "1":
        state["reset_requested"] = "hanging"
        print("reset requested: hanging")

    elif key == "2":
        state["reset_requested"] = "wide_open"
        print("reset requested: wide_open")

    elif key == "3":
        state["reset_requested"] = "folded"
        print("reset requested: folded")

    elif key == "4":
        state["reset_requested"] = "moving_test"
        print("reset requested: moving_test")

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

    elif key == "f":
        state["camera_mode"] = "free"
        print("camera mode request: free")

    elif key == "t":
        state["camera_mode"] = "track_link2"
        print("camera mode request: track_link2")

    elif key == "g":
        state["camera_mode"] = "track_link1"
        print("camera mode request: track_link1")

    elif key == "k":
        # handled in main loop, because viewer object is only available there.
        state["print_camera_requested"] = True

    elif key == "v":
        if state["recording"]:
            state["recording"] = False
            state["save_requested"] = True
            print("recording stopped, save requested")
        else:
            state["frames"] = []
            state["recording"] = True
            print("recording started")

    elif key == "l":
        state["clear_recording_requested"] = True
        print("clear recorded frames requested")

    elif key == "h":
        print_key_help()

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")


# Extra key state initialized after callback definition for readability.
state["print_camera_requested"] = False


# -----------------------------
# Initialize
# -----------------------------
reset_to_keyframe("wide_open")
print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu)
print("record_fps:", state["record_fps"], "record_stride:", record_stride)
print("recording resolution:", state["record_width"], "x", state["record_height"])
print_key_help()

print_interval = 1.0
print_every = max(1, int(print_interval / model.opt.timestep))
step_count = 0
last_camera_mode = None

# Offscreen renderer used only for recording frames.
renderer = mujoco.Renderer(
    model,
    height=state["record_height"],
    width=state["record_width"],
)
render_camera = mujoco.MjvCamera()


# -----------------------------
# Main viewer loop
# -----------------------------
try:
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        apply_camera_mode(viewer)
        last_camera_mode = state["camera_mode"]

        while viewer.is_running():
            step_start = time.time()

            if state["quit_requested"]:
                if state["recording"]:
                    state["recording"] = False
                    state["save_requested"] = True
                break

            if state["reset_requested"] is not None:
                reset_to_keyframe(state["reset_requested"])
                state["reset_requested"] = None
                step_count = 0

            if state["camera_mode"] != last_camera_mode:
                apply_camera_mode(viewer)
                last_camera_mode = state["camera_mode"]

            if state["print_camera_requested"]:
                print_camera_info(viewer)
                state["print_camera_requested"] = False

            if state["clear_recording_requested"]:
                state["frames"] = []
                state["clear_recording_requested"] = False
                print("recorded frames cleared")

            # torque command clipping + data.ctrl input
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

            # Capture RGB frame from observer camera.
            if state["recording"] and step_count % record_stride == 0:
                frame = capture_frame(renderer, viewer, render_camera)
                state["frames"].append(frame)

                if len(state["frames"]) % state["record_fps"] == 0:
                    print(f"recording... frames={len(state['frames'])}, sim_time={data.time:.3f}")

            if state["save_requested"]:
                save_recording()
                state["save_requested"] = False

            if step_count > 0 and step_count % print_every == 0:
                q1 = data.joint("shoulder_hinge").qpos[0]
                dq1 = data.joint("shoulder_hinge").qvel[0]
                q2 = data.joint("elbow_hinge").qpos[0]
                dq2 = data.joint("elbow_hinge").qvel[0]

                print(
                    f"time={data.time:.3f}, "
                    f"q1={q1:.4f}, dq1={dq1:.4f}, "
                    f"q2={q2:.4f}, dq2={dq2:.4f}, "
                    f"camera={state['camera_mode']}, "
                    f"recording={state['recording']}, frames={len(state['frames'])}"
                )

            viewer.sync()

            # real-time pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

finally:
    renderer.close()

print("simulation finished")
