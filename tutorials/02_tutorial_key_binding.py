# 02_tutorial_key_binding.py
# 해당 예제는 시뮬레이션 루프를 키입력으로 제어하는 기능을 설명하기 위함.
# 키보드 입력을 외부에서 실시간으로 받아 시뮬레이션에 반영함 (정지, 리셋, 재개 등..)

import mujoco
import mujoco.viewer
import time


xml = """
<mujoco model="basic_box">
  <option timestep="0.005" gravity="0 0 -9.81"/>

  <worldbody>
    <light name="top_light" pos="0 0 3"/>

    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>

    <body name="falling_box" pos="0 0 1">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0" rgba="0.2 0.4 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("nq:", model.nq) # Body의 위치(3) 및 자세 (4) 
print("nv:", model.nv) # Body의 선속도(3) 및 각속도 (3)
print("nu:", model.nu) # Actuator (Joint)의 개수에 따라 정해짐
print("timestep:", model.opt.timestep)

state = {
    "paused": False,
    "reset_requested": False,
    "show_contact": False,
    "quit_requested": False,
}

# 시뮬레이션 루프 제어를 위한 키 바인딩
def key_callback(keycode):
    # keycode는 정수로 들어오므로 chr로 변환해서 사용
    key = chr(keycode).lower()

    if key == " ":
        state["paused"] = not state["paused"]
        print("paused:", state["paused"])

    elif key == "r":
        state["reset_requested"] = True
        print("reset requested")

    elif key == "c":
        state["show_contact"] = not state["show_contact"]
        print("show_contact:", state["show_contact"])

    elif key == "q":
        state["quit_requested"] = True
        print("quit requested")

# Key callback을 입력인자로 받아 GUI 포함 시뮬레이션 실행
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20

    start = time.time()

    while viewer.is_running():
        step_start = time.time()

        # "q" : 시뮬레이션 종료 처리
        if state["quit_requested"]:
            viewer.close()
            break
        
        # "r" : 시뮬레이션 리셋 처리
        if state["reset_requested"]:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            state["reset_requested"] = False
            print("simulation reset")

        # "c" : Contact Point 표시 처리
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(state["show_contact"])

        # "Space" : 시뮬레이션 pause 처리
        if not state["paused"]:
            mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    
print("simulation finished")