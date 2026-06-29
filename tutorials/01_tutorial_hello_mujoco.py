# 02_tutorial_key_binding.py
# 해당 예제는 MuJoCo GUI를 실행하고 간단한 기능 테스트를 수행하기 위함.

import mujoco
import mujoco.viewer
import time

# MuJoCo에 띄울 물리적 객체를 설정하는 파일 (주로 .xml 확장자를 가지며, .mjcf 확장자도 가능함)
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

# Make model and data from xml file
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)  
print("timestep:", model.opt.timestep)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    # Camera 설정
    with viewer.lock():
        # 시점 설정
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        # Contact point 표시
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

    while viewer.is_running():
        step_start = time.time()
        # [필수] Physics step
        mujoco.mj_step(model, data)
        # [필수] Rendered State와 Simulation state 동기화
        viewer.sync()

        # [NOTE: 권장] 시뮬레이션 시간과 실제 Wall-clock time 간의 동기화가 되게끔 하는 time sleep 처리 (시뮬레이션에 대부분 Wall Clock 보다 빠르게 돌기 때문)
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)