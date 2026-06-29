"""Unified MuJoCo simulation wrapper (no ROS dependency).

Owns the MjModel/MjData pair and integrates three concerns in one place:
- physics: model load, reset, step, ctrl I/O
- passive viewer: open/sync/close (lazy import so headless needs no GUI)
- key bindings: space (pause), r (reset), q (quit) -> flags only

Being ROS-free means it can be reused from the standalone tutorials and
exercised directly in unit tests.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import mujoco

logger = logging.getLogger(__name__)


@dataclass
class SimConfig:
    model_path: str
    use_viewer: bool = False
    home_keyframe: Optional[str] = None


class MujocoSim:
    """MjModel/MjData owner: physics + passive viewer + key bindings."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        self.data = mujoco.MjData(self.model)

        # Key-binding flags. key_callback runs in the viewer thread while the
        # node consumes them in its own thread, so guard with a lock.
        self._lock = threading.Lock()
        self._paused = False
        self._reset_requested = False
        self._quit_requested = False

        self._viewer = None  # set by open_viewer()

        # Resolve home keyframe id once (<0 -> mj_resetData on reset()).
        self._home_key_id = -1
        if config.home_keyframe:
            self._home_key_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, config.home_keyframe
            )
            if self._home_key_id < 0:
                logger.warning("home_keyframe '%s' not found; using mj_resetData",
                               config.home_keyframe)

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    @property
    def timestep(self) -> float:
        return self.model.opt.timestep

    @property
    def sim_time(self) -> float:
        return self.data.time

    @property
    def nq(self) -> int:
        return self.model.nq

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nu(self) -> int:
        return self.model.nu

    def reset(self) -> None:
        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def set_ctrl(self, values) -> None:
        """Write a full-length ctrl vector, clipped to each actuator's range."""
        ctrl = np.asarray(values, dtype=float)
        if ctrl.shape != (self.nu,):
            raise ValueError(f"ctrl length {ctrl.shape} != nu ({self.nu})")
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        limited = self.model.actuator_ctrllimited.astype(bool)
        self.data.ctrl[:] = np.where(limited, np.clip(ctrl, low, high), ctrl)

    def joint_names(self) -> List[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]

    def actuator_names(self) -> List[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

    # ------------------------------------------------------------------ #
    # Inspection (see tutorials/04_tutorial_model_data_inspection.py)
    # ------------------------------------------------------------------ #
    def inspect(self) -> str:
        """Build + log a one-shot report of model/data structure.

        Logs joint and actuator ordering (the basis for name<->index mapping
        used by ros_bridge), plus ctrl index mapping and ranges.
        """
        mujoco.mj_forward(self.model, self.data)
        model_name = self.model.names.split(b"\x00")[0].decode(errors="ignore")
        lines = [
            "===== MujocoSim inspection =====",
            f"model='{model_name}' nq={self.nq} nv={self.nv} nu={self.nu} "
            f"njnt={self.model.njnt} timestep={self.timestep}",
            "--- joints (order) ---",
        ]
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            lines.append(
                f"  [{jid}] {name} type={int(self.model.jnt_type[jid])} "
                f"qposadr={self.model.jnt_qposadr[jid]} "
                f"dofadr={self.model.jnt_dofadr[jid]}"
            )
        lines.append("--- actuators (ctrl index order) ---")
        for aid in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            lines.append(
                f"  ctrl[{aid}] {name} "
                f"ctrlrange={self.model.actuator_ctrlrange[aid]} "
                f"gear={self.model.actuator_gear[aid][0]}"
            )
        report = "\n".join(lines)
        logger.info("\n%s", report)
        return report

    # ------------------------------------------------------------------ #
    # Viewer (lazy import: headless runs need no GUI bindings)
    # ------------------------------------------------------------------ #
    def open_viewer(self) -> None:
        import mujoco.viewer  # noqa: PLC0415 (intentional lazy import)
        self._viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        )

    @property
    def is_viewer_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def sync(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def close_viewer(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------ #
    # Key bindings -- set flags only (runs in the viewer thread)
    # ------------------------------------------------------------------ #
    def _key_callback(self, keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except ValueError:
            return
        with self._lock:
            if key == " ":
                self._paused = not self._paused
                logger.info("paused: %s", self._paused)
            elif key == "r":
                self._reset_requested = True
                logger.info("reset requested")
            elif key == "q":
                self._quit_requested = True
                logger.info("quit requested")

    # ------------------------------------------------------------------ #
    # Flag consumption (called from the node thread)
    # ------------------------------------------------------------------ #
    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def is_quit_requested(self) -> bool:
        with self._lock:
            return self._quit_requested

    def consume_reset_request(self) -> bool:
        with self._lock:
            if self._reset_requested:
                self._reset_requested = False
                return True
            return False
