"""Humanoid walk on the MuJoCo CPU backend, for non-episodic online learning."""

from typing import Any, Mapping

import mujoco
import numpy as np
from mujoco_playground._src.dm_control_suite import common, humanoid

from ss2r.benchmark_suites.mujoco_playground.humanoid import recovery
from ss2r.benchmark_suites.mujoco_playground.humanoid.humanoid import _name_to_id


def _gaussian_tolerance(distance: float) -> float:
    # rewards.tolerance outside of the bounds, gaussian sigmoid with the default
    # value_at_margin of 0.1. `distance` is relative to the margin.
    return 0.1 ** (distance**2)


def _linear_tolerance(distance: float) -> float:
    # rewards.tolerance outside of the bounds, linear sigmoid with a
    # value_at_margin of 0.
    return 1.0 - distance if distance < 1.0 else 0.0


class HumanoidWalkCPU:
    """Same model, observation and task reward as playground's HumanoidWalk.

    An episode never ends, unless the humanoid falls over. In this case
    `terminated` is returned and the caller should reset.
    """

    def __init__(
        self,
        ctrl_dt: float = 0.025,
        sim_dt: float = 0.0025,
        hard_upright: float = recovery.HARD_UPRIGHT,
        hard_head_height: float = recovery.HARD_HEAD_HEIGHT,
        recovery_upright: float = recovery.RECOVERY_UPRIGHT,
        recovery_head_height: float = recovery.RECOVERY_HEAD_HEIGHT,
        recovery_tilt_rate: float = recovery.RECOVERY_TILT_RATE,
        reset_noise: float = 0.01,
        action_repeat: int = 1,
        real_params: Mapping[str, Any] | None = None,
        seed: int = 0,
    ):
        self.model = mujoco.MjModel.from_xml_string(
            humanoid._XML_PATH.read_text(), common.get_assets()
        )
        self.model.opt.timestep = sim_dt
        self._apply_real_params(real_params or {})
        self.data = mujoco.MjData(self.model)
        self._n_substeps = int(round(ctrl_dt / sim_dt))
        self._action_repeat = action_repeat
        self._hard_upright = hard_upright
        self._hard_head_height = hard_head_height
        self._recovery_upright = recovery_upright
        self._recovery_head_height = recovery_head_height
        self._recovery_tilt_rate = recovery_tilt_rate
        self._reset_noise = reset_noise
        self._rng = np.random.default_rng(seed)
        self._head_id = self.model.body("head").id
        self._torso_id = self.model.body("torso").id
        self._extremities_ids = [
            self.model.body(side + limb).id
            for side in ("left_", "right_")
            for limb in ("hand", "foot")
        ]
        # All geoms except the floor are capsules or spheres.
        self._body_geoms = np.arange(1, self.model.ngeom)
        self.observation_size = (
            (self.model.nq - 7) + 1 + 3 * len(self._extremities_ids) + 3 + 3
        ) + self.model.nv
        self.action_size = self.model.nu
        self.dt = ctrl_dt * action_repeat

    def _apply_real_params(self, params: Mapping[str, Any]) -> None:
        """Perturbs the nominal model, with the semantics of the sim randomization
        (humanoid.domain_randomization): an offset to the friction of the floor
        and to the gears of the hips and knees, the same for both legs."""
        model = self.model
        friction = model.geom_friction[0, 0] + params.get("friction", 0.0)
        model.geom_friction[0, 0] = np.clip(friction, 0.0, 1.0)
        gear_hip = params.get("gear_hip") or {}
        offsets = {
            "hip_x": gear_hip.get("x", 0.0),
            "hip_y": gear_hip.get("y", 0.0),
            "hip_z": gear_hip.get("z", 0.0),
            "knee": params.get("gear_knee", 0.0),
        }
        for name, actuator_id in _name_to_id.items():
            joint = name.split("_", 1)[1]
            model.actuator_gear[actuator_id, 0] += offsets[joint]

    def _place_on_ground(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        geoms = self._body_geoms
        size = self.model.geom_size[geoms]
        axis = self.data.geom_xmat[geoms].reshape(-1, 3, 3)[:, 2, 2]
        lowest = np.min(
            self.data.geom_xpos[geoms, 2] - np.abs(axis) * size[:, 1] - size[:, 0]
        )
        self.data.qpos[2] -= lowest
        self._forward()

    def _forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        # Warm start the solver as MJX does after a forward, such that the first
        # step matches the simulation (the model only runs 4 solver iterations).
        self.data.qacc_warmstart[:] = self.data.qacc

    def reset(self) -> np.ndarray:
        """Standing upright on the ground, with a little noise on the joint angles."""
        mujoco.mj_resetData(self.model, self.data)
        noise = self._reset_noise * self._rng.normal(size=self.model.nq - 7)
        self.data.qpos[7:] = noise
        self._place_on_ground()
        return self._get_obs()

    def get_state(self) -> dict[str, np.ndarray]:
        """The state of the simulation, as taken by `set_state`."""
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self._forward()
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        reward = 0.0
        for _ in range(self._action_repeat):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data, self._n_substeps)
            reward += self._task_reward(action)
        obs = self._get_obs()
        hard_bounds = (self._hard_upright, self._hard_head_height)
        terminated = bool(recovery.violates(obs, *hard_bounds)) or not bool(
            np.all(np.isfinite(obs))
        )
        info: dict[str, Any] = {
            "in_recovery_set": bool(
                recovery.in_recovery_set(
                    obs,
                    self._recovery_upright,
                    self._recovery_head_height,
                    self._recovery_tilt_rate,
                )
            ),
        }
        return obs, reward, terminated, info

    def _get_obs(self) -> np.ndarray:
        data = self.data
        torso_frame = data.xmat[self._torso_id].reshape(3, 3)
        torso_to_limb = data.xpos[self._extremities_ids] - data.xpos[self._torso_id]
        return np.hstack(
            [
                data.qpos[7:],
                data.xpos[self._head_id, 2],
                (torso_to_limb @ torso_frame).ravel(),
                torso_frame[2],
                data.sensor("torso_subtreelinvel").data,
                data.qvel,
            ]
        ).astype(np.float32)

    def _task_reward(self, action: np.ndarray) -> float:
        """The walk reward of playground's humanoid."""
        stand_height = humanoid._STAND_HEIGHT
        head_height = self.data.xpos[self._head_id, 2]
        standing = (
            1.0
            if head_height >= stand_height
            else _gaussian_tolerance((stand_height - head_height) / (stand_height / 4))
        )
        torso_upright = self.data.xmat[self._torso_id].reshape(3, 3)[2, 2]
        upright = (
            1.0
            if torso_upright >= 0.9
            else _linear_tolerance((0.9 - torso_upright) / 1.9)
        )
        # Quadratic tolerance of the actions with bounds (0, 0) and a margin of 1.
        small_control = np.mean(np.where(np.abs(action) < 1.0, 1.0 - action**2, 0.0))
        small_control = (4.0 + small_control) / 5.0
        speed = humanoid.WALK_SPEED
        velocity = np.linalg.norm(self.data.sensor("torso_subtreelinvel").data[:2])
        move = (
            1.0 if velocity >= speed else _linear_tolerance((speed - velocity) / speed)
        )
        move = (5.0 * move + 1.0) / 6.0
        return float(standing * upright * move * small_control)
