"""Walker walk on the MuJoCo CPU backend, for non-episodic online learning."""

from typing import Any, Mapping

import mujoco
import numpy as np
from mujoco_playground._src.dm_control_suite import common, walker

from ss2r.benchmark_suites.mujoco_playground.walker import recovery

_TORSO_ID = 1
# Of the stand and move rewards, see mujoco_playground's walker.
_STAND_HEIGHT = 1.2


def _gaussian_tolerance(distance: float) -> float:
    # rewards.tolerance outside of the bounds, gaussian sigmoid with the default
    # value_at_margin of 0.1. `distance` is relative to the margin.
    return 0.1 ** (distance**2)


def _linear_tolerance(distance: float, value_at_margin: float) -> float:
    # rewards.tolerance outside of the bounds, linear sigmoid.
    scaled = distance * (1.0 - value_at_margin)
    return 1.0 - scaled if scaled < 1.0 else 0.0


class WalkerWalkCPU:
    """Same model, observation and task reward as playground's WalkerWalk.

    An episode never ends, unless the walker falls over. In this case
    `terminated` is returned and the caller should reset.
    """

    def __init__(
        self,
        ctrl_dt: float = 0.025,
        sim_dt: float = 0.0025,
        hard_upright: float = recovery.HARD_UPRIGHT,
        recovery_upright: float = recovery.RECOVERY_UPRIGHT,
        recovery_pitch_rate: float = recovery.RECOVERY_PITCH_RATE,
        recovery_leg_upright: float = recovery.RECOVERY_LEG_UPRIGHT,
        reset_noise: float = 0.01,
        action_repeat: int = 1,
        real_params: Mapping[str, float] | None = None,
        seed: int = 0,
    ):
        self.model = mujoco.MjModel.from_xml_string(
            walker._XML_PATH.read_text(), common.get_assets()
        )
        self.model.opt.timestep = sim_dt
        self._apply_real_params(real_params or {})
        self.data = mujoco.MjData(self.model)
        self._n_substeps = int(round(ctrl_dt / sim_dt))
        self._action_repeat = action_repeat
        self._hard_upright = hard_upright
        self._recovery_upright = recovery_upright
        self._recovery_pitch_rate = recovery_pitch_rate
        self._recovery_leg_upright = recovery_leg_upright
        self._reset_noise = reset_noise
        self._rng = np.random.default_rng(seed)
        self.observation_size = 2 * (self.model.nbody - 1) + 1 + self.model.nv
        self.action_size = self.model.nu
        self.dt = ctrl_dt * action_repeat

    def _apply_real_params(self, params: Mapping[str, float]) -> None:
        """Perturbs the nominal model, with the semantics of the sim randomization."""
        model = self.model
        model.body_ipos[_TORSO_ID, -1] += params.get("torso_length", 0.0) / 2.0
        model.geom_friction[:, 0] += params.get("friction", 0.0)
        model.dof_damping[3:] += params.get("joint_damping", 0.0)
        model.actuator_gear[:, 0] += params.get("gear", 0.0)

    def reset(self) -> np.ndarray:
        """Standing upright, with a little noise on the joint angles."""
        mujoco.mj_resetData(self.model, self.data)
        noise = self._reset_noise * self._rng.normal(size=self.model.nq - 3)
        self.data.qpos[3:] = noise
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def get_state(self) -> dict[str, np.ndarray]:
        """The state of the simulation, as taken by `set_state`."""
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        reward = 0.0
        for _ in range(self._action_repeat):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data, self._n_substeps)
            reward += self._task_reward()
        obs = self._get_obs()
        terminated = bool(recovery.violates(obs, self._hard_upright))
        info: dict[str, Any] = {
            "in_recovery_set": bool(
                recovery.in_recovery_set(
                    obs,
                    self._recovery_upright,
                    self._recovery_pitch_rate,
                    self._recovery_leg_upright,
                )
            ),
        }
        return obs, reward, terminated, info

    def _get_obs(self) -> np.ndarray:
        xmat = self.data.xmat.reshape(-1, 3, 3)
        orientations = xmat[1:, 0, [0, 2]].ravel()
        upright = xmat[_TORSO_ID, 2, 2]
        return np.hstack([orientations, upright, self.data.qvel]).astype(np.float32)

    def _task_reward(self) -> float:
        """The walk reward of playground's walker."""
        xmat = self.data.xmat.reshape(-1, 3, 3)
        torso_height = self.data.xpos[_TORSO_ID, 2]
        standing = (
            1.0
            if torso_height >= _STAND_HEIGHT
            else _gaussian_tolerance(
                (_STAND_HEIGHT - torso_height) / (_STAND_HEIGHT / 2)
            )
        )
        upright = (1 + xmat[_TORSO_ID, 2, 2]) / 2
        stand_reward = (3 * standing + upright) / 4
        speed = walker.WALK_SPEED
        velocity = self.data.sensor("torso_subtreelinvel").data[0]
        move_reward = (
            1.0
            if velocity >= speed
            else _linear_tolerance((speed - velocity) / (speed / 2), 0.5)
        )
        return float(stand_reward * (5 * move_reward + 1) / 6)
