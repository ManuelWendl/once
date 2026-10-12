"""Cartpole swingup on the MuJoCo CPU backend, for non-episodic online learning."""

from typing import Any, Mapping, Protocol

import mujoco
import numpy as np
from mujoco_playground._src.dm_control_suite import cartpole, common

from ss2r.benchmark_suites.mujoco_playground.cartpole import recovery

_POLE_ID = -1


class RealEnv(Protocol):
    """Interface of the (single, non-episodic) system that ONCE is deployed on."""

    observation_size: int
    action_size: int
    dt: float

    def reset(self) -> np.ndarray:
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        ...


def _gaussian_tolerance(x: float, margin: float) -> float:
    # Equivalent to rewards.tolerance(x, margin=margin) with default arguments.
    return 0.1 ** ((x / margin) ** 2)


class CartpoleSwingupCPU:
    """Same model, observation and task reward as playground's CartpoleSwingup.

    An episode never ends, unless the hard constraint on the cart position is
    violated. In this case `terminated` is returned and the caller should reset.
    """

    def __init__(
        self,
        ctrl_dt: float = 0.01,
        sim_dt: float = 0.01,
        action_repeat: int = 1,
        hard_position_bound: float = recovery.HARD_POSITION_BOUND,
        recovery_position_bound: float = recovery.RECOVERY_POSITION_BOUND,
        recovery_velocity_bound: float = recovery.RECOVERY_VELOCITY_BOUND,
        real_params: Mapping[str, float] | None = None,
        seed: int = 0,
    ):
        self.model = mujoco.MjModel.from_xml_string(
            cartpole._XML_PATH.read_text(), common.get_assets()
        )
        self.model.opt.timestep = sim_dt
        self._apply_real_params(real_params or {})
        self.data = mujoco.MjData(self.model)
        self._n_substeps = int(round(ctrl_dt / sim_dt))
        self._action_repeat = action_repeat
        self._slider_qposadr = self.model.jnt_qposadr[self.model.joint("slider").id]
        self._hinge_qposadr = self.model.jnt_qposadr[self.model.joint("hinge_1").id]
        self._hard_position_bound = hard_position_bound
        self._recovery_position_bound = recovery_position_bound
        self._recovery_velocity_bound = recovery_velocity_bound
        self._rng = np.random.default_rng(seed)
        self.observation_size = 5
        self.action_size = self.model.nu
        self.dt = ctrl_dt * action_repeat

    def _apply_real_params(self, params: Mapping[str, float]) -> None:
        """Perturbs the nominal model, with the semantics of the sim randomization."""
        model = self.model
        pole_length = params.get("pole_length", 0.0)
        if pole_length != 0.0:
            scale = (0.5 + pole_length) / 0.5
            model.geom_size[_POLE_ID, 1] = 0.5 + pole_length
            model.body_mass[_POLE_ID] *= scale
            model.body_inertia[_POLE_ID] *= scale**3
            model.body_ipos[_POLE_ID, -1] += pole_length / 2.0
        pole_mass = params.get("pole_mass", 0.0)
        if pole_mass != 0.0:
            scale = (model.body_mass[_POLE_ID] + pole_mass) / model.body_mass[_POLE_ID]
            model.body_mass[_POLE_ID] *= scale
            model.body_inertia[_POLE_ID] *= scale
        model.actuator_gear[0, 0] += params.get("gear", 0.0)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._slider_qposadr] = 0.01 * self._rng.normal()
        self.data.qpos[self._hinge_qposadr] = np.pi + 0.01 * self._rng.normal()
        self.data.qvel[:] = 0.01 * self._rng.normal(size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

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
            reward += self._task_reward(action)
        obs = self._get_obs()
        terminated = bool(recovery.violates(obs, self._hard_position_bound))
        info: dict[str, Any] = {
            "in_recovery_set": bool(
                recovery.in_recovery_set(
                    obs, self._recovery_position_bound, self._recovery_velocity_bound
                )
            ),
        }
        return obs, reward, terminated, info

    def _get_obs(self) -> np.ndarray:
        xmat = self.data.xmat.reshape(-1, 3, 3)
        return np.array(
            [
                self.data.qpos[self._slider_qposadr],
                xmat[2, 2, 2],
                xmat[2, 0, 2],
                *self.data.qvel,
            ],
            dtype=np.float32,
        )

    def _task_reward(self, action: np.ndarray) -> float:
        xmat = self.data.xmat.reshape(-1, 3, 3)
        upright = (xmat[2, 2, 2] + 1) / 2
        cart_position = self.data.qpos[self._slider_qposadr]
        centered = (1 + _gaussian_tolerance(cart_position, 2.0)) / 2
        small_control = (4 + max(1.0 - action[0] ** 2, 0.0)) / 5
        small_velocity = (1 + _gaussian_tolerance(self.data.qvel[1], 5.0)) / 2
        return float(upright * small_control * small_velocity * centered)
