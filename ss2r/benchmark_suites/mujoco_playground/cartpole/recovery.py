"""Cartpole recovery task: bring the cart back into the safe operable set R.

The predicates below operate on the cartpole observation
`[cart_position, cos(pole), sin(pole), cart_velocity, pole_velocity]` and only
use operators, so that they can be shared between jax (simulation, learned
model) and numpy (CPU environment) code.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground import dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import cartpole

from ss2r.benchmark_suites.mujoco_playground import mjx_data

_CART_POSITION = 0
_CART_VELOCITY = 3

RECOVERY_POSITION_BOUND = 1.0
RECOVERY_VELOCITY_BOUND = 0.5
HARD_POSITION_BOUND = 1.5


def in_recovery_set(
    obs,
    position_bound: float = RECOVERY_POSITION_BOUND,
    velocity_bound: float = RECOVERY_VELOCITY_BOUND,
):
    """Whether the cart is in the safe operable region R."""
    return (abs(obs[..., _CART_POSITION]) <= position_bound) & (
        abs(obs[..., _CART_VELOCITY]) <= velocity_bound
    )


def violates(obs, hard_position_bound: float = HARD_POSITION_BOUND):
    """Whether the hard constraint on the cart position is violated."""
    return abs(obs[..., _CART_POSITION]) >= hard_position_bound


def recovery_reward_fn(
    next_obs,
    position_bound: float = RECOVERY_POSITION_BOUND,
    velocity_bound: float = RECOVERY_VELOCITY_BOUND,
    hard_position_bound: float = HARD_POSITION_BOUND,
):
    """1 if the cart entered R, 0 otherwise. Not to be mixed with the task reward."""
    reached = in_recovery_set(next_obs, position_bound, velocity_bound)
    reached = reached & ~violates(next_obs, hard_position_bound)
    return reached.astype("float32")


def default_config() -> config_dict.ConfigDict:
    config = cartpole.default_config()
    config.recovery_position_bound = RECOVERY_POSITION_BOUND
    config.recovery_velocity_bound = RECOVERY_VELOCITY_BOUND
    config.hard_position_bound = HARD_POSITION_BOUND
    config.reset_cart_velocity = 2.0
    config.reset_pole_velocity = 5.0
    return config


class CartpoleRecovery(cartpole.Balance):
    """Cartpole that terminates when reaching R or violating the hard constraint.

    `state.reward` remains the swingup task reward. The recovery signal is
    written to `state.info["recovery_reward"]`.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            swing_up=True,
            sparse=False,
            config=config,
            config_overrides=config_overrides,
        )
        assert not self._vision, "CartpoleRecovery does not support vision."
        defaults = default_config()
        get = lambda name: self._config.get(name, defaults[name])
        self._recovery_position_bound = get("recovery_position_bound")
        self._recovery_velocity_bound = get("recovery_velocity_bound")
        self._hard_position_bound = get("hard_position_bound")
        self._reset_cart_velocity = get("reset_cart_velocity")
        self._reset_pole_velocity = get("reset_pole_velocity")

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Samples the initial state from the whole constraint set."""
        state = super().reset(rng)
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        bound = self._hard_position_bound
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[self._slider_qposadr].set(
            jax.random.uniform(rng1, (), minval=-bound, maxval=bound)
        )
        qpos = qpos.at[self._hinge_1_qposadr].set(
            jax.random.uniform(rng2, (), minval=-jp.pi, maxval=jp.pi)
        )
        qvel = jp.zeros(self.mjx_model.nv)
        qvel = qvel.at[0].set(
            jax.random.uniform(
                rng3,
                (),
                minval=-self._reset_cart_velocity,
                maxval=self._reset_cart_velocity,
            )
        )
        qvel = qvel.at[1].set(
            jax.random.uniform(
                rng4,
                (),
                minval=-self._reset_pole_velocity,
                maxval=self._reset_pole_velocity,
            )
        )
        data = mjx_data.init(self.mjx_model, qpos=qpos, qvel=qvel)
        obs = self._get_obs(data, state.info)
        state.info["recovery_reward"] = jp.zeros(())
        state.metrics["recovery/reached"] = jp.zeros(())
        state.metrics["recovery/violated"] = jp.zeros(())
        return state.replace(data=data, obs=obs)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        nstate = super().step(state, action)
        violated = violates(nstate.obs, self._hard_position_bound)
        recovery_reward = recovery_reward_fn(
            nstate.obs,
            self._recovery_position_bound,
            self._recovery_velocity_bound,
            self._hard_position_bound,
        )
        nstate.info["recovery_reward"] = recovery_reward
        nstate.metrics["recovery/reached"] = recovery_reward
        nstate.metrics["recovery/violated"] = violated.astype(jp.float32)
        terminate = (recovery_reward > 0.0) | violated
        done = jp.maximum(nstate.done, terminate.astype(nstate.done.dtype))
        return nstate.replace(done=done)


dm_control_suite.register_environment(
    "CartpoleRecovery", CartpoleRecovery, default_config
)
