from typing import Any

import flax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics
from brax.training.types import Params


@flax.struct.dataclass
class OnceTrainingState:
    """Contains training state for the learner.

    Fields prefixed with `task_` and `behavior_` belong to the task reward,
    fields prefixed with `recovery_` to the recovery prior. The two are never
    mixed.
    """

    behavior_policy_params: Params
    behavior_policy_optimizer_state: optax.OptState
    task_q_params: Params
    task_target_q_params: Params
    task_q_optimizer_state: optax.OptState
    alpha_params: Params
    alpha_optimizer_state: optax.OptState
    model_params: Params
    model_optimizer_state: optax.OptState
    normalizer_params: running_statistics.RunningStatisticsState
    recovery_policy_params: Params
    recovery_q_params: Params
    recovery_target_q_params: Params
    recovery_q_optimizer_state: optax.OptState
    recovery_normalizer_params: running_statistics.RunningStatisticsState
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray


@flax.struct.dataclass
class RecoveryRolloutState:
    """State of the recovery rollouts in the learned model."""

    obs: jnp.ndarray
    # Number of steps taken in the current rollout. Zero means that the rollout
    # (re)starts from a transition of the behavior buffer.
    step: jnp.ndarray


@flax.struct.dataclass
class RingBufferState:
    data: Any
    insert_position: jnp.ndarray
    size: jnp.ndarray
