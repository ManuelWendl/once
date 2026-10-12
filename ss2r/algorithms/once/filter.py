from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.types import PRNGKey

from ss2r.algorithms.mbpo.networks import MBPONetworks
from ss2r.algorithms.once.types import OnceTrainingState
from ss2r.algorithms.sac.networks import SafeSACNetworks


def get_filter_params(training_state: OnceTrainingState, recovery_threshold) -> Any:
    return (
        training_state.normalizer_params,
        training_state.behavior_policy_params,
        training_state.recovery_normalizer_params,
        training_state.recovery_policy_params,
        training_state.recovery_q_params,
        recovery_threshold,
    )


def make_filter_params_template(
    mbpo_network: MBPONetworks,
    recovery_network: SafeSACNetworks,
    observation_size: int,
) -> Any:
    """Parameters with the structure of `get_filter_params`, to (un)flatten them."""
    key = jax.random.PRNGKey(0)
    normalizer_params = running_statistics.init_state(
        specs.Array((observation_size,), jnp.dtype("float32"))
    )
    return (
        normalizer_params,
        mbpo_network.policy_network.init(key),
        normalizer_params,
        recovery_network.policy_network.init(key),
        recovery_network.qr_network.init(key),
        jnp.zeros(()),
    )


def make_recovery_filter_fn(
    mbpo_network: MBPONetworks,
    recovery_network: SafeSACNetworks,
) -> Callable[[Any, bool], types.Policy]:
    """Takes the behavior action only if the recovery prior can still recover."""

    def make_policy(params: Any, deterministic: bool = False) -> types.Policy:
        (
            normalizer_params,
            policy_params,
            recovery_normalizer_params,
            recovery_policy_params,
            recovery_q_params,
            recovery_threshold,
        ) = params

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = mbpo_network.policy_network.apply(
                normalizer_params, policy_params, observations
            )
            mode_a = mbpo_network.parametric_action_distribution.mode(logits)
            if deterministic:
                behavioral_action = mode_a
            else:
                behavioral_action = mbpo_network.parametric_action_distribution.sample(
                    logits, key_sample
                )
            recovery_logits = recovery_network.policy_network.apply(
                recovery_normalizer_params, recovery_policy_params, observations
            )
            recovery_action = recovery_network.parametric_action_distribution.mode(
                recovery_logits
            )
            q_r = recovery_network.qr_network.apply(
                recovery_normalizer_params,
                recovery_q_params,
                observations,
                behavioral_action,
            ).mean(axis=-1)
            can_recover = q_r > recovery_threshold
            action = jnp.where(
                can_recover[..., None], behavioral_action, recovery_action
            )
            extras = {
                "intervention": 1 - can_recover.astype(jnp.float32),
                "policy_distance": jnp.linalg.norm(mode_a - action, axis=-1),
                "q_r": q_r,
                "behavior_action": behavioral_action,
            }
            return action, extras

        return policy

    return make_policy
