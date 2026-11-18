import functools
from typing import Any, Mapping, Optional, Tuple

import jax.nn as jnn
import jax.numpy as jnp
from brax.training import types as brax_types
from brax.training.acme import running_statistics

import ss2r.algorithms.mbpo.networks as mbpo_networks
import ss2r.algorithms.sac.networks as sac_networks


def recovery_observation_indices(
    agent_cfg: Mapping[str, Any], observation_size: int
) -> Optional[Tuple[int, ...]]:
    """The values of the observation that the recovery prior sees: the ones of
    `recovery_observation_indices`, the first `recovery_observation_size`, or None
    for all of them."""
    indices = agent_cfg.get("recovery_observation_indices")
    size = agent_cfg.get("recovery_observation_size")
    if indices is not None and size is not None:
        raise ValueError(
            "Set either recovery_observation_indices or recovery_observation_size."
        )
    if indices is not None:
        indices = tuple(int(i) for i in indices)
        if not all(0 <= i < observation_size for i in indices):
            raise ValueError(
                f"recovery_observation_indices {indices} are not all in an "
                f"observation of size {observation_size}"
            )
        return indices
    if size is not None and size < observation_size:
        return tuple(range(size))
    return None


def recovery_input_size(agent_cfg: Mapping[str, Any], observation_size: int) -> int:
    """The size of the observation that the recovery prior sees."""
    indices = recovery_observation_indices(agent_cfg, observation_size)
    return observation_size if indices is None else len(indices)


def expand_normalizer(
    normalizer_params: running_statistics.RunningStatisticsState,
    indices: Tuple[int, ...],
    observation_size: int,
) -> running_statistics.RunningStatisticsState:
    """The statistics of the prior, which sees `indices` of the observation, for
    the whole observation. The values it does not see stay unnormalized: mean 0
    and std 1, with the prior's count, such that they stay about so online."""
    idx = jnp.asarray(indices)
    count = normalizer_params.count
    if isinstance(count, brax_types.UInt64):
        count = jnp.float32(count.hi) * jnp.float32(2.0**32) + jnp.float32(count.lo)

    def full(fill, values):
        values = jnp.asarray(values)
        return jnp.full((observation_size,), fill, values.dtype).at[idx].set(values)

    return normalizer_params.replace(  # type: ignore[attr-defined]
        mean=full(0.0, normalizer_params.mean),
        summed_variance=full(count, normalizer_params.summed_variance),
        std=full(1.0, normalizer_params.std),
    )


def make_networks(
    agent_cfg: Mapping[str, Any],
    prior_agent_cfg: Mapping[str, Any],
    observation_size: int,
    action_size: int,
) -> Tuple[mbpo_networks.MBPONetworks, sac_networks.SafeSACNetworks, Any]:
    """Creates the behavior/model networks and the recovery prior networks.

    The recovery networks are created exactly as when training the prior, so
    that its parameters and observation normalization can be used as is. With
    `recovery_observation_indices` or `recovery_observation_size`, they see only
    those values of the observation.
    """
    normalize_fn = lambda x, y: x
    if agent_cfg["normalize_observations"]:
        normalize_fn = functools.partial(
            running_statistics.normalize, max_abs_value=5.0
        )
    mbpo_network = mbpo_networks.make_mbpo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        policy_hidden_layer_sizes=agent_cfg["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=agent_cfg["value_hidden_layer_sizes"],
        model_hidden_layer_sizes=agent_cfg["model_hidden_layer_sizes"],
        activation=getattr(jnn, agent_cfg["activation"]),
        use_bro=agent_cfg["use_bro"],
        n_critics=agent_cfg["n_critics"],
        n_heads=agent_cfg["n_heads"],
        safe=False,
    )
    indices = recovery_observation_indices(agent_cfg, observation_size)
    recovery_normalize_fn = lambda x, y: x
    if prior_agent_cfg["normalize_observations"]:
        recovery_normalize_fn = running_statistics.normalize
    if indices is not None:
        normalize = recovery_normalize_fn
        idx = jnp.asarray(indices)
        recovery_normalize_fn = lambda x, y: normalize(jnp.take(x, idx, axis=-1), y)
    recovery_network = sac_networks.make_sac_networks(
        observation_size=recovery_input_size(agent_cfg, observation_size),
        action_size=action_size,
        preprocess_observations_fn=recovery_normalize_fn,
        policy_hidden_layer_sizes=prior_agent_cfg["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=prior_agent_cfg["value_hidden_layer_sizes"],
        activation=getattr(jnn, prior_agent_cfg["activation"]),
        use_bro=prior_agent_cfg["use_bro"],
        n_critics=prior_agent_cfg["n_critics"],
        n_heads=prior_agent_cfg["n_heads"],
        safe=False,
    )
    return mbpo_network, recovery_network, normalize_fn
