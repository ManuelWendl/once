import functools
from typing import Any, Mapping, Tuple

import jax.nn as jnn
from brax.training.acme import running_statistics

import ss2r.algorithms.mbpo.networks as mbpo_networks
import ss2r.algorithms.sac.networks as sac_networks


def make_networks(
    agent_cfg: Mapping[str, Any],
    prior_agent_cfg: Mapping[str, Any],
    observation_size: int,
    action_size: int,
) -> Tuple[mbpo_networks.MBPONetworks, sac_networks.SafeSACNetworks, Any]:
    """Creates the behavior/model networks and the recovery prior networks.

    The recovery networks are created exactly as when training the prior, so
    that its parameters and observation normalization can be used as is.
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
    recovery_normalize_fn = lambda x, y: x
    if prior_agent_cfg["normalize_observations"]:
        recovery_normalize_fn = running_statistics.normalize
    recovery_network = sac_networks.make_sac_networks(
        observation_size=observation_size,
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
