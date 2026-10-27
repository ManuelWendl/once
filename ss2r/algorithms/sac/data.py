import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey, Transition

from ss2r.algorithms.sac.types import CollectDataFn, ReplayBufferState, float16
from ss2r.rl.types import MakePolicyFn, UnrollFn
from ss2r.rl.utils import quantize_images, remove_pixels


def get_collection_fn(cfg):
    if cfg.agent.data_collection.name == "step":
        random_action_prob = cfg.agent.get("random_action_prob", 0.0)
        random_first_action_prob = cfg.agent.get("random_first_action_prob", 0.0)
        if random_action_prob > 0.0 or random_first_action_prob > 0.0:
            collect = make_collection_fn(
                make_random_actor_step(random_action_prob, random_first_action_prob)
            )
            # One step per call, as collect_single_step.
            collect.single_step = True  # type: ignore[attr-defined]
            return collect
        return collect_single_step
    elif cfg.agent.data_collection.name == "episodic":

        def generate_episodic_unroll(
            env,
            env_state,
            make_policy_fn,
            policy_params,
            key,
            extra_fields,
        ):
            env_state, transitions = acting.generate_unroll(
                env,
                env_state,
                make_policy_fn(policy_params),
                key,
                cfg.training.episode_length,
                extra_fields,
            )
            transitions = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), transitions
            )
            return env_state, transitions

        return make_collection_fn(generate_episodic_unroll)
    elif cfg.agent.data_collection.name == "hardware":
        data_collection_cfg = cfg.agent.data_collection
        if "Go1" in cfg.environment.task_name or "Go2" in cfg.environment.task_name:
            from ss2r.algorithms.sac.go1_sac_to_onnx import (
                go1_postprocess_data,
                make_go1_policy,
            )
            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = functools.partial(make_go1_policy, cfg=cfg)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                go1_postprocess_data,
                data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        elif "rccar" in cfg.environment.task_name:
            import cloudpickle

            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = lambda _, params: cloudpickle.dumps(params)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                lambda data, _: Transition(*data),
                address=data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        elif "PandaPickCubeCartesian" in cfg.environment.task_name:
            from ss2r.algorithms.sac.franka_sac_to_onnx import (
                make_franka_policy,
                postprocess_data,
            )
            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = functools.partial(make_franka_policy, cfg=cfg)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                postprocess_data,
                address=data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        else:
            raise ValueError(
                f"Environment {cfg.environment.task_name} not supported for hardware data collection."
            )
    else:
        raise ValueError(f"Unknown data collection {cfg.agent.data_collection.name}")


def actor_step(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.actor_step(env, env_state, policy, key, extra_fields)


def make_random_actor_step(random_action_prob: float, random_first_action_prob: float):
    """`actor_step` that executes a uniformly random action instead of the one of
    the policy: in the first step of an episode with random_first_action_prob,
    in the other steps with random_action_prob. The executed action is stored.

    For a critic that rates the actions of another policy, followed by this one
    (as the filter of ONCE does with the recovery prior), not only the policy's
    own actions."""

    def step(env, env_state, make_policy_fn, policy_params, key, extra_fields):
        policy = make_policy_fn(policy_params)
        key_policy, key_random, key_use = jax.random.split(key, 3)
        actions, policy_extras = policy(env_state.obs, key_policy)
        # The auto reset keeps the step count until the next step, but marks the
        # first state of a new episode with done.
        first = env_state.done > 0
        if "steps" in env_state.info:
            first = first | (env_state.info["steps"] == 0)
        prob = jnp.where(first, random_first_action_prob, random_action_prob)
        use = jax.random.uniform(key_use, prob.shape) < prob
        random_actions = jax.random.uniform(
            key_random, actions.shape, minval=-1.0, maxval=1.0
        )
        actions = jnp.where(use[..., None], random_actions, actions)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(  # pytype: disable=wrong-arg-types
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={"policy_extras": policy_extras, "state_extras": state_extras},
        )

    return step


def generate_unroll(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    unroll_length,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.generate_unroll(
        env, env_state, policy, key, unroll_length, extra_fields
    )


def make_collection_fn(unroll_fn: UnrollFn) -> CollectDataFn:
    def collect_data(
        env: envs.Env,
        make_policy_fn: MakePolicyFn,
        params: Params,
        normalizer_params: running_statistics.RunningStatisticsState,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        env_state, transitions = unroll_fn(
            env,
            env_state,
            make_policy_fn,
            (normalizer_params, params),
            key,
            extra_fields=extra_fields,
        )
        normalizer_params = running_statistics.update(
            normalizer_params, remove_pixels(transitions.observation)
        )
        transitions = float16(transitions)
        transitions = transitions._replace(
            observation=quantize_images(transitions.observation),
            next_observation=quantize_images(transitions.next_observation),
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    return collect_data


collect_single_step = make_collection_fn(actor_step)
