"""Learner of ONCE: non-episodic MBPO with a recovery prior.

The task (behavior) pipeline and the recovery pipeline are kept separate:
they have their own buffers, losses and reward signals. `Transition.reward`
always holds the task reward, the recovery signal lives in
`extras["state_extras"]["recovery_reward"]`.
"""

from typing import Any, Callable, Mapping, Tuple

import jax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics, specs
from brax.training.types import PRNGKey

import ss2r.algorithms.mbpo.losses as mbpo_losses
import ss2r.algorithms.sac.losses as sac_losses
from ss2r.algorithms.mbpo.networks import MBPONetworks
from ss2r.algorithms.once import buffer as ring_buffer
from ss2r.algorithms.once.types import (
    OnceTrainingState,
    RecoveryRolloutState,
    RingBufferState,
)
from ss2r.algorithms.sac import gradients
from ss2r.algorithms.sac.networks import SafeSACNetworks
from ss2r.algorithms.sac.q_transforms import PessimisticReward, SACBase
from ss2r.algorithms.sac.types import Metrics, Transition


def make_dummy_transition(
    observation_size: int,
    action_size: int,
    state_extras: Tuple[str, ...] = (),
    behavior: bool = False,
) -> Transition:
    dummy_obs = jnp.zeros((observation_size,))
    policy_extras = {}
    if behavior:
        policy_extras = {
            "behavior_action": jnp.zeros((action_size,)),
            "intervention": jnp.zeros(()),
        }
    extras = {
        "state_extras": {
            "truncation": jnp.zeros(()),
            **{name: jnp.zeros(()) for name in state_extras},
        },
        "policy_extras": policy_extras,
    }
    return Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=jnp.zeros((action_size,)),
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        next_observation=dummy_obs,
        extras=extras,
    )


def make_learner(
    *,
    mbpo_network: MBPONetworks,
    recovery_network: SafeSACNetworks,
    normalize_fn: Any,
    observation_size: int,
    action_size: int,
    agent_cfg: Mapping[str, Any],
    recovery_reward_fn: Callable[[jax.Array], jax.Array],
    violates_fn: Callable[[jax.Array], jax.Array],
    action_repeat: int,
):
    """Creates the initialization, data insertion and learner step functions."""
    c = agent_cfg
    model_updates = c["model_grad_updates_per_step"]
    critic_updates = c["critic_grad_updates_per_step"]
    recovery_updates = c["recovery_grad_updates_per_step"]
    num_actor_updates = -(-critic_updates // c["num_critic_updates_per_actor_update"])
    num_task_transitions = c["num_model_rollouts"] * c["unroll_length"]
    if num_task_transitions < c["sac_batch_size"] * critic_updates:
        raise ValueError(
            "Not enough model rollouts: num_model_rollouts * unroll_length must be "
            "at least sac_batch_size * critic_grad_updates_per_step"
        )
    recovery_threshold = c["recovery_threshold"]
    recovery_horizon = c["recovery_horizon"]
    policy_network = mbpo_network.policy_network
    task_q_network = mbpo_network.qr_network
    model_network = mbpo_network.model_network
    action_distribution = mbpo_network.parametric_action_distribution
    recovery_policy_network = recovery_network.policy_network
    recovery_q_network = recovery_network.qr_network
    recovery_action_distribution = recovery_network.parametric_action_distribution

    make_optimizer = lambda lr: optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr)
    )
    alpha_optimizer = optax.adam(learning_rate=c["alpha_learning_rate"])
    policy_optimizer = make_optimizer(c["learning_rate"])
    task_q_optimizer = make_optimizer(c["critic_learning_rate"])
    model_optimizer = make_optimizer(c["model_learning_rate"])
    recovery_q_optimizer = make_optimizer(c["recovery_q_learning_rate"])

    # Task losses (infinite horizon, discounted).
    alpha_loss, task_critic_loss, actor_loss, model_loss = mbpo_losses.make_losses(
        mbpo_network=mbpo_network,
        reward_scaling=c["reward_scaling"],
        cost_scaling=1.0,
        discounting=c["discounting"],
        safety_discounting=1.0,
        action_size=action_size,
        use_bro=c["use_bro"],
        normalize_fn=normalize_fn,
        target_entropy=c["target_entropy"],
    )
    # Recovery loss (finite horizon), the same as when training the prior.
    _, recovery_critic_loss, _ = sac_losses.make_losses(
        sac_network=recovery_network,
        reward_scaling=1.0,
        cost_scaling=1.0,
        discounting=c["recovery_discounting"],
        safety_discounting=1.0,
        action_size=action_size,
        use_bro=c["use_bro"],
    )
    task_q_transform = SACBase(
        pessimistic_q=c["pessimistic_q"] if c["use_bro"] else True
    )
    recovery_q_transform = PessimisticReward(
        penalty_scale=c["penalty_scale"], margin=c["margin"]
    )
    alpha_update = gradients.gradient_update_fn(
        alpha_loss, alpha_optimizer, pmap_axis_name=None
    )
    task_critic_update = gradients.gradient_update_fn(
        task_critic_loss, task_q_optimizer, pmap_axis_name=None
    )
    actor_update = gradients.gradient_update_fn(
        actor_loss, policy_optimizer, pmap_axis_name=None, has_aux=True
    )
    model_update = gradients.gradient_update_fn(
        model_loss, model_optimizer, pmap_axis_name=None
    )
    recovery_critic_update = gradients.gradient_update_fn(
        recovery_critic_loss, recovery_q_optimizer, pmap_axis_name=None
    )
    polyak = lambda target, new: jax.tree_util.tree_map(
        lambda x, y: x * (1 - c["tau"]) + y * c["tau"], target, new
    )
    behavior_dummy_transition = make_dummy_transition(
        observation_size, action_size, behavior=True
    )
    recovery_dummy_transition = make_dummy_transition(
        observation_size, action_size, PessimisticReward.required_extras
    )

    def init(
        key: PRNGKey, prior_params: Tuple[Any, ...]
    ) -> Tuple[
        OnceTrainingState, RingBufferState, RecoveryRolloutState, RingBufferState
    ]:
        key_policy, key_q, key_model = jax.random.split(key, 3)
        recovery_normalizer_params, recovery_policy_params = prior_params[:2]
        recovery_q_params = prior_params[3]
        _check_same_structure(
            recovery_policy_params,
            recovery_policy_network.init(key_policy),
            "recovery policy",
        )
        _check_same_structure(
            recovery_q_params, recovery_q_network.init(key_q), "recovery Q function"
        )
        log_alpha = jnp.asarray(jnp.log(c["init_alpha"]), dtype=jnp.float32)
        policy_params = policy_network.init(key_policy)
        task_q_params = task_q_network.init(key_q)
        model_keys = jax.random.split(key_model, c["model_ensemble_size"])
        model_params = jax.vmap(model_network.init)(model_keys)
        normalizer_params = running_statistics.init_state(
            specs.Array((observation_size,), jnp.dtype("float32"))
        )
        training_state = OnceTrainingState(  # type: ignore
            behavior_policy_params=policy_params,
            behavior_policy_optimizer_state=policy_optimizer.init(policy_params),
            task_q_params=task_q_params,
            task_target_q_params=task_q_params,
            task_q_optimizer_state=task_q_optimizer.init(task_q_params),
            alpha_params=log_alpha,
            alpha_optimizer_state=alpha_optimizer.init(log_alpha),
            model_params=model_params,
            model_optimizer_state=model_optimizer.init(model_params),
            normalizer_params=normalizer_params,
            recovery_policy_params=recovery_policy_params,
            recovery_q_params=recovery_q_params,
            recovery_target_q_params=recovery_q_params,
            recovery_q_optimizer_state=recovery_q_optimizer.init(recovery_q_params),
            recovery_normalizer_params=recovery_normalizer_params,
            gradient_steps=jnp.zeros(()),
            env_steps=jnp.zeros(()),
        )
        behavior_buffer_state = ring_buffer.init(
            behavior_dummy_transition, c["max_replay_size"]
        )
        recovery_buffer_state = ring_buffer.init(
            recovery_dummy_transition, c["recovery_replay_size"]
        )
        recovery_rollout_state = RecoveryRolloutState(  # type: ignore
            obs=jnp.zeros((c["num_recovery_rollouts"], observation_size)),
            step=jnp.zeros((c["num_recovery_rollouts"],), jnp.int32),
        )
        return (
            training_state,
            behavior_buffer_state,
            recovery_rollout_state,
            recovery_buffer_state,
        )

    def insert(
        training_state: OnceTrainingState,
        behavior_buffer_state: RingBufferState,
        transitions: Transition,
    ) -> Tuple[OnceTrainingState, RingBufferState]:
        """Inserts a chunk of real transitions, collected with the behavior."""
        normalizer_params = running_statistics.update(
            training_state.normalizer_params, transitions.observation
        )
        behavior_buffer_state = ring_buffer.insert(behavior_buffer_state, transitions)
        num_steps = transitions.reward.shape[0] * action_repeat
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + num_steps,
        )
        return training_state, behavior_buffer_state

    def predict(
        training_state: OnceTrainingState,
        obs: jax.Array,
        action: jax.Array,
        key: PRNGKey,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Ensemble predictions and their disagreement."""
        vmap_pred_fn = jax.vmap(model_network.apply, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, _ = vmap_pred_fn(
            training_state.normalizer_params, training_state.model_params, obs, action
        )
        disagreement = next_obs_pred.std(axis=0).mean(-1)
        next_obs_mean = next_obs_pred.mean(axis=0)
        if c["model_propagation"] == "nominal":
            next_obs_sample = next_obs_mean
        elif c["model_propagation"] == "random":
            idx = jax.random.randint(key, (obs.shape[0],), 0, next_obs_pred.shape[0])
            next_obs_sample = next_obs_pred[idx, jnp.arange(obs.shape[0])]
        else:
            raise ValueError(f"Unknown model propagation: {c['model_propagation']}")
        return next_obs_mean, next_obs_sample, reward_pred.mean(axis=0), disagreement

    def model_sgd_step(
        training_state: OnceTrainingState, transitions: Transition
    ) -> Tuple[OnceTrainingState, Metrics]:
        loss, model_params, model_optimizer_state = model_update(
            training_state.model_params,
            training_state.normalizer_params,
            transitions,
            optimizer_state=training_state.model_optimizer_state,
            params=training_state.model_params,
        )
        training_state = training_state.replace(  # type: ignore
            model_params=model_params, model_optimizer_state=model_optimizer_state
        )
        return training_state, {"model_loss": loss}

    def generate_task_data(
        training_state: OnceTrainingState, start_obs: jax.Array, key: PRNGKey
    ) -> Tuple[Transition, Metrics]:
        """Rolls out the behavior policy in the model, labeled with the task reward."""

        def f(obs, key):
            key_action, key_model = jax.random.split(key)
            logits = policy_network.apply(
                training_state.normalizer_params,
                training_state.behavior_policy_params,
                obs,
            )
            action = action_distribution.sample(logits, key_action)
            next_obs_mean, next_obs_sample, reward, disagreement = predict(
                training_state, obs, action, key_model
            )
            q_r = recovery_q_network.apply(
                training_state.recovery_normalizer_params,
                training_state.recovery_q_params,
                obs,
                action,
            ).mean(axis=-1)
            # The recovery prior takes over when it might not be able to recover
            # anymore. The behavior gets the worst task reward and terminates.
            intervention = q_r < recovery_threshold
            terminated = intervention | violates_fn(next_obs_mean)
            task_reward = jnp.where(
                terminated,
                c["intervention_task_reward"],
                reward + c["optimism"] * disagreement,
            )
            transitions = Transition(
                observation=obs,
                action=action,
                reward=task_reward,
                discount=1.0 - terminated.astype(jnp.float32),
                next_observation=next_obs_mean,
                extras={
                    "state_extras": {"truncation": jnp.zeros_like(task_reward)},
                    "policy_extras": {},
                },
            )
            next_obs = jnp.where(
                terminated[:, None], jnp.roll(start_obs, 1, axis=0), next_obs_sample
            )
            metrics = {
                "disagreement": disagreement.mean(),
                "model_intervention": intervention.astype(jnp.float32).mean(),
                "model_q_r": q_r.mean(),
            }
            return next_obs, (transitions, metrics)

        keys = jax.random.split(key, c["unroll_length"])
        _, (transitions, metrics) = jax.lax.scan(f, start_obs, keys)
        transitions = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), transitions
        )
        return transitions, jax.tree_util.tree_map(jnp.mean, metrics)

    def task_critic_sgd_step(
        carry: Tuple[OnceTrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[OnceTrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_critic = jax.random.split(key)
        alpha = jnp.exp(training_state.alpha_params) + c["min_alpha"]
        loss, task_q_params, task_q_optimizer_state = task_critic_update(
            training_state.task_q_params,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.task_target_q_params,
            alpha,
            transitions,
            key_critic,
            task_q_transform,
            optimizer_state=training_state.task_q_optimizer_state,
            params=training_state.task_q_params,
        )
        training_state = training_state.replace(  # type: ignore
            task_q_params=task_q_params,
            task_q_optimizer_state=task_q_optimizer_state,
            task_target_q_params=polyak(
                training_state.task_target_q_params, task_q_params
            ),
            gradient_steps=training_state.gradient_steps + 1,
        )
        return (training_state, key), {"critic_loss": loss}

    def actor_sgd_step(
        carry: Tuple[OnceTrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[OnceTrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_alpha, key_actor = jax.random.split(key, 3)
        loss_alpha, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params) + c["min_alpha"]
        (loss_actor, _), policy_params, policy_optimizer_state = actor_update(
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.task_q_params,
            None,
            alpha,
            transitions,
            key_actor,
            float("inf"),
            None,
            None,
            optimizer_state=training_state.behavior_policy_optimizer_state,
            params=training_state.behavior_policy_params,
        )
        training_state = training_state.replace(  # type: ignore
            behavior_policy_params=policy_params,
            behavior_policy_optimizer_state=policy_optimizer_state,
            alpha_params=alpha_params,
            alpha_optimizer_state=alpha_optimizer_state,
        )
        metrics = {
            "actor_loss": loss_actor,
            "alpha_loss": loss_alpha,
            "alpha": jnp.exp(alpha_params),
        }
        return (training_state, key), metrics

    def generate_recovery_data(
        training_state: OnceTrainingState,
        behavior_buffer_state: RingBufferState,
        rollout_state: RecoveryRolloutState,
        key: PRNGKey,
    ) -> Tuple[RecoveryRolloutState, Transition, Metrics]:
        """Rolls out the recovery prior in the model, labeled with recovery rewards.

        A rollout starts with a state-action pair of the behavior policy, taken
        from the behavior buffer, and then follows the recovery prior for at most
        `recovery_horizon` steps. If the behavior action was executed, the first
        step is the real transition. If the recovery prior took over instead,
        the outcome of the behavior action is predicted with the model.

        With `recovery_restart_rollouts`, all rollouts start anew in every call:
        the model is never unrolled for more than `recovery_unroll_length` steps.
        """
        if c["recovery_restart_rollouts"]:
            rollout_state = rollout_state.replace(  # type: ignore
                step=jnp.zeros_like(rollout_state.step)
            )

        def f(rollout_state, key):
            key_start, key_action, key_model = jax.random.split(key, 3)
            start = ring_buffer.sample(
                behavior_buffer_state, key_start, c["num_recovery_rollouts"]
            )
            is_start = rollout_state.step == 0
            obs = jnp.where(is_start[:, None], start.observation, rollout_state.obs)
            logits = recovery_policy_network.apply(
                training_state.recovery_normalizer_params,
                training_state.recovery_policy_params,
                obs,
            )
            action = recovery_action_distribution.sample(logits, key_action)
            start_extras = start.extras["policy_extras"]
            action = jnp.where(
                is_start[:, None], start_extras["behavior_action"], action
            )
            _, next_obs, _, disagreement = predict(
                training_state, obs, action, key_model
            )
            is_real = is_start & (start_extras["intervention"] == 0.0)
            next_obs = jnp.where(is_real[:, None], start.next_observation, next_obs)
            valid = jnp.isfinite(next_obs).all(axis=-1)
            next_obs = jnp.where(valid[:, None], next_obs, obs)
            recovery_reward = recovery_reward_fn(next_obs) * valid
            violated = violates_fn(next_obs) | ~valid
            step = rollout_state.step + 1
            # Reaching the horizon is a true terminal state, no bootstrapping.
            done = (recovery_reward > 0.0) | violated | (step >= recovery_horizon)
            transitions = Transition(
                observation=obs,
                action=action,
                # The task reward is not used by the recovery pipeline.
                reward=jnp.zeros_like(recovery_reward),
                discount=1.0 - done.astype(jnp.float32),
                next_observation=next_obs,
                extras={
                    "state_extras": {
                        "truncation": jnp.zeros_like(recovery_reward),
                        "recovery_reward": recovery_reward,
                        "disagreement": disagreement,
                    },
                    "policy_extras": {},
                },
            )
            rollout_state = RecoveryRolloutState(  # type: ignore
                obs=next_obs, step=jnp.where(done, 0, step)
            )
            metrics = {
                "recovery_reached": recovery_reward.mean(),
                "recovery_violated": violated.astype(jnp.float32).mean(),
                "recovery_disagreement": disagreement.mean(),
            }
            return rollout_state, (transitions, metrics)

        keys = jax.random.split(key, c["recovery_unroll_length"])
        rollout_state, (transitions, metrics) = jax.lax.scan(f, rollout_state, keys)
        transitions = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), transitions
        )
        return rollout_state, transitions, jax.tree_util.tree_map(jnp.mean, metrics)

    def recovery_critic_sgd_step(
        carry: Tuple[OnceTrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[OnceTrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_critic = jax.random.split(key)
        loss, recovery_q_params, recovery_q_optimizer_state = recovery_critic_update(
            training_state.recovery_q_params,
            training_state.recovery_policy_params,
            training_state.recovery_normalizer_params,
            training_state.recovery_target_q_params,
            0.0,
            transitions,
            key_critic,
            recovery_q_transform,
            optimizer_state=training_state.recovery_q_optimizer_state,
            params=training_state.recovery_q_params,
        )
        training_state = training_state.replace(  # type: ignore
            recovery_q_params=recovery_q_params,
            recovery_q_optimizer_state=recovery_q_optimizer_state,
            recovery_target_q_params=polyak(
                training_state.recovery_target_q_params, recovery_q_params
            ),
        )
        return (training_state, key), {"recovery_critic_loss": loss}

    def split_batches(transitions: Transition, num_batches: int) -> Transition:
        return jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_batches, -1) + x.shape[1:]), transitions
        )

    def learner_step(
        training_state: OnceTrainingState,
        behavior_buffer_state: RingBufferState,
        recovery_rollout_state: RecoveryRolloutState,
        recovery_buffer_state: RingBufferState,
        key: PRNGKey,
    ) -> Tuple[OnceTrainingState, RecoveryRolloutState, RingBufferState, Metrics]:
        (
            key_model,
            key_start,
            key_rollout,
            key_task,
            key_recovery_rollout,
            key_recovery_sample,
            key_recovery,
        ) = jax.random.split(key, 7)
        # Learn the model from real data.
        transitions = ring_buffer.sample(
            behavior_buffer_state, key_model, c["batch_size"] * model_updates
        )
        training_state, model_metrics = jax.lax.scan(
            model_sgd_step, training_state, split_batches(transitions, model_updates)
        )
        # Behavior (task) pipeline.
        start = ring_buffer.sample(
            behavior_buffer_state, key_start, c["num_model_rollouts"]
        )
        task_transitions, rollout_metrics = generate_task_data(
            training_state, start.observation, key_rollout
        )
        task_transitions = jax.tree_util.tree_map(
            lambda x: x[: c["sac_batch_size"] * critic_updates], task_transitions
        )
        task_transitions = split_batches(task_transitions, critic_updates)
        (training_state, _), critic_metrics = jax.lax.scan(
            task_critic_sgd_step, (training_state, key_task), task_transitions
        )
        actor_transitions = jax.tree_util.tree_map(
            lambda x: x[:num_actor_updates], task_transitions
        )
        (training_state, _), actor_metrics = jax.lax.scan(
            actor_sgd_step, (training_state, key_task), actor_transitions
        )
        # Recovery pipeline.
        (
            recovery_rollout_state,
            recovery_transitions,
            recovery_rollout_metrics,
        ) = generate_recovery_data(
            training_state,
            behavior_buffer_state,
            recovery_rollout_state,
            key_recovery_rollout,
        )
        recovery_buffer_state = ring_buffer.insert(
            recovery_buffer_state, recovery_transitions
        )
        recovery_transitions = ring_buffer.sample(
            recovery_buffer_state,
            key_recovery_sample,
            c["recovery_batch_size"] * recovery_updates,
        )
        (training_state, _), recovery_metrics = jax.lax.scan(
            recovery_critic_sgd_step,
            (training_state, key_recovery),
            split_batches(recovery_transitions, recovery_updates),
        )
        metrics = {
            **model_metrics,
            **critic_metrics,
            **actor_metrics,
            **recovery_metrics,
        }
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics |= rollout_metrics
        metrics |= recovery_rollout_metrics
        metrics["task_reward"] = task_transitions.reward.mean()
        metrics["buffer_current_size"] = behavior_buffer_state.size
        metrics["recovery_buffer_current_size"] = recovery_buffer_state.size
        return training_state, recovery_rollout_state, recovery_buffer_state, metrics

    return init, insert, learner_step


def _check_same_structure(params: Any, expected: Any, name: str) -> None:
    shapes = lambda tree: [jnp.shape(x) for x in jax.tree_util.tree_leaves(tree)]
    if shapes(params) != shapes(expected):
        raise ValueError(
            f"The parameters of the {name} do not match the networks of the prior. "
            "Make sure that the prior configuration is the one used for training it."
        )
