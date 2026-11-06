"""Handover adversary for the recovery prior, after RARL (Pinto et al., 2017): a second SAC agent
sends the commands in the env's adversary phase to make the ball fall, then the backup takes
over. Every transition trains the backup as usual; the adversary learns the probability of a
fall from its own transitions and from the backup's, and the two are updated in turns."""

from typing import Any, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.training.acme import running_statistics, specs
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey, Transition

from ss2r.algorithms.sac import networks as sac_networks
from ss2r.algorithms.sac.types import ReplayBufferState, float16
from ss2r.rl.utils import quantize_images, remove_pixels


@flax.struct.dataclass
class AdversaryState:
    policy_params: Params
    policy_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    q_optimizer_state: optax.OptState
    log_alpha: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    normalizer_params: running_statistics.RunningStatisticsState


class Adversary:
    """The adversary's networks, data collection and update."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        backup_network: sac_networks.SafeSACNetworks,
        alternate_every_steps: int,
        learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        init_alpha: float = 1.0,
        tau: float = 0.005,
        hidden_layer_sizes: Sequence[int] = (256, 256, 256),
        activation=jax.nn.swish,
        n_critics: int = 2,
        use_bro: bool = True,
    ):
        if alternate_every_steps < 1:
            raise ValueError(
                f"alternate_every_steps {alternate_every_steps} is below 1"
            )
        self.network = sac_networks.make_sac_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_hidden_layer_sizes=hidden_layer_sizes,
            value_hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            use_bro=use_bro,
            n_critics=n_critics,
        )
        self.make_policy = sac_networks.make_inference_fn(self.network)
        self.backup_network = backup_network
        self.observation_size = observation_size
        self.alternate_every_steps = alternate_every_steps
        self.init_alpha = init_alpha
        self.tau = tau
        self.target_entropy = -0.5 * action_size
        clipped = lambda lr: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
        self.policy_optimizer = clipped(learning_rate)
        self.q_optimizer = clipped(critic_learning_rate)
        self.alpha_optimizer = optax.adam(alpha_learning_rate)

    def init(self, key: PRNGKey) -> AdversaryState:
        k_policy, k_q = jax.random.split(key)
        policy_params = self.network.policy_network.init(k_policy)
        q_params = self.network.qr_network.init(k_q)
        log_alpha = jnp.asarray(jnp.log(self.init_alpha), dtype=jnp.float32)
        spec = specs.Array((self.observation_size,), jnp.dtype("float32"))
        return AdversaryState(
            policy_params=policy_params,
            policy_optimizer_state=self.policy_optimizer.init(policy_params),
            q_params=q_params,
            target_q_params=q_params,
            q_optimizer_state=self.q_optimizer.init(q_params),
            log_alpha=log_alpha,
            alpha_optimizer_state=self.alpha_optimizer.init(log_alpha),
            normalizer_params=running_statistics.init_state(spec),
        )

    def turn(self, env_steps: jax.Array) -> jax.Array:
        """Whether the adversary is the one updated: blocks of alternate_every_steps
        environment steps, the backup's first."""
        return (env_steps // self.alternate_every_steps) % 2 == 1

    def dummy_extras(self) -> dict[str, jax.Array]:
        obs = jnp.zeros((self.observation_size,))
        scalar = jnp.zeros(())
        return {
            "adversary_obs": obs,
            "next_adversary_obs": obs,
            "adversary_acting": scalar,
            "next_adversary_acting": scalar,
            "violated": scalar,
        }

    def collect(
        self,
        env: envs.Env,
        make_backup_policy,
        backup_params: Params,
        backup_normalizer: running_statistics.RunningStatisticsState,
        state: AdversaryState,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str],
    ) -> Tuple[Any, AdversaryState, envs.State, ReplayBufferState]:
        """One step of all envs: the adversary acts where its phase lasts, the backup
        elsewhere; the executed action goes into the replay buffer."""
        k_backup, k_adversary = jax.random.split(key)
        backup_policy = make_backup_policy((backup_normalizer, backup_params))
        backup_action, _ = backup_policy(env_state.obs, k_backup)
        adversary_obs = env_state.info["adversary_obs"]
        adversary_policy = self.make_policy(
            (state.normalizer_params, state.policy_params)
        )
        adversary_action, _ = adversary_policy(adversary_obs, k_adversary)
        acting = env_state.info["adversary_left"] > 0
        action = jnp.where(acting[..., None], adversary_action, backup_action)
        nstate = env.step(env_state, action)
        state_extras = {name: nstate.info[name] for name in extra_fields}
        state_extras.update(
            adversary_obs=adversary_obs,
            next_adversary_obs=nstate.info["adversary_obs"],
            adversary_acting=acting.astype(jnp.float32),
            next_adversary_acting=(nstate.info["adversary_left"] > 0).astype(
                jnp.float32
            ),
            violated=nstate.metrics["recovery/violated"],
        )
        transitions = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=env_state.obs,
            action=action,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={"policy_extras": {}, "state_extras": state_extras},
        )
        backup_normalizer = running_statistics.update(
            backup_normalizer, remove_pixels(transitions.observation)
        )
        state = state.replace(  # type: ignore[attr-defined]
            normalizer_params=running_statistics.update(
                state.normalizer_params, adversary_obs
            )
        )
        transitions = float16(transitions)
        transitions = transitions._replace(
            observation=quantize_images(transitions.observation),
            next_observation=quantize_images(transitions.next_observation),
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return backup_normalizer, state, nstate, buffer_state

    def _sample(self, params, normalizer, obs, key):
        """A reparameterized action of the adversary and its log-probability."""
        dist = self.network.parametric_action_distribution
        dist_params = self.network.policy_network.apply(normalizer, params, obs)
        raw = dist.sample_no_postprocessing(dist_params, key)
        return dist.postprocess(raw), dist.log_prob(dist_params, raw)

    def update(
        self,
        state: AdversaryState,
        backup_params: Params,
        backup_normalizer: running_statistics.RunningStatisticsState,
        transitions: Transition,
        key: PRNGKey,
    ) -> Tuple[AdversaryState, dict[str, jax.Array]]:
        """One gradient step of the adversary's critic, actor and temperature."""
        extras = transitions.extras["state_extras"]
        obs, next_obs = extras["adversary_obs"], extras["next_adversary_obs"]
        acting, next_acting = (
            extras["adversary_acting"],
            extras["next_adversary_acting"],
        )
        k_next, k_backup, k_actor, k_alpha = jax.random.split(key, 4)
        alpha = jnp.exp(state.log_alpha)
        q_apply = self.network.qr_network.apply
        norm = state.normalizer_params

        # Target: a fall now, else the next step's value, with the adversary's action
        # while its phase lasts and the backup's after.
        next_adversary, next_log_prob = self._sample(
            state.policy_params, norm, next_obs, k_next
        )
        backup_dist = self.backup_network.parametric_action_distribution
        backup_dist_params = self.backup_network.policy_network.apply(
            backup_normalizer, backup_params, transitions.next_observation
        )
        next_backup = backup_dist.postprocess(
            backup_dist.sample_no_postprocessing(backup_dist_params, k_backup)
        )
        next_action = jnp.where(next_acting[:, None] > 0, next_adversary, next_backup)
        next_q = q_apply(norm, state.target_q_params, next_obs, next_action).min(-1)
        next_v = next_q - next_acting * alpha * next_log_prob
        target = jax.lax.stop_gradient(
            extras["violated"] + transitions.discount * next_v
        )
        keep = 1.0 - extras["truncation"]

        def critic_loss(q_params):
            q = q_apply(norm, q_params, obs, transitions.action)
            return 0.5 * jnp.mean(keep[:, None] * (q - target[:, None]) ** 2)

        weight = acting / jnp.maximum(acting.sum(), 1.0)

        def actor_loss(policy_params):
            action, log_prob = self._sample(policy_params, norm, obs, k_actor)
            q = q_apply(norm, state.q_params, obs, action).mean(-1)
            return jnp.sum(weight * (alpha * log_prob - q))

        def alpha_loss(log_alpha):
            _, log_prob = self._sample(state.policy_params, norm, obs, k_alpha)
            entropy_gap = jax.lax.stop_gradient(-log_prob - self.target_entropy)
            return jnp.sum(weight * jnp.exp(log_alpha) * entropy_gap)

        q_loss, q_grads = jax.value_and_grad(critic_loss)(state.q_params)
        q_updates, q_opt = self.q_optimizer.update(
            q_grads, state.q_optimizer_state, state.q_params
        )
        q_params = optax.apply_updates(state.q_params, q_updates)
        pi_loss, pi_grads = jax.value_and_grad(actor_loss)(state.policy_params)
        pi_updates, pi_opt = self.policy_optimizer.update(
            pi_grads, state.policy_optimizer_state, state.policy_params
        )
        policy_params = optax.apply_updates(state.policy_params, pi_updates)
        a_loss, a_grad = jax.value_and_grad(alpha_loss)(state.log_alpha)
        a_updates, a_opt = self.alpha_optimizer.update(
            a_grad, state.alpha_optimizer_state, state.log_alpha
        )
        log_alpha = optax.apply_updates(state.log_alpha, a_updates)
        target_q_params = jax.tree.map(
            lambda t, q: t * (1 - self.tau) + q * self.tau,
            state.target_q_params,
            q_params,
        )
        state = state.replace(  # type: ignore[attr-defined]
            policy_params=policy_params,
            policy_optimizer_state=pi_opt,
            q_params=q_params,
            target_q_params=target_q_params,
            q_optimizer_state=q_opt,
            log_alpha=log_alpha,
            alpha_optimizer_state=a_opt,
        )
        metrics = {
            "adversary_critic_loss": q_loss,
            "adversary_actor_loss": pi_loss,
            "adversary_alpha_loss": a_loss,
            "adversary_alpha": jnp.exp(log_alpha),
        }
        return state, metrics

    def zero_metrics(self) -> dict[str, jax.Array]:
        names = (
            "adversary_critic_loss",
            "adversary_actor_loss",
            "adversary_alpha_loss",
            "adversary_alpha",
        )
        return {name: jnp.zeros(()) for name in names}
