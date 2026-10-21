"""Actor of ONCE: runs the policy on the (single, non-episodic) real system."""

import time
from typing import Any, Callable, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ss2r.algorithms.once.filter import (
    make_filter_params_template,
    make_recovery_filter_fn,
)
from ss2r.algorithms.once.networks import make_networks
from ss2r.algorithms.once.shared_params import SharedParams
from ss2r.benchmark_suites.mujoco_cpu import RealEnv


def make_policy_step(
    mbpo_network, recovery_network, observation_size: int
) -> Tuple[Callable, int]:
    """Creates the policy step, which takes the parameters as one flat vector.

    A single flat vector is cheap to move between devices and processes, and
    updating it never triggers a recompilation.
    """
    make_policy = make_recovery_filter_fn(mbpo_network, recovery_network)
    template = make_filter_params_template(
        mbpo_network, recovery_network, observation_size
    )
    flat_template, unravel_fn = ravel_pytree(template)

    @jax.jit
    def policy_step(flat_params, obs, key):
        key, key_sample = jax.random.split(key)
        action, extras = make_policy(unravel_fn(flat_params))(obs, key_sample)
        return (
            action,
            extras["behavior_action"],
            extras["intervention"],
            extras["q_r"],
            key,
        )

    return policy_step, flat_template.size


@jax.jit
def flatten_params(params: Any) -> jax.Array:
    return ravel_pytree(params)[0]


def _new_stats() -> dict[str, float]:
    return {
        "steps": 0.0,
        "task_reward": 0.0,
        "intervention": 0.0,
        "q_r": 0.0,
        "in_recovery_set": 0.0,
        "violations": 0.0,
        "step_time": 0.0,
        "max_step_time": 0.0,
        "overruns": 0.0,
    }


class Actor:
    """Steps the environment and accumulates fixed-size chunks of transitions."""

    def __init__(
        self,
        env: RealEnv,
        policy_step: Callable,
        chunk_size: int,
        seed: int,
        record_state: bool = False,
    ):
        self.env = env
        self.policy_step = policy_step
        self.chunk_size = chunk_size
        self.record_state = record_state
        self.key = jax.random.PRNGKey(seed)
        self.params: Any = None
        self.obs = env.reset()
        self.index = 0
        self.chunk = self._new_chunk()
        self.stats = _new_stats()

    def _new_chunk(self) -> dict[str, np.ndarray]:
        n, obs_size = self.chunk_size, self.env.observation_size
        chunk = {
            "observation": np.zeros((n, obs_size), np.float32),
            "action": np.zeros((n, self.env.action_size), np.float32),
            "reward": np.zeros((n,), np.float32),
            "discount": np.zeros((n,), np.float32),
            "next_observation": np.zeros((n, obs_size), np.float32),
            # The action proposed by the behavior policy, and whether the
            # recovery prior took over instead of executing it.
            "behavior_action": np.zeros((n, self.env.action_size), np.float32),
            "intervention": np.zeros((n,), np.float32),
        }
        if self.record_state:
            # The state of the system before every action and the value of the
            # filter, to replay and render the run afterwards.
            chunk["q_r"] = np.zeros((n,), np.float32)
            for name, value in self.env.get_state().items():  # type: ignore
                chunk[f"state_{name}"] = np.zeros((n, *value.shape), value.dtype)
        return chunk

    def set_params(self, params: Any) -> None:
        self.params = params

    def warmup(self) -> None:
        """Compiles the policy, without acting."""
        self.policy_step(self.params, self.obs, self.key)

    def step(self) -> tuple[dict[str, np.ndarray], dict[str, float]] | None:
        """Takes one step. Returns a chunk of transitions once it is full."""
        start = time.perf_counter()
        action, behavior_action, intervention, q_r, self.key = self.policy_step(
            self.params, self.obs, self.key
        )
        action = np.asarray(action)
        intervention = float(intervention)
        i = self.index
        if self.record_state:
            self.chunk["q_r"][i] = float(q_r)
            for name, value in self.env.get_state().items():  # type: ignore
                self.chunk[f"state_{name}"][i] = value
        next_obs, reward, terminated, info = self.env.step(action)
        self.chunk["observation"][i] = self.obs
        self.chunk["action"][i] = action
        self.chunk["reward"][i] = reward
        self.chunk["discount"][i] = 0.0 if terminated else 1.0
        self.chunk["next_observation"][i] = next_obs
        self.chunk["behavior_action"][i] = np.asarray(behavior_action)
        self.chunk["intervention"][i] = intervention
        stats = self.stats
        stats["steps"] += 1
        stats["task_reward"] += reward
        stats["intervention"] += intervention
        stats["q_r"] += float(q_r)
        stats["in_recovery_set"] += float(info["in_recovery_set"])
        if "cost" in info:
            stats["cost"] = stats.get("cost", 0.0) + float(info["cost"])
        if "out_of_arena" in info:
            stats["out_of_arena"] = stats.get("out_of_arena", 0.0) + float(
                info["out_of_arena"]
            )
        if terminated:
            # The hard constraint was violated, the system has to be reset.
            stats["violations"] += 1
            next_obs = self.env.reset()
        self.obs = next_obs
        self.index += 1
        step_time = time.perf_counter() - start
        stats["step_time"] += step_time
        stats["max_step_time"] = max(stats["max_step_time"], step_time)
        if self.index < self.chunk_size:
            return None
        out = self.chunk, self.stats
        self.chunk = self._new_chunk()
        self.stats = _new_stats()
        self.index = 0
        return out


def run_actor(
    environment_cfg: Mapping[str, Any],
    agent_cfg: Mapping[str, Any],
    prior_agent_cfg: Mapping[str, Any],
    action_repeat: int,
    seed: int,
    shared_params_name: str,
    transitions_queue,
    stop_event,
    record_state: bool = False,
) -> None:
    """Entry point of the actor process. Never waits for the learner."""
    from ss2r.benchmark_suites import mujoco_cpu

    env = mujoco_cpu.make(environment_cfg, action_repeat, seed)
    mbpo_network, recovery_network, _ = make_networks(
        agent_cfg, prior_agent_cfg, env.observation_size, env.action_size
    )
    policy_step, num_params = make_policy_step(
        mbpo_network, recovery_network, env.observation_size
    )
    shared_params = SharedParams(num_params, shared_params_name)
    actor = Actor(env, policy_step, agent_cfg["chunk_size"], seed, record_state)
    params = None
    while params is None:
        params = shared_params.read()
    actor.set_params(jnp.asarray(params))
    actor.warmup()
    # Signal the learner that the actor is ready.
    transitions_queue.put(None)
    realtime_factor = agent_cfg["realtime_factor"]
    period = env.dt / realtime_factor if realtime_factor else 0.0
    next_tick = time.perf_counter()
    try:
        while not stop_event.is_set():
            # Deploy the newest policy as soon as it is available.
            params = shared_params.read()
            if params is not None:
                actor.set_params(jnp.asarray(params))
            out = actor.step()
            if out is not None:
                transitions_queue.put(out)
            if period:
                next_tick += period
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    actor.stats["overruns"] += 1
                    next_tick = time.perf_counter()
    finally:
        transitions_queue.cancel_join_thread()
        shared_params.close()
