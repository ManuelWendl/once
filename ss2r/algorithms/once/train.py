"""Online Non-episodic Constrained Exploration (ONCE).

Non-episodic MBPO on a single trajectory of the real system. The actor never
stops to wait for training: policies are trained asynchronously and deployed
as soon as they are available.
"""

import multiprocessing
import os
import queue
import shutil
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import jax
import numpy as np
from absl import logging
from brax.training.agents.sac import checkpoint
from ml_collections import config_dict

from ss2r.algorithms.once.actor import (
    Actor,
    flatten_params,
    make_policy_step,
    run_actor,
)
from ss2r.algorithms.once.filter import get_filter_params
from ss2r.algorithms.once.learner import make_learner
from ss2r.algorithms.once.networks import make_networks
from ss2r.algorithms.once.shared_params import SharedParams
from ss2r.algorithms.once.trajectory import TrajectoryWriter
from ss2r.algorithms.sac.types import Metrics, Transition
from ss2r.benchmark_suites import mujoco_cpu


def _to_transition(chunk: Mapping[str, np.ndarray]) -> Transition:
    return Transition(
        observation=chunk["observation"],
        action=chunk["action"],
        reward=chunk["reward"],
        discount=chunk["discount"],
        next_observation=chunk["next_observation"],
        extras={
            "state_extras": {"truncation": np.zeros_like(chunk["reward"])},
            "policy_extras": {
                "behavior_action": chunk["behavior_action"],
                "intervention": chunk["intervention"],
            },
        },
    )


class _SyncActor:
    """Runs the actor in the learner's process, for debugging and tests."""

    def __init__(
        self,
        environment_cfg,
        agent_cfg,
        networks,
        action_repeat,
        seed,
        record_state=False,
    ):
        env = mujoco_cpu.make(environment_cfg, action_repeat, seed)
        policy_step, _ = make_policy_step(*networks, env.observation_size)
        self._actor = Actor(
            env, policy_step, agent_cfg["chunk_size"], seed, record_state
        )

    def publish(self, params: Any) -> None:
        self._actor.set_params(flatten_params(params))

    def get(self, block: bool) -> list:
        del block
        while True:
            out = self._actor.step()
            if out is not None:
                return [out]

    def close(self) -> None:
        pass


class _AsyncActor:
    """Runs the actor in its own process, on the CPU."""

    def __init__(
        self,
        environment_cfg,
        agent_cfg,
        prior_agent_cfg,
        action_repeat,
        seed,
        num_params,
        record_state=False,
    ):
        ctx = multiprocessing.get_context("spawn")
        self._single_threaded = agent_cfg["actor_single_threaded"]
        self._shared_params = SharedParams(num_params)
        self._transitions_queue = ctx.Queue()
        self._stop_event = ctx.Event()
        self._process = ctx.Process(
            target=run_actor,
            args=(
                environment_cfg,
                agent_cfg,
                prior_agent_cfg,
                action_repeat,
                seed,
                self._shared_params.name,
                self._transitions_queue,
                self._stop_event,
                record_state,
            ),
            daemon=True,
        )
        self._started = False

    def start(self, params: Any) -> None:
        # The actor only uses the CPU, such that it never competes with the
        # learner for the accelerator.
        actor_environ = {"JAX_PLATFORMS": "cpu"}
        if self._single_threaded:
            actor_environ[
                "XLA_FLAGS"
            ] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        previous = {name: os.environ.get(name) for name in actor_environ}
        os.environ.update(actor_environ)
        try:
            self._process.start()
        finally:
            for name, value in previous.items():
                if value is None:
                    del os.environ[name]
                else:
                    os.environ[name] = value
        self._write(params)
        while True:
            try:
                assert self._transitions_queue.get(timeout=1.0) is None
                break
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError("The actor process died while starting.")
        self._started = True

    def _write(self, params: Any) -> None:
        flat_params = np.asarray(flatten_params(params))
        assert flat_params.size == self._shared_params.size
        self._shared_params.write(flat_params)

    def publish(self, params: Any) -> None:
        if not self._started:
            self.start(params)
        else:
            self._write(params)

    def get(self, block: bool) -> list:
        out: list = []
        if block:
            while not out:
                try:
                    out.append(self._transitions_queue.get(timeout=1.0))
                except queue.Empty:
                    if not self._process.is_alive():
                        raise RuntimeError("The actor process died.")
        while True:
            try:
                out.append(self._transitions_queue.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self._stop_event.set()
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
        self._transitions_queue.cancel_join_thread()
        self._shared_params.close()


def train(
    *,
    environment_cfg: Mapping[str, Any],
    agent_cfg: Mapping[str, Any],
    prior_agent_cfg: Mapping[str, Any],
    prior_params: Tuple[Any, ...],
    num_timesteps: int,
    action_repeat: int = 1,
    seed: int = 0,
    log_every: int = 1000,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    trajectory_logdir: Optional[str] = None,
):
    """Trains online. `num_timesteps` and `log_every` are in environment steps."""
    c = agent_cfg
    # Only used to query the sizes, the actor owns the environment.
    env = mujoco_cpu.make(environment_cfg, action_repeat, seed)
    observation_size, action_size, control_dt = (
        env.observation_size,
        env.action_size,
        env.dt,
    )
    del env
    mbpo_network, recovery_network, normalize_fn = make_networks(
        agent_cfg, prior_agent_cfg, observation_size, action_size
    )
    recovery_reward_fn, violates_fn = mujoco_cpu.get_recovery_fns(environment_cfg)
    init, insert, learner_step = make_learner(
        mbpo_network=mbpo_network,
        recovery_network=recovery_network,
        normalize_fn=normalize_fn,
        observation_size=observation_size,
        action_size=action_size,
        agent_cfg=agent_cfg,
        recovery_reward_fn=recovery_reward_fn,
        violates_fn=violates_fn,
        action_repeat=action_repeat,
    )
    insert = jax.jit(insert)
    learner_step = jax.jit(learner_step)
    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    (
        training_state,
        behavior_buffer_state,
        recovery_rollout_state,
        recovery_buffer_state,
    ) = init(init_key, prior_params)
    policy_params = lambda ts: get_filter_params(ts, c["recovery_threshold"])

    # Compile before the system starts moving. The outputs are discarded.
    t = time.time()
    dummy_chunk = {
        "observation": np.zeros((c["chunk_size"], observation_size), np.float32),
        "action": np.zeros((c["chunk_size"], action_size), np.float32),
        "reward": np.zeros((c["chunk_size"],), np.float32),
        "discount": np.zeros((c["chunk_size"],), np.float32),
        "next_observation": np.zeros((c["chunk_size"], observation_size), np.float32),
        "behavior_action": np.zeros((c["chunk_size"], action_size), np.float32),
        "intervention": np.zeros((c["chunk_size"],), np.float32),
    }
    insert(training_state, behavior_buffer_state, _to_transition(dummy_chunk))
    jax.block_until_ready(
        learner_step(
            training_state,
            behavior_buffer_state,
            recovery_rollout_state,
            recovery_buffer_state,
            key,
        )
    )
    logging.info("Compiled the learner in %.1f seconds", time.time() - t)

    trajectory_writer = None
    if trajectory_logdir:
        trajectory_writer = TrajectoryWriter(
            trajectory_logdir,
            {
                "environment": environment_cfg,
                "action_repeat": action_repeat,
                "control_dt": control_dt,
                "seed": seed,
            },
        )
    record_state = trajectory_writer is not None
    actor: _AsyncActor | _SyncActor
    if c["asynchronous"]:
        _, num_params = make_policy_step(
            mbpo_network, recovery_network, observation_size
        )
        actor = _AsyncActor(
            environment_cfg,
            agent_cfg,
            prior_agent_cfg,
            action_repeat,
            seed,
            num_params,
            record_state,
        )
    else:
        actor = _SyncActor(
            environment_cfg,
            agent_cfg,
            (mbpo_network, recovery_network),
            action_repeat,
            seed,
            record_state,
        )

    def get_params() -> Tuple[Any, ...]:
        return (
            training_state.normalizer_params,
            training_state.behavior_policy_params,
            None,
            training_state.task_q_params,
            None,
            training_state.alpha_params,
            training_state.recovery_normalizer_params,
            training_state.recovery_policy_params,
            training_state.recovery_q_params,
            training_state.model_params,
        )

    def save(step: int) -> None:
        if not checkpoint_logdir:
            return
        dummy_ckpt_config = config_dict.ConfigDict({"network_factory_kwargs": {}})
        checkpoint.save(checkpoint_logdir, step, get_params(), dummy_ckpt_config)
        # Only keep the latest checkpoint, such that the disk usage does not
        # grow with the number of logs.
        if last_save and last_save != step:
            shutil.rmtree(
                os.path.join(checkpoint_logdir, f"{last_save:012d}"),
                ignore_errors=True,
            )

    env_steps = 0
    last_save = 0
    control_steps = 0
    learner_steps = 0
    violations = 0.0
    costs = 0.0
    next_log = log_every
    window: dict[str, float] = {}
    window_learner_steps = 0
    window_start = time.time()
    learner_metrics: Metrics = {}
    start_time = time.time()
    metrics: Metrics = {}
    try:
        actor.publish(policy_params(training_state))
        while env_steps < num_timesteps:
            # Never wait for data, unless there is not enough to start learning.
            waiting = control_steps < c["min_replay_size"]
            for chunk, stats in actor.get(block=waiting or not c["asynchronous"]):
                training_state, behavior_buffer_state = insert(
                    training_state, behavior_buffer_state, _to_transition(chunk)
                )
                if trajectory_writer:
                    trajectory_writer.add(chunk)
                control_steps += c["chunk_size"]
                env_steps += c["chunk_size"] * action_repeat
                violations += stats["violations"]
                costs += stats.get("cost", 0.0)
                for name, value in stats.items():
                    if name == "max_step_time":
                        window[name] = max(window.get(name, 0.0), value)
                    else:
                        window[name] = window.get(name, 0.0) + value
            if control_steps >= c["min_replay_size"]:
                key, step_key = jax.random.split(key)
                (
                    training_state,
                    recovery_rollout_state,
                    recovery_buffer_state,
                    learner_metrics,
                ) = learner_step(
                    training_state,
                    behavior_buffer_state,
                    recovery_rollout_state,
                    recovery_buffer_state,
                    step_key,
                )
                learner_steps += 1
                window_learner_steps += 1
                if learner_steps % c["publish_every"] == 0:
                    actor.publish(policy_params(training_state))
                else:
                    jax.block_until_ready(training_state.gradient_steps)
            if env_steps >= next_log and window.get("steps", 0.0) > 0:
                next_log = (env_steps // log_every + 1) * log_every
                steps = window["steps"]
                elapsed = time.time() - window_start
                metrics = {
                    f"training/{name}": float(value)
                    for name, value in jax.device_get(learner_metrics).items()
                }
                metrics |= {
                    "online/task_reward": window["task_reward"] / steps,
                    "online/intervention": window["intervention"] / steps,
                    "online/q_r": window["q_r"] / steps,
                    "online/in_recovery_set": window["in_recovery_set"] / steps,
                    "online/violations": violations,
                    "online/step_time": window["step_time"] / steps,
                    "online/max_step_time": window["max_step_time"],
                    "online/overruns": window["overruns"],
                    "online/control_frequency": steps / elapsed,
                    "online/realtime_factor": steps * control_dt / elapsed,
                    "learner/steps_per_second": window_learner_steps / elapsed,
                    "learner/update_to_data": window_learner_steps
                    * c["critic_grad_updates_per_step"]
                    / steps,
                    "learner/steps": learner_steps,
                    "training/walltime": time.time() - start_time,
                }
                if "cost" in window:
                    # Environment steps with a cost, for environments with one.
                    metrics["online/costs"] = costs
                logging.info(metrics)
                progress_fn(env_steps, metrics)
                save(env_steps)
                last_save = env_steps
                if trajectory_writer:
                    trajectory_writer.flush(env_steps)
                window = {}
                window_learner_steps = 0
                window_start = time.time()
    finally:
        actor.close()
        if trajectory_writer:
            trajectory_writer.flush(env_steps)
    if last_save != env_steps:
        save(env_steps)
    logging.info("total steps: %s", env_steps)
    return get_params(), metrics
