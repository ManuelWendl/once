"""Calibration of a recovery prior's Q_R against the backup's outcome in simulation."""

import pathlib
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf
from scipy import stats

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import (
    cyberrunner,
    evaluation,
    randomize,
    recovery,
)
from ss2r.benchmark_suites.wrappers import BraxDomainRandomizationVmapWrapper

# Shield thresholds on Q_R.
TAUS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
NUM_BINS = 10
# Bands of the ball's distance to the nearest hole centre (m).
DISTANCE_BANDS = (0.0, recovery.RECOVERY_HOLE_DISTANCE, 0.025, float("inf"))
# Options of the behavior run that shape its observation.
OBSERVATION_KEYS = ("history_length", "num_path_points", "path_step", "path_distance")


def _config(run_dir) -> Any:
    return OmegaConf.load(pathlib.Path(run_dir) / ".hydra" / "config.yaml")


def recovery_task(run_dir) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    """A recovery prior run's task options, horizon and evaluation randomization."""
    cfg = _config(run_dir)
    task = OmegaConf.to_container(cfg.environment.task_params)
    task["adversary_fraction"] = 0.0  # type: ignore[index]  # the backup alone
    ranges = OmegaConf.to_container(cfg.environment.eval_params)
    return task, int(cfg.training.episode_length), ranges  # type: ignore


def physics(run_dir) -> Dict[str, Any]:
    """What a run's simulated worlds depend on: model options, randomization, delay."""
    task, _, ranges = recovery_task(run_dir)
    return {
        **evaluation.run_model(run_dir),
        "ranges": ranges,
        "delay_ms": task["delay_ms"],
    }


def _randomized(env, ranges: Mapping[str, Any], key: jax.Array, num: int) -> Any:
    """env over num randomized worlds; the same key gives the same worlds."""
    keys = jax.random.split(key, num)
    ranges = OmegaConf.create(dict(ranges))
    return BraxDomainRandomizationVmapWrapper(
        env,
        lambda sys: randomize.domain_randomization(sys, keys, ranges),
        augment_state=False,
    )


def make_recovery_env(task, ranges, key: jax.Array, num: int) -> Any:
    """The recovery task (MJX) over num randomized worlds."""
    cfg = recovery.default_config()
    for name, value in task.items():
        cfg[name] = value
    cfg.impl = "jax"
    return _randomized(recovery.CyberRunnerRecovery(cfg), ranges, key, num)


def make_behavior_env(behavior_dir, task, ranges, key: jax.Array, num: int) -> Any:
    """The behavior run's maze and observation, from the maze's start, with the recovery
    task's delay (MJX) over the same worlds as make_recovery_env."""
    behavior = _config(behavior_dir).environment.task_params
    cfg = cyberrunner.default_config()
    for name, value in evaluation.run_model(behavior_dir).items():
        cfg[name] = value
    for name in OBSERVATION_KEYS:
        cfg[name] = behavior[name]
    cfg.impl = "jax"
    cfg.random_start = False
    cfg.delay_ms = list(task["delay_ms"])
    return _randomized(cyberrunner.CyberRunner(cfg), ranges, key, num)


def _select(mask: jax.Array, new: Any, old: Any) -> Any:
    """new where mask (over the leading axis), else old."""
    return jax.tree.map(
        lambda a, b: jp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), a, b),
        new,
        old,
    )


def keep(key: jax.Array, active: jax.Array, count: jax.Array) -> jax.Array:
    """Reservoir sampling: the count-th active step replaces the kept one with
    probability 1 / count, so the kept step is uniform over the episode."""
    return (active > 0) & (jax.random.uniform(key, active.shape) * count < 1.0)


def behavior_states(
    env,
    policy: Callable,
    num: int,
    steps: int,
    key: jax.Array,
    within: Sequence[float] = (float("inf"),),
) -> Tuple[Tuple[Tuple[Any, jax.Array, jax.Array], ...], jax.Array]:
    """For each distance in `within` (m), one state of each behavior episode, uniform
    over its steps with the ball centre closer than that to a hole centre, the
    behavior's action there and whether the episode had such a step; and whether each
    episode fell into a hole."""
    base = env.unwrapped
    k_reset, k_policy, k_keep = jax.random.split(key, 3)

    def run():
        state = env.reset(jax.random.split(k_reset, num))

        def step(carry, keys):
            state, active, kept, fell = carry
            action, _ = policy(state.obs, keys[0])
            ball = jax.vmap(base.ball_position)(state.data)
            distance = recovery.hole_distance(ball, base.holes)
            new_kept = []
            for i, (d, (states, actions, count)) in enumerate(zip(within, kept)):
                near = active * (distance < d)
                count = count + near
                take = keep(jax.random.fold_in(keys[1], i), near, count)
                states = _select(take, state, states)
                actions = jp.where(take[:, None], action, actions)
                new_kept.append((states, actions, count))
            state = env.step(state, action)
            fell = fell + active * state.done * (1.0 - state.metrics["success"])
            active = active * (1.0 - state.done)
            return (state, active, tuple(new_kept), fell), None

        zeros = jp.zeros(num)
        kept = tuple((state, jp.zeros((num, 2)), zeros) for _ in within)
        carry = (state, jp.ones(num), kept, zeros)
        keys = (jax.random.split(k_policy, steps), jax.random.split(k_keep, steps))
        (_, _, kept, fell), _ = jax.lax.scan(step, carry, keys)
        return tuple((s, a, count > 0) for s, a, count in kept), fell

    return jax.jit(run)()


def as_recovery_states(states: Any, history_size: int) -> Any:
    """Behavior states as the recovery task's: the history as observation, nothing held
    in R yet."""
    zeros = jp.zeros_like(states.done)
    info = {
        **states.info,
        "recovery_reward": zeros,
        "recovery_hold": jp.zeros(zeros.shape, jp.int32),
    }
    metrics = {"recovery/reached": zeros, "recovery/violated": zeros}
    return states.replace(
        obs=states.obs[:, :history_size],
        reward=zeros,
        done=zeros,
        info=info,
        metrics=metrics,
    )


def hole_distance(env, states: Any) -> np.ndarray:
    """Distance from the ball centre to the nearest hole centre in each state."""
    base = env.unwrapped
    ball = jax.vmap(base.ball_position)(states.data)
    return np.asarray(recovery.hole_distance(ball, base.holes))


class Calibrator:
    """Q_R of a recovery prior's checkpoints, and whether the ball then reaches R or
    falls into a hole, from fixed recovery task states."""

    def __init__(self, env, networks, horizon: int):
        def backup(normalizer, policy_params, obs):
            logits = networks.policy_network.apply(normalizer, policy_params, obs)
            return networks.parametric_action_distribution.mode(logits)

        def q(normalizer, qr_params, obs, action):
            return networks.qr_network.apply(normalizer, qr_params, obs, action)

        def outcome(normalizer, policy_params, states, first_action, use_backup):
            """first_action, then the backup (use_backup 1) or zero commands (0):
            whether the ball reaches R, and whether it falls first."""

            def step(carry, t):
                state, active, reached, fell = carry
                later = use_backup * backup(normalizer, policy_params, state.obs)
                state = env.step(state, jp.where(t == 0, first_action, later))
                reached = reached + active * state.metrics["recovery/reached"]
                fell = fell + active * state.metrics["recovery/violated"]
                active = active * (1.0 - state.done)
                return (state, active, reached, fell), None

            ones = jp.ones_like(states.done)
            carry = (states, ones, 0.0 * ones, 0.0 * ones)
            carry, _ = jax.lax.scan(step, carry, jp.arange(horizon))
            return carry[2], carry[3]

        self._backup, self._q, self._outcome = (
            jax.jit(backup),
            jax.jit(q),
            jax.jit(outcome),
        )

    def __call__(
        self, params: Tuple[Any, Any, Any], states: Any, actions: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """Q_R (lowest and mean critic) of the backup's action, or of `actions`, and
        whether the ball reaches R within the horizon with that action, then the backup,
        or falls first (neither: a timeout)."""
        normalizer, policy_params, qr_params = params
        if actions is None:
            actions = self._backup(normalizer, policy_params, states.obs)
        q = np.asarray(self._q(normalizer, qr_params, states.obs, actions))
        reached, fell = self._outcome(normalizer, policy_params, states, actions, 1.0)
        return {
            "q_min": q.min(-1),
            "q_mean": q.mean(-1),
            "reached": np.asarray(reached),
            "fell": np.asarray(fell),
        }

    def zero_commands(self, params: Tuple[Any, Any, Any], states: Any) -> np.ndarray:
        """Whether the ball reaches R within the horizon with zero commands."""
        normalizer, policy_params, _ = params
        actions = jp.zeros((states.done.shape[0], 2))
        reached, _ = self._outcome(normalizer, policy_params, states, actions, 0.0)
        return np.asarray(reached)


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    """Probability that a state reaching R has a higher p than one that does not."""
    positive = y > 0
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(p)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration(
    p: np.ndarray, y: np.ndarray, distance: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Reliability of the predicted probabilities p of reaching R against the outcomes y:
    per bin of p its count, mean p and share reached; ECE, Brier score and AUROC; per
    shield threshold TAU the share of states let through (p >= TAU), of those the share
    that then falls short of R, the share of the failures let through and of the
    successes stopped; and per band of hole distance its count, mean p and share reached."""
    raw, y = np.asarray(p, float), np.asarray(y, float) > 0
    p = np.clip(raw, 0.0, 1.0)
    bins = np.minimum((p * NUM_BINS).astype(int), NUM_BINS - 1)
    count = np.bincount(bins, minlength=NUM_BINS)
    mean_p = np.bincount(bins, p, NUM_BINS) / np.maximum(count, 1)
    share = np.bincount(bins, y, NUM_BINS) / np.maximum(count, 1)
    result: Dict[str, Any] = {
        "count": int(p.size),
        "reached": _mean(y),
        "mean_q": _mean(p),
        "ece": float((count * np.abs(share - mean_p)).sum() / p.size),
        "brier": _mean((p - y) ** 2),
        "auroc": auroc(raw, y),
        "bins": {
            "count": count.tolist(),
            "mean_q": mean_p.tolist(),
            "reached": share.tolist(),
        },
        "tau": {},
    }
    for tau in TAUS:
        passed = p >= tau
        result["tau"][str(tau)] = {
            "passed": _mean(passed),
            "fail_when_passed": _mean(~y[passed]),
            "failures_passed": _mean(passed[~y]),
            "successes_stopped": _mean(~passed[y]),
        }
    if distance is not None:
        result["distance_bands"] = [
            {
                "band_m": [low, high],
                "count": int(band.sum()),
                "mean_q": _mean(p[band]),
                "reached": _mean(y[band]),
            }
            for low, high in zip(DISTANCE_BANDS[:-1], DISTANCE_BANDS[1:])
            for band in [(distance >= low) & (distance < high)]
        ]
    return result


def upper_95(k: int, n: int) -> float:
    """One-sided 95% upper bound of a rate seen k times in n (Clopper-Pearson)."""
    if n == 0:
        return float("nan")
    return 1.0 if k >= n else float(stats.beta.ppf(0.95, k + 1, n - k))


def interventions(
    q: np.ndarray, behavior_fell: np.ndarray, backup_fell: np.ndarray
) -> Dict[str, Any]:
    """The shield's takeovers against what would have happened, per threshold TAU.

    For each state, q is Q_R(obs, behavior action), behavior_fell whether the behavior
    alone falls from there, and backup_fell whether the ball falls with the backup
    taking over. The shield takes over where q < TAU: usefully if the behavior alone
    falls and the backup does not, in vain if both fall, superfluously if the behavior
    alone does not fall (harmfully if the backup then falls). It misses the states it
    lets through where the behavior alone falls; missed_upper_95 bounds the share of
    the behavior's falls it misses."""
    q = np.asarray(q, float)
    behavior_fell = np.asarray(behavior_fell) > 0
    backup_fell = np.asarray(backup_fell) > 0
    falls = int(behavior_fell.sum())
    result: Dict[str, Any] = {"count": int(q.size), "behavior_falls": falls, "tau": {}}
    for tau in TAUS:
        takeover = q < tau
        missed = int((~takeover & behavior_fell).sum())
        result["tau"][str(tau)] = {
            "takeovers": int(takeover.sum()),
            "useful": int((takeover & behavior_fell & ~backup_fell).sum()),
            "in_vain": int((takeover & behavior_fell & backup_fell).sum()),
            "superfluous": int((takeover & ~behavior_fell).sum()),
            "harmful": int((takeover & ~behavior_fell & backup_fell).sum()),
            "missed": missed,
            "missed_upper_95": upper_95(missed, falls),
        }
    return result
