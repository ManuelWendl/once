"""Is a go to goal recovery prior good enough for the filter of ONCE?

On the CPU system, with the filter of go_to_goal_once (recovery_threshold,
recovery_hold) between a driver and the system, two drivers drive: one straight
to the goals (as a goal seeking policy) and one with random actions, each held
for 10 control steps (as an exploring one). Per prior and driver:

- goals, violations (with the driver or the backup in control), the share of
  blocked steps, and the steps stuck: with the backup in control for 150 steps in
  a row or more, as when the filter locks up;
- Q_R of the driver's actions that ran into an obstacle within the control step,
  against the one of the other passed actions;
- calibration: Q_R(s, a) of passed actions against how often the backup recovers
  after them (Monte Carlo: a, then the backup's mode action, recovered with
  recovery_dwell control steps in a row in R within recovery_horizon and no
  violation);
- for priors that see the goal lidar: how much Q_R moves with it alone.

Then, from rest just outside R (0.31 to 0.5 m from an obstacle), how often the
backup recovers and how often it stands still, which locks up the filter.

A good prior has few violations and no steps stuck, gives actions into an
obstacle a Q_R well below the threshold, its Q_R above the threshold means a
recovery, and it never stands still outside R. Priors with and without the goal
lidar (go_to_goal_recovery_prior_driver) and the time spent of their horizon
(go_to_goal_recovery_prior_time) are told apart by the size of their observation
normalizer. As ONCE, Q_R is asked with no time spent, and the backup acts with
the steps of its current recovery.

Usage: check_gtg_prior.py [--layouts N] [--steps N] <wandb run id or checkpoint directory> [...]
"""

import argparse
import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.sac import checkpoint
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ss2r.algorithms.once.networks import make_networks, recovery_input
from ss2r.benchmark_suites import mujoco_cpu
from ss2r.benchmark_suites.wrappers import (
    GO_TO_GOAL_GOAL_LIDAR,
    go_to_goal_indices_without_goal,
)

repository = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
entity = "manuelwendl-eth-z-rich"
# Steps in a row with the backup in control from which the system counts as stuck.
stuck_steps = 150
# Every this many passed steps, one is evaluated by Monte Carlo.
calibration_every = 40
bins = ((0.0, 0.85), (0.85, 0.9), (0.9, 0.93), (0.93, 0.96), (0.96, 1.01))

with initialize_config_dir(
    version_base=None, config_dir=os.path.join(repository, "ss2r", "configs")
):
    cfg = compose("train_once", overrides=["+experiment=go_to_goal_once"])
    prior_cfg = compose("train_brax", overrides=[f"+experiment={cfg.prior.experiment}"])
environment_cfg = OmegaConf.to_container(cfg.environment, resolve=True)
agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
prior_agent_cfg = OmegaConf.to_container(prior_cfg.agent, resolve=True)
action_repeat = cfg.training.action_repeat
threshold = agent_cfg["recovery_threshold"]
hold = agent_cfg.get("recovery_hold", 1)
horizon = agent_cfg["recovery_horizon"]
dwell = agent_cfg["recovery_dwell"]


def merged(path):
    """A checkpoint with the database of the saving process at the top level."""
    process = os.path.join(path, "ocdbt.process_0")
    if os.path.exists(os.path.join(path, "manifest.ocdbt")) or not os.path.isdir(
        process
    ):
        return path
    copy = os.path.join(tempfile.mkdtemp(), "checkpoint")
    shutil.copytree(path, copy)
    shutil.copy(os.path.join(process, "manifest.ocdbt"), copy)
    shutil.copytree(os.path.join(process, "d"), os.path.join(copy, "d"))
    return copy


def load_prior(source, observation_size, action_size):
    """The prior's functions (the backup's mode action at the steps spent of its
    recovery, Q_R with none spent, both for the filter), and what it sees."""
    if not os.path.exists(source):
        import wandb

        artifact = wandb.Api().artifact(f"{entity}/ss2r/checkpoint:{source}")
        source = artifact.download(os.path.join(tempfile.mkdtemp(), source))
    params = list(checkpoint.load(merged(source)))
    if isinstance(params[0], dict):
        params[0] = running_statistics.RunningStatisticsState(**params[0])
    normalizer, policy, q_params = jax.tree_util.tree_map(
        jnp.asarray, (params[0], params[1], params[3])
    )
    size = normalizer.mean.shape[-1]
    without_goal = go_to_goal_indices_without_goal()
    sizes = {
        observation_size: (None, None),
        len(without_goal): (without_goal, None),
        observation_size + 1: (None, horizon),
        len(without_goal) + 1: (without_goal, horizon),
    }
    if size not in sizes:
        raise ValueError(f"A prior with an observation of size {size}")
    indices, time_horizon = sizes[size]
    _, network, _ = make_networks(
        {
            **agent_cfg,
            "recovery_observation_indices": indices,
            "recovery_time_horizon": time_horizon,
        },
        prior_agent_cfg,
        observation_size,
        action_size,
    )
    dist = network.parametric_action_distribution

    def mode(obs, spent):
        obs = recovery_input(obs, spent, time_horizon)
        return dist.mode(network.policy_network.apply(normalizer, policy, obs))

    def q_r(obs, action):
        obs = recovery_input(obs, 0, time_horizon)
        return network.qr_network.apply(normalizer, q_params, obs, action).mean(-1)

    @jax.jit
    def filter_values(obs, action, spent):
        """Q_R of the action and the backup's action at `spent`."""
        return q_r(obs[None], action[None])[0], mode(obs[None], spent)[0]

    sees = [
        f"{'with' if indices is None else 'without'} the goal lidar",
        f"{'with' if time_horizon else 'without'} the time",
    ]
    return jax.jit(mode), jax.jit(q_r), filter_values, indices is None, sees


def goal_direction(env):
    """Angle of the goal in the frame of the robot, from the simulator."""
    data = env.data
    rotation = data.xmat[env._robot_body_id].reshape(3, 3)
    offset = data.mocap_pos[env._goal_mocap_id] - data.xpos[env._robot_body_id]
    local = rotation.T @ offset
    return np.arctan2(local[1], local[0])


def recovered(env, state, action, mode):
    """Whether the backup recovers after `action` (Monte Carlo, mode actions), and
    whether it stood still."""
    obs = env.set_state(**state)
    start = env.data.qpos[env._robot_qpos_ids][:2].copy()
    in_r = 0
    for spent in range(1, horizon + 1):
        obs, _, terminated, info = env.step(np.asarray(action))
        if terminated:
            return 0.0, False
        in_r = in_r + 1 if info["in_recovery_set"] else 0
        if in_r >= dwell:
            return 1.0, False
        action = np.asarray(mode(jnp.asarray(obs)[None], spent))[0]
    moved = np.linalg.norm(env.data.qpos[env._robot_qpos_ids][:2] - start)
    return 0.0, moved < 0.05


def rest_starts(mode, layouts, per_layout=10, low=0.31, high=0.5):
    """From rest, `low` to `high` from the closest obstacle (outside R): how often
    the backup recovers within the horizon, and how often it stands still."""
    outcomes = []
    for seed in range(layouts):
        rng = np.random.default_rng(seed)
        env = mujoco_cpu.make(environment_cfg, action_repeat, seed)
        env.reset()
        data = env.data
        obstacles = np.array(
            [data.mocap_pos[m][:2] for m in env._hazard_mocap_ids]
            + [data.xpos[b][:2] for b in env._vase_body_ids]
        )
        layout = env.get_state()
        found = 0
        for _ in range(100 * per_layout):
            if found == per_layout:
                break
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(low, high + 0.3)
            center = obstacles[rng.integers(len(obstacles))]
            xy = center + radius * np.array([np.cos(angle), np.sin(angle)])
            if not low <= np.linalg.norm(obstacles - xy, axis=1).min() < high:
                continue
            qpos = layout["qpos"].copy()
            qpos[env._robot_qpos_ids] = [*xy, rng.uniform(-np.pi, np.pi)]
            state = dict(
                qpos=qpos,
                qvel=np.zeros_like(layout["qvel"]),
                mocap_pos=layout["mocap_pos"],
            )
            obs = env.set_state(**state)
            # The backup's first action, as when the filter blocks from the start.
            action = np.asarray(mode(jnp.asarray(obs)[None], 0))[0]
            outcomes.append(recovered(env, state, action, mode))
            found += 1
    return np.array(outcomes, dtype=float).reshape(-1, 2)


def drive(driver, seed, steps, mode, filter_values):
    """Drives behind the filter. The counts and the passed actions."""
    rng = np.random.default_rng(seed)
    env = mujoco_cpu.make(environment_cfg, action_repeat, seed)
    sim = mujoco_cpu.make(environment_cfg, action_repeat, seed)
    obs = env.reset()
    held = np.zeros(env.action_size, np.float32)
    # Steps of the backup's current recovery, as the actor of ONCE counts them.
    left = run = spent = 0
    counts = dict(goals=0, driver=0, backup=0, blocked=0, stuck=0, passed=0)
    crash, passed, calibration = [], [], []
    for t in range(steps):
        if driver == "goal":
            angle = goal_direction(env)
            action = np.array(
                [1.0 if abs(angle) < 0.5 else 0.0, np.clip(2 * angle, -1, 1)],
                np.float32,
            )
        else:
            if t % 10 == 0:
                held = rng.uniform(-1, 1, env.action_size).astype(np.float32)
            action = held
        q, backup = filter_values(jnp.asarray(obs), jnp.asarray(action), spent)
        q = float(q)
        through = False
        if left > 0:
            left -= 1
        elif q < threshold:
            left = hold - 1
        else:
            through = True
        if not through:
            action = np.asarray(backup)
        counts["blocked"] += not through
        spent = 1 if through else (spent + 1) % horizon
        run = 0 if through else run + 1
        counts["stuck"] += run >= stuck_steps
        state, previous = env.get_state(), obs
        obs, reward, terminated, _ = env.step(action)
        counts["goals"] += reward > 0.5
        if through:
            counts["passed"] += 1
            if terminated:
                crash.append(q)
            else:
                passed.append((q, previous, action))
            if counts["passed"] % calibration_every == 0:
                outcome = 0.0
                if not terminated:
                    outcome, _ = recovered(sim, state, action, mode)
                calibration.append((q, outcome))
        if terminated:
            counts["driver" if through else "backup"] += 1
            obs = env.reset()
            left = run = spent = 0
    return counts, crash, passed, calibration


def goal_sensitivity(passed, q_r, swaps=32, states=150):
    """The range of Q_R(s, a) over swaps of the goal lidar from other states."""
    rng = np.random.default_rng(0)
    pool = np.stack([obs for _, obs, _ in passed])
    goal = list(GO_TO_GOAL_GOAL_LIDAR)
    ranges = []
    for i in rng.choice(len(passed), min(states, len(passed)), replace=False):
        _, obs, action = passed[i]
        batch = np.repeat(obs[None], swaps, 0)
        batch[:, goal] = pool[rng.integers(len(pool), size=swaps)][:, goal]
        q = np.asarray(
            q_r(jnp.asarray(batch), jnp.repeat(jnp.asarray(action)[None], swaps, 0))
        )
        ranges.append(q.max() - q.min())
    return np.median(ranges), np.percentile(ranges, 90)


def check(source, layouts, steps):
    probe_env = mujoco_cpu.make(environment_cfg, action_repeat, 0)
    mode, q_r, filter_values, sees_goal, sees = load_prior(
        source, probe_env.observation_size, probe_env.action_size
    )
    print(
        f"\nPrior {source} ({', '.join(sees)}): threshold {threshold}, hold {hold}, "
        f"{layouts} layouts x {steps} control steps per driver"
    )
    print("  driver  goals  violations (driver/backup)  blocked  steps stuck")
    crashes, passes, calibration = [], [], []
    for driver in ("goal", "random"):
        total = dict(goals=0, driver=0, backup=0, blocked=0, stuck=0)
        for seed in range(layouts):
            counts, crash, passed, calib = drive(
                driver, seed, steps, mode, filter_values
            )
            for name in total:
                total[name] += counts[name]
            crashes += crash
            passes += passed
            calibration += calib
        violations = total["driver"] + total["backup"]
        print(
            f"  {driver:6}  {total['goals']:5d}  {violations:10d} "
            f"({total['driver']}/{total['backup']}){'':12}"
            f"{100 * total['blocked'] / (layouts * steps):5.1f}%  {total['stuck']:11d}"
        )
    if not passes:
        print("  No action passed the filter.")
        return
    others = np.array([q for q, _, _ in passes])
    if crashes:
        print(
            f"  Q_R of passed actions into an obstacle: median "
            f"{np.median(crashes):.3f} (n {len(crashes)}, max {np.max(crashes):.3f});"
            f" of the other passed actions: median {np.median(others):.3f}"
            f" (p10 {np.percentile(others, 10):.3f})"
        )
    calibration = np.array(calibration).reshape(-1, 2)
    print("  Q_R of passed actions -> the backup recovers after them (Monte Carlo):")
    if not len(calibration):
        print("    (no sample)")
    for low, high in bins:
        inside = (calibration[:, 0] >= low) & (calibration[:, 0] < high)
        if inside.any():
            print(
                f"    {low:.2f}-{min(high, 1.0):.2f}: {100 * calibration[inside, 1].mean():5.1f}%"
                f" (n {inside.sum()})"
            )
    if sees_goal:
        median, p90 = goal_sensitivity(passes, q_r)
        print(
            f"  The goal lidar alone moves Q_R(s, a) by {median:.3f} "
            f"(median, p90 {p90:.3f}) at passed states."
        )
    rest = rest_starts(mode, 2 * layouts)
    print(
        f"  From rest 0.31-0.5 m from an obstacle (outside R, {len(rest)} starts): "
        f"recovered {100 * rest[:, 0].mean():.0f}%, stood still "
        f"{100 * rest[:, 1].mean():.0f}%."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--layouts", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()
    for source in args.sources:
        check(source, args.layouts, args.steps)
