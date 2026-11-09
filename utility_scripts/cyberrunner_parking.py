"""Where and why does the ball of a CyberRunner ONCE run stay still ("park")?

Reads the trajectory a run stores (training.store_trajectory: <run dir>/trajectory) and
finds the stretches in which the ball does not move. Per stretch: where the ball is, for
how long, what the behavior proposes and what the filter does with it, also in the 10 s
before the stretch.

With --prior (the default is the run's prior.wandb_id), also at the middle of each
stretch, from the stored state:
  - the share of a grid of actions that the filter lets through (Q_R >= threshold), and
    Q_R of 8 directions at full command and of the behavior's action;
  - where the ball goes in the simulation when each of these 8 actions is held for 1 s,
    unfiltered and through the filter (the prior's action where Q_R < threshold).

With the run's newest ONCE checkpoint (<run dir>/ckpt, or --checkpoint), also at the
middle of each stretch: the behavior's task critic Q(s, a) over the action grid (staying,
the behavior's own action, the best and worst action) and for the 8 held actions, and the
model's disagreement (the exploration bonus over optimism) over the same actions. The
checkpoint is the newest, not the one of the stretch: the last stretch fits it best.

With --info-gain [N], also the information gain per real transition of the newest N
(20000) with the checkpoint's model, as the optimism schedule (agent.optimism_schedule)
counts it, and its confidence width beta after 100k, 250k and 500k steps: to set R and
max of the schedule. The model has already learned these transitions: online the
information is larger, so this is a lower bound.

What it tells:
  - the filter blocks the behavior (interventions during or right before the stretch):
    the prior is too conservative there;
  - the filter would let an escape through and the ball can move (the sim test reaches
    a larger path fraction), but the behavior does not propose it: the behavior is stuck
    in a local optimum (its critic or the model);
  - no action moves the ball, or only into a hole: the ball is pinned by the physics;
  - the critic rates staying (a=0) or the behavior's action above the held actions that
    escape in the sim test: the critic does not know that escaping pays; if the
    disagreement of these actions is not above the one of a=0, optimism cannot help.

Usage:
  cyberrunner_parking.py <run dir> [<run dir> ...] [--prior ID|PATH] [--threshold 0.95]
      [--min-duration 30] [--plot parking.png] [--no-sim] [--checkpoint PATH]
      [--no-critic] [--info-gain [N]]

<run dir> is the run's directory, the one with .hydra/ and trajectory/, e.g.
$ONCE_LOG_DIR/train_once/<overrides>/seed=0.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np

repository = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repository)

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import cyberrunner  # noqa: E402
from ss2r.benchmark_suites.mujoco_playground.cyberrunner.sim import (  # noqa: E402
    maze,
)

ENTITY = "manuelwendl-eth-z-rich"
# The ball counts as still in a second in which its mean position moves less than this.
STILL_MM_PER_S = 1.0
BEFORE_S = 10
# Escape test: the actions held, as (command 0, command 1).
DIRECTIONS = [
    (np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
]
HOLD_S = 1.0


def load_trajectory(run_dir):
    """The stored transitions, in order; parts still being written are skipped."""
    paths = sorted(glob.glob(os.path.join(run_dir, "trajectory", "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No trajectory in {run_dir}/trajectory")
    parts = []
    for path in paths:
        try:
            with np.load(path) as part:
                parts.append({name: part[name] for name in part.files})
        except Exception as error:  # A part being written right now.
            print(f"  skipping {os.path.basename(path)}: {error}")
    data = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    with open(os.path.join(run_dir, "trajectory", "metadata.json")) as file:
        metadata = json.load(file)
    return data, metadata


def run_config(run_dir):
    from omegaconf import OmegaConf

    path = os.path.join(run_dir, ".hydra", "config.yaml")
    return OmegaConf.load(path) if os.path.exists(path) else None


def observed_balls(observation, history_length):
    """The newest observed ball of every step (board coordinates, m)."""
    frames = observation[:, : history_length * cyberrunner.FRAME_SIZE]
    frames = frames.reshape(len(observation), history_length, cyberrunner.FRAME_SIZE)
    return frames[:, -1, 2:4]


def still_stretches(ball, discount, steps_per_s, min_duration_s):
    """[start, end) step ranges of at least `min_duration_s` in which the ball is still."""
    n = len(ball) // steps_per_s
    means = ball[: n * steps_per_s].reshape(n, steps_per_s, 2).mean(1)
    resets = discount[: n * steps_per_s].reshape(n, steps_per_s).min(1) == 0
    moved = np.r_[np.inf, np.linalg.norm(np.diff(means, axis=0), axis=1) * 1e3]
    still = (moved < STILL_MM_PER_S) & ~resets
    stretches, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_duration_s:
                stretches.append((start * steps_per_s, i * steps_per_s))
            start = None
    return stretches


def describe(ball, layout):
    """Distances of a ball to the nearest hole and wall (mm)."""
    hole = np.linalg.norm(layout.holes - ball, axis=1).min()
    walls = []
    for x0, x1, y in layout.walls_h:
        walls.append(np.hypot(max(x0 - ball[0], 0, ball[0] - x1), ball[1] - y))
    for y0, y1, x in layout.walls_v:
        walls.append(np.hypot(max(y0 - ball[1], 0, ball[1] - y1), ball[0] - x))
    return hole * 1e3, min(walls) * 1e3


def fmt(v):
    return "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"


def make_run_networks(cfg, env):
    """The behavior/model networks and the recovery networks of a run."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from ss2r.algorithms.once.networks import make_networks

    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(repository, "ss2r", "configs")
    ):
        prior_cfg = compose(
            "train_brax",
            overrides=[
                f"+experiment={cfg.prior.experiment}",
                *cfg.prior.get("overrides", []),
            ],
        )
    mbpo_network, recovery_network, _ = make_networks(
        OmegaConf.to_container(cfg.agent, resolve=True),
        OmegaConf.to_container(prior_cfg.agent, resolve=True),
        env.observation_size,
        env.action_size,
    )
    return mbpo_network, recovery_network


def load_params(path, normalizer_indices=(0,)):
    """The parameters of a checkpoint, the normalizers as their state."""
    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics
    from brax.training.agents.sac import checkpoint

    params = list(checkpoint.load(merged(os.path.abspath(path))))
    for i in normalizer_indices:
        if isinstance(params[i], dict):
            params[i] = running_statistics.RunningStatisticsState(**params[i])
    return [
        None if p is None else jax.tree_util.tree_map(jnp.asarray, p) for p in params
    ]


def latest_checkpoint(run_dir):
    """The newest ONCE checkpoint of a run (ckpt/<12-digit step>), or None."""
    steps = [
        p
        for p in glob.glob(os.path.join(run_dir, "ckpt", "*"))
        if os.path.basename(p).isdigit() and len(os.path.basename(p)) == 12
    ]
    return max(steps, key=lambda p: int(os.path.basename(p))) if steps else None


class Prior:
    """Q_R and the action of a recovery prior, as the filter of the run uses them."""

    def __init__(self, source, cfg, env):
        import jax
        import jax.numpy as jnp

        _, network = make_run_networks(cfg, env)
        if not os.path.exists(source):
            import wandb

            artifact = wandb.Api().artifact(f"{ENTITY}/ss2r/checkpoint:{source}")
            source = artifact.download(os.path.join(tempfile.mkdtemp(), source))
        params = load_params(source)
        normalizer, policy, q_params = params[0], params[1], params[3]

        @jax.jit
        def q_r(obs, actions):
            obs = jnp.broadcast_to(obs, actions.shape[:-1] + obs.shape[-1:])
            return network.qr_network.apply(normalizer, q_params, obs, actions).mean(-1)

        @jax.jit
        def action(obs):
            logits = network.policy_network.apply(normalizer, policy, obs)
            return network.parametric_action_distribution.mode(logits)

        self.q_r = lambda obs, actions: np.asarray(
            q_r(jnp.asarray(obs), jnp.asarray(actions, jnp.float32))
        )
        self.action = lambda obs: np.asarray(action(jnp.asarray(obs)))


class Behavior:
    """The behavior policy, its task critic and the model of a ONCE checkpoint."""

    def __init__(self, path, cfg, env):
        import jax
        import jax.numpy as jnp

        network, _ = make_run_networks(cfg, env)
        # (normalizer, policy, -, task Q, -, log alpha, recovery normalizer, recovery
        # policy, Q_R, model), as ONCE's get_params.
        params = load_params(path, normalizer_indices=(0, 6))
        normalizer, policy, q_params, model_params = (
            params[0],
            params[1],
            params[3],
            params[9],
        )
        self.alpha = float(jnp.exp(params[5]))

        @jax.jit
        def q(obs, actions):
            # The actor maximizes the mean of the critics (use_bro), or their minimum.
            obs = jnp.broadcast_to(obs, actions.shape[:-1] + obs.shape[-1:])
            values = network.qr_network.apply(normalizer, q_params, obs, actions)
            return values.mean(-1) if cfg.agent.use_bro else values.min(-1)

        @jax.jit
        def model(obs, actions):
            # One step of the ensemble, as the learner's predict: the disagreement is
            # the std over the members, averaged over the observation.
            obs = jnp.broadcast_to(obs, actions.shape[:-1] + obs.shape[-1:])
            next_obs, reward, _ = jax.vmap(
                network.model_network.apply, in_axes=(None, 0, None, None)
            )(normalizer, model_params, obs, actions)
            return next_obs.std(0).mean(-1), reward.mean(0)

        @jax.jit
        def variance(obs, actions):
            # The ensemble's variance of every output, as the learner's information
            # gain uses it.
            next_obs, _, _ = jax.vmap(
                network.model_network.apply, in_axes=(None, 0, None, None)
            )(normalizer, model_params, obs, actions)
            return next_obs.var(0)

        @jax.jit
        def action(obs):
            logits = network.policy_network.apply(normalizer, policy, obs)
            return network.parametric_action_distribution.mode(logits)

        as_np = lambda f: lambda *x: jax.tree_util.tree_map(
            np.asarray, f(*(jnp.asarray(v, jnp.float32) for v in x))
        )
        self.q, self.model, self.action = as_np(q), as_np(model), as_np(action)
        self.variance = as_np(variance)


def current_schedule(run_dir):
    """The optimism schedule of the repository's configuration, with the overrides
    the run was started with."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = OmegaConf.load(os.path.join(run_dir, ".hydra", "overrides.yaml"))
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(repository, "ss2r", "configs")
    ):
        cfg = compose("train_once", overrides=list(overrides))
    return cfg.agent.optimism_schedule


def info_gain_calibration(behavior, data, cfg, run_dir, steps_per_s, num_transitions):
    """Information per real transition of the newest stored ones, with the model of
    the checkpoint, and the confidence width of the optimism schedule it implies.

    The model has already learned these transitions, so online, where they count
    before it learns them, the information is larger: the projection is a lower
    bound on the growth of beta."""
    schedule = cfg.agent.get("optimism_schedule")
    if schedule is None:
        schedule = current_schedule(run_dir)
        print(
            "\nthe run predates the optimism schedule: using the repository's, with "
            "the run's overrides"
        )
    obs = data["observation"][-num_transitions:]
    actions = data["action"][-num_transitions:]
    variance = np.concatenate(
        [
            behavior.variance(obs[i : i + 1024], actions[i : i + 1024])
            for i in range(0, len(obs), 1024)
        ]
    )
    dims = schedule.get("dims")
    dims = list(range(variance.shape[-1])) if dims is None else list(dims)
    noise_var = np.broadcast_to(
        np.asarray(schedule.get("noise_std", 1.0), np.float64) ** 2, (len(dims),)
    )
    info = 0.5 * np.log1p(variance[:, dims] / noise_var).sum(-1)
    print(
        f"\ninformation gain per transition, over the newest {len(obs)} "
        f"({len(obs) / steps_per_s / 60:.1f} min), outputs {dims}: mean "
        f"{info.mean():.3f}, median {np.median(info):.3f}, 90th percentile "
        f"{np.percentile(info, 90):.3f}"
    )
    print(
        "  per output, mean sigma/noise_std: "
        + ", ".join(
            f"{d}: {np.sqrt(variance[:, d] / v).mean():.2f}"
            for d, v in zip(dims, noise_var)
        )
    )
    b, r = schedule.get("B", 0.0), schedule.get("R", 1.0)
    delta, cap = schedule.get("delta", 0.1), schedule.get("max")
    log_delta = np.log(1.0 / delta)
    start = cfg.agent.min_replay_size
    for steps in (100_000, 250_000, 500_000):
        gamma = info.mean() * max(steps - start, 0)
        beta = b + r * np.sqrt(2 * (gamma + 1 + log_delta))
        capped = beta if cap is None else min(beta, cap)
        print(
            f"  after {steps} steps: gamma {gamma:.3g}, beta {beta:.3g}"
            + ("" if cap is None else f" (capped at {cap}: {capped:.3g})")
        )
    if cap is not None and info.mean() > 0:
        gamma_cap = ((cap - b) / r) ** 2 / 2 - 1 - log_delta
        steps = start + gamma_cap / info.mean()
        print(
            f"  beta reaches the cap {cap} at gamma {gamma_cap:.3g}, after about "
            f"{steps:.3g} steps ({steps / steps_per_s / 60:.1f} min)"
        )


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


def hold(env, state, action, steps, prior=None, threshold=None):
    """Holds an action from a stored state; with a prior, through the filter.
    Returns the path fraction at the end, its maximum, the ball's displacement (mm),
    whether it fell and the share of filtered steps."""
    obs = env.set_state(state["qpos"], state["qvel"])
    start = env._ball_position().copy()
    best, fell, blocked = env._path_fraction(), False, 0
    for _ in range(steps):
        a = np.asarray(action, np.float32)
        if prior is not None and prior.q_r(obs, a[None])[0] < threshold:
            a, blocked = prior.action(obs), blocked + 1
        obs, _, fell, info = env.step(a)
        best = max(best, info["maxima"]["progress"])
        if fell:
            break
    moved = (env._ball_position() - start) * 1e3
    return env._path_fraction(), best, moved, fell, blocked / steps


def diagnose(run_dir, args, plots):
    from ss2r.benchmark_suites import mujoco_cpu

    print(f"=== {run_dir}")
    data, metadata = load_trajectory(run_dir)
    cfg = run_config(run_dir)
    environment_cfg = metadata["environment"]
    params = environment_cfg["task_params"]
    layout = maze.get_layout(params.get("layout", "easy"))
    history_length = params.get("history_length", 5)
    steps_per_s = int(round(1.0 / metadata["control_dt"]))
    env = mujoco_cpu.make(environment_cfg, metadata["action_repeat"], 0)

    ball = observed_balls(data["observation"], history_length)
    stretches = still_stretches(ball, data["discount"], steps_per_s, args.min_duration)
    total_s = len(ball) / steps_per_s
    still_s = sum(e - s for s, e in stretches) / steps_per_s
    print(
        f"{len(ball)} steps ({total_s / 60:.1f} min), ball still for "
        f"{still_s / 60:.1f} min in {len(stretches)} stretches of >= "
        f"{args.min_duration} s"
    )

    prior, threshold = None, args.threshold
    if cfg is not None and threshold is None:
        threshold = float(cfg.agent.recovery_threshold)
    threshold = 0.95 if threshold is None else threshold
    source = args.prior or (cfg.prior.get("wandb_id") if cfg is not None else None)
    if source and cfg is not None:
        prior = Prior(source, cfg, env)
        print(f"prior {source}, threshold {threshold}")
    elif source:
        print("no .hydra/config.yaml in the run directory: skipping the prior")
    behavior = None
    path = None if args.no_critic else args.checkpoint or latest_checkpoint(run_dir)
    if path and cfg is not None:
        behavior = Behavior(path, cfg, env)
        name = os.path.basename(path.rstrip("/"))
        when = (
            f" (at {int(name) / metadata['action_repeat'] / steps_per_s / 60:.1f} min)"
            if name.isdigit()
            else ""
        )
        print(f"behavior checkpoint {path}{when}, alpha {behavior.alpha:.4f}")
        if args.info_gain:
            info_gain_calibration(
                behavior, data, cfg, run_dir, steps_per_s, args.info_gain
            )

    grid = (
        np.stack(np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21)), -1)
        .reshape(-1, 2)
        .astype(np.float32)
    )
    rows = []
    for start, end in stretches:
        middle = (start + end) // 2
        b = ball[start:end].mean(0)
        hole, wall = describe(b, layout)
        progress = max(env._progress(b), 0.0) / env._total_progress
        before = slice(max(start - BEFORE_S * steps_per_s, 0), start)
        proposed = data["behavior_action"][start:end]
        print(
            f"\n- {start / steps_per_s / 60:6.1f} to {end / steps_per_s / 60:6.1f} min "
            f"({(end - start) / steps_per_s:5.0f} s) at ({b[0] * 1e3:.1f}, "
            f"{b[1] * 1e3:.1f}) mm, path fraction {progress:.3f}, hole {hole:.1f} mm, "
            f"wall {wall:.1f} mm"
        )
        print(
            f"  interventions {data['intervention'][start:end].mean():.3f} "
            f"(10 s before: {data['intervention'][before].mean():.3f}), "
            f"Q_R of the behavior {data['q_r'][start:end].mean():.3f} "
            f"(min {data['q_r'][start:end].min():.3f})"
        )
        print(
            f"  behavior action mean {fmt(proposed.mean(0))} std "
            f"{fmt(proposed.std(0))}, |a| {np.linalg.norm(proposed, axis=1).mean():.2f}"
            f"; executed {fmt(data['action'][start:end].mean(0))}"
            f"; board angles {fmt(np.degrees(data['state_qpos'][start:end, :2].mean(0)))}"
            f" deg"
        )
        row = dict(start=start, end=end, ball=b, progress=progress)
        if prior is not None:
            obs = data["observation"][middle]
            passed = prior.q_r(obs, grid) >= threshold
            q_dirs = prior.q_r(obs, np.asarray(DIRECTIONS, np.float32))
            q_behavior = prior.q_r(obs, proposed.mean(0, keepdims=True))[0]
            print(
                f"  filter at the middle: {passed.mean():.0%} of the action grid "
                f"passes; Q_R of the mean behavior action {q_behavior:.3f}; the "
                f"prior's action {fmt(prior.action(obs))}"
            )
            row["passed"] = passed
        if behavior is not None:
            obs = data["observation"][middle]
            zero = np.zeros((1, 2), np.float32)
            q_grid = behavior.q(obs, grid)
            disagreement, _ = behavior.model(obs, grid)
            q_dirs_task = behavior.q(obs, np.asarray(DIRECTIONS, np.float32))
            disagreement_dirs, _ = behavior.model(
                obs, np.asarray(DIRECTIONS, np.float32)
            )
            print(
                f"  critic: Q(a=0) {behavior.q(obs, zero)[0]:.2f}, Q(mean behavior "
                f"action) {behavior.q(obs, proposed.mean(0, keepdims=True))[0]:.2f}, "
                f"best of the grid {q_grid.max():.2f} at a={fmt(grid[q_grid.argmax()])},"
                f" worst {q_grid.min():.2f}; the policy now {fmt(behavior.action(obs))}"
            )
            print(
                f"  model disagreement: a=0 {behavior.model(obs, zero)[0][0]:.4f}, the "
                f"grid min {disagreement.min():.4f} median "
                f"{np.median(disagreement):.4f} max {disagreement.max():.4f} at "
                f"a={fmt(grid[disagreement.argmax()])}"
            )
            if args.no_sim:
                for k, direction in enumerate(DIRECTIONS):
                    print(
                        f"    a={fmt(direction)}: Q {q_dirs_task[k]:.2f}, "
                        f"disagreement {disagreement_dirs[k]:.4f}"
                    )
        if not args.no_sim:
            state = {
                "qpos": data["state_qpos"][middle],
                "qvel": data["state_qvel"][middle],
            }
            steps = int(HOLD_S * steps_per_s)
            print(
                f"  holding an action for {HOLD_S:.0f} s from the middle (path "
                f"fraction at the end / max, ball moved mm, fell"
                + (", Q_R, filtered share" if prior is not None else "")
                + "):"
            )
            for k, direction in enumerate(DIRECTIONS):
                end_fraction, best, moved, fell, _ = hold(env, state, direction, steps)
                line = (
                    f"    a={fmt(direction)}: {end_fraction:.3f}/{best:.3f} "
                    f"moved {fmt(moved)}{' FELL' if fell else ''}"
                )
                if prior is not None:
                    f_end, f_best, _, f_fell, blocked = hold(
                        env, state, direction, steps, prior, threshold
                    )
                    line += (
                        f" | Q_R {q_dirs[k]:.3f}, filtered {blocked:.0%}: "
                        f"{f_end:.3f}/{f_best:.3f}{' FELL' if f_fell else ''}"
                    )
                if behavior is not None:
                    line += (
                        f" | Q {q_dirs_task[k]:.2f}, disagreement "
                        f"{disagreement_dirs[k]:.4f}"
                    )
                print(line)
        rows.append(row)
    plots.append((run_dir, layout, ball, rows))


def plot(plots, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(plots), figsize=(6 * len(plots), 5.5), squeeze=False
    )
    for ax, (run_dir, layout, ball, rows) in zip(axes[0], plots):
        ax.add_patch(
            plt.Rectangle(
                (0, 0), maze.BOARD_WIDTH * 1e3, maze.BOARD_HEIGHT * 1e3, fill=False
            )
        )
        for x0, x1, y in layout.walls_h:
            ax.plot([x0 * 1e3, x1 * 1e3], [y * 1e3, y * 1e3], "k", lw=2)
        for y0, y1, x in layout.walls_v:
            ax.plot([x * 1e3, x * 1e3], [y0 * 1e3, y1 * 1e3], "k", lw=2)
        for h in layout.holes:
            ax.add_patch(plt.Circle(h * 1e3, 7.5, color="tab:red", alpha=0.6))
        w = layout.waypoints * 1e3
        ax.plot(w[:, 0], w[:, 1], ":", color="tab:blue", lw=1)
        ax.plot(ball[::20, 0] * 1e3, ball[::20, 1] * 1e3, ",", color="0.6")
        for row in rows:
            minutes = (row["end"] - row["start"]) / 3600
            x, y = row["ball"] * 1e3
            ax.scatter(x, y, s=20 + 20 * minutes, color="tab:orange", zorder=3)
            ax.annotate(
                f"{minutes:.0f} min", (x, y), (4, 4), textcoords="offset points"
            )
        ax.set_aspect("equal")
        ax.set_title(
            os.path.basename(os.path.dirname(run_dir.rstrip("/")))[-60:]
            + "/"
            + os.path.basename(run_dir.rstrip("/")),
            fontsize=7,
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nplot: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--prior", help="W&B run id or checkpoint directory")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--plot", default="parking.png")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument(
        "--checkpoint", help="ONCE checkpoint (default: the newest in <run dir>/ckpt)"
    )
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument(
        "--info-gain",
        type=int,
        nargs="?",
        const=20000,
        default=0,
        metavar="N",
        help="information gain of the newest N transitions (20000) and the beta of "
        "the optimism schedule it implies",
    )
    args = parser.parse_args()
    plots = []
    for run_dir in args.run_dirs:
        diagnose(run_dir, args, plots)
    if args.plot:
        plot(plots, args.plot)


if __name__ == "__main__":
    main()
