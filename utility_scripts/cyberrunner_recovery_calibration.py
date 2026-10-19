"""Calibrate CyberRunner recovery prior runs: Q_R against the backup's outcome in simulation.

Each checkpoint is asked on sets of states, the same for every checkpoint and run: the run's
own resets (with the backup's action), and one state from each episode of a behavior policy
(with the backup's action, and with the behavior's action as the shield asks), from all its
steps and, with --near-holes, from its steps near a hole. Writes --out/calibration.jsonl,
--out/<run>.npz with the predictions and outcomes, and with --plot --out/<run>.png.
"""

import argparse
import json
import pathlib
import time
from typing import Any, Dict, Tuple

import jax
import numpy as np

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import (
    calibration,
    cyberrunner,
    evaluation,
)

CRITICS = ("q_min", "q_mean")
SHIELD_TAU = "0.9"


def _name(run_dir: pathlib.Path) -> str:
    return "_".join(run_dir.parts[-3:])


def _set_name(within_mm: float) -> str:
    return "behavior" if within_mm == float("inf") else f"behavior_{within_mm:g}mm"


def _checkpoints(run_dir: pathlib.Path, every: int) -> list:
    ckpts = sorted(p for p in (run_dir / "ckpt").iterdir() if p.is_dir())
    return ckpts[::-1][::every][::-1]  # the last one always


def _load(run_dir, ckpt) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """networks, (normalizer, policy params, Q_R params) of a checkpoint."""
    networks, normalizer, policy_params, qr_params = evaluation.load_recovery_params(
        run_dir, ckpt
    )
    return networks, (normalizer, policy_params, qr_params)


def evaluate_run(run_dir, args, behavior: Dict[str, Any], keys, out) -> list:
    """Rows of every checkpoint of one run, also saved as arrays in <run>.npz.

    `behavior` maps a set's name to its behavior states, the behavior's actions there
    and which episodes gave a state."""
    k_physics, k_reset = keys
    task, horizon, ranges = calibration.recovery_task(run_dir)
    env = calibration.make_recovery_env(task, ranges, k_physics, args.states)
    history = task["history_length"] * cyberrunner.FRAME_SIZE
    resets = jax.jit(env.reset)(jax.random.split(k_reset, args.states))
    sets = {"reset": (resets, None, np.ones(args.states, bool))}
    for name, (states, actions, valid) in behavior.items():
        states = calibration.as_recovery_states(states, history)
        sets[name] = (states, actions, np.asarray(valid))
    # (states, action) pairs asked of Q_R; the behavior's action is the shield's question.
    queries = [("reset", "backup")] + [
        (name, action) for name in behavior for action in ("backup", "behavior")
    ]
    ckpts = _checkpoints(run_dir, args.every)
    networks, params = _load(run_dir, ckpts[0])
    if params[0].mean.shape[-1] != history:
        raise ValueError(f"{run_dir}: Q_R sees {params[0].mean.shape[-1]} values")
    calibrator = calibration.Calibrator(env, networks, horizon)
    arrays: dict = {}
    for name, (states, _, valid) in sets.items():
        arrays[f"{name}_valid"] = valid
        arrays[f"{name}_distance"] = calibration.hole_distance(env, states)
        arrays[f"{name}_zero_commands"] = calibrator.zero_commands(params, states)
    rows = []
    for ckpt in ckpts:
        start = time.time()
        _, params = _load(run_dir, ckpt)
        for name, action in queries:
            states, actions, valid = sets[name]
            result = calibrator(
                params, states, actions if action == "behavior" else None
            )
            for key, value in result.items():
                arrays.setdefault(f"{name}_{action}_{key}", []).append(value)
            for critic in CRITICS:
                row = {
                    "run_dir": str(run_dir),
                    "step": int(ckpt.name),
                    "states": name,
                    "action": action,
                    "critic": critic,
                    "zero_commands_reached": float(
                        arrays[f"{name}_zero_commands"][valid].mean()
                    ),
                    **calibration.calibration(
                        result[critic][valid],
                        result["reached"][valid],
                        arrays[f"{name}_distance"][valid],
                    ),
                }
                rows.append(row)
                out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"{_name(run_dir)} step {int(ckpt.name)} ({time.time() - start:.0f} s)")
        for name, action in queries:
            _print(_pick(rows, name, action)[-1])
    arrays["steps"] = [int(c.name) for c in ckpts]
    np.savez(
        args.out / f"{_name(run_dir)}.npz",
        **{k: np.asarray(v) for k, v in arrays.items()},  # type: ignore[arg-type]
    )
    return rows


def _pick(rows, states, action, critic="q_min") -> list:
    return [
        r
        for r in rows
        if (r["states"], r["action"], r["critic"]) == (states, action, critic)
    ]


def _print(row) -> None:
    tau = row["tau"][SHIELD_TAU]
    failures = round(row["count"] * (1 - row["reached"]))
    print(
        f"  {row['states'] + '/' + row['action']:<28} n {row['count']:>5} "
        f"reached {row['reached']:.3f} (zero {row['zero_commands_reached']:.3f}) "
        f"ECE {row['ece']:.3f} Brier {row['brier']:.3f} AUROC {row['auroc']:.3f} | "
        f"TAU {SHIELD_TAU}: fail when passed {tau['fail_when_passed']:.3f}, failures "
        f"passed {tau['failures_passed']:.3f} of {failures}, successes stopped "
        f"{tau['successes_stopped']:.3f}",
        flush=True,
    )


def best_step(rows, shield_set: str) -> int:
    """The checkpoint with the lowest Brier score on the shield's question."""
    return min(_pick(rows, shield_set, "behavior"), key=lambda r: r["brier"])["step"]


def plot(rows, path: pathlib.Path, shield_set: str) -> None:
    """Reliability at the best checkpoint, ECE and Brier over training, and the shield's
    errors over training (lowest critic)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = best_step(rows, shield_set)
    queries = list(dict.fromkeys((r["states"], r["action"]) for r in rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for states, action in queries:
        label = f"{states} states, {action} action"
        series = _pick(rows, states, action)
        bins = next(r for r in series if r["step"] == best)["bins"]
        filled = [i for i, n in enumerate(bins["count"]) if n > 0]
        axes[0, 0].plot(
            [bins["mean_q"][i] for i in filled],
            [bins["reached"][i] for i in filled],
            "o-",
            label=label,
        )
        steps = [r["step"] / 1e6 for r in series]
        axes[0, 1].plot(steps, [r["ece"] for r in series], label=label)
        axes[1, 0].plot(steps, [r["brier"] for r in series], label=label)
        if action == "behavior":
            tau = [r["tau"][SHIELD_TAU] for r in series]
            for key, style in (("fail_when_passed", "-"), ("successes_stopped", "--")):
                axes[1, 1].plot(
                    steps, [t[key] for t in tau], style, label=f"{key}, {states}"
                )
    axes[0, 0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0, 0].set(
        xlabel="Q_R (lowest critic)", ylabel="share reaching R", title=f"step {best}"
    )
    axes[0, 1].set(xlabel="env steps (M)", title="ECE")
    axes[1, 0].set(xlabel="env steps (M)", title="Brier score")
    axes[1, 1].set(xlabel="env steps (M)", title=f"shield at TAU {SHIELD_TAU}")
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=pathlib.Path,
        help="recovery prior Hydra run directories",
    )
    parser.add_argument(
        "--behavior",
        required=True,
        type=pathlib.Path,
        help="behavior policy (PPO) Hydra run directory, best checkpoint",
    )
    parser.add_argument(
        "--out", required=True, type=pathlib.Path, help="results directory"
    )
    parser.add_argument("--states", type=int, default=4096, help="states per set")
    parser.add_argument(
        "--behavior-steps", type=int, default=3000, help="behavior episode length"
    )
    parser.add_argument(
        "--every", type=int, default=1, help="every N-th checkpoint (and the last)"
    )
    parser.add_argument(
        "--near-holes",
        type=float,
        nargs="*",
        default=[],
        help="also, per distance (mm), behavior states closer than that to a hole "
        "centre; the closest is the one the best checkpoint is picked on",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    within_mm = [float("inf")] + sorted(args.near_holes, reverse=True)
    shield_set = _set_name(within_mm[-1])

    worlds = {json.dumps(calibration.physics(d), sort_keys=True) for d in args.run_dirs}
    if len(worlds) > 1:
        raise ValueError(f"Runs with different physics: {worlds}")
    model = evaluation.run_model(args.run_dirs[0])
    if evaluation.run_model(args.behavior) != model:
        raise ValueError(f"The behavior run's model is not the runs' {model}")
    args.out.mkdir(parents=True, exist_ok=True)
    k_physics, k_reset, k_behavior = jax.random.split(jax.random.PRNGKey(args.seed), 3)

    start = time.time()
    task, _, ranges = calibration.recovery_task(args.run_dirs[0])
    env = calibration.make_behavior_env(
        args.behavior, task, ranges, k_physics, args.states
    )
    make_policy, params = evaluation.load_policy_params(
        args.behavior, evaluation.best_checkpoint(args.behavior)
    )
    kept, fell = calibration.behavior_states(
        env,
        make_policy(params),
        args.states,
        args.behavior_steps,
        k_behavior,
        within=[mm / 1000 for mm in within_mm],
    )
    behavior = {_set_name(mm): sets for mm, sets in zip(within_mm, kept)}
    print(
        f"behavior states: {float(fell.mean()):.3f} of the episodes fell, episodes with "
        + ", ".join(f"{n}: {float(v.mean()):.3f}" for n, (_, _, v) in behavior.items())
        + f" ({time.time() - start:.0f} s)",
        flush=True,
    )
    with open(args.out / "calibration.jsonl", "a") as out:
        for run_dir in args.run_dirs:
            rows = evaluate_run(run_dir, args, behavior, (k_physics, k_reset), out)
            best = best_step(rows, shield_set)
            print(
                f"{_name(run_dir)}: best calibrated checkpoint on {shield_set} {best}"
            )
            if args.plot:
                plot(rows, args.out / f"{_name(run_dir)}.png", shield_set)


if __name__ == "__main__":
    main()
