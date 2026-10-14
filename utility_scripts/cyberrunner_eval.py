"""Evaluate trained CyberRunner policies on the fixed simulated conditions."""

import argparse
import json
import pathlib

from omegaconf import OmegaConf

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import evaluation

CONFIGS = pathlib.Path(__file__).parents[1] / "ss2r/configs"
ENV_CONFIG = CONFIGS / "environment/cyberrunner.yaml"
EXPERIMENT_CONFIG = CONFIGS / "experiment/cyberrunner_ppo.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Hydra run directories")
    parser.add_argument("--out", required=True, help="JSON lines file to append to")
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--episode-length", type=int, default=3000)
    parser.add_argument("--impl", default="warp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args()

    full = OmegaConf.to_container(OmegaConf.load(ENV_CONFIG).train_params)
    delay_ms = list(OmegaConf.load(EXPERIMENT_CONFIG).environment.task_params.delay_ms)
    pick = {"best": evaluation.best_checkpoint, "last": evaluation.last_checkpoint}
    ckpts = [pick[args.checkpoint](d) for d in args.run_dirs]
    policies = [
        evaluation.load_policy_params(d, c) for d, c in zip(args.run_dirs, ckpts)
    ]
    make_policy = policies[0][0]
    # evaluate with the maze, ball and motor models the runs were trained with
    models = {
        json.dumps(evaluation.run_model(d), sort_keys=True) for d in args.run_dirs
    }
    if len(models) > 1:
        raise ValueError(f"Runs trained with different models: {models}")
    task = json.loads(models.pop())
    with open(args.out, "a") as out:
        for name, (params, delay) in evaluation.eval_conditions(full, delay_ms).items():
            evaluator = evaluation.Evaluator(
                make_policy,
                params,
                delay,
                num_episodes=args.episodes,
                episode_length=args.episode_length,
                impl=args.impl,
                seed=args.seed,
                task=task,
            )
            for run_dir, ckpt, (_, policy_params) in zip(
                args.run_dirs, ckpts, policies
            ):
                result = evaluator(policy_params)
                success = (result["outcome"] == "success").mean()
                print(f"{run_dir} {name}: success {success:.3f}", flush=True)
                row = {
                    "run_dir": str(run_dir),
                    "condition": name,
                    "checkpoint_step": int(ckpt.name),
                    "outcome": result["outcome"].tolist(),
                    "progress": result["progress"].round(4).tolist(),
                    "length": result["length"].tolist(),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
