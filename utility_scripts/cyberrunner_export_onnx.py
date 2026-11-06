"""Export a CyberRunner PPO run, or a recovery prior run (--recovery), to ONNX for the robot."""

import argparse

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Hydra run directory")
    parser.add_argument(
        "out", help="ONNX file to write (with --recovery: directory for both files)"
    )
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="recovery prior: backup_policy.onnx and backup_q.onnx",
    )
    args = parser.parse_args()

    if args.recovery:
        best, export = evaluation.best_recovery_checkpoint, evaluation.export_recovery
    else:
        best, export = evaluation.best_checkpoint, evaluation.export
    pick = {"best": best, "last": evaluation.last_checkpoint}
    ckpt = export(args.run_dir, args.out, pick[args.checkpoint](args.run_dir))
    print(f"{args.out}: checkpoint {ckpt}")


if __name__ == "__main__":
    main()
