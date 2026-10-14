"""Export a CyberRunner PPO run to ONNX for the robot (macos/run.sh eval_onnx)."""

import argparse

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import evaluation, export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Hydra run directory")
    parser.add_argument("out", help="ONNX file to write")
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    args = parser.parse_args()

    pick = {"best": evaluation.best_checkpoint, "last": evaluation.last_checkpoint}
    ckpt = export.export(args.run_dir, args.out, pick[args.checkpoint](args.run_dir))
    print(f"{args.out}: checkpoint {ckpt}")


if __name__ == "__main__":
    main()
