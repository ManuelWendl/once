"""The CPU simulator behind the rig's bridge: tests everything but the rig. Paces itself at
60 Hz as the camera, loses sight of the ball when it falls, and waits at a goal as the operator.

    python sim_bridge.py [--goal-pause-s S] [--fall-reset-s S] [--seed N]
"""

import argparse
import time

import numpy as np
from bridge_server import serve

from ss2r.benchmark_suites.mujoco_cpu.cyberrunner import CTRL_DT, CyberRunnerCPU


class SimBridge:
    def __init__(self, sim: CyberRunnerCPU, goal_pause_s: float, fall_reset_s: float):
        self.sim = sim
        self.goal_pause_s = goal_pause_s
        self.fall_reset_s = fall_reset_s
        self.next_frame = time.monotonic()
        self.ball_seen = np.zeros(2)
        self.detected = True

    def measurement(self) -> dict:
        return {
            "ball": np.asarray(self.ball_seen, float).tolist(),
            "joints": np.asarray(self.sim._history[-1][:2], float).tolist(),
            "detected": self.detected,
            "time": time.time(),
        }

    def handle(self, request: dict) -> dict:
        command = request["command"]
        if command == "step":
            return self.step(request["action"])
        if command == "reset":
            return self.reset(request["cause"])
        if command == "wait_ball":
            return self.measurement()
        raise ValueError(f"Unknown command {command}")

    def step(self, action: list) -> dict:
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        *_, ball_seen, in_hole, _, _ = self.sim._control_step(action)
        self.ball_seen = ball_seen
        self.detected = not in_hole
        self.next_frame += CTRL_DT
        wait = self.next_frame - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        else:
            self.next_frame = time.monotonic()
        return self.measurement()

    def reset(self, cause: str) -> dict:
        time.sleep(
            {"goal": self.goal_pause_s, "fall": self.fall_reset_s}.get(cause, 0.0)
        )
        self.sim.reset()
        self.ball_seen = self.sim._history[-1][2:4]
        self.detected = True
        self.next_frame = time.monotonic()
        return self.measurement()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--goal-pause-s", type=float, default=5.0)
    parser.add_argument("--fall-reset-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import pathlib

    configs = pathlib.Path(__file__).resolve().parents[1] / "ss2r/configs"
    with initialize_config_dir(str(configs), version_base=None):
        cfg = compose("train_once", overrides=["+experiment=cyberrunner_once"])
    environment = OmegaConf.to_container(cfg.environment, resolve=True)
    sim = CyberRunnerCPU(
        **environment["task_params"],
        real_params=environment["real_params"],
        seed=args.seed,
    )
    bridge = SimBridge(sim, args.goal_pause_s, args.fall_reset_s)
    serve(bridge.handle, lambda: None, args.port)


if __name__ == "__main__":
    main()
