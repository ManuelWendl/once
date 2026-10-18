"""Safe backup on the rig, on top of rig_eval.py (LAYOUT, CALIB and WAIT_DEG work the same).

MODE=backup    the backup acts from the first frame the ball is seen after a throw (test B)
MODE=zero      zero commands instead: the do-nothing baseline of test B
MODE=shield    the --onnx policy acts while Q_R(obs, its action) >= TAU, else the backup (test A,
               as ONCE's recovery filter, decided at every step, but with the pessimistic Q_R
               of the export: the lowest critic, where ONCE's filter takes their mean)
MODE=behavior  the --onnx policy alone, with Q_R logged (test A without the shield)

Run with the rig stack up (macos/run.sh hw) and macos/env.sh sourced, BACKUP=DIR holding
backup_policy.onnx and backup_q.onnx (cyberrunner_export_onnx.py --recovery):
    LAYOUT=easy MODE=backup BACKUP=DIR python shield_eval.py --onnx DIR/backup_policy.onnx --num-episodes N \
        --max-steps 300 --num-wait-steps 1 --output-dir D
    LAYOUT=easy MODE=shield BACKUP=DIR TAU=0.9 [WAIT_DEG=..] python shield_eval.py --onnx S31.onnx \
        --num-episodes N --output-dir D
With MODE backup or zero an episode starts once the ball, picked up after the last one, is seen
again more than START_MM from the start corner (balls the lift brings back are skipped). The
backup and Q_R see the first 30 observations (the history), the behavior policy all 50. Per step,
the action, the behavior's action, Q_R and whether the backup acted go to D/shield/episode_NNN.npz,
row by row with the runner's own episode_NNN.npz.
"""

import os
import pathlib
import sys

import numpy as np
import onnxruntime as ort

HISTORY = (
    30  # the backup's observation: 5 frames of [alpha, beta, ball_x, ball_y, a0, a1]
)
MODE = os.environ.get("MODE", "shield")
TAU = float(os.environ.get("TAU", "0.9"))
START = (
    np.array([float(x) for x in os.environ.get("START", "154,224").split(",")]) / 1000
)
START_MM = float(os.environ.get("START_MM", "30"))
GONE_FRAMES = (
    30  # the ball must be out of sight this long (0.5 s) before a throw counts
)


def _session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class ShieldPolicy:
    """Behavior policy, backup and Q_R; called once per step with the runner's observation."""

    def __init__(self, backup_dir, behavior_onnx=None, mode=MODE, tau=TAU):
        if mode not in ("backup", "zero", "shield", "behavior"):
            raise ValueError(f"Unknown MODE {mode!r}")
        backup_dir = pathlib.Path(backup_dir)
        self.mode, self.tau = mode, tau
        self.backup = _session(backup_dir / "backup_policy.onnx")
        self.q = _session(backup_dir / "backup_q.onnx")
        self.behavior = None
        if mode in ("shield", "behavior"):
            self.behavior = _session(behavior_onnx)
        self.rows: list = []

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs["state"] if isinstance(obs, dict) else obs, np.float32)
        obs = obs.reshape(-1)
        history = obs[:HISTORY]
        backup = self.backup.run(["action"], {"observation": history})[0]
        behavior = np.full(2, np.nan, np.float32)
        if self.mode == "backup":
            action = backup
        elif self.mode == "zero":
            action = np.zeros(2, np.float32)
        else:
            behavior = self.behavior.run(["action"], {"observation": obs})[0]
            action = behavior
        q = self.q.run(["q"], {"observation": history, "action": action})[0][0]
        intervened = self.mode == "shield" and q < self.tau
        if intervened:
            action = backup
        self.rows.append((action, behavior, q, intervened))
        return action

    def save(self, path) -> None:
        """Write the steps since the last save, then start over."""
        if not self.rows:
            return
        action, behavior, q, intervened = (np.array(x) for x in zip(*self.rows))
        np.savez(
            path, action=action, behavior_action=behavior, q=q, intervened=intervened
        )
        self.rows = []


def _wait(env, until, frames):
    """Read frames until `until(env)` held for `frames` frames in a row."""
    count = 0
    while count < frames:
        env._get_obs()
        count = count + 1 if until(env) else 0


def main():
    import atexit
    from datetime import datetime

    import rig_eval  # noqa: F401  (LAYOUT, CALIB, WAIT_DEG)
    from cyberrunner_dreamer import eval_onnx

    args = sys.argv[1:]
    behavior_onnx = args[args.index("--onnx") + 1]
    out = args[args.index("--output-dir") + 1] if "--output-dir" in args else "."
    log_dir = pathlib.Path(out) / "shield" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    policy = ShieldPolicy(os.environ["BACKUP"], behavior_onnx)
    print(f"MODE={policy.mode} TAU={policy.tau} logs in {log_dir}")
    episodes = []

    def save():
        episodes.append(len(episodes) + 1)
        policy.save(log_dir / f"episode_{episodes[-1]:03d}.npz")

    atexit.register(save)
    eval_onnx.load_onnx_model = lambda path: None
    eval_onnx.ONNXPolicy = lambda session: policy
    env_class = eval_onnx.CyberrunnerGymONNX
    reset = env_class.reset

    def reset_and_save(self):
        if policy.rows:
            save()
        if policy.mode not in ("backup", "zero"):
            return reset(self)
        while True:
            print("Pick up the ball and throw it onto the board")
            _wait(self, lambda e: not e.ball_detected, GONE_FRAMES)
            obs = reset(self)
            if np.linalg.norm(self.current_ball_pos - START) * 1000 > START_MM:
                return obs

    env_class.reset = reset_and_save
    eval_onnx.main()


if __name__ == "__main__":
    main()
