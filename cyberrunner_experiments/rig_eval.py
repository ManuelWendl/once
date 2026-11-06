"""eval_onnx on the maze given by LAYOUT (easy, medium or hard; required). Easy and medium run
the runner's env with that layout and without the hard maze's goal assist (it forces beta = 1 in
the last 4 cm, the hard maze's final direction); hard is Thomas's runner as it is, as in
macos/run.sh eval_onnx.

Run with the rig stack up (macos/run.sh hw) and macos/env.sh sourced:
    LAYOUT=easy [WAIT_DEG=a,b START=x,y LEVEL_AT_START=0] python rig_eval.py --onnx F --num-episodes N --output-dir D
With WAIT_DEG set, the board holds that tilt while the ball arrives (level_eval's reset).
With CALIB=FILE (calib_fit.py), the camera's ball position is mapped onto the plate before the
policy (and the rollout log) sees it.
"""

import os

import numpy as np
from cyberrunner_dreamer import cyberrunner_layout, eval_onnx

LAYOUTS = ("easy", "medium", "hard")
LAYOUT = os.environ.get("LAYOUT")
if LAYOUT not in LAYOUTS:
    raise SystemExit(f"Set LAYOUT to one of {', '.join(LAYOUTS)} (got {LAYOUT!r})")

if "WAIT_DEG" in os.environ:
    import level_eval  # noqa: F401  (replaces the runner's reset)

if "CALIB" in os.environ:
    import json

    _calib = json.load(open(os.environ["CALIB"]))
    _A, _b = np.array(_calib["A"]), np.array(_calib["b"])
    _msg_to_obs = eval_onnx.CyberrunnerGymONNX._msg_to_obs

    def _calibrated(self, msg):
        _msg_to_obs(self, msg)
        if self.ball_detected:
            ball = _A @ np.asarray(self.current_ball_pos, dtype=float) + _b
            self.current_ball_pos = ball.astype(np.float32)

    eval_onnx.CyberrunnerGymONNX._msg_to_obs = _calibrated


class NoAssistEnv(eval_onnx.CyberrunnerGymONNX):
    def __init__(self, *args, **kwargs):
        kwargs["layout"] = getattr(cyberrunner_layout, f"cyberrunner_{LAYOUT}_layout")
        super().__init__(*args, **kwargs)
        self._policy_action = None

    def step(self, action):
        self._policy_action = np.array(action, dtype=np.float32)
        return super().step(action)

    def _send_action(self, action):
        """The step's own send gets the policy's action (no goal assist); other sends pass."""
        if self._policy_action is not None:
            action, self._policy_action = self._policy_action, None
            self.last_action = action.copy()
        return super()._send_action(action)


if LAYOUT != "hard":
    eval_onnx.CyberrunnerGymONNX = NoAssistEnv

if __name__ == "__main__":
    eval_onnx.main()
