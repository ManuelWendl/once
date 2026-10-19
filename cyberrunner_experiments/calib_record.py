"""Record the camera's ball positions for the plate calibration: the board is levelled once, then the
motors are off so it stays still; put the ball still in each corner of the plate for about 5 s,
lifting it out in between.

Run with the rig stack up (macos/run.sh hw) and macos/env.sh sourced:
    python calib_record.py OUT.npz [SECONDS]
"""

import sys
import time

import level_eval  # noqa: F401  (board angles also when no ball is seen)
import numpy as np
from cyberrunner_dreamer import eval_onnx
from level_eval import level

if __name__ == "__main__":
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
    env = eval_onnx.CyberrunnerGymONNX(num_wait_steps=30, eval_mode=True)
    level(env, 180)  # at most 3 s, then the motors are off
    rows, t0 = [], time.time()
    print(
        f"recording {seconds:.0f} s: put the ball still in each corner for about 5 s",
        flush=True,
    )
    while time.time() - t0 < seconds:
        env._get_obs()
        angle = np.asarray(env.current_joint_pos, dtype=float)
        ball = (
            np.asarray(env.current_ball_pos, dtype=float)
            if env.ball_detected
            else np.full(2, np.nan)
        )
        rows.append([time.time() - t0, *ball, *angle])
        if len(rows) % 300 == 0:
            print(
                f"  {rows[-1][0]:5.0f} s  ball {np.round(1000 * ball)} mm", flush=True
            )
    np.savez(sys.argv[1], data=np.array(rows))
    print(f"saved {len(rows)} frames to {sys.argv[1]}")
