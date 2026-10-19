"""eval_onnx starting level, as in the sim: while the ball arrives the board is held slightly tilted
towards the start corner (Thomas's reset leaves it at about -4.3/-1.1 deg), then levelled.

Run with the rig stack up (macos/run.sh hw) and macos/env.sh sourced:
    [WAIT_DEG=a,b] [START=x,y] [START_MM=r] [LEVEL_AT_START=0] python level_eval.py --onnx F --num-episodes N --output-dir D
"""

import collections
import os
import time

import numpy as np
import rclpy
from cyberrunner_dreamer import eval_onnx
from std_srvs.srv import Trigger

LEVEL_DEG = 0.5  # levelled once both axes are within this for 10 frames
# tilt that rolls the ball into the start corner while it arrives (WAIT_DEG="alpha,beta")
WAIT_DEG = tuple(float(x) for x in os.environ.get("WAIT_DEG", "-3.0,-1.0").split(","))
# start corner; the ball must rest within START_MM of it (START="x,y" in mm)
START = (
    np.array([float(x) for x in os.environ.get("START", "154,224").split(",")]) / 1000
)
START_MM = float(os.environ.get("START_MM", "15"))
# still: moved less than STILL_MM in the last STILL_FRAMES
STILL_MM, STILL_FRAMES = 2.0, 15
WAIT_FRAMES = 60  # the ball must rest still at the start this long (1 s) before the episode starts
VALID_DEG = 12.0  # larger readings are estimator errors: ignored
LEVEL_FRAMES = 120  # active levelling at most (2 s) once the ball is in place
# 0: start from the corner tilt
LEVEL_AT_START = os.environ.get("LEVEL_AT_START", "1") == "1"
# levelled target: a hair towards the corner keeps the ball there (inside the training tilt DR)
LEVEL_TARGET_DEG = (-0.3, -0.2)
KP, KI = 0.5, 0.03  # Thomas's levelling P gain, plus a faster integral
# command limits while tilted and while levelling
TILT_COMMAND, LEVEL_COMMAND = 0.4, 0.25
Env = eval_onnx.CyberrunnerGymONNX
original_msg_to_obs = Env._msg_to_obs


def msg_to_obs(self, msg):
    """Also update the board angles when no ball is seen."""
    original_msg_to_obs(self, msg)
    angles = np.array([msg.state.alpha, msg.state.beta], dtype=np.float32)
    if np.all(np.isfinite(angles)):
        self.current_joint_pos = angles


class Tilt:
    """P+I control of the board angles towards a target, one frame at a time."""

    def __init__(self, target_deg, max_command):
        self.target, self.integral, self.max = (
            np.radians(target_deg),
            np.zeros(2),
            max_command,
        )

    def step(self, env):
        env._get_obs()
        angle = np.asarray(env.current_joint_pos, dtype=float)
        if not np.all(np.abs(angle) < np.radians(VALID_DEG)):
            env._send_action(np.zeros(2))
            return angle, False
        error = angle - self.target
        self.integral = np.clip(self.integral - KI * error / 0.15, -0.2, 0.2)
        env._send_action(
            np.clip(-KP * error / 0.15 + self.integral, -self.max, self.max)
        )
        return angle, bool(np.all(np.abs(error) < np.radians(LEVEL_DEG)))


def level(env, frames):
    """Drive the board level for at most `frames`, then zero the motors; returns the angles."""
    tilt, calm = Tilt(LEVEL_TARGET_DEG, LEVEL_COMMAND), 0
    for _ in range(frames):
        angle, close = tilt.step(env)
        calm = calm + 1 if close else 0
        if calm >= 10:
            break
    env._send_action(np.zeros(2))
    return angle


def at_start(env):
    return np.linalg.norm(np.asarray(env.current_ball_pos) - START) < START_MM / 1000


def reset(self):
    print("Resetting ...")
    self.episodes += 1
    print(f"Previous episode length: {self.steps / 60.0:.2f}s")
    print(f"Episodes: {self.episodes}")
    self.steps, self.success, self.ball_detected = 0, False, False
    if self.future is not None:  # Thomas's board reset
        rclpy.spin_until_future_complete(self.node, self.future, timeout_sec=10)
    frames = 0
    while True:
        # hold the tilt until the ball rests in the corner
        tilt, seen = Tilt(WAIT_DEG, TILT_COMMAND), 0
        recent = collections.deque(maxlen=STILL_FRAMES)
        while seen < max(self.num_wait_steps, WAIT_FRAMES):
            tilt.step(self)
            frames += 1
            ball = np.asarray(self.current_ball_pos, dtype=float)
            recent.append(ball)
            still = (
                len(recent) == STILL_FRAMES
                and np.linalg.norm(ball - recent[0]) < STILL_MM / 1000
            )
            seen = seen + 1 if self.ball_detected and at_start(self) and still else 0
            if frames % 300 == 0:  # every 5 s: why the episode has not started yet
                a = np.degrees(np.asarray(self.current_joint_pos, dtype=float))
                print(
                    f"  waiting {frames // 60} s: ball {'seen' if self.ball_detected else 'NOT seen'} at "
                    f"({1000 * ball[0]:.0f}, {1000 * ball[1]:.0f}) mm, at start {at_start(self)}, still {still}, "
                    f"board ({a[0]:+.1f}, {a[1]:+.1f}) deg",
                    flush=True,
                )
            if (
                frames % (30 * 60) == 0 and seen < self.num_wait_steps
            ):  # as Thomas: shake, reset
                for client in (self.shake_client, self.client):
                    rclpy.spin_until_future_complete(
                        self.node, client.call_async(Trigger.Request()), timeout_sec=10
                    )
                tilt = Tilt(WAIT_DEG, TILT_COMMAND)
        if not LEVEL_AT_START:
            self._send_action(np.zeros(2))
            angle = np.asarray(self.current_joint_pos, dtype=float)
            break
        angle = level(self, LEVEL_FRAMES)
        self._get_obs()
        if self.ball_detected and at_start(self):
            break
        print("  ball left the corner while levelling: tilting again")
    ball = 1000 * np.asarray(self.current_ball_pos, dtype=float)
    print(
        f"  start: alpha {np.degrees(angle[0]):+.2f}, beta {np.degrees(angle[1]):+.2f} deg, "
        f"ball ({ball[0]:.0f}, {ball[1]:.0f}) mm"
        f"{'' if np.all(np.abs(angle) < np.radians(2 * LEVEL_DEG)) else ' (NOT level)'}, after {frames} frames"
    )

    self._reset_history()
    self.prev_path_idx, _ = self.p.closest_point(self.current_ball_pos)
    self.last_action = np.zeros(2, dtype=np.float32)
    self.last_time = time.time()
    return self._build_observation()


Env._msg_to_obs = msg_to_obs
Env.reset = reset

if __name__ == "__main__":
    eval_onnx.main()
