"""The rig for ONCE's actor: steps, resets and measurements over a local socket.

Run with the rig stack up (macos/run.sh hw) and macos/env.sh sourced, before rig_actor.py:
    LAYOUT=easy LEVEL_AT_START=0 CALIB=$PWD/easy_plate.json WAIT_DEG=-2.5,-0.75 START=148,222 \
        START_MM=15 python rig_bridge.py --log-dir D
The motors go to zero if the actor is silent for 100 ms, disconnects, or the bridge stops.
The lift runs only while the ball is not seen on the board. At a goal the board levels and a sound plays: take the ball off the board and put it at the
start. Training stays paused until it rests there.
"""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import time
from typing import Any, Callable

import numpy as np
from bridge_server import serve

SOUND = "/System/Library/Sounds/Glass.aiff"
# Frames to wait for a lost ball to be seen again (1 s).
WAIT_BALL_FRAMES = 60
ZERO = np.zeros(2)


class Bridge:
    """Requests of `CyberRunnerRig` on the rig's env. `hold_level` keeps the board level
    until the ball is taken off it."""

    def __init__(self, env: Any, hold_level: Callable, log_dir: pathlib.Path):
        self.env = env
        self.hold_level = hold_level
        self.log_dir = log_dir
        self.silences = 0
        self.resets = 0

    def measurement(self) -> dict:
        env = self.env
        return {
            "ball": np.asarray(env.current_ball_pos, float).tolist(),
            "joints": np.asarray(env.current_joint_pos, float).tolist(),
            "detected": bool(env.ball_detected),
            "time": time.time(),
        }

    def handle(self, request: dict) -> dict:
        command = request["command"]
        if command == "step":
            return self.step(request["action"])
        if command == "wait_ball":
            return self.wait_ball()
        if command == "reset":
            return self.reset(request["cause"])
        raise ValueError(f"Unknown command {command}")

    def step(self, action: list) -> dict:
        env = self.env
        action = np.asarray(action, np.float32)
        env.last_action = action.copy()
        env._send_action(action)
        env._get_obs()
        # As the runner: the ball may be missed for 1 or 2 frames.
        for _ in range(2):
            if env.ball_detected:
                break
            env._get_obs()
        return self.measurement()

    def wait_ball(self) -> dict:
        env = self.env
        env._send_action(ZERO)
        for _ in range(WAIT_BALL_FRAMES):
            env._get_obs()
            if env.ball_detected:
                break
        return self.measurement()

    def reset(self, cause: str) -> dict:
        env = self.env
        env._send_action(ZERO)
        self.log(cause)
        if cause == "goal":
            subprocess.Popen(["afplay", SOUND])
            print("GOAL: take the ball and put it at the start", flush=True)
            self.hold_level(env)
            # No board reset and no shake: the ball comes back by hand.
            env.future = None
            env.allow_shake = False
        else:
            env.allow_shake = True
            if cause != "start":
                env._reset_board()
        env.reset()
        self.resets += 1
        return self.measurement()

    def log(self, cause: str) -> None:
        """A line per reset, and the camera's last frame at a fall."""
        env = self.env
        record = {"time": time.time(), "cause": cause, **self.measurement()}
        with open(self.log_dir / "resets.jsonl", "a") as out:
            out.write(json.dumps(record) + "\n")
        image = getattr(env, "last_image", None)
        if cause == "fall" and image is not None:
            import cv2

            name = self.log_dir / f"fall_{self.resets:05d}.png"
            cv2.imwrite(str(name), env.br.imgmsg_to_cv2(image))

    def idle(self) -> None:
        self.silences += 1
        self.env._send_action(ZERO)


def lift(env: Any, command: str) -> None:
    """Calls the motor node's `start_reload` or `stop_reload`."""
    import rclpy
    from std_srvs.srv import Trigger

    client = env.node.create_client(Trigger, f"cyberrunner_dynamixel/{command}")
    if not client.wait_for_service(timeout_sec=2.0):
        print(f"WARNING: no {command} service", flush=True)
        return
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(env.node, future, timeout_sec=2.0)
    if future.result() is None:
        print(f"WARNING: {command} did not answer", flush=True)


class Lift:
    """Runs the lift while the ball is not seen on the board, after `frames` frames of
    the change, such that a missed frame does not switch it."""

    def __init__(self, env: Any, frames: int = 30):
        from std_srvs.srv import Trigger

        self.request = Trigger.Request
        self.clients = {
            running: env.node.create_client(Trigger, f"cyberrunner_dynamixel/{command}")
            for running, command in ((True, "start_reload"), (False, "stop_reload"))
        }
        self.env = env
        self.frames = frames
        self.running = False
        self.count = 0

    def update(self) -> None:
        """Called after every frame; the request is not waited for."""
        if self.env.ball_detected != self.running:
            self.count = 0
            return
        self.count += 1
        if self.count >= self.frames:
            self.running = not self.running
            self.count = 0
            self.clients[self.running].call_async(self.request())


def stop(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-wait-steps", type=int, default=60)
    args = parser.parse_args()
    if "WAIT_DEG" not in os.environ:
        raise SystemExit("Set WAIT_DEG: the start reset needs level_eval's tilt")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    import level_eval
    import rig_eval

    runner = rig_eval.eval_onnx.CyberrunnerGymONNX
    msg_to_obs = runner._msg_to_obs

    def keep_image(self, msg):
        self.last_image = msg.subimg
        msg_to_obs(self, msg)

    runner._msg_to_obs = keep_image

    def hold_level(env) -> None:
        tilt = level_eval.Tilt(level_eval.LEVEL_TARGET_DEG, level_eval.LEVEL_COMMAND)
        unseen = 0
        while unseen < 30:  # 0.5 s
            tilt.step(env)
            unseen = unseen + 1 if not env.ball_detected else 0
        env._send_action(ZERO)

    env = runner(num_wait_steps=args.num_wait_steps)
    # rclpy's own handlers shut ROS down at Ctrl-C, after which the zero below cannot be
    # sent and the motors keep their last command.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    bridge = Bridge(env, hold_level, args.log_dir)
    # The motor node starts with the lift running.
    lift(env, "stop_reload")
    lift_control = Lift(env)
    get_obs = env._get_obs

    def get_obs_and_lift():
        observation = get_obs()
        lift_control.update()
        return observation

    env._get_obs = get_obs_and_lift
    try:
        serve(bridge.handle, bridge.idle, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        env._send_action(ZERO)
        lift(env, "stop_reload")
        print(f"Resets {bridge.resets}, silences {bridge.silences}", flush=True)


if __name__ == "__main__":
    main()
