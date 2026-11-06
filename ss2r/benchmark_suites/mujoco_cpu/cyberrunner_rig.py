"""The real CyberRunner as ONCE's real system, through a bridge to the rig's ROS stack."""

import json
import socket
import time
from typing import Any, Callable

import numpy as np

from ss2r.benchmark_suites.mujoco_cpu.cyberrunner import CyberRunner
from ss2r.benchmark_suites.mujoco_playground.cyberrunner import cyberrunner, recovery

_STEP_TIMEOUT_S = 2.0


class CyberRunnerRig(CyberRunner):
    """The task on the rig, with the measured ball.

    It connects to the bridge at the first `reset`, so that the learner can build it to
    query the sizes. A ball no longer seen within `fall_distance` of a hole has fallen,
    and the last frame has it in that hole, which is what `violates` reads. A ball lost
    elsewhere is the camera's fault: the episode restarts, without a violation. At the
    goal the system is paused until the ball is back at the start, in the same step:
    for the learner the ball is put back at once, as in the simulator.
    """

    def __init__(
        self,
        bridge_port: int = 5600,
        fall_distance: float = 0.02,
        real_params: None = None,
        seed: int = 0,
        **kwargs: Any,
    ):
        del real_params, seed  # The physics are real, and nothing here is random.
        super().__init__(**kwargs)
        self._bridge_port = bridge_port
        self._fall_distance = fall_distance
        self._connection: socket.socket | None = None
        self._reader: Any = None
        self.on_pause: Callable[[bool], None] = lambda paused: None
        self._ball: np.ndarray | None = None
        self._joints = np.zeros(2)
        self._time = 0.0
        self._detected = False
        self._lost = False
        self._reset_s = 0.0

    def _call(self, command: str, timeout: float | None = _STEP_TIMEOUT_S, **kwargs):
        """A request to the bridge, and its reply."""
        if self._connection is None:
            self._connection = socket.create_connection(
                ("127.0.0.1", self._bridge_port)
            )
            self._connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._reader = self._connection.makefile("r")
        self._connection.settimeout(timeout)
        message = json.dumps({"command": command, **kwargs}) + "\n"
        self._connection.sendall(message.encode())
        line = self._reader.readline()
        if not line:
            raise ConnectionError("The bridge closed the connection")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"Bridge: {reply['error']}")
        return reply

    def close(self) -> None:
        """Disconnects, which stops the motors."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _apply(self, measurement: dict) -> None:
        self._detected = measurement["detected"]
        self._joints = np.asarray(measurement["joints"], np.float64)
        self._time = measurement["time"]
        if self._detected:
            self._ball = np.asarray(measurement["ball"], np.float64)

    def reset(self) -> np.ndarray:
        cause = self._reset_cause()
        if cause == "goal":
            self.on_pause(True)
        start = time.perf_counter()
        # Blocks until the ball is at the start: no timeout.
        self._apply(self._call("reset", timeout=None, cause=cause))
        self._reset_s += time.perf_counter() - start
        if cause == "goal":
            self.on_pause(False)
        return self._start_episode()

    def _reset_cause(self) -> str:
        ball = self._ball
        if ball is None:
            return "start"
        if recovery.violates(ball, self._layout.holes):
            return "fall"
        goal = np.linalg.norm(ball - self._layout.waypoints[-1])
        return "goal" if goal < cyberrunner.GOAL_THRESHOLD else "lost"

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self._lost = False
        self._reset_s = 0.0
        obs, reward, terminated, info = super().step(action)
        lost = self._lost
        if lost:
            obs = self.reset()
        # The time of a reset that happened within the step, for the actor's metrics.
        info["paused_s"] = self._reset_s
        info["counts"]["paused_s"] = self._reset_s
        info["counts"]["tracking_losses"] = float(lost)
        return obs, reward, terminated, info

    def get_state(self) -> dict[str, np.ndarray]:
        """What was measured: the rig has no simulation state."""
        return {
            "ball": np.asarray(self._ball, np.float64),
            "joints": self._joints.copy(),
            "time": np.asarray(self._time, np.float64),
        }

    def _physics(self, action: np.ndarray) -> None:
        self._apply(self._call("step", action=action.tolist()))
        if self._detected:
            return
        holes = np.asarray(self._layout.holes)
        distances = np.linalg.norm(holes - self._ball, axis=-1)
        if distances.min() < self._fall_distance:
            self._ball = holes[np.argmin(distances)].astype(np.float64)
            return
        self._apply(self._call("wait_ball"))
        self._lost = not self._detected

    def _ball_position(self) -> np.ndarray:
        return self._ball

    def _measure(self, ball: np.ndarray, action: np.ndarray):
        frame = np.concatenate([self._joints, ball, action]).astype(np.float32)
        return frame, ball.astype(np.float32)
