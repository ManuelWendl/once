"""CyberRunner recovery task: bring the ball to rest away from the holes, the safe operable set R.

The predicates below take the ball position and velocity in board coordinates and only use
operators and array methods, so that they can be shared between jax and numpy code.
"""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import dm_control_suite
from mujoco_playground._src import mjx_env

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import cyberrunner
from ss2r.benchmark_suites.mujoco_playground.cyberrunner.sim import (
    constants,
    dynamics,
    maze,
)

# R: the ball centre farther than one ball radius from every hole's edge, and slower than this.
RECOVERY_HOLE_DISTANCE = constants.HOLE_RADIUS + constants.MARBLE_RADIUS
RECOVERY_SPEED = 0.05  # m/s
# Reset: ball speed range (m/s); its direction and the board tilt are uniform.
RESET_SPEED = (0.05, 0.4)
# Candidate ball positions drawn per reset; the first one clear of holes and walls is used.
NUM_CANDIDATES = 64


def hole_distance(ball, holes):
    """Distance from the ball centre to the nearest hole centre."""
    return (((ball[..., None, :] - holes) ** 2).sum(-1) ** 0.5).min(-1)


def in_recovery_set(
    ball,
    velocity,
    holes,
    distance: float = RECOVERY_HOLE_DISTANCE,
    speed: float = RECOVERY_SPEED,
):
    """Whether the ball is in the safe operable region R: away from the holes and slow."""
    slow = (velocity**2).sum(-1) < speed**2
    return (hole_distance(ball, holes) > distance) & slow


def violates(ball, holes):
    """Whether the ball fell into a hole."""
    return hole_distance(ball, holes) < constants.HOLE_RADIUS


def recovery_reward_fn(
    ball,
    velocity,
    holes,
    distance: float = RECOVERY_HOLE_DISTANCE,
    speed: float = RECOVERY_SPEED,
):
    """1 if the ball entered R, 0 otherwise. Not to be mixed with the task reward."""
    reached = in_recovery_set(ball, velocity, holes, distance, speed)
    reached = reached & ~violates(ball, holes)
    return reached.astype("float32")


def default_config() -> config_dict.ConfigDict:
    config = cyberrunner.default_config()
    config.recovery_hole_distance = RECOVERY_HOLE_DISTANCE
    config.recovery_speed = RECOVERY_SPEED
    config.reset_speed = list(RESET_SPEED)
    return config


class CyberRunnerRecovery(cyberrunner.CyberRunner):
    """CyberRunner that terminates when the ball reaches R or falls into a hole.

    `state.reward` remains the maze reward. The recovery signal is written to
    `state.info["recovery_reward"]`.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        defaults = default_config()
        get = lambda name: self._config.get(name, defaults[name])
        self._recovery_hole_distance = get("recovery_hole_distance")
        self._recovery_speed = get("recovery_speed")
        self._reset_speed = tuple(get("reset_speed"))
        # Height of the resting ball's centre above the board's origin.
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, data)
        board_height = data.xpos[self._board_body_id][2]
        self._ball_height = cyberrunner.BALL_START_HEIGHT - float(board_height)

    def _initial_state(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Board tilted at random and still, ball away from holes and walls, sliding."""
        k_ball, k_tilt, k_dir, k_speed = jax.random.split(key, 4)
        tilt_range = jp.array([constants.RANGE_ALPHA[1], constants.RANGE_BETA[1]])
        tilt = jax.random.uniform(k_tilt, (2,), minval=-tilt_range, maxval=tilt_range)
        qpos = jp.asarray(self._mj_model.qpos0).at[:2].set(tilt)
        data = mjx.forward(self.mjx_model, self._make_data().replace(qpos=qpos))
        board_pos = data.xpos[self._board_body_id]
        board_mat = data.xmat[self._board_body_id].reshape(3, 3)
        ball = self._sample_ball(k_ball)
        position = board_pos + board_mat @ jp.append(ball, self._ball_height)
        qpos = qpos.at[2:5].set(position)
        qpos = qpos.at[5:9].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        angle = jax.random.uniform(k_dir, (), maxval=2 * jp.pi)
        low, high = self._reset_speed
        speed = jax.random.uniform(k_speed, (), minval=low, maxval=high)
        velocity = speed * jp.array([jp.cos(angle), jp.sin(angle), 0.0])
        qvel = jp.zeros(self._mj_model.nv).at[2:5].set(board_mat @ velocity)
        return qpos, qvel

    def _sample_ball(self, key: jax.Array) -> jax.Array:
        """A board position away from the holes and clear of the walls."""
        r = constants.MARBLE_RADIUS
        candidates = jax.random.uniform(
            key,
            (NUM_CANDIDATES, 2),
            minval=jp.array([r, r]),
            maxval=jp.array([maze.BOARD_WIDTH - r, maze.BOARD_HEIGHT - r]),
        )
        gaps = jax.vmap(lambda b: dynamics.wall_gaps(b, *self._walls)[2].min())(
            candidates
        )
        far = hole_distance(candidates, self._holes) > self._recovery_hole_distance
        valid = far & (gaps > 0.0)
        return jp.where(valid.any(), candidates[jp.argmax(valid)], self._waypoints[0])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        state.info["recovery_reward"] = jp.zeros(())
        state.metrics["recovery/reached"] = jp.zeros(())
        state.metrics["recovery/violated"] = jp.zeros(())
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        ball_before = self.ball_position(state.data)
        nstate = super().step(state, action)
        ball = self.ball_position(nstate.data)
        velocity = (ball - ball_before) / self.dt
        violated = violates(ball, self._holes)
        recovery_reward = recovery_reward_fn(
            ball,
            velocity,
            self._holes,
            self._recovery_hole_distance,
            self._recovery_speed,
        )
        nstate.info["recovery_reward"] = recovery_reward
        nstate.metrics["recovery/reached"] = recovery_reward
        nstate.metrics["recovery/violated"] = violated.astype(jp.float32)
        # The goal does not end a recovery episode, only R and the holes do.
        done = ((recovery_reward > 0.0) | violated).astype(nstate.done.dtype)
        return nstate.replace(done=done)


dm_control_suite.register_environment(
    "CyberRunnerRecovery", CyberRunnerRecovery, default_config
)
