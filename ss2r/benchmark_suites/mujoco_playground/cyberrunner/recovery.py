"""CyberRunner recovery task: bring the ball to rest away from the holes, the safe operable set R.

The predicates below take the ball position and velocity in board coordinates and only use
operators and array methods, so that they can be shared between jax and numpy code.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
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
# R counts once the ball has stayed in it this many steps in a row (1: entering it is enough).
RECOVERY_HOLD_STEPS = 1
# Reset: ball speed range (m/s); its direction and the board tilt are uniform.
RESET_SPEED = (0.05, 0.4)
# Reset: the ball centre at least this far from every hole centre (m). Below R's distance,
# down to just outside a hole, the backup also starts where it can barely recover.
RESET_HOLE_DISTANCE = RECOVERY_HOLE_DISTANCE
# Reset near a hole: this share of the resets puts the ball at most RESET_NEAR_HOLE_MAX (m)
# from a random hole's centre, rolling towards the nearest hole within RESET_TOWARD_HOLE_DEG,
# where recovering is unlikely. 0 keeps all starts uniform over the board.
RESET_NEAR_HOLE_FRACTION = 0.0
RESET_NEAR_HOLE_MAX = 0.025
RESET_TOWARD_HOLE_DEG = 60.0
# Candidate ball positions drawn per reset; the first one clear of holes and walls is used.
NUM_CANDIDATES = 64
# Handover starts: this share of the resets starts where another controller hands over,
# after RESET_WARMUP_STEPS steps of a generic driver from the sampled start, the board first
# set turning at up to RESET_WARMUP_BOARD_SPEED (rad/s) per axis. The driver is an
# Ornstein-Uhlenbeck process per axis, its std and time constant drawn per episode.
RESET_WARMUP_FRACTION = 0.0
RESET_WARMUP_STEPS = 5
RESET_WARMUP_BOARD_SPEED = 1.0
WARMUP_DRIVER_STD = (0.1, 1.0)
WARMUP_DRIVER_TIME_CONSTANT_S = (0.1, 2.0)
# Starts whose ball falls during the warm-up are discarded: 13% fall (25% of those near a
# hole), so all of 4 candidates fall in about 0.03% of the resets.
WARMUP_CANDIDATES = 4
# Adversary: this share of the episodes starts with a phase of 1 to ADVERSARY_MAX_STEPS steps
# in which an adversary trained alongside the backup sends the commands, then the backup
# takes over. info["adversary_left"] counts the phase's steps left, info["adversary_obs"] is
# the adversary's observation: the true state, which only the simulation has.
ADVERSARY_FRACTION = 0.0
ADVERSARY_MAX_STEPS = 30


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
    config.recovery_hold_steps = RECOVERY_HOLD_STEPS
    config.reset_speed = list(RESET_SPEED)
    config.reset_hole_distance = RESET_HOLE_DISTANCE
    config.reset_near_hole_fraction = RESET_NEAR_HOLE_FRACTION
    config.reset_near_hole_max = RESET_NEAR_HOLE_MAX
    config.reset_toward_hole_deg = RESET_TOWARD_HOLE_DEG
    config.reset_warmup_fraction = RESET_WARMUP_FRACTION
    config.reset_warmup_steps = RESET_WARMUP_STEPS
    config.reset_warmup_board_speed = RESET_WARMUP_BOARD_SPEED
    config.adversary_fraction = ADVERSARY_FRACTION
    config.adversary_max_steps = ADVERSARY_MAX_STEPS
    return config


class CyberRunnerRecovery(cyberrunner.CyberRunner):
    """CyberRunner that terminates when the ball reaches R or falls into a hole.

    R is reached once the ball has stayed in it `recovery_hold_steps` steps in a row. The
    reward is the recovery reward, also in `state.info["recovery_reward"]` for the
    pessimistic critic, and only the recovery metrics are logged, not the maze's.
    Episodes start from the sampler of `_initial_state`, a share of them moved on by a
    generic driver (`reset_warmup_fraction`, see `_handover`).
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
        self._recovery_hold = int(get("recovery_hold_steps"))
        if self._recovery_hold < 1:
            raise ValueError(f"recovery_hold_steps {self._recovery_hold} is below 1")
        self._reset_speed = tuple(get("reset_speed"))
        self._reset_hole_distance = get("reset_hole_distance")
        if self._reset_hole_distance <= constants.HOLE_RADIUS:
            raise ValueError(
                f"reset_hole_distance {self._reset_hole_distance} m starts the ball in a "
                f"hole (radius {constants.HOLE_RADIUS} m)"
            )
        self._near_hole_fraction = get("reset_near_hole_fraction")
        self._near_hole_max = get("reset_near_hole_max")
        self._toward_hole = math.radians(get("reset_toward_hole_deg"))
        if not 0.0 <= self._near_hole_fraction <= 1.0:
            raise ValueError(
                f"reset_near_hole_fraction {self._near_hole_fraction} is not in [0, 1]"
            )
        if self._near_hole_max <= self._reset_hole_distance:
            raise ValueError(
                f"reset_near_hole_max {self._near_hole_max} m is not above "
                f"reset_hole_distance {self._reset_hole_distance} m"
            )
        self._warmup_fraction = get("reset_warmup_fraction")
        self._warmup_steps = int(get("reset_warmup_steps"))
        self._warmup_board_speed = get("reset_warmup_board_speed")
        if not 0.0 <= self._warmup_fraction <= 1.0:
            raise ValueError(
                f"reset_warmup_fraction {self._warmup_fraction} is not in [0, 1]"
            )
        if self._warmup_fraction > 0.0 and self._warmup_steps < 1:
            raise ValueError(f"reset_warmup_steps {self._warmup_steps} is below 1")
        if self._warmup_fraction > 0.0:
            # A new episode takes these from its handover start too.
            self.episode_info_keys = (
                *self.episode_info_keys,
                "start_commands",
                "start_delay",
                "start_stuck",
            )
        self._adversary_fraction = get("adversary_fraction")
        self._adversary_max_steps = int(get("adversary_max_steps"))
        if not 0.0 <= self._adversary_fraction <= 1.0:
            raise ValueError(
                f"adversary_fraction {self._adversary_fraction} is not in [0, 1]"
            )
        if self._adversary_fraction > 0.0 and self._adversary_max_steps < 1:
            raise ValueError(
                f"adversary_max_steps {self._adversary_max_steps} is below 1"
            )
        if self.has_adversary:
            self.episode_info_keys = (
                *self.episode_info_keys,
                "adversary_left",
                "adversary_obs",
            )
        # Height of the resting ball's centre above the board's origin.
        data = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, data)
        board_height = data.xpos[self._board_body_id][2]
        self._ball_height = cyberrunner.BALL_START_HEIGHT - float(board_height)

    def _initial_state(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Board tilted at random and still, ball away from holes and walls, sliding; a
        share of the balls starts near a hole, rolling towards it."""
        k_ball, k_tilt, k_dir, k_speed = jax.random.split(key, 4)
        # Own keys, so that the uniform starts stay as they were.
        k_near, k_place, k_cone = jax.random.split(jax.random.fold_in(key, 1), 3)
        tilt_range = jp.array([constants.RANGE_ALPHA[1], constants.RANGE_BETA[1]])
        tilt = jax.random.uniform(k_tilt, (2,), minval=-tilt_range, maxval=tilt_range)
        qpos = jp.asarray(self._mj_model.qpos0).at[:2].set(tilt)
        data = mjx.forward(self.mjx_model, self._make_data().replace(qpos=qpos))
        board_pos = data.xpos[self._board_body_id]
        board_mat = data.xmat[self._board_body_id].reshape(3, 3)
        ball = self._sample_ball(k_ball)
        near = jax.random.uniform(k_near) < self._near_hole_fraction
        ball = jp.where(near, self._sample_near_hole(k_place, ball), ball)
        position = board_pos + board_mat @ jp.append(ball, self._ball_height)
        qpos = qpos.at[2:5].set(position)
        qpos = qpos.at[5:9].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        angle = jax.random.uniform(k_dir, (), maxval=2 * jp.pi)
        nearest = self._holes[jp.argmin(jp.linalg.norm(self._holes - ball, axis=-1))]
        to_hole = jp.arctan2(nearest[1] - ball[1], nearest[0] - ball[0])
        spread = jax.random.uniform(k_cone, (), minval=-1.0, maxval=1.0)
        angle = jp.where(near, to_hole + self._toward_hole * spread, angle)
        low, high = self._reset_speed
        speed = jax.random.uniform(k_speed, (), minval=low, maxval=high)
        velocity = speed * jp.array([jp.cos(angle), jp.sin(angle), 0.0])
        qvel = jp.zeros(self._mj_model.nv).at[2:5].set(board_mat @ velocity)
        return qpos, qvel

    def _sample_ball(self, key: jax.Array) -> jax.Array:
        """A board position reset_hole_distance from the holes and clear of the walls."""
        r = constants.MARBLE_RADIUS
        candidates = jax.random.uniform(
            key,
            (NUM_CANDIDATES, 2),
            minval=jp.array([r, r]),
            maxval=jp.array([maze.BOARD_WIDTH - r, maze.BOARD_HEIGHT - r]),
        )
        return self._first_clear(candidates, self._waypoints[0])

    def _sample_near_hole(self, key: jax.Array, fallback: jax.Array) -> jax.Array:
        """A position reset_hole_distance to reset_near_hole_max from a random hole's
        centre and clear of the walls, else `fallback`."""
        k_hole, k_radius, k_angle = jax.random.split(key, 3)
        holes = self._holes[
            jax.random.randint(k_hole, (NUM_CANDIDATES,), 0, self._holes.shape[0])
        ]
        radius = jax.random.uniform(
            k_radius,
            (NUM_CANDIDATES, 1),
            minval=self._reset_hole_distance,
            maxval=self._near_hole_max,
        )
        angle = jax.random.uniform(k_angle, (NUM_CANDIDATES,), maxval=2 * jp.pi)
        offset = jp.stack([jp.cos(angle), jp.sin(angle)], axis=-1)
        return self._first_clear(holes + radius * offset, fallback)

    def _first_clear(self, candidates: jax.Array, fallback: jax.Array) -> jax.Array:
        """The first candidate on the board, clear of the walls and reset_hole_distance
        from every hole, else `fallback`."""
        r = constants.MARBLE_RADIUS
        board = jp.array([maze.BOARD_WIDTH - r, maze.BOARD_HEIGHT - r])
        inside = jp.all((candidates >= r) & (candidates <= board), axis=-1)
        gaps = jax.vmap(lambda b: dynamics.wall_gaps(b, *self._walls)[2].min())(
            candidates
        )
        far = hole_distance(candidates, self._holes) > self._reset_hole_distance
        valid = inside & far & (gaps > 0.0)
        return jp.where(valid.any(), candidates[jp.argmax(valid)], fallback)

    def _sampled_start(self, rng: jax.Array) -> mjx_env.State:
        """A start of the sampler, no commands pending."""
        state = super().reset(rng)
        info = state.info
        return state.replace(
            info={
                "start_commands": jp.zeros_like(info["commands"]),
                "start_delay": info["delay"],
                "start_stuck": info["stuck"],
                **info,
            }
        )

    def _handover(self, rng: jax.Array) -> mjx_env.State:
        """A share of the starts moved on by a generic driver, so that the episode starts
        as after a handover: the board turning, the driver's commands pending and in the
        history. These steps are not training transitions. Starts whose ball falls
        meanwhile are discarded: of WARMUP_CANDIDATES sampled starts, the first that does
        not fall is used (the sampled start only if all fall)."""
        keys = jax.random.split(rng, WARMUP_CANDIDATES)
        plain = jax.vmap(self._sampled_start)(keys)
        warm, fell = jax.vmap(self._warm_up)(plain, keys)
        kept = ~fell.any(-1)
        use = (
            jax.random.uniform(jax.random.fold_in(rng, 8)) < self._warmup_fraction
        ) & kept.any()
        index = jp.where(use, jp.argmax(kept), 0)
        candidates = jax.tree.map(lambda w, p: jp.where(use, w, p), warm, plain)
        return jax.tree.map(lambda x: x[index], candidates)

    def _warm_up(
        self, state: mjx_env.State, key: jax.Array
    ) -> Tuple[mjx_env.State, jax.Array]:
        """The start after the driver's steps, and whether the ball fell at each."""
        k_speed, k_std, k_tau, k_first, k_drive = jax.random.split(
            jax.random.fold_in(key, 7), 5
        )
        speed = self._warmup_board_speed
        qvel = state.data.qvel.at[:2].set(
            jax.random.uniform(k_speed, (2,), minval=-speed, maxval=speed)
        )
        low, high = WARMUP_DRIVER_STD
        std = jax.random.uniform(k_std, (2,), minval=low, maxval=high)
        low, high = np.log(WARMUP_DRIVER_TIME_CONSTANT_S)
        tau = jp.exp(jax.random.uniform(k_tau, (2,), minval=low, maxval=high))
        decay = jp.exp(-self.dt / tau)

        def drive(carry, key):
            state, command = carry
            noise = jax.random.normal(key, (2,))
            command = decay * command + std * jp.sqrt(1.0 - decay**2) * noise
            # The maze's step: the recovery one would end the episode on R.
            state = cyberrunner.CyberRunner.step(self, state, command)
            return (state, command), violates(
                self.ball_position(state.data), self._holes
            )

        first = std * jax.random.normal(k_first, (2,))
        (warm, _), fell = jax.lax.scan(
            drive,
            (state.replace(data=state.data.replace(qvel=qvel)), first),
            jax.random.split(k_drive, self._warmup_steps),
        )
        info = warm.info
        warm = warm.replace(
            info={
                **info,
                "start_history": info["history"],
                "start_commands": info["commands"],
                "start_delay": info["delay"],
                "start_stuck": info["stuck"],
            },
            # The maze's reward and metrics of the last step are not the episode's.
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
        )
        return warm, fell

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self._warmup_fraction > 0.0:
            state = self._handover(rng)
        else:
            state = super().reset(rng)
        state.info["recovery_reward"] = jp.zeros(())
        state.info["recovery_hold"] = jp.zeros((), jp.int32)
        state.info["spidr_state"] = self.spidr_state(state.data)
        metrics = {"recovery/reached": jp.zeros(()), "recovery/violated": jp.zeros(())}
        if self.has_adversary:
            k_use, k_steps = jax.random.split(jax.random.fold_in(rng, 9))
            steps = jax.random.randint(k_steps, (), 1, self._adversary_max_steps + 1)
            use = jax.random.uniform(k_use) < self._adversary_fraction
            left = jp.where(use, steps, 0)
            state.info["adversary_left"] = left
            state.info["adversary_obs"] = self.adversary_observation(
                state.data, state.info, left
            )
            metrics["adversary/acting"] = jp.zeros(())
            metrics["adversary/violated"] = jp.zeros(())
        return state.replace(metrics=metrics)

    @property
    def has_adversary(self) -> bool:
        return self._adversary_fraction > 0.0

    @property
    def adversary_observation_size(self) -> int:
        return 11 + 2 * self._num_commands

    def adversary_observation(
        self, data: mjx.Data, info: Dict[str, Any], left: jax.Array
    ) -> jax.Array:
        """The true state: tilts and their velocities, ball position and velocity on the
        board, pending commands, delay, whether the ball is stuck, and the share of the
        adversary's phase left."""
        board_mat = data.xmat[self._board_body_id].reshape(3, 3)
        ball_velocity = (board_mat.T @ data.qvel[2:5])[:2]
        low, high = self._delay_substeps
        scalars = jp.stack(
            [
                (info["delay"] - low) / max(high - low, 1),
                info["stuck"].astype(jp.float32),
                left / self._adversary_max_steps,
            ]
        )
        return jp.concatenate(
            [
                data.qpos[:2],
                data.qvel[:2],
                self.ball_position(data),
                ball_velocity,
                info["commands"].reshape(-1),
                scalars,
            ]
        ).astype(jp.float32)

    def spidr_state(self, data: mjx.Data) -> jax.Array:
        """What SPiDR's copies disagree on with disagreement_on "state": the tilts, the
        ball position on the board, and how far the tilts' velocities move them in one
        step (the copies start from one state, so the ball's moves show in its position)."""
        return jp.concatenate(
            [data.qpos[:2], self.ball_position(data), self.dt * data.qvel[:2]]
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # The episode wrapper zeroes `steps` after an auto-reset.
        new_episode = state.info.get("steps", jp.ones(())) == 0
        ball_before = self.ball_position(state.data)
        nstate = super().step(state, action)
        ball = self.ball_position(nstate.data)
        velocity = (ball - ball_before) / self.dt
        violated = violates(ball, self._holes)
        in_r = recovery_reward_fn(
            ball,
            velocity,
            self._holes,
            self._recovery_hole_distance,
            self._recovery_speed,
        )
        # Steps in R in a row, this one included.
        hold = jp.where(new_episode, 0, state.info["recovery_hold"]) + 1
        hold = jp.where(in_r > 0.0, hold, 0)
        recovery_reward = (hold >= self._recovery_hold).astype(jp.float32)
        nstate.info["recovery_hold"] = hold
        nstate.info["recovery_reward"] = recovery_reward
        if "spidr_state" in state.info:  # states built elsewhere (calibration) lack it
            nstate.info["spidr_state"] = self.spidr_state(nstate.data)
        # The maze's metrics go; those of wrappers (e.g. SPiDR's disagreement) stay.
        metrics = {k: v for k, v in nstate.metrics.items() if k in state.metrics}
        metrics["recovery/reached"] = recovery_reward
        metrics["recovery/violated"] = violated.astype(jp.float32)
        if self.has_adversary:
            # This step's commands came from the adversary while its phase lasted.
            acting = state.info["adversary_left"] > 0
            left = jp.maximum(state.info["adversary_left"] - 1, 0)
            nstate.info["adversary_left"] = left
            nstate.info["adversary_obs"] = self.adversary_observation(
                nstate.data, nstate.info, left
            )
            metrics["adversary/acting"] = acting.astype(jp.float32)
            metrics["adversary/violated"] = (violated & acting).astype(jp.float32)
        # The goal does not end a recovery episode, only R and the holes do.
        done = ((recovery_reward > 0.0) | violated).astype(nstate.done.dtype)
        return nstate.replace(reward=recovery_reward, done=done, metrics=metrics)


dm_control_suite.register_environment(
    "CyberRunnerRecovery", CyberRunnerRecovery, default_config
)
