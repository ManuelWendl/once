"""Go to goal recovery task: slow down and get clear of all obstacles.

The predicates below operate on the go to goal observation
`[obstacle_lidar (16), goal_lidar (16), ..., accelerometer (2), velocimeter (2),
gyro (1), magnetometer (2)]`. They index the lidar from the front and the
sensors from the back, such that they hold with and without
`GoToGoalObservationWrapper`, and only use operators and array methods, so that
they can be shared between jax (simulation, learned model) and numpy code.
"""

from typing import Sequence

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env

from ss2r.benchmark_suites.mujoco_playground import mjx_data
from ss2r.benchmark_suites.safety_gym import go_to_goal, lidar

_OBSTACLE_LIDAR = slice(0, lidar.NUM_LIDAR_BINS)
_VELOCIMETER = slice(-5, -3)

# Distances are measured between the centers of the robot and of an obstacle.
# Hazards have a radius of 0.2 and have a cost when the center of the robot is
# inside. The robot touches a vase from a distance of 0.3. At its top speed the
# robot moves 0.12 per control step, which the hard constraint leaves room for.
HARD_DISTANCE = 0.35
# Safe operable region R: slow, with a buffer to the hard constraint. From
# 0.5 m/s the robot brakes to a stop within 0.03 (and coasts for 0.25), such
# that it can always stop inside of the buffer.
RECOVERY_DISTANCE = 0.5
RECOVERY_SPEED = 0.5
# Leaving the arena: the arena has no walls, and beyond the range of the lidar
# the robot sees neither the obstacles nor the goal. The observation has no
# absolute position, so leaving is detected as being farther than this from
# every obstacle; off (None) by default.
MAX_DISTANCE = None
# R requires an obstacle within this distance: a buffer to leaving the arena,
# as RECOVERY_DISTANCE is to the hard constraint. Off (None) by default.
RECOVERY_MAX_DISTANCE = None
# The nominal robot reaches 1.5 m/s and 3 rad/s.
RESET_SPEED = 1.5
RESET_ANGULAR_VELOCITY = 3.0


def obstacle_distance(obs):
    """Distance to the closest obstacle, saturating at the range of the lidar.

    The strongest lidar reading is exactly the one of the closest obstacle.
    """
    closest = obs[..., _OBSTACLE_LIDAR].max(axis=-1)
    return lidar.LIDAR_MAX_DIST * (1.0 - closest)


def speed(obs):
    """Planar speed of the robot."""
    return (obs[..., _VELOCIMETER] ** 2).sum(axis=-1) ** 0.5


def out_of_arena(obs, max_distance: float | None = MAX_DISTANCE):
    """Whether the robot left the arena: no obstacle within `max_distance`."""
    if max_distance is None:
        # Never: the distance saturates at the range of the lidar.
        return obstacle_distance(obs) > lidar.LIDAR_MAX_DIST
    return obstacle_distance(obs) >= max_distance


def violates(
    obs,
    hard_distance: float = HARD_DISTANCE,
    max_distance: float | None = MAX_DISTANCE,
):
    """Whether the hard constraint on the distance to the obstacles is violated:
    too close to an obstacle or, with `max_distance`, out of the arena."""
    return (obstacle_distance(obs) < hard_distance) | out_of_arena(obs, max_distance)


def in_recovery_set(
    obs,
    recovery_distance: float = RECOVERY_DISTANCE,
    recovery_speed: float = RECOVERY_SPEED,
    max_distance: float | None = MAX_DISTANCE,
    recovery_max_distance: float | None = RECOVERY_MAX_DISTANCE,
):
    """Whether the robot is in the safe operable region R."""
    inside = (
        (obstacle_distance(obs) >= recovery_distance)
        & (speed(obs) <= recovery_speed)
        & ~out_of_arena(obs, max_distance)
    )
    if recovery_max_distance is not None:
        inside = inside & (obstacle_distance(obs) <= recovery_max_distance)
    return inside


def recovery_reward_fn(
    next_obs,
    recovery_distance: float = RECOVERY_DISTANCE,
    recovery_speed: float = RECOVERY_SPEED,
    hard_distance: float = HARD_DISTANCE,
    max_distance: float | None = MAX_DISTANCE,
    recovery_max_distance: float | None = RECOVERY_MAX_DISTANCE,
):
    """1 if the robot entered R, 0 otherwise. Not to be mixed with the task reward."""
    reached = in_recovery_set(
        next_obs, recovery_distance, recovery_speed, max_distance, recovery_max_distance
    )
    reached = reached & ~violates(next_obs, hard_distance, max_distance)
    return reached.astype("float32")


class GoToGoalRecovery(go_to_goal.GoToGoal):
    """Go to goal that terminates when reaching R or violating the hard constraint.

    An episode starts with the robot in motion, anywhere in the constraint set.
    To recover, the robot has to slow down and to get away from the obstacles.

    `state.reward` remains the go to goal task reward. The recovery signal is
    written to `state.info["recovery_reward"]`.
    """

    def __init__(
        self,
        *,
        recovery_distance: float = RECOVERY_DISTANCE,
        recovery_speed: float = RECOVERY_SPEED,
        hard_distance: float = HARD_DISTANCE,
        max_distance: float | None = MAX_DISTANCE,
        recovery_max_distance: float | None = RECOVERY_MAX_DISTANCE,
        reset_speed: float = RESET_SPEED,
        reset_angular_velocity: float = RESET_ANGULAR_VELOCITY,
        far_start_prob: float = 0.0,
        near_start_prob: float = 0.0,
        near_start_width: float = 0.3,
        stop_start_prob: float = 0.0,
        slow_start_prob: float = 0.0,
        start_bands: Sequence[Sequence[float]] = ((0.36, 0.8, 1.0),),
        align_velocity: bool = False,
        heading_spread: float = 0.35,
        slow_angular_velocity: float = 1.0,
        recovery_dwell: int = 1,
        action_repeat: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._recovery_distance = recovery_distance
        self._recovery_speed = recovery_speed
        self._hard_distance = hard_distance
        self._max_distance = max_distance
        self._recovery_max_distance = recovery_max_distance
        self._reset_speed = reset_speed
        self._reset_angular_velocity = reset_angular_velocity
        self._far_start_prob = far_start_prob
        self._near_start_prob = near_start_prob
        self._near_start_width = near_start_width
        # Starts at rest and slow, at a distance to the closest obstacle from
        # one of start_bands: (lower, upper, weight). With align_velocity, the
        # other starts move along the heading (forward or backward, within
        # heading_spread) as the robot does, instead of in any direction.
        self._stop_start_prob = stop_start_prob
        self._slow_start_prob = slow_start_prob
        self._start_bands = tuple(tuple(float(x) for x in band) for band in start_bands)
        self._align_velocity = align_velocity
        self._heading_spread = heading_spread
        self._slow_angular_velocity = slow_angular_velocity
        # A recovery needs the robot in R at the end of this many control steps
        # in a row. One control step is action_repeat steps of this environment,
        # as the episode wrapper repeats the action.
        self._recovery_dwell = recovery_dwell
        self._action_repeat = action_repeat
        # Start as close to the obstacles as the hard constraint allows.
        obstacle_keepouts = [
            self.spec[name].keepout
            for name in ("hazards", "vases")
            if self.spec[name].num_objects > 0
        ]
        if obstacle_keepouts:
            keepout = max(hard_distance - min(obstacle_keepouts), 0.0)
            self.spec["robot"] = go_to_goal.ObjectSpec(keepout, 1)
        self._robot_qvel_ids = mjx_env.get_qvel_ids(self._mj_model, ["x", "y", "z"])

    def reset(self, rng: jax.Array) -> go_to_goal.State:
        """Samples the layout, and the velocity of the robot in any direction."""
        rng, rng_speed, rng_direction, rng_angular = jax.random.split(rng, 4)
        state = super().reset(rng)
        speed_ = jax.random.uniform(rng_speed, (), maxval=self._reset_speed)
        direction = jax.random.uniform(rng_direction, (), minval=-jp.pi, maxval=jp.pi)
        if self._align_velocity:
            heading = state.data.qpos[jp.asarray(self._robot_qpos_ids)[2]]
            direction = self._along_heading(jax.random.fold_in(rng, 3), heading)
        angular_velocity = jax.random.uniform(
            rng_angular,
            (),
            minval=-self._reset_angular_velocity,
            maxval=self._reset_angular_velocity,
        )
        qvel = state.data.qvel.at[self._robot_qvel_ids].set(
            jp.hstack(
                [
                    speed_ * jp.cos(direction),
                    speed_ * jp.sin(direction),
                    angular_velocity,
                ]
            )
        )
        far_start = (
            self._far_start_prob > 0.0
            and self._recovery_max_distance is not None
            and self._max_distance is not None
        )
        near_start = self._near_start_prob > 0.0
        stop_start = self._stop_start_prob > 0.0 or self._slow_start_prob > 0.0
        qpos = state.data.qpos
        far = jp.zeros((), bool)
        near = jp.zeros((), bool)
        if far_start:
            qpos, far = self._far_start(jax.random.fold_in(rng, 1), state.data)
        if near_start:
            qpos, qvel, near = self._near_start(
                jax.random.fold_in(rng, 2), state.data, qpos, qvel, far
            )
        if stop_start:
            qpos, qvel = self._stop_start(
                jax.random.fold_in(rng, 4), state.data, qpos, qvel, far | near
            )
        data = mjx_data.init(
            self.mjx_model,
            qpos=qpos,
            qvel=qvel,
            mocap_pos=state.data.mocap_pos,
            mocap_quat=state.data.mocap_quat,
        )
        if far_start or near_start or stop_start:
            state.info["last_goal_dist"] = jp.linalg.norm(self._robot_to_goal(data)[:2])
        obs = self.get_obs(data)
        state.info["recovery_reward"] = jp.zeros(())
        # Control steps in a row that ended in R, and the step within the
        # current control step.
        state.info["recovery_dwell"] = jp.zeros((), jp.int32)
        state.info["substep"] = jp.zeros((), jp.int32)
        state.metrics["recovery/reached"] = jp.zeros(())
        state.metrics["recovery/violated"] = jp.zeros(())
        return state.replace(data=data, obs=obs)  # type: ignore

    def _sample_position(
        self, rng: jax.Array, data, lower: float, upper: float
    ) -> tuple[jax.Array, jax.Array]:
        """A position from `lower` to `upper` from the closest obstacle, and
        whether one was found among the candidates."""
        obstacles = [
            data.qpos[jp.asarray(ids[:2])][None] for ids in self._vases_qpos_ids
        ]
        if self._hazard_mocap_id:
            obstacles.append(data.mocap_pos[jp.asarray(self._hazard_mocap_id), :2])
        obstacles = jp.concatenate(obstacles)
        bound = max(go_to_goal._EXTENTS) + upper
        candidates = jax.random.uniform(rng, (512, 2), minval=-bound, maxval=bound)
        distance = jp.linalg.norm(candidates[:, None] - obstacles[None], axis=-1)
        distance = distance.min(-1)
        valid = (distance >= lower) & (distance < upper)
        return candidates[jp.argmax(valid)], valid.any()

    def _far_start(self, rng: jax.Array, data) -> tuple[jax.Array, jax.Array]:
        """With probability far_start_prob, moves the robot outside of R, far from
        the obstacles: between recovery_max_distance and max_distance from the
        closest one. The layout places the robot close to the obstacles, such that
        the prior would otherwise never learn to return from there."""
        assert self._recovery_max_distance is not None
        assert self._max_distance is not None
        rng_use, rng_xy = jax.random.split(rng)
        xy, found = self._sample_position(
            rng_xy, data, self._recovery_max_distance, self._max_distance
        )
        use = (jax.random.uniform(rng_use) < self._far_start_prob) & found
        xy_ids = jp.asarray(self._robot_qpos_ids)[:2]
        xy = jp.where(use, xy, data.qpos[xy_ids])
        return data.qpos.at[xy_ids].set(xy), use

    def _near_start(
        self, rng: jax.Array, data, qpos: jax.Array, qvel: jax.Array, far: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """With probability near_start_prob, of the episodes that do not start far,
        starts the robot inside of R close to the obstacles: from recovery_distance
        to near_start_width beyond it, slow and not rotating. The prior parks the
        robot there after recovering, and its Q_R hardly sees these states
        otherwise: episodes end on entering R. Its Q_R was below 0.9 for every
        action at 11% of the parking spots 0.5-0.6 from an obstacle."""
        rng_use, rng_xy, rng_speed, rng_direction = jax.random.split(rng, 4)
        lower = self._recovery_distance
        xy, found = self._sample_position(
            rng_xy, data, lower, lower + self._near_start_width
        )
        use = (jax.random.uniform(rng_use) < self._near_start_prob) & found & ~far
        speed_ = jax.random.uniform(rng_speed, (), maxval=self._recovery_speed)
        direction = jax.random.uniform(rng_direction, (), minval=-jp.pi, maxval=jp.pi)
        velocity = jp.hstack(
            [speed_ * jp.cos(direction), speed_ * jp.sin(direction), 0.0]
        )
        xy_ids = jp.asarray(self._robot_qpos_ids)[:2]
        qpos = qpos.at[xy_ids].set(jp.where(use, xy, qpos[xy_ids]))
        qvel_ids = jp.asarray(self._robot_qvel_ids)
        qvel = qvel.at[qvel_ids].set(jp.where(use, velocity, qvel[qvel_ids]))
        return qpos, qvel, use

    def _along_heading(self, rng: jax.Array, heading: jax.Array) -> jax.Array:
        """A direction of motion along the heading, forward or backward, within
        heading_spread: the robot can only thrust along its heading."""
        rng_back, rng_spread = jax.random.split(rng)
        backward = jax.random.uniform(rng_back) < 0.5
        spread = jax.random.uniform(
            rng_spread, (), minval=-self._heading_spread, maxval=self._heading_spread
        )
        return heading + jp.where(backward, jp.pi, 0.0) + spread

    def _stop_start(
        self, rng: jax.Array, data, qpos: jax.Array, qvel: jax.Array, taken: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """With probability stop_start_prob, of the episodes that do not start far
        or near, starts the robot at rest, with slow_start_prob slow (up to
        recovery_speed along its heading, turning up to slow_angular_velocity).
        The distance to the closest obstacle is drawn from one of start_bands.

        A deadlock of the filter is the robot stopped where the prior never
        learned to act, so every place the robot can stop has to be a start."""
        keys = jax.random.split(rng, 6)
        u = jax.random.uniform(keys[0])
        rest = (u < self._stop_start_prob) & ~taken
        slow = (
            (u >= self._stop_start_prob)
            & (u < self._stop_start_prob + self._slow_start_prob)
            & ~taken
        )
        bands = jp.asarray(self._start_bands)
        band = jax.random.choice(keys[1], len(self._start_bands), p=bands[:, 2])
        xy, found = self._sample_position(keys[2], data, bands[band, 0], bands[band, 1])
        use = (rest | slow) & found
        heading = qpos[jp.asarray(self._robot_qpos_ids)[2]]
        speed_ = jp.where(
            slow,
            jax.random.uniform(keys[3], (), maxval=self._recovery_speed),
            0.0,
        )
        direction = self._along_heading(keys[4], heading)
        angular_velocity = jp.where(
            slow,
            jax.random.uniform(
                keys[5],
                (),
                minval=-self._slow_angular_velocity,
                maxval=self._slow_angular_velocity,
            ),
            0.0,
        )
        velocity = jp.hstack(
            [speed_ * jp.cos(direction), speed_ * jp.sin(direction), angular_velocity]
        )
        xy_ids = jp.asarray(self._robot_qpos_ids)[:2]
        qpos = qpos.at[xy_ids].set(jp.where(use, xy, qpos[xy_ids]))
        qvel_ids = jp.asarray(self._robot_qvel_ids)
        qvel = qvel.at[qvel_ids].set(jp.where(use, velocity, qvel[qvel_ids]))
        return qpos, qvel

    def step(self, state: go_to_goal.State, action: jax.Array) -> go_to_goal.State:
        dwell, substep = state.info["recovery_dwell"], state.info["substep"]
        # The auto reset does not reset the info: a new episode starts with zero
        # steps. The step within the control step needs no reset, as episodes
        # end with a control step.
        if "steps" in state.info:
            dwell = jp.where(state.info["steps"] == 0, 0, dwell)
        nstate = super().step(state, action)
        violated = violates(nstate.obs, self._hard_distance, self._max_distance)
        in_recovery = recovery_reward_fn(
            nstate.obs,
            self._recovery_distance,
            self._recovery_speed,
            self._hard_distance,
            self._max_distance,
            self._recovery_max_distance,
        )
        # Recovered once the robot ended recovery_dwell control steps in a row in
        # R. Only the end of a control step counts, as only its last step is kept
        # by the episode wrapper.
        end_of_control_step = substep == self._action_repeat - 1
        dwell = jp.where(
            end_of_control_step, jp.where(in_recovery > 0.0, dwell + 1, 0), dwell
        )
        recovery_reward = (
            end_of_control_step & (dwell >= self._recovery_dwell)
        ).astype(jp.float32)
        terminate = (recovery_reward > 0.0) | violated
        nstate.info["recovery_dwell"] = jp.where(terminate, 0, dwell)
        nstate.info["substep"] = (substep + 1) % self._action_repeat
        nstate.info["recovery_reward"] = recovery_reward
        nstate.metrics["recovery/reached"] = recovery_reward
        nstate.metrics["recovery/violated"] = violated.astype(jp.float32)
        done = jp.maximum(nstate.done, terminate.astype(nstate.done.dtype))
        return nstate.replace(done=done)  # type: ignore
