"""Go to goal recovery task: slow down and get clear of all obstacles.

The predicates below operate on the go to goal observation
`[obstacle_lidar (16), goal_lidar (16), ..., accelerometer (2), velocimeter (2),
gyro (1), magnetometer (2)]`. They index the lidar from the front and the
sensors from the back, such that they hold with and without
`GoToGoalObservationWrapper`, and only use operators and array methods, so that
they can be shared between jax (simulation, learned model) and numpy code.
"""

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


def violates(obs, hard_distance: float = HARD_DISTANCE):
    """Whether the hard constraint on the distance to the obstacles is violated."""
    return obstacle_distance(obs) < hard_distance


def in_recovery_set(
    obs,
    recovery_distance: float = RECOVERY_DISTANCE,
    recovery_speed: float = RECOVERY_SPEED,
):
    """Whether the robot is in the safe operable region R."""
    return (obstacle_distance(obs) >= recovery_distance) & (
        speed(obs) <= recovery_speed
    )


def recovery_reward_fn(
    next_obs,
    recovery_distance: float = RECOVERY_DISTANCE,
    recovery_speed: float = RECOVERY_SPEED,
    hard_distance: float = HARD_DISTANCE,
):
    """1 if the robot entered R, 0 otherwise. Not to be mixed with the task reward."""
    reached = in_recovery_set(next_obs, recovery_distance, recovery_speed)
    reached = reached & ~violates(next_obs, hard_distance)
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
        reset_speed: float = RESET_SPEED,
        reset_angular_velocity: float = RESET_ANGULAR_VELOCITY,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._recovery_distance = recovery_distance
        self._recovery_speed = recovery_speed
        self._hard_distance = hard_distance
        self._reset_speed = reset_speed
        self._reset_angular_velocity = reset_angular_velocity
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
        data = mjx_data.init(
            self.mjx_model,
            qpos=state.data.qpos,
            qvel=qvel,
            mocap_pos=state.data.mocap_pos,
            mocap_quat=state.data.mocap_quat,
        )
        obs = self.get_obs(data)
        state.info["recovery_reward"] = jp.zeros(())
        state.metrics["recovery/reached"] = jp.zeros(())
        state.metrics["recovery/violated"] = jp.zeros(())
        return state.replace(data=data, obs=obs)  # type: ignore

    def step(self, state: go_to_goal.State, action: jax.Array) -> go_to_goal.State:
        nstate = super().step(state, action)
        violated = violates(nstate.obs, self._hard_distance)
        recovery_reward = recovery_reward_fn(
            nstate.obs,
            self._recovery_distance,
            self._recovery_speed,
            self._hard_distance,
        )
        nstate.info["recovery_reward"] = recovery_reward
        nstate.metrics["recovery/reached"] = recovery_reward
        nstate.metrics["recovery/violated"] = violated.astype(jp.float32)
        terminate = (recovery_reward > 0.0) | violated
        done = jp.maximum(nstate.done, terminate.astype(nstate.done.dtype))
        return nstate.replace(done=done)  # type: ignore
