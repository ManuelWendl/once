"""Go to goal recovery task: brake to a (near) full stop, clear of all obstacles.

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

from ss2r.benchmark_suites.safety_gym import go_to_goal, lidar

_OBSTACLE_LIDAR = slice(0, lidar.NUM_LIDAR_BINS)
_VELOCIMETER = slice(-5, -3)

# Distances are measured between the centers of the robot and of an obstacle.
# Hazards have a radius of 0.2 and count as violated when the center of the
# robot is inside. The robot touches a vase from a distance of 0.3.
RECOVERY_MIN_DISTANCE = 0.35
RECOVERY_STOP_SPEED = 0.05
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


def violates(obs, min_distance: float = RECOVERY_MIN_DISTANCE):
    """Whether the robot is closer to an obstacle than the minimum distance."""
    return obstacle_distance(obs) < min_distance


def in_recovery_set(
    obs,
    min_distance: float = RECOVERY_MIN_DISTANCE,
    stop_speed: float = RECOVERY_STOP_SPEED,
):
    """Whether the robot is (nearly) standing still, clear of all obstacles."""
    return (speed(obs) <= stop_speed) & ~violates(obs, min_distance)


def recovery_reward_fn(
    next_obs,
    min_distance: float = RECOVERY_MIN_DISTANCE,
    stop_speed: float = RECOVERY_STOP_SPEED,
):
    """1 if the robot stopped, 0 otherwise. Not to be mixed with the task reward."""
    return in_recovery_set(next_obs, min_distance, stop_speed).astype("float32")


class GoToGoalRecovery(go_to_goal.GoToGoal):
    """Go to goal that terminates when the robot stopped or got too close.

    An episode starts with the robot in motion, anywhere it keeps the minimum
    distance. The minimum distance has to be kept until the robot stopped: the
    episode ends without a recovery reward as soon as it is violated.

    `state.reward` remains the go to goal task reward. The recovery signal is
    written to `state.info["recovery_reward"]`.
    """

    def __init__(
        self,
        *,
        recovery_min_distance: float = RECOVERY_MIN_DISTANCE,
        recovery_stop_speed: float = RECOVERY_STOP_SPEED,
        reset_speed: float = RESET_SPEED,
        reset_angular_velocity: float = RESET_ANGULAR_VELOCITY,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._recovery_min_distance = recovery_min_distance
        self._recovery_stop_speed = recovery_stop_speed
        self._reset_speed = reset_speed
        self._reset_angular_velocity = reset_angular_velocity
        # Start as close to the obstacles as the minimum distance allows.
        obstacle_keepouts = [
            self.spec[name].keepout
            for name in ("hazards", "vases")
            if self.spec[name].num_objects > 0
        ]
        if obstacle_keepouts:
            keepout = max(recovery_min_distance - min(obstacle_keepouts), 0.0)
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
        data = mjx_env.init(
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
        violated = violates(nstate.obs, self._recovery_min_distance)
        recovery_reward = recovery_reward_fn(
            nstate.obs, self._recovery_min_distance, self._recovery_stop_speed
        )
        nstate.info["recovery_reward"] = recovery_reward
        nstate.metrics["recovery/reached"] = recovery_reward
        nstate.metrics["recovery/violated"] = violated.astype(jp.float32)
        terminate = (recovery_reward > 0.0) | violated
        done = jp.maximum(nstate.done, terminate.astype(nstate.done.dtype))
        return nstate.replace(done=done)  # type: ignore
