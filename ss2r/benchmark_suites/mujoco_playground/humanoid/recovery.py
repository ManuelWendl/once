"""Humanoid recovery task: stand back up straight, without falling over.

The predicates below operate on the humanoid observation
`[joint angles (21), head height (1), extremities (12), torso z (3),
center of mass velocity (3), qvel (27)]`. `torso z` is the bottom row of the
torso orientation, its last entry the cosine of the tilt of the torso (upright).
`qvel` starts with the linear (world frame) and the angular (torso frame)
velocity of the root. They only use operators, so that they can be shared
between jax (simulation, learned model) and numpy (CPU environment) code.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from brax import math as brax_math
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import humanoid

from ss2r.benchmark_suites.mujoco_playground import mjx_data

_HEAD_HEIGHT = 21
_UPRIGHT = 36
# Angular velocity of the torso about its x and y axes, the rate at which it
# tilts.
_TILT_RATES = (43, 44)

# The humanoid falls over when its torso tilts by more than 60 degrees, or when
# its head is below 1 m (standing straight it is at 1.475 m, kneeling at about
# 1.07 m).
HARD_UPRIGHT = 0.5
HARD_HEAD_HEIGHT = 1.0
# Safe operable region R: standing up straight, with the torso tilted by less
# than about 25 degrees, the head at least at the height of the standing reward
# (about 7 cm below standing straight, which excludes crouching), and only
# slowly tilting.
RECOVERY_UPRIGHT = 0.9
RECOVERY_HEAD_HEIGHT = 1.4
RECOVERY_TILT_RATE = 1.0


def violates(
    obs,
    hard_upright: float = HARD_UPRIGHT,
    hard_head_height: float = HARD_HEAD_HEIGHT,
):
    """Whether the humanoid fell over."""
    return (obs[..., _UPRIGHT] < hard_upright) | (
        obs[..., _HEAD_HEIGHT] < hard_head_height
    )


def in_recovery_set(
    obs,
    recovery_upright: float = RECOVERY_UPRIGHT,
    recovery_head_height: float = RECOVERY_HEAD_HEIGHT,
    recovery_tilt_rate: float = RECOVERY_TILT_RATE,
):
    """Whether the humanoid is in the safe operable region R."""
    x, y = _TILT_RATES
    tilt_rate_sq = obs[..., x] ** 2 + obs[..., y] ** 2
    return (
        (obs[..., _UPRIGHT] >= recovery_upright)
        & (obs[..., _HEAD_HEIGHT] >= recovery_head_height)
        & (tilt_rate_sq <= recovery_tilt_rate**2)
    )


def recovery_reward_fn(
    next_obs,
    recovery_upright: float = RECOVERY_UPRIGHT,
    recovery_head_height: float = RECOVERY_HEAD_HEIGHT,
    recovery_tilt_rate: float = RECOVERY_TILT_RATE,
    hard_upright: float = HARD_UPRIGHT,
    hard_head_height: float = HARD_HEAD_HEIGHT,
):
    """1 if the humanoid entered R, 0 otherwise. Not to be mixed with the task reward."""
    reached = in_recovery_set(
        next_obs, recovery_upright, recovery_head_height, recovery_tilt_rate
    )
    reached = reached & ~violates(next_obs, hard_upright, hard_head_height)
    return reached.astype("float32")


def default_config() -> config_dict.ConfigDict:
    config = humanoid.default_config()
    config.hard_upright = HARD_UPRIGHT
    config.hard_head_height = HARD_HEAD_HEIGHT
    config.recovery_upright = RECOVERY_UPRIGHT
    config.recovery_head_height = RECOVERY_HEAD_HEIGHT
    config.recovery_tilt_rate = RECOVERY_TILT_RATE
    # Consecutive steps in R until the humanoid counts as recovered: 1 ends an
    # episode on entering R, more also require staying in R.
    config.recovery_dwell_steps = 1
    # Initial states: largest tilt of the torso (in a random direction, with a
    # random heading), share of the joint ranges around the standing pose, and
    # the largest initial velocities (of the root, uniform per axis: horizontal
    # and vertical in the world frame, angular in the torso frame).
    # `reset_standing_share` of the starts are near standing instead, with
    # `reset_standing_noise` on the angles and velocities. Sampled states that
    # already fell over start near standing as well.
    config.reset_tilt = 0.5
    config.reset_joint_range = 0.5
    config.reset_horizontal_velocity = 1.5
    config.reset_vertical_velocity = 0.5
    config.reset_angular_velocity = 2.0
    config.reset_joint_velocity = 5.0
    config.reset_standing_share = 0.0
    config.reset_standing_noise = (0.05, 0.1)
    return config


class HumanoidRecovery(humanoid.Humanoid):
    """Humanoid that terminates when reaching R or falling over.

    With `recovery_dwell_steps` > 1, reaching R means staying in R for that many
    consecutive steps.

    `state.reward` remains the walk task reward. The recovery signal is written
    to `state.info["recovery_reward"]`.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            move_speed=humanoid.WALK_SPEED,
            config=config,
            config_overrides=config_overrides,
        )
        defaults = default_config()
        get = lambda name: self._config.get(name, defaults[name])
        self._hard_upright = get("hard_upright")
        self._hard_head_height = get("hard_head_height")
        self._recovery_upright = get("recovery_upright")
        self._recovery_head_height = get("recovery_head_height")
        self._recovery_tilt_rate = get("recovery_tilt_rate")
        self._recovery_dwell_steps = int(get("recovery_dwell_steps"))
        self._reset_tilt = get("reset_tilt")
        self._reset_horizontal_velocity = get("reset_horizontal_velocity")
        self._reset_vertical_velocity = get("reset_vertical_velocity")
        self._reset_angular_velocity = get("reset_angular_velocity")
        self._reset_joint_velocity = get("reset_joint_velocity")
        self._reset_standing_share = float(get("reset_standing_share"))
        self._reset_standing_noise = tuple(
            float(x) for x in get("reset_standing_noise")
        )
        # Bounds of the joint angles: a share of the joint ranges, all of which
        # contain the standing pose (zero).
        joint_range = get("reset_joint_range")
        ranges = self.mj_model.jnt_range[1:]
        self._joint_angle_bounds = (
            joint_range * ranges[:, 0],
            joint_range * ranges[:, 1],
        )
        # All geoms except the floor are capsules or spheres: (radius, half
        # length, unused), with a half length of zero for spheres.
        self._body_geoms = jp.arange(1, self.mj_model.ngeom)

    def _lowest_point(self, data: mjx.Data) -> jax.Array:
        """Height of the lowest point of the humanoid."""
        geoms = self._body_geoms
        center = data.geom_xpos[geoms, 2]
        axis = data.geom_xmat[geoms, 2, 2]
        size = self.mjx_model.geom_size[geoms]
        return jp.min(center - jp.abs(axis) * size[:, 1] - size[:, 0])

    def _on_ground(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """The state with the lowest point of the humanoid on the ground."""
        data = mjx_data.init(self.mjx_model, qpos=qpos, qvel=qvel)
        qpos = qpos.at[2].add(-self._lowest_point(data))
        return mjx_data.init(self.mjx_model, qpos=qpos, qvel=qvel)

    def _sample(
        self,
        rng: jax.Array,
        max_tilt: float,
        joint_bounds: tuple[Any, Any],
        velocities: tuple[float, float, float, float],
    ) -> tuple[jax.Array, jax.Array]:
        """A pose with a random heading and tilt, and random velocities."""
        rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
        nq, nv = self.mjx_model.nq, self.mjx_model.nv
        yaw, direction = jax.random.uniform(rng1, (2,), minval=-jp.pi, maxval=jp.pi)
        tilt = jax.random.uniform(rng2, (), minval=0.0, maxval=max_tilt)
        heading = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        tilt_axis = jp.array([jp.cos(direction), jp.sin(direction), 0.0])
        tilt_quat = jp.concatenate(
            [jp.cos(tilt / 2)[None], jp.sin(tilt / 2) * tilt_axis]
        )
        qpos = jp.zeros(nq)
        qpos = qpos.at[3:7].set(brax_math.quat_mul(tilt_quat, heading))
        lowers, uppers = joint_bounds
        qpos = qpos.at[7:].set(
            jax.random.uniform(rng3, (nq - 7,), minval=lowers, maxval=uppers)
        )
        horizontal, vertical, angular, joint = velocities
        root_bounds = jp.array([horizontal, horizontal, vertical, *(angular,) * 3])
        qvel = jp.zeros(nv)
        qvel = qvel.at[:6].set(
            jax.random.uniform(rng4, (6,), minval=-root_bounds, maxval=root_bounds)
        )
        qvel = qvel.at[6:].set(
            jax.random.uniform(rng5, (nv - 6,), minval=-joint, maxval=joint)
        )
        return qpos, qvel

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Samples a perturbed standing state, anywhere it did not fall over."""
        state = super().reset(rng)
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        qpos, qvel = self._sample(
            rng1,
            self._reset_tilt,
            self._joint_angle_bounds,
            (
                self._reset_horizontal_velocity,
                self._reset_vertical_velocity,
                self._reset_angular_velocity,
                self._reset_joint_velocity,
            ),
        )
        data = self._on_ground(qpos, qvel)
        # A share of the starts near standing, the initial state of the task,
        # such that the prior also learns to act in and around R.
        angle_noise, velocity_noise = self._reset_standing_noise
        standing_qpos, standing_qvel = self._sample(
            rng2,
            angle_noise,
            (-angle_noise, angle_noise),
            (velocity_noise,) * 4,
        )
        standing = jax.random.uniform(rng3) < self._reset_standing_share
        fallen = violates(self._get_obs(data, state.info), *self._hard_bounds)
        use_standing = standing | fallen
        qpos = jp.where(use_standing, standing_qpos, data.qpos)
        qvel = jp.where(use_standing, standing_qvel, qvel)
        data = self._on_ground(qpos, qvel)
        obs = self._get_obs(data, state.info)
        state.info["recovery_reward"] = jp.zeros(())
        # Consecutive steps in R so far.
        state.info["recovery_dwell"] = jp.zeros(())
        state.metrics["recovery/reached"] = jp.zeros(())
        state.metrics["recovery/violated"] = jp.zeros(())
        return state.replace(data=data, obs=obs)

    @property
    def _hard_bounds(self) -> tuple[float, float]:
        return self._hard_upright, self._hard_head_height

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        dwell = state.info["recovery_dwell"]
        # The auto reset does not reset the info: a new episode starts with
        # zero steps.
        if "steps" in state.info:
            dwell = jp.where(state.info["steps"] == 0, 0.0, dwell)
        nstate = super().step(state, action)
        violated = violates(nstate.obs, *self._hard_bounds)
        inside = recovery_reward_fn(
            nstate.obs,
            self._recovery_upright,
            self._recovery_head_height,
            self._recovery_tilt_rate,
            *self._hard_bounds,
        )
        dwell = jp.where(inside > 0.0, dwell + 1.0, 0.0)
        recovery_reward = (dwell >= self._recovery_dwell_steps).astype(jp.float32)
        terminate = (recovery_reward > 0.0) | violated
        nstate.info["recovery_dwell"] = jp.where(terminate, 0.0, dwell)
        nstate.info["recovery_reward"] = recovery_reward
        nstate.metrics["recovery/reached"] = recovery_reward
        nstate.metrics["recovery/violated"] = violated.astype(jp.float32)
        done = jp.maximum(nstate.done, terminate.astype(nstate.done.dtype))
        return nstate.replace(done=done)


dm_control_suite.register_environment(
    "HumanoidRecovery", HumanoidRecovery, default_config
)
