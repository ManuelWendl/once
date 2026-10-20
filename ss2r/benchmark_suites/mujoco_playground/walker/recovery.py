"""Walker recovery task: stand back up straight, without falling over.

The predicates below operate on the walker observation
`[orientations (14), upright (1), qvel (9)]`. The orientations are the xx and xz
components of the bodies torso, right thigh, right leg, right foot, left thigh,
left leg and left foot, i.e. the cosine and sine of their absolute pitch.
`upright` is the zz component of the torso orientation (the cosine of its tilt)
and `qvel` starts with the velocities of `rootz`, `rootx` and `rooty` (the pitch
rate). They only use operators, so that they can be shared between jax
(simulation, learned model) and numpy (CPU environment) code.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import walker

from ss2r.benchmark_suites.mujoco_playground import mjx_data

_UPRIGHT = 14
_PITCH_RATE = 17
# Cosine of the pitch of the thighs and the (lower) legs.
_LEG_COSINES = (2, 4, 8, 10)

# The walker falls over when its torso tilts by more than 60 degrees.
HARD_UPRIGHT = 0.5
# Safe operable region R: standing up straight, with the torso, the thighs and
# the legs tilted by less than about 25 degrees, and only slowly rotating. The
# condition on the thighs and legs excludes crouching and sitting with an
# upright torso.
RECOVERY_UPRIGHT = 0.9
RECOVERY_LEG_UPRIGHT = 0.9
RECOVERY_PITCH_RATE = 1.0


def violates(obs, hard_upright: float = HARD_UPRIGHT):
    """Whether the walker fell over."""
    return obs[..., _UPRIGHT] < hard_upright


def in_recovery_set(
    obs,
    recovery_upright: float = RECOVERY_UPRIGHT,
    recovery_pitch_rate: float = RECOVERY_PITCH_RATE,
    recovery_leg_upright: float = RECOVERY_LEG_UPRIGHT,
):
    """Whether the walker is in the safe operable region R."""
    inside = (obs[..., _UPRIGHT] >= recovery_upright) & (
        abs(obs[..., _PITCH_RATE]) <= recovery_pitch_rate
    )
    for index in _LEG_COSINES:
        inside = inside & (obs[..., index] >= recovery_leg_upright)
    return inside


def recovery_reward_fn(
    next_obs,
    recovery_upright: float = RECOVERY_UPRIGHT,
    recovery_pitch_rate: float = RECOVERY_PITCH_RATE,
    hard_upright: float = HARD_UPRIGHT,
    recovery_leg_upright: float = RECOVERY_LEG_UPRIGHT,
):
    """1 if the walker entered R, 0 otherwise. Not to be mixed with the task reward."""
    reached = in_recovery_set(
        next_obs, recovery_upright, recovery_pitch_rate, recovery_leg_upright
    )
    reached = reached & ~violates(next_obs, hard_upright)
    return reached.astype("float32")


def _bounds(value) -> tuple[float, float]:
    """(low, high) from a symmetric bound or from [low, high]."""
    if isinstance(value, (int, float)):
        return -float(value), float(value)
    low, high = value
    return float(low), float(high)


def _per_joint(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds of all joints from [low, high] of the hip, knee
    and ankle, the same for both legs (joint order: right, then left)."""
    lows, highs = zip(*(_bounds(b) for b in bounds))
    return np.tile(lows, 2), np.tile(highs, 2)


def default_config() -> config_dict.ConfigDict:
    config = walker.default_config()
    config.hard_upright = HARD_UPRIGHT
    config.recovery_upright = RECOVERY_UPRIGHT
    config.recovery_pitch_rate = RECOVERY_PITCH_RATE
    config.recovery_leg_upright = RECOVERY_LEG_UPRIGHT
    # Initial states: share of the joint ranges around the standing pose, and
    # the largest initial velocities. Optionally, `reset_pitch` (largest tilt,
    # by default all tilts that did not fall), `reset_forward_velocity` as
    # [low, high], and `reset_joint_angles` and `reset_joint_velocities` as
    # [low, high] for the hip, knee and ankle of both legs. `reset_standing_share`
    # of the starts are near standing instead, with `reset_standing_noise` on the
    # angles and velocities.
    # Consecutive steps in R until the walker counts as recovered: 1 ends an
    # episode on entering R, more also require staying in R.
    config.recovery_dwell_steps = 1
    config.reset_joint_range = 0.8
    config.reset_pitch_rate = 3.0
    config.reset_forward_velocity = 2.0
    config.reset_vertical_velocity = 0.5
    config.reset_joint_velocity = 5.0
    return config


class WalkerRecovery(walker.PlanarWalker):
    """Walker that terminates when reaching R or falling over.

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
            move_speed=walker.WALK_SPEED,
            config=config,
            config_overrides=config_overrides,
        )
        defaults = default_config()
        get = lambda name: self._config.get(name, defaults[name])
        self._hard_upright = get("hard_upright")
        self._recovery_upright = get("recovery_upright")
        self._recovery_pitch_rate = get("recovery_pitch_rate")
        self._recovery_leg_upright = get("recovery_leg_upright")
        self._recovery_dwell_steps = int(get("recovery_dwell_steps"))
        self._reset_joint_range = get("reset_joint_range")
        self._reset_pitch_rate = get("reset_pitch_rate")
        self._reset_forward_velocity = get("reset_forward_velocity")
        self._reset_vertical_velocity = get("reset_vertical_velocity")
        self._reset_joint_velocity = get("reset_joint_velocity")
        optional = lambda name, default: self._config.get(name, default)
        # Share of the starts near standing, and the noise of their angles and
        # velocities.
        self._reset_standing_share = float(optional("reset_standing_share", 0.0))
        self._reset_standing_noise = tuple(
            float(x) for x in optional("reset_standing_noise", (0.05, 0.1))
        )
        self._reset_pitch = optional(
            "reset_pitch", float(np.arccos(self._hard_upright))
        )
        # Bounds of the joint angles and velocities, (low, high) per joint.
        joint_angles = optional("reset_joint_angles", None)
        if joint_angles is None:
            self._joint_angle_bounds = (
                self._reset_joint_range * self._lowers,
                self._reset_joint_range * self._uppers,
            )
        else:
            self._joint_angle_bounds = _per_joint(joint_angles)
        joint_velocities = optional("reset_joint_velocities", None)
        if joint_velocities is None:
            velocity = self._reset_joint_velocity
            self._joint_velocity_bounds = (
                np.full(self.mjx_model.nv - 3, -velocity),
                np.full(self.mjx_model.nv - 3, velocity),
            )
        else:
            self._joint_velocity_bounds = _per_joint(joint_velocities)
        self._forward_velocity_bounds = _bounds(self._reset_forward_velocity)
        # All geoms except the floor are capsules: (radius, half length).
        self._body_geoms = jp.arange(1, self.mj_model.ngeom)

    def _lowest_point(self, data: mjx.Data) -> jax.Array:
        """Height of the lowest point of the walker."""
        geoms = self._body_geoms
        center = data.geom_xpos[geoms, 2]
        axis = data.geom_xmat[geoms, 2, 2]
        size = self.mjx_model.geom_size[geoms]
        return jp.min(center - jp.abs(axis) * size[:, 1] - size[:, 0])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Samples a perturbed standing state, anywhere it did not fall over."""
        state = super().reset(rng)
        rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8, rng9 = jax.random.split(rng, 9)
        nq, nv = self.mjx_model.nq, self.mjx_model.nv
        max_pitch = self._reset_pitch
        qpos = jp.zeros(nq)
        qpos = qpos.at[2].set(
            jax.random.uniform(rng1, (), minval=-max_pitch, maxval=max_pitch)
        )
        # Joint angles around the standing pose (zero).
        lowers, uppers = self._joint_angle_bounds
        qpos = qpos.at[3:].set(
            jax.random.uniform(rng2, (nq - 3,), minval=lowers, maxval=uppers)
        )
        qvel = jp.zeros(nv)
        qvel = qvel.at[0].set(
            jax.random.uniform(
                rng6,
                (),
                minval=-self._reset_vertical_velocity,
                maxval=self._reset_vertical_velocity,
            )
        )
        low, high = self._forward_velocity_bounds
        qvel = qvel.at[1].set(jax.random.uniform(rng3, (), minval=low, maxval=high))
        qvel = qvel.at[2].set(
            jax.random.uniform(
                rng4, (), minval=-self._reset_pitch_rate, maxval=self._reset_pitch_rate
            )
        )
        lowers, uppers = self._joint_velocity_bounds
        qvel = qvel.at[3:].set(
            jax.random.uniform(rng5, (nv - 3,), minval=lowers, maxval=uppers)
        )
        # A share of the starts near standing, the initial state of the task,
        # such that the prior also learns to act in and around R.
        standing = jax.random.uniform(rng7) < self._reset_standing_share
        angle_noise, velocity_noise = self._reset_standing_noise
        standing_qpos = (
            jp.zeros(nq)
            .at[2:]
            .set(
                jax.random.uniform(
                    rng8, (nq - 2,), minval=-angle_noise, maxval=angle_noise
                )
            )
        )
        standing_qvel = jax.random.uniform(
            rng9, (nv,), minval=-velocity_noise, maxval=velocity_noise
        )
        qpos = jp.where(standing, standing_qpos, qpos)
        qvel = jp.where(standing, standing_qvel, qvel)
        # Place the lowest point of the walker on the ground.
        data = mjx_data.init(self.mjx_model, qpos=qpos, qvel=qvel)
        qpos = qpos.at[0].add(-self._lowest_point(data))
        data = mjx_data.init(self.mjx_model, qpos=qpos, qvel=qvel)
        obs = self._get_obs(data, state.info)
        state.info["recovery_reward"] = jp.zeros(())
        # Consecutive steps in R so far.
        state.info["recovery_dwell"] = jp.zeros(())
        state.metrics["recovery/reached"] = jp.zeros(())
        state.metrics["recovery/violated"] = jp.zeros(())
        return state.replace(data=data, obs=obs)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        dwell = state.info["recovery_dwell"]
        # The auto reset does not reset the info: a new episode starts with
        # zero steps.
        if "steps" in state.info:
            dwell = jp.where(state.info["steps"] == 0, 0.0, dwell)
        nstate = super().step(state, action)
        violated = violates(nstate.obs, self._hard_upright)
        inside = recovery_reward_fn(
            nstate.obs,
            self._recovery_upright,
            self._recovery_pitch_rate,
            self._hard_upright,
            self._recovery_leg_upright,
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


dm_control_suite.register_environment("WalkerRecovery", WalkerRecovery, default_config)
