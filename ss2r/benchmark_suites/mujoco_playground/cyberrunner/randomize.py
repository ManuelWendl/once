"""Domain randomization for the CyberRunner environment.

Ranges are [low, high] multipliers on the nominal values, except
gravity_tilt_deg, which tilts gravity by that many degrees along x and y, and the ball
stiction tilts, in degrees.
"""

from typing import Any, Mapping, Tuple

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

SCALE_GROUPS = (
    "motor_gain",
    "motor_time_constant",
    "motor_speed_limit",
    "joint_damping",
    "joint_friction",
    "ball_friction",  # sliding only: the condim-3 contact ignores spin and roll
    "ball_mass",  # inertia scaled with it
    "ball_contact_damping",
)
TILT_GROUP = "gravity_tilt_deg"
# Ball stiction (needs ball_stiction), in degrees of board tilt; [0, 0] keeps the model's
# value: the tilt that starts a still ball, and the resistance while it rolls.
STATIC_GROUP = "ball_static_deg"
ROLLING_GROUP = "ball_rolling_deg"
_MOTORS = ("alpha_motor", "beta_motor")
_JOINTS = ("alpha_joint", "beta_joint")

SAMPLE_NAMES = (
    "motor_gain_alpha",
    "motor_gain_beta",
    "motor_time_constant_alpha",
    "motor_time_constant_beta",
    "motor_speed_limit_alpha",
    "motor_speed_limit_beta",
    "joint_damping_alpha",
    "joint_damping_beta",
    "joint_friction_alpha",
    "joint_friction_beta",
    "ball_friction",
    "ball_mass",
    "ball_contact_damping",
    "gravity_tilt_x_deg",
    "gravity_tilt_y_deg",
    "ball_static_deg",
    "ball_rolling_deg",
)


def domain_randomization(
    sys: mjx.Model, rng: jax.Array, cfg: Mapping[str, Any]
) -> Tuple[mjx.Model, mjx.Model, jax.Array]:
    """Returns the batched model, its vmap axes and the sampled values."""
    ranges = _ranges(cfg)
    act = jp.array([_id(sys, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _MOTORS])
    stiction = bool(sys.geom_friction[ball_geom_id(sys), 1] > 0)
    for name in (STATIC_GROUP, ROLLING_GROUP):
        if ranges[name][1] > 0 and not stiction:
            raise ValueError(f"{name} needs the ball_stiction model")
    # DC motors (motor_dc): motor_speed_limit scales the voltage limit (current limit and
    # no-load speed) instead of the command cap.
    dc_axes = sys.actuator_gainprm[act, 2] > 0
    dc = bool(jp.any(dc_axes))
    joints = [_id(sys, mujoco.mjtObj.mjOBJ_JOINT, n) for n in _JOINTS]
    dof = jp.array([int(sys.jnt_dofadr[j]) for j in joints])
    ball_body = _id(sys, mujoco.mjtObj.mjOBJ_BODY, "marble")
    ball_geom = int(sys.body_geomadr[ball_body])
    g = jp.linalg.norm(sys.opt.gravity)

    @jax.vmap
    def randomize(key):
        keys = jax.random.split(key, len(SCALE_GROUPS) + 1)
        k = dict(zip(SCALE_GROUPS + (TILT_GROUP,), keys))

        def draw(name, shape=()):
            low, high = ranges[name]
            return jax.random.uniform(k[name], shape, minval=low, maxval=high)

        gain = draw("motor_gain", (2,))
        tau = draw("motor_time_constant", (2,))
        speed_limit = draw("motor_speed_limit", (2,))
        damping = draw("joint_damping", (2,))
        friction_loss = draw("joint_friction", (2,))
        ball_friction = draw("ball_friction")
        ball_mass = draw("ball_mass")
        contact_damping = draw("ball_contact_damping")
        tilt = draw(TILT_GROUP, (2,))
        # Own keys, so the draws of the other groups stay as they were.
        stiction_deg, stiction_coef = [], []
        for slot, name in ((1, STATIC_GROUP), (2, ROLLING_GROUP)):
            low, high = ranges[name]
            deg = jax.random.uniform(
                jax.random.fold_in(key, 3 + slot), minval=low, maxval=high
            )
            nominal = sys.geom_friction[ball_geom, slot]
            stiction_deg.append(deg)
            stiction_coef.append(jp.where(high > 0, jp.tan(jp.radians(deg)), nominal))

        horizontal = g * jp.sin(jp.deg2rad(tilt))
        gravity = jp.append(horizontal, -jp.sqrt(g**2 - jp.sum(horizontal**2)))
        fields = {
            "actuator_gear": sys.actuator_gear.at[act, 0].multiply(gain),
            "actuator_dynprm": sys.actuator_dynprm.at[act, 0].multiply(tau),
            "actuator_ctrlrange": sys.actuator_ctrlrange.at[act].multiply(
                jp.where(dc_axes, 1.0, speed_limit)[:, None]
            ),
            "dof_damping": sys.dof_damping.at[dof].multiply(damping),
            "dof_frictionloss": sys.dof_frictionloss.at[dof].multiply(friction_loss),
            "geom_friction": sys.geom_friction.at[ball_geom, 0]
            .multiply(ball_friction)
            .at[ball_geom, 1:]
            .set(jp.stack(stiction_coef)),
            "body_mass": sys.body_mass.at[ball_body].multiply(ball_mass),
            "body_inertia": sys.body_inertia.at[ball_body].multiply(ball_mass),
            "geom_solref": sys.geom_solref.at[ball_geom, 1].multiply(contact_damping),
            "opt.gravity": gravity,
        }
        if dc:
            fields["actuator_gainprm"] = sys.actuator_gainprm.at[act, 1:3].multiply(
                speed_limit[:, None]
            )
        samples = jp.concatenate(
            [
                gain,
                tau,
                speed_limit,
                damping,
                friction_loss,
                jp.stack([ball_friction, ball_mass, contact_damping]),
                tilt,
                jp.stack(stiction_deg),
            ]
        )
        return fields, samples

    fields, samples = randomize(rng)
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace({name: 0 for name in fields})
    return sys.tree_replace(fields), in_axes, samples


def ball_geom_id(sys: mjx.Model) -> int:
    return int(sys.body_geomadr[_id(sys, mujoco.mjtObj.mjOBJ_BODY, "marble")])


def _id(sys: mjx.Model, obj: mujoco.mjtObj, name: str) -> int:
    idx = mjx.name2id(sys, obj, name)
    if idx < 0:
        raise ValueError(f"CyberRunner model has no {obj.name} named {name!r}")
    return idx


def _ranges(cfg: Mapping[str, Any]) -> dict:
    """Validated [low, high] ranges."""
    ranges = {}
    for name in SCALE_GROUPS + (TILT_GROUP, STATIC_GROUP, ROLLING_GROUP):
        if name not in cfg:
            raise ValueError(f"Missing domain randomization range {name!r}")
        low, high = (float(v) for v in cfg[name])
        if low > high:
            raise ValueError(f"{name}: low {low} is above high {high}")
        if name in SCALE_GROUPS and low <= 0:
            raise ValueError(f"{name}: multipliers must be positive, got {low}")
        if name in (STATIC_GROUP, ROLLING_GROUP) and not 0 <= low <= high < 45:
            raise ValueError(f"{name}: must lie in [0, 45) deg, got {low}, {high}")
        ranges[name] = (low, high)
    return ranges
