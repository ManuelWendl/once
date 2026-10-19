"""Forces MuJoCo lacks: static and rolling friction of the ball, and the voltage-limited
DC motor with breakaway and sliding joint friction."""

from typing import Tuple

import jax
import jax.numpy as jp

from ss2r.benchmark_suites.mujoco_playground.cyberrunner.sim.constants import (
    MARBLE_RADIUS,
)

STICK_SPEED = 0.002  # m/s: a slower ball sticks when the tilt allows it
HOLD_STEPS = 2.0  # physics steps over which a stuck ball loses its speed
WALL_GAP = 0.0005  # m: the ball touches a wall closer than this
ROLLING_MASS = 7.0 / 5.0  # inertia of a rolling solid ball, in units of its mass


def wall_gaps(
    ball: jax.Array, segments: jax.Array, half_width: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Offset from each wall to the ball, its length, and the gap between ball and wall."""
    start, span = segments[:, :2], segments[:, 2:] - segments[:, :2]
    along = jp.sum((ball - start) * span, -1) / jp.maximum(jp.sum(span**2, -1), 1e-12)
    offset = ball - (start + jp.clip(along, 0.0, 1.0)[:, None] * span)
    dist = jp.linalg.norm(offset, axis=-1)
    return offset, dist, dist - half_width - MARBLE_RADIUS


def wall_normal(
    ball: jax.Array, segments: jax.Array, half_width: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Unit normal from the nearest wall towards the ball, and whether the ball touches it."""
    offset, dist, gap = wall_gaps(ball, segments, half_width)
    i = jp.argmin(gap)
    return offset[i] / jp.maximum(dist[i], 1e-9), gap[i] < WALL_GAP


def stiction_force(
    ball: jax.Array,
    velocity: jax.Array,
    gravity: jax.Array,
    stuck: jax.Array,
    static: jax.Array,
    rolling: jax.Array,
    wall_friction: jax.Array,
    mass: jax.Array,
    walls: Tuple[jax.Array, jax.Array],
    dt: float,
) -> Tuple[jax.Array, jax.Array]:
    """Force on the ball (board x, y) and whether it is stuck.

    ball, velocity: the ball's position and velocity relative to the board (x, y)
    gravity: gravity in board coordinates (x, y, z)
    static, rolling: tan of the tilt that starts a still ball, and of the resistance
        while it rolls
    wall_friction: friction coefficient between the ball and a wall it is pushed against
    """
    downhill, pressure = gravity[:2], jp.abs(gravity[2])
    normal, touching = wall_normal(ball, *walls)
    push = jp.where(touching, jp.maximum(-downhill @ normal, 0.0), 0.0)
    load = downhill + push * normal  # the wall carries the rest
    limit = static * pressure + wall_friction * push
    speed = jp.linalg.norm(velocity)
    stuck = (jp.linalg.norm(load) <= limit) & (stuck | (speed < STICK_SPEED))
    hold = -mass * load - ROLLING_MASS * mass * velocity / (HOLD_STEPS * dt)
    resist = -mass * rolling * pressure * velocity / jp.maximum(speed, STICK_SPEED)
    return jp.where(stuck, hold, resist), stuck


def current_bounds(
    speed: jax.Array, limit: jax.Array, no_load_speed: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Motor current range (command units) the voltage limit leaves at this speed."""
    back_emf = speed / no_load_speed
    return limit * (-1.0 - back_emf), limit * (1.0 - back_emf)


def friction_assist(
    speed: jax.Array,
    static: jax.Array,
    sliding_fraction: jax.Array,
    stribeck_speed: jax.Array,
) -> jax.Array:
    """Joint torque along the motion that lowers the breakaway friction to the sliding level."""
    moving = 1.0 - jp.exp(-jp.square(speed / stribeck_speed))
    return static * (1.0 - sliding_fraction) * jp.sign(speed) * moving
