"""DC motor under a voltage limit, and joint friction with separate breakaway and sliding levels."""

from typing import Tuple

import jax
import jax.numpy as jp


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
