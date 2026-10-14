"""Static and rolling friction of the ball on the board, as a force on the ball."""

from typing import Tuple

import jax
import jax.numpy as jp
import numpy as np

from ss2r.benchmark_suites.mujoco_playground.cyberrunner import layout as maze_layout
from ss2r.benchmark_suites.mujoco_playground.cyberrunner.model import (
    MARBLE_RADIUS,
    WALL_RADIUS,
)

STICK_SPEED = 0.002  # m/s: a slower ball sticks when the tilt allows it
HOLD_STEPS = 2.0  # physics steps over which a stuck ball loses its speed
WALL_GAP = 0.0005  # m: the ball touches a wall closer than this
ROLLING_MASS = 7.0 / 5.0  # inertia of a rolling solid ball, in units of its mass


def wall_segments(maze: maze_layout.Layout) -> Tuple[np.ndarray, np.ndarray]:
    """Maze walls and board edges as segments [x0, y0, x1, y1], with their half-widths."""
    walls = [[x0, y, x1, y] for x0, x1, y in maze.walls_h]
    walls += [[x, y0, x, y1] for y0, y1, x in maze.walls_v]
    w, h = maze_layout.BOARD_WIDTH, maze_layout.BOARD_HEIGHT
    edges = [[0, 0, w, 0], [0, h, w, h], [0, 0, 0, h], [w, 0, w, h]]
    half_width = [WALL_RADIUS] * len(walls) + [0.0] * len(edges)
    return np.array(walls + edges, np.float32), np.array(half_width, np.float32)


def wall_normal(
    ball: jax.Array, segments: jax.Array, half_width: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Unit normal from the nearest wall towards the ball, and whether the ball touches it."""
    start, span = segments[:, :2], segments[:, 2:] - segments[:, :2]
    along = jp.sum((ball - start) * span, -1) / jp.maximum(jp.sum(span**2, -1), 1e-12)
    offset = ball - (start + jp.clip(along, 0.0, 1.0)[:, None] * span)
    dist = jp.linalg.norm(offset, axis=-1)
    gap = dist - half_width - MARBLE_RADIUS
    i = jp.argmin(gap)
    return offset[i] / jp.maximum(dist[i], 1e-9), gap[i] < WALL_GAP


def force(
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
