"""Path points relative to the ball, computed as on the real robot."""

import jax
import jax.numpy as jp
import numpy as np


def interpolate_path(waypoints: np.ndarray, distance: float) -> np.ndarray:
    """Points every `distance` metres along the waypoints."""
    waypoints = np.asarray(waypoints, dtype=np.float32)
    segments = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        v = end - start
        num_points = int(np.floor(np.linalg.norm(v) / distance)) + 1
        line = np.linspace(
            0.0, 1.0, num_points, endpoint=False, dtype=np.float32
        ).reshape(num_points, 1)
        segments.append(start + line * v)
    return np.concatenate(segments, axis=0)


def relative_path(
    points: jax.Array, ball_pos: jax.Array, num_points: int, step: int
) -> jax.Array:
    """Flattened [num_points * 2] path points relative to the ball."""
    nearest = jp.argmin(jp.sum((points - ball_pos) ** 2, axis=1))
    indices = jp.clip(nearest + step * jp.arange(num_points), 0, points.shape[0] - 1)
    return (points[indices] - ball_pos).reshape(-1)
