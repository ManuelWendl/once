"""Maze layouts, walls and path geometry of the CyberRunner labyrinth, in board coordinates (m)."""

import json
import pathlib
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ss2r.benchmark_suites.mujoco_playground.cyberrunner.sim.constants import (
    HOLE_RADIUS,
    WALL_RADIUS,
)

BOARD_WIDTH = 0.276
BOARD_HEIGHT = 0.231
_LAYOUTS_FILE = pathlib.Path(__file__).with_name("layouts.json")


class Layout(NamedTuple):
    walls_h: np.ndarray  # [H, 3] horizontal walls (x_start, x_end, y)
    walls_v: np.ndarray  # [V, 3] vertical walls (y_start, y_end, x)
    holes: np.ndarray  # [K, 2] hole centres
    waypoints: np.ndarray  # [N, 2] path from start to goal


def get_layout(name: str) -> Layout:
    """Return the walls, holes and waypoints of the hard, medium or easy maze."""
    layouts = json.loads(_LAYOUTS_FILE.read_text())
    if name not in layouts:
        raise ValueError(
            f"Unknown maze layout {name!r}, expected one of {sorted(layouts)}"
        )
    entry = layouts[name]
    return Layout(*(np.array(entry[f], dtype=entry["dtype"]) for f in Layout._fields))


def wall_segments(maze: Layout) -> Tuple[np.ndarray, np.ndarray]:
    """Maze walls and board edges as segments [x0, y0, x1, y1], with their half-widths."""
    walls = [[x0, y, x1, y] for x0, x1, y in maze.walls_h]
    walls += [[x, y0, x, y1] for y0, y1, x in maze.walls_v]
    w, h = BOARD_WIDTH, BOARD_HEIGHT
    edges = [[0, 0, w, 0], [0, h, w, h], [0, 0, 0, h], [w, 0, w, h]]
    half_width = [WALL_RADIUS] * len(walls) + [0.0] * len(edges)
    return np.array(walls + edges, np.float32), np.array(half_width, np.float32)


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
    nearest = jnp.argmin(jnp.sum((points - ball_pos) ** 2, axis=1))
    indices = jnp.clip(nearest + step * jnp.arange(num_points), 0, points.shape[0] - 1)
    return (points[indices] - ball_pos).reshape(-1)


def compute_waypoint_distances(waypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute segment lengths and cumulative distances along waypoints."""
    seg_vectors = waypoints[1:] - waypoints[:-1]
    seg_lengths = np.linalg.norm(seg_vectors, axis=1)
    cum_distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return seg_lengths.astype(np.float32), cum_distances.astype(np.float32)


def ray_segment_intersection(ray_origin, ray_dir, seg_start, seg_end):
    """Distance from ray_origin to ray-segment intersection, or inf."""
    seg_vec = seg_end - seg_start
    denom = ray_dir[0] * seg_vec[1] - ray_dir[1] * seg_vec[0]
    parallel = jnp.abs(denom) < 1e-8
    diff = seg_start - ray_origin
    t = (diff[0] * seg_vec[1] - diff[1] * seg_vec[0]) / (denom + 1e-10)
    s = (diff[0] * ray_dir[1] - diff[1] * ray_dir[0]) / (denom + 1e-10)
    valid = (~parallel) & (t > 0) & (s >= 0) & (s <= 1)
    return jnp.where(valid, t, jnp.inf)


def ray_circle_intersection(ray_origin, ray_dir, circle_center, radius):
    """Distance from ray_origin to ray-circle entry, or inf."""
    oc = ray_origin - circle_center
    a = jnp.dot(ray_dir, ray_dir)
    b = 2.0 * jnp.dot(oc, ray_dir)
    c = jnp.dot(oc, oc) - radius**2
    disc = b * b - 4 * a * c
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t = jnp.where(t1 > 0, t1, jnp.where(t2 > 0, t2, jnp.inf))
    return jnp.where(disc < 0, jnp.inf, t)


@partial(jax.jit, static_argnames=("num_rays",))
def compute_path_progress_jax(
    marble_pos,
    waypoints,
    seg_lengths,
    cum_distances,
    walls_h,
    walls_v,
    holes,
    num_rays: int = 32,
    hole_radius: float = HOLE_RADIUS,
    proj_threshold: float = 0.002,
):
    """Path progress with direct-projection fallback + raycast occlusion check.

    Mirrors the algorithm in env_mujoco.py:264-434:
      1. Direct projection onto every segment; if marble is within `proj_threshold`
         of the path, return the projection result directly (cleaned env L319-323).
      2. Otherwise, cast `num_rays` rays. For each ray, find the closest path
         intersection that is closer than the nearest wall/hole obstacle
         (cleaned env L401). Pick the ray with the minimum visible path distance.
      3. If no ray finds a visible path, return -1 sentinel for progress.

    Returns:
        progress: scalar float (path-distance * 10), -1 if not detected.
        seg_idx: int — which path segment was hit (0 if not detected).
        param: float in [0, 1] along that segment.
        closest_point: [2] float position on the path.
        found: bool — whether a valid path point was located.
    """
    seg_starts = waypoints[:-1]  # [S, 2]
    seg_ends = waypoints[1:]  # [S, 2]
    seg_vecs = seg_ends - seg_starts  # [S, 2]

    # === DIRECT PROJECTION (cleaned env L294-323) ===
    to_marble = marble_pos - seg_starts  # [S, 2]
    seg_len_sq = jnp.sum(seg_vecs**2, axis=1)  # [S]
    t_proj = jnp.sum(to_marble * seg_vecs, axis=1) / jnp.maximum(seg_len_sq, 1e-10)
    t_proj_clipped = jnp.clip(t_proj, 0.0, 1.0)  # [S]
    closest_on_seg = seg_starts + t_proj_clipped[:, None] * seg_vecs  # [S, 2]
    dist_to_seg = jnp.linalg.norm(marble_pos - closest_on_seg, axis=1)  # [S]
    proj_seg = jnp.argmin(dist_to_seg)
    proj_min_dist = dist_to_seg[proj_seg]
    proj_param = t_proj_clipped[proj_seg]
    proj_progress = cum_distances[proj_seg] + proj_param * seg_lengths[proj_seg]
    proj_closest = closest_on_seg[proj_seg]

    # === RAYCAST (vmap-of-vmap, mirroring cyberrunner_mjx/utils/raycast.py:145-221) ===
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, num_rays, endpoint=False)
    ray_dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)  # [R, 2]

    def cast_single_ray(direction):
        # Walls — vmap over horizontal walls (x_start, x_end, y)
        h_dists = jax.vmap(
            lambda w: ray_segment_intersection(
                marble_pos,
                direction,
                jnp.array([w[0], w[2]]),
                jnp.array([w[1], w[2]]),
            )
        )(walls_h)
        # Walls — vmap over vertical walls (y_start, y_end, x)
        v_dists = jax.vmap(
            lambda w: ray_segment_intersection(
                marble_pos,
                direction,
                jnp.array([w[2], w[0]]),
                jnp.array([w[2], w[1]]),
            )
        )(walls_v)
        wall_dist = jnp.min(jnp.concatenate([h_dists, v_dists]))

        # Holes — vmap over hole centers
        hole_dists = jax.vmap(
            lambda hp: ray_circle_intersection(marble_pos, direction, hp, hole_radius)
        )(holes)
        hole_dist = jnp.min(hole_dists)

        obstacle_dist = jnp.minimum(wall_dist, hole_dist)

        # Path intersections — vmap over segments, return (dist_along_ray, seg_param)
        def intersect_seg(start, end):
            d = ray_segment_intersection(marble_pos, direction, start, end)
            seg_v = end - start
            seg_l_sq = jnp.sum(seg_v**2)
            hit = marble_pos + direction * d
            s = jnp.dot(hit - start, seg_v) / jnp.maximum(seg_l_sq, 1e-8)
            return d, jnp.clip(s, 0.0, 1.0)

        path_dists, path_params = jax.vmap(intersect_seg)(seg_starts, seg_ends)
        # Occlusion: discard intersections behind walls/holes (cleaned env L401)
        valid = path_dists < obstacle_dist
        path_dists_filtered = jnp.where(valid, path_dists, jnp.inf)
        min_idx = jnp.argmin(path_dists_filtered)
        return path_dists_filtered[min_idx], min_idx, path_params[min_idx]

    ray_min_dists, ray_seg_idxs, ray_params_per_ray = jax.vmap(cast_single_ray)(
        ray_dirs
    )
    # Across all rays, pick the one with the closest visible path hit
    closest_ray = jnp.argmin(ray_min_dists)
    ray_dist = ray_min_dists[closest_ray]
    ray_seg = ray_seg_idxs[closest_ray]
    ray_param = ray_params_per_ray[closest_ray]
    ray_found = ray_dist < jnp.inf

    ray_progress = cum_distances[ray_seg] + ray_param * seg_lengths[ray_seg]
    ray_closest = seg_starts[ray_seg] + ray_param * (
        seg_ends[ray_seg] - seg_starts[ray_seg]
    )

    # === SELECT BRANCH (replaces both Python `if`s in cleaned env L319, L414) ===
    use_proj = proj_min_dist < proj_threshold
    progress = jnp.where(
        use_proj,
        proj_progress * 10.0,
        jnp.where(ray_found, ray_progress * 10.0, -1.0),
    )
    seg_idx = jnp.where(use_proj, proj_seg, jnp.where(ray_found, ray_seg, 0))
    param = jnp.where(use_proj, proj_param, jnp.where(ray_found, ray_param, 0.0))
    closest = jnp.where(
        use_proj,
        proj_closest,
        jnp.where(ray_found, ray_closest, jnp.zeros(2)),
    )
    found = use_proj | ray_found
    return progress, seg_idx, param, closest, found
