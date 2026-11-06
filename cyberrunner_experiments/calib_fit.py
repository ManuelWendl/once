"""Fit the camera-to-plate transformation from a calib_record recording: the still ball in the
plate's corners (centre 6.3 mm from both edges, as in the sim) against the camera's positions.
Writes maze = A @ camera + b as JSON.

Usage: python calib_fit.py RECORDING.npz OUT.json
"""

import json
import sys

import numpy as np

from ss2r.benchmark_suites.mujoco_playground.cyberrunner.sim import constants, maze

R = constants.MARBLE_RADIUS
W, H = maze.BOARD_WIDTH, maze.BOARD_HEIGHT
CORNERS = np.array([[R, R], [W - R, R], [W - R, H - R], [R, H - R]])
MAX_OFF_M = 0.025  # a still ball farther than this from every corner is not a reference
# still: 90% within 2.5 mm of the 1 s median, 2 s at least
STILL_M, STILL_S, MIN_S = 0.0025, 1.0, 2.0


def still_clusters(data):
    t, ball = data[:, 0], data[:, 1:3]
    ok = np.isfinite(ball).all(1)
    still = np.zeros(len(t), bool)
    for i in np.flatnonzero(ok):
        w = ok & (t >= t[i] - STILL_S) & (t <= t[i])
        spread = np.percentile(
            np.linalg.norm(ball[w] - np.median(ball[w], axis=0), axis=1), 90
        )
        still[i] = t[i] - t[w].min() > 0.8 * STILL_S and spread < STILL_M
    clusters, start = [], None
    for i in range(len(t) + 1):
        if i < len(t) and still[i]:
            start = i if start is None else start
        elif start is not None:
            if t[i - 1] - t[start] >= MIN_S:
                clusters.append((np.median(ball[start:i], axis=0), t[start], t[i - 1]))
            start = None
    return clusters


def similarity(src, dst):
    """Least-squares rotation, uniform scale and shift (Umeyama)."""
    ms, md = src.mean(0), dst.mean(0)
    u, s, vt = np.linalg.svd((dst - md).T @ (src - ms))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    scale = np.trace(np.diag(s) @ d) / ((src - ms) ** 2).sum()
    return scale * rot, md - scale * rot @ ms


if __name__ == "__main__":
    clusters = still_clusters(np.load(sys.argv[1])["data"])
    for c, t0, t1 in clusters:
        print(
            f"still ball at camera ({1000 * c[0]:6.1f}, {1000 * c[1]:6.1f}) mm, {t0:5.1f}-{t1:5.1f} s"
        )
    if not clusters:
        sys.exit("no still ball found in the recording")
    cam = np.array([c for c, _, _ in clusters])
    dist = np.linalg.norm(cam[:, None] - CORNERS[None], axis=-1)
    # -1: not at a corner
    near = np.where(dist.min(1) < MAX_OFF_M, np.argmin(dist, axis=1), -1)
    pairs = {}
    for k in range(len(CORNERS)):  # per corner, the longest still period next to it
        idx = [i for i in range(len(cam)) if near[i] == k]
        if idx:
            pairs[k] = max(idx, key=lambda i: clusters[i][2] - clusters[i][1])
    if len(pairs) < 3:
        sys.exit(f"only {len(pairs)} corners found; need at least 3")
    src = np.array([cam[i] for i in pairs.values()])
    dst = CORNERS[list(pairs)]
    A, b = similarity(src, dst)
    res = 1000 * np.linalg.norm(src @ A.T + b - dst, axis=1)
    angle = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    print(
        f"corners used: {len(pairs)}; rotation {angle:+.2f} deg, scale {np.sqrt(abs(np.linalg.det(A))):.4f}, "
        f"shift ({1000 * b[0]:+.1f}, {1000 * b[1]:+.1f}) mm; residuals {np.round(res, 1).tolist()} mm"
    )
    json.dump(
        dict(
            A=A.tolist(),
            b=b.tolist(),
            residual_mm=res.tolist(),
            corners=[int(k) for k in pairs],
        ),
        open(sys.argv[2], "w"),
        indent=2,
    )
