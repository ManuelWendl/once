"""Stores the trajectory of the real system, to replay and render it afterwards."""

import glob
import json
import os
from typing import Any, Mapping

import numpy as np

# Already stored as the observation of the next step.
_SKIP = ("next_observation",)


class TrajectoryWriter:
    """Writes the transitions of the actor to compressed parts on disk.

    Every row is a control step. Fields prefixed with `state_` hold the state
    of the system before the action was taken, `intervention` whether the
    recovery prior took over, and `discount` is zero when the system was reset
    after the step.
    """

    def __init__(self, logdir: str, metadata: Mapping[str, Any]) -> None:
        self._logdir = logdir
        self._chunks: list[Mapping[str, np.ndarray]] = []
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "metadata.json"), "w") as file:
            json.dump(metadata, file, indent=2)

    def add(self, chunk: Mapping[str, np.ndarray]) -> None:
        self._chunks.append(chunk)

    def flush(self, env_steps: int) -> None:
        """Writes the transitions since the last flush, named by the last step."""
        if not self._chunks:
            return
        data = {
            name: np.concatenate([chunk[name] for chunk in self._chunks])
            for name in self._chunks[0]
            if name not in _SKIP
        }
        np.savez_compressed(os.path.join(self._logdir, f"{env_steps:012d}.npz"), **data)
        self._chunks = []


def load(logdir: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Loads a stored trajectory and its metadata."""
    parts = [np.load(path) for path in sorted(glob.glob(f"{logdir}/*.npz"))]
    data = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    with open(os.path.join(logdir, "metadata.json")) as file:
        metadata = json.load(file)
    return data, metadata
