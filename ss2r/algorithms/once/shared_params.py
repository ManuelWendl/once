"""Lock-free sharing of the newest policy parameters between two processes."""

from multiprocessing import shared_memory

import numpy as np

_HEADER_BYTES = 8


class SharedParams:
    """A flat parameter vector in shared memory, guarded by a sequence lock.

    There is a single writer (the learner) and a single reader (the actor).
    Neither of them ever blocks: the reader skips an update that is being
    written and picks it up at its next attempt.
    """

    def __init__(self, size: int, name: str | None = None):
        self.size = size
        if name is None:
            self._shm = shared_memory.SharedMemory(
                create=True, size=_HEADER_BYTES + 4 * size
            )
            self._owner = True
        else:
            self._shm = shared_memory.SharedMemory(name=name)
            self._owner = False
        self._version = np.ndarray((1,), np.int64, buffer=self._shm.buf)
        self._data = np.ndarray(
            (size,), np.float32, buffer=self._shm.buf, offset=_HEADER_BYTES
        )
        if self._owner:
            self._version[0] = 0
        self._last_read = 0

    @property
    def name(self) -> str:
        return self._shm.name

    def write(self, params: np.ndarray) -> None:
        self._version[0] += 1
        self._data[:] = params
        self._version[0] += 1

    def read(self) -> np.ndarray | None:
        """Returns the parameters if there are new ones, otherwise None."""
        version = int(self._version[0])
        if version == self._last_read or version % 2 == 1:
            return None
        params = self._data.copy()
        if int(self._version[0]) != version:
            return None
        self._last_read = version
        return params

    def close(self) -> None:
        del self._version, self._data
        self._shm.close()
        if self._owner:
            self._shm.unlink()
