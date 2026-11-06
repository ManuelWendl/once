"""Link between the learner and an actor on another machine, a robot.

There is one exchange: the learner sends the newest policy, and the robot swaps it in and
replies at once with all the transitions since the last exchange. Neither of them waits for
the other: the robot's loop only touches a slot and a buffer, and the learner's a slot and a
queue. Only what changed in the policy travels, block by block. Messages are raw arrays and
a JSON header with a session token, never pickles.
"""

import hmac
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from typing import Any

import numpy as np
import zmq

from ss2r.algorithms.once.actor import flatten_params

TOKEN_ENV = "ONCE_REMOTE_TOKEN"
_BLOCK = 4096
_REPLY_TIMEOUT_MS = 5000
_SEND_TIMEOUT_MS = 1000
# Without a reply for this long the robot counts as paused, and after this long as dead.
_SILENT_S = 5.0
_DEAD_S = 600.0


def _token() -> str:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise RuntimeError(f"Set {TOKEN_ENV} to the same secret on both machines")
    return token


def changed_blocks(
    new: np.ndarray, old: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """The ids of the blocks of `new` that differ from `old`, and their values."""
    num_blocks = math.ceil(new.size / _BLOCK)
    block = lambda x, i: x[i * _BLOCK : (i + 1) * _BLOCK]
    ids = [
        i
        for i in range(num_blocks)
        if old is None or not np.array_equal(block(new, i), block(old, i))
    ]
    values = [block(new, i) for i in ids]
    return (
        np.asarray(ids, np.int32),
        np.concatenate(values) if values else np.zeros(0, np.float32),
    )


def apply_blocks(params: np.ndarray, ids: np.ndarray, values: np.ndarray) -> None:
    """Writes blocks into `params`, in place."""
    offset = 0
    for i in ids:
        start = int(i) * _BLOCK
        length = min(_BLOCK, params.size - start)
        if start < 0 or length <= 0 or offset + length > values.size:
            raise ValueError(f"Block {i} does not fit")
        params[start : start + length] = values[offset : offset + length]
        offset += length
    if offset != values.size:
        raise ValueError(f"{values.size - offset} values left over")


def encode_chunks(items: list) -> tuple[dict, list[bytes]]:
    """The metadata and the raw frames of (chunk, stats) pairs."""
    meta: dict[str, Any] = {"stats": [stats for _, stats in items], "arrays": []}
    frames = []
    for name in sorted(items[0][0]) if items else []:
        array = np.stack([chunk[name] for chunk, _ in items])
        meta["arrays"].append([name, array.dtype.str, list(array.shape)])
        frames.append(array.tobytes())
    return meta, frames


def decode_chunks(meta: dict, frames: list[bytes]) -> list:
    arrays = {
        name: np.frombuffer(frame, dtype).reshape(shape)
        for (name, dtype, shape), frame in zip(meta["arrays"], frames, strict=True)
    }
    return [
        ({name: array[i] for name, array in arrays.items()}, stats)
        for i, stats in enumerate(meta["stats"])
    ]


class RobotServer:
    """The robot's end: the newest policy in a slot, the new transitions in a buffer.

    Has the `read` of `SharedParams` and the `put` of a queue, for `actor_loop`. A reply
    stays until the next request confirms it, so a dropped reply loses no transitions.
    """

    def __init__(self, address: str, num_params: int):
        self._address = address
        self._num_params = num_params
        self._token = _token()
        self._session = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._params: np.ndarray | None = None
        self._version = 0
        self._last_read = 0
        self._buffer: list = []
        self._paused = False
        self._seq = 0
        self._pending: list[bytes] | None = None
        self._last_request: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._params is None or self._version == self._last_read:
                return None
            self._last_read = self._version
            return self._params.copy()

    def put(self, item: Any) -> None:
        if item is None:  # The signal that the actor is ready: its first reply says so.
            return
        with self._lock:
            self._buffer.append(item)

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def seconds_since_request(self) -> float:
        """Seconds since the learner last asked, 0 before its first request."""
        last = self._last_request
        return 0.0 if last is None else time.monotonic() - last

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        # The robot dials out, as the learner's machine cannot reach it.
        socket.connect(self._address)
        try:
            while not self._stop.is_set():
                if not socket.poll(100):
                    continue
                frames = socket.recv_multipart()
                try:
                    reply = self._handle(frames)
                except (ValueError, KeyError) as error:
                    logging.warning("Bad request: %s", error)
                    reply = [json.dumps({"error": str(error)}).encode()]
                socket.send_multipart(reply)
        finally:
            socket.close()
            context.term()

    def _handle(self, frames: list[bytes]) -> list[bytes]:
        header = json.loads(frames[0])
        if not hmac.compare_digest(str(header["token"]), self._token):
            raise ValueError("Wrong token")
        if header["size"] != self._num_params:
            raise ValueError(f"{header['size']} parameters, not {self._num_params}")
        self._last_request = time.monotonic()
        ids = np.frombuffer(frames[1], np.int32)
        values = np.frombuffer(frames[2], np.float32)
        with self._lock:
            if header["full"]:
                self._params = np.zeros(self._num_params, np.float32)
            if self._params is not None and ids.size:
                apply_blocks(self._params, ids, values)
                self._version += 1
            paused = self._paused
            if self._pending is None or header["ack"] == self._seq:
                self._pending = None
                items, self._buffer = self._buffer, []
            else:
                items = []
        if self._pending is None and items:
            self._seq += 1
            meta, data = encode_chunks(items)
            meta["seq"] = self._seq
            self._pending = [json.dumps(meta).encode(), *data]
        status = {
            "token": self._token,
            "session": self._session,
            "paused": paused,
        }
        if self._pending is None:
            meta, data = {"seq": self._seq, "stats": [], "arrays": []}, []
            return [json.dumps(status).encode(), json.dumps(meta).encode()]
        return [json.dumps(status).encode(), *self._pending]


class RemoteActor:
    """The learner's end, with the interface of the actor in `train`.

    A thread does the exchanges, so that the learner never waits on the network. The
    robot can restart meanwhile: it then gets the full policy again.
    """

    def __init__(self, address: str, num_params: int, exchange_hz: float = 10.0):
        self._address = address
        self._num_params = num_params
        self._period = 1.0 / exchange_hz
        self._token = _token()
        self._lock = threading.Lock()
        self._newest: np.ndarray | None = None
        self._sent: np.ndarray | None = None
        self._chunks: queue.Queue = queue.Queue()
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._session: str | None = None
        self._seq = 0
        self._last_reply = 0.0
        self._robot_paused = False
        self._thread = threading.Thread(target=self._exchange_loop, daemon=True)
        self._started = False

    @property
    def paused(self) -> bool:
        """Whether the robot paused, or has not been heard of for a while."""
        return self._robot_paused or time.monotonic() - self._last_reply > _SILENT_S

    def publish(self, params: Any) -> None:
        flat = np.asarray(flatten_params(params))
        assert flat.size == self._num_params
        with self._lock:
            self._newest = flat
        if not self._started:
            self._started = True
            self._thread.start()
            logging.info("Waiting for the robot at %s", self._address)
            while not self._connected.wait(timeout=1.0):
                self._check()
            logging.info("The robot is connected.")

    def get(self, block: bool) -> list:
        out: list = []
        if block:
            while not out:
                self._check()
                try:
                    out.append(self._chunks.get(timeout=1.0))
                except queue.Empty:
                    pass
        while True:
            try:
                out.append(self._chunks.get_nowait())
            except queue.Empty:
                self._check()
                return out

    def close(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=2 * _REPLY_TIMEOUT_MS / 1000)

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("The link to the robot failed") from self._error
        heard = self._connected.is_set()
        if heard and time.monotonic() - self._last_reply > _DEAD_S:
            raise RuntimeError(f"No reply from the robot for {_DEAD_S} seconds")

    def _exchange_loop(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        # A request without reply can be followed by another one.
        socket.setsockopt(zmq.REQ_RELAXED, 1)
        socket.setsockopt(zmq.REQ_CORRELATE, 1)
        socket.setsockopt(zmq.SNDTIMEO, _SEND_TIMEOUT_MS)
        try:
            socket.bind(self._address)
            while not self._stop.is_set():
                start = time.monotonic()
                self._exchange(socket)
                time.sleep(max(self._period - (time.monotonic() - start), 0.0))
        except Exception as error:  # Reported to the learner, in its thread.
            self._error = error
        finally:
            socket.close()
            context.term()

    def _exchange(self, socket: zmq.Socket) -> None:
        with self._lock:
            newest = self._newest
        assert newest is not None
        ids, values = changed_blocks(newest, self._sent)
        header = {
            "token": self._token,
            "ack": self._seq,
            "size": self._num_params,
            "full": self._sent is None,
        }
        try:
            socket.send_multipart(
                [json.dumps(header).encode(), ids.tobytes(), values.tobytes()]
            )
        except zmq.Again:  # No robot yet.
            return
        if not socket.poll(_REPLY_TIMEOUT_MS):
            # The policy may or may not have arrived.
            self._sent = None
            return
        frames = socket.recv_multipart()
        status = json.loads(frames[0])
        if "error" in status:
            raise RuntimeError(f"The robot refused the request: {status['error']}")
        if not hmac.compare_digest(str(status["token"]), self._token):
            raise RuntimeError("Wrong token in the robot's reply")
        self._sent = newest
        if status["session"] != self._session:
            if self._session is not None:
                logging.warning("The robot restarted, sending the full policy.")
                self._sent = None
            self._session = status["session"]
            self._seq = 0
        meta = json.loads(frames[1])
        if meta["seq"] != self._seq:
            for item in decode_chunks(meta, frames[2:]):
                self._chunks.put(item)
            self._seq = meta["seq"]
        self._robot_paused = status["paused"]
        self._last_reply = time.monotonic()
        self._connected.set()
