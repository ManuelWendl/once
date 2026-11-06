"""The bridge's server: one JSON request and one JSON reply per line on a local socket.

Standard library only, as it runs in the rig's conda env too.
"""

import json
import socket
from typing import Callable


def serve(
    handle: Callable[[dict], dict],
    idle: Callable[[], None],
    port: int,
    idle_s: float = 0.1,
) -> None:
    """Serves one client at a time. `idle` is called when a client is silent for `idle_s`,
    and when it disconnects."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    print(f"Bridge listening on {port}", flush=True)
    try:
        while True:
            connection, _ = server.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(idle_s)
            pending = b""
            print("Actor connected", flush=True)
            try:
                while True:
                    try:
                        data = connection.recv(65536)
                    except TimeoutError:
                        idle()
                        continue
                    if not data:
                        break
                    pending += data
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        try:
                            reply = handle(json.loads(line))
                        except (ValueError, KeyError, RuntimeError) as error:
                            reply = {"error": repr(error)}
                        connection.sendall((json.dumps(reply) + "\n").encode())
            except ConnectionError:
                pass
            finally:
                connection.close()
                idle()
                print("Actor disconnected: motors at zero", flush=True)
    finally:
        server.close()
