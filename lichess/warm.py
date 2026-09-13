#!/usr/bin/env python3
"""
A pre-warmed engine, so a game does not start by waiting for numba.

Importing engine/agent.py compiles the search. That costs about 7 s on a fast laptop and was
measured at 28 s on the Ampere A1 the bot is hosted on, where the cores are roughly four times
slower. lichess-bot starts a fresh engine process for each game and only then arms its own
30 s abort timer, so on that box every game was aborted before the first move.

Nothing about the engine is slow once it is compiled, so the fix is to compile once and keep
it. This daemon imports the engine at boot and then sits on a unix socket. For each game,
lichess/uci_client.py connects, hands over its own stdin, stdout and stderr, and this forks.
The child inherits an already-compiled engine and starts answering UCI immediately.

fork() is what makes this safe rather than clever. The parent never searches, so every child
begins from the same untouched module state the platform gave a fresh process, and two games
running at once cannot see each other's transposition table. Copy-on-write means they share
the ~500 MB of weights and compiled code rather than each paying for it.

Run it under deploy/nnue-warm.service. Without it, uci_client.py falls back to running the
engine in-process, which is exactly what lichess-bot did before.
"""

import contextlib
import os
import signal
import socket
import sys
import threading
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing this imports the engine, which is the 28 s. It happens once, here, at boot.
import uci

DEFAULT_SOCKET = "/run/nnue/warm.sock"

# Three descriptors arrive per game: the game's stdin, stdout and stderr.
FD_COUNT = 3


# What the forked child becomes: an ordinary uci.py, but one that did not have to compile.
#
# The descriptors belong to the client process lichess-bot is talking to, so they are put where
# uci.py expects to find them, and uci.py's own split is redone on top - the protocol keeps a
# private handle on the real stdout and fd 1 is pointed at stderr, so no diagnostic from the
# engine or a library can land in the middle of the UCI stream.
def serve_game(fds: list[int], conn: socket.socket) -> None:
    stdin_fd, stdout_fd, stderr_fd = fds
    os.dup2(stdin_fd, 0)
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    for fd in fds:
        os.close(fd)

    uci.protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdin = os.fdopen(0, "r")

    # The parent ran agent.warm_up() to compile, which leaves entries in the transposition
    # table and the history and killer tables. A game starts from none of that.
    uci.board = chess.Board()
    uci.move_overhead_ms = uci.DEFAULT_MOVE_OVERHEAD_MS
    uci.new_game()

    # The socket cuts both ways. The client blocks on it, so lichess-bot sees the engine exit
    # when and only when this child does - and this watches the other direction, so a client
    # that dies without closing the game's stdin cannot leave a child behind holding a core
    # and half a gigabyte. Reading it returns empty the moment the client is gone.
    def exit_with_client() -> None:
        with contextlib.suppress(OSError):
            conn.recv(1)
        os._exit(0)

    threading.Thread(target=exit_with_client, daemon=True).start()

    try:
        uci.main()
    finally:
        conn.close()


def main() -> None:
    path = os.environ.get("NNUE_WARM_SOCKET", DEFAULT_SOCKET)

    # Children are never waited on, so let the kernel reap them.
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    if os.path.exists(path):
        os.unlink(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o600)
    server.listen(8)
    print(f"warm: engine compiled, listening on {path}", file=sys.stderr, flush=True)

    while True:
        conn, _ = server.accept()
        try:
            _, fds, _, _ = socket.recv_fds(conn, 16, FD_COUNT)
        except OSError as exc:
            print(f"warm: bad handover ({exc!r})", file=sys.stderr, flush=True)
            conn.close()
            continue

        if len(fds) != FD_COUNT:
            for fd in fds:
                os.close(fd)
            conn.close()
            continue

        if os.fork() == 0:
            server.close()
            serve_game(fds, conn)
            os._exit(0)

        # The parent keeps neither: the child owns the game, and the client's socket must not
        # be held open here or it would never see the game end.
        for fd in fds:
            os.close(fd)
        conn.close()


if __name__ == "__main__":
    main()
