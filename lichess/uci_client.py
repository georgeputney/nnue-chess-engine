#!/usr/bin/env python3
"""
What lichess-bot starts for a game: a few milliseconds instead of half a minute.

This hands its own stdin, stdout and stderr to the pre-warmed engine in lichess/warm.py, which
forks a child already holding the compiled search, and then waits. The child does the talking
straight down these descriptors, so nothing is relayed through here and this process exists
only to be something for lichess-bot to start and stop.

If the daemon is not running, this becomes uci.py - the engine in this process, compiling on
import, exactly as before. That keeps a laptop, a UCI GUI and `make lichess-check` working
with nothing to set up, and means a dead daemon costs a slow first game rather than no game.
"""

import contextlib
import os
import socket
import sys
from pathlib import Path

DEFAULT_SOCKET = "/run/nnue/warm.sock"


# The engine in this process. execv rather than an import, so lichess-bot's pipes and its
# signals land on a plain uci.py with nothing of this wrapper left in the way.
def run_in_process() -> None:
    uci = Path(__file__).resolve().parent / "uci.py"
    os.execv(sys.executable, [sys.executable, str(uci)])


def main() -> None:
    path = os.environ.get("NNUE_WARM_SOCKET", DEFAULT_SOCKET)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
        socket.send_fds(sock, [b"game"], [0, 1, 2])
    except OSError as exc:
        print(f"uci_client: no warm engine at {path} ({exc!r}); compiling in-process",
              file=sys.stderr, flush=True)
        run_in_process()
        return

    # The child holds the other end for the life of the game, so this returns empty exactly
    # when the game is over. Closing our own descriptors first would pull them out from under
    # the child - it is using these, not copies.
    with contextlib.suppress(OSError, KeyboardInterrupt):
        sock.recv(1)


if __name__ == "__main__":
    main()
