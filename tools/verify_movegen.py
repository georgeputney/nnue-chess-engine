"""Checks the jitted move generator against the plain-Python one it's meant to replace - the
exact legal move set at each position, not just perft's leaf counts (two different bugs could
still produce the same node count by coincidence; comparing the actual moves can't miss that).

Walks random games from the starting position using python-chess for the actual move choice (so
the positions visited are realistic and varied) and compares legal_moves_reference against
legal_moves at every ply.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from board import from_chess_board  # noqa: E402
from movegen import legal_moves, legal_moves_reference  # noqa: E402

GAMES = 60
PLIES = 60


def main() -> None:
    random.seed(5)
    positions_checked = 0
    mismatches = 0

    for _game in range(GAMES):
        board = chess.Board()
        for _ply in range(PLIES):
            moves = list(board.legal_moves)
            if not moves:
                break

            pos = from_chess_board(board)
            plain = sorted(legal_moves_reference(pos))
            jitted_arr, jitted_count = legal_moves(pos)
            jitted = sorted(int(jitted_arr[i]) for i in range(jitted_count))

            positions_checked += 1
            if plain != jitted:
                mismatches += 1
                only_plain = sorted(set(plain) - set(jitted))
                only_jitted = sorted(set(jitted) - set(plain))
                print(f"MISMATCH fen={board.fen()}")
                print(f"  only in plain:  {only_plain}")
                print(f"  only in jitted: {only_jitted}")

            board.push(random.choice(moves))

    print(f"checked {positions_checked} positions, {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
