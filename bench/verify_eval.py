"""Checks agent.evaluate (numba, on a Board) against reference.evaluate (plain, on a
chess.Board) - bit-exact, across many positions from random games off several seeds.

Only the classical tapered eval has a plain-python twin (reference.py); the shipped NNUE
engine's equivalent check is bench/verify_nnue.py's oracle comparison against engine/net.py.
So this checks the frozen classical build in stages/07-numba-classical, the numba port's own
claim to bit-exactness against the reference.py it was ported from.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSICAL = ROOT / "stages" / "07-numba-classical"
if str(CLASSICAL) not in sys.path:
    sys.path.insert(0, str(CLASSICAL))

import agent  # noqa: E402
import chess  # noqa: E402
import reference  # noqa: E402
from board import from_chess_board  # noqa: E402

SEEDS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r1bqk2r/pp1n1ppp/2n1p3/2bpP3/5P2/2NB4/PPP3PP/R1BQK1NR w KQkq - 0 8",
]
GAMES_PER_SEED = 25
PLIES = 80


def main() -> None:
    random.seed(13)
    checked = 0
    mismatches = 0

    for seed in SEEDS:
        for _game in range(GAMES_PER_SEED):
            board = chess.Board(seed)
            for _ply in range(PLIES):
                plain = reference.evaluate(board, board.turn)
                bb = agent.evaluate(from_chess_board(board))
                checked += 1
                if plain != bb:
                    mismatches += 1
                    print(f"MISMATCH fen={board.fen()} plain={plain} bb={bb}")
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))

    print(f"checked {checked}, {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
