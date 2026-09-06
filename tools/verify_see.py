"""Checks the jitted static exchange evaluation against the plain-Python one it mirrors: for
every capture in every position of some random games, agent.see must equal reference.see.

Same idea as verify_movegen - walk varied positions with python-chess, then compare the two
implementations move for move. SEE is pure arithmetic over the attack tables, so an exact match
is the bar; any divergence is a bug in one side.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

import agent  # noqa: E402
import reference  # noqa: E402
from board import from_chess_board  # noqa: E402
from move import PROMOTION_NONE, move_from_square, move_promotion_raw, move_to_square  # noqa: E402
from movegen import legal_moves  # noqa: E402

GAMES = 80
PLIES = 70
PROMO = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


def _packed(pos: object, mv: chess.Move) -> int | None:
    """The bitboard layer's packed-int move matching a python-chess Move, or None."""
    encoded, count = legal_moves(pos)
    want_promo = PROMO[mv.promotion] if mv.promotion else PROMOTION_NONE
    for i in range(count):
        m = int(encoded[i])
        if (move_from_square(m) == mv.from_square
                and move_to_square(m) == mv.to_square
                and move_promotion_raw(m) == want_promo):
            return m
    return None


def main() -> None:
    random.seed(11)
    checked = 0
    mismatches = 0

    for _game in range(GAMES):
        board = chess.Board()
        for _ply in range(PLIES):
            moves = list(board.legal_moves)
            if not moves:
                break

            pos = from_chess_board(board)
            for mv in moves:
                if not board.is_capture(mv):
                    continue
                packed = _packed(pos, mv)
                if packed is None:
                    print(f"NO PACKED MOVE for {mv.uci()} in {board.fen()}")
                    mismatches += 1
                    continue
                got = agent.see(pos, packed)
                want = reference.see(board, mv)
                checked += 1
                if got != want:
                    mismatches += 1
                    print(f"MISMATCH {mv.uci()}  agent={got} reference={want}\n  {board.fen()}")

            board.push(random.choice(moves))

    print(f"checked {checked} captures, {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
