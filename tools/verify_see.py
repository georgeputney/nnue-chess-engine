"""Checks the jitted static exchange evaluation against the plain-Python one it mirrors: for
every capture in every position of some random games, agent.see must equal reference.see.

see_ge is checked two ways: the jitted and plain ports must return the same bool at every
threshold, and at threshold 0 - the only one the engine actually uses - see_ge must equal
(see >= 0). It is the Stockfish running-balance form, which is deliberately not identical to
(see >= t) at arbitrary t (the defender is assumed to keep recapturing), so we do not hold it
to that.

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

                # the two ports must agree at every threshold; at 0 - all the engine uses -
                # see_ge must also equal (see >= 0)
                for t in (want - 1, want, want + 1, -900, -100, 0, 100, 900):
                    a_ge = agent.see_ge(pos, packed, t)
                    r_ge = reference.see_ge(board, mv, t)
                    if a_ge != r_ge:
                        mismatches += 1
                        print(f"SEE_GE PORTS {mv.uci()} t={t}  agent={a_ge} reference={r_ge}"
                              f"\n  {board.fen()}")
                    if t == 0 and a_ge != (want >= 0):
                        mismatches += 1
                        print(f"SEE_GE@0 {mv.uci()}  see_ge={a_ge} (see={want})\n  {board.fen()}")

            board.push(random.choice(moves))

    print(f"checked {checked} captures, {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
