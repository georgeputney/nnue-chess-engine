"""Perft: the standard way a move generator proves itself. Count leaf positions at a fixed
depth from a known position and compare against published reference counts - if they match, the
generator produced neither too many moves (illegal ones slipping through) nor too few (legal
ones missed) at every ply up to that depth, not just the top one.

Reference counts are the standard, widely-published perft results used to test virtually every
chess engine (see https://www.chessprogramming.org/Perft_Results) - not specific to any one
engine's implementation.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.board import Board, parse_fen  # noqa: E402
from engine.movegen import (  # noqa: E402
    legal_moves,
    legal_moves_reference,
    make_move,
    make_move_reference,
)

# (name, fen, [perft(1), perft(2), ...]) - depth 1 first.
POSITIONS: list[tuple[str, str, list[int]]] = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        [20, 400, 8902, 197281, 4865609],
    ),
    (
        "kiwipete",  # heavy on captures, castling, promotions, en passant
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [48, 2039, 97862, 4085603],
    ),
    (
        "position 3",  # sparse, pawn-endgame-heavy
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        [14, 191, 2812, 43238, 674624],
    ),
    (
        "position 4",  # castling rights, promotions, pins
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [6, 264, 9467, 422333],
    ),
    (
        "position 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [44, 1486, 62379, 2103487],
    ),
]


def perft(pos: Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in legal_moves_reference(pos):
        nodes += perft(make_move_reference(pos, move), depth - 1)
    return nodes


def perft_nb(pos: Board, depth: int) -> int:
    """Same as perft(), but calling the jitted movegen / make_move. The recursion here is
    still plain Python - only legal_moves and make_move themselves are compiled -
    which is already enough to see the win, since that's where nearly all the work is."""
    if depth == 0:
        return 1
    moves, count = legal_moves(pos)
    if depth == 1:
        return count
    nodes = 0
    for i in range(count):
        child: Board = make_move(pos, int(moves[i]))  # type: ignore[assignment, type-var, call-arg]
        nodes += perft_nb(child, depth - 1)
    return nodes


def main() -> None:
    args = sys.argv[1:]
    use_jit = "--jit" in args
    args = [a for a in args if a != "--jit"]
    max_depth = int(args[0]) if args else 3
    run = perft_nb if use_jit else perft
    all_ok = True

    for name, fen, expected in POSITIONS:
        pos = parse_fen(fen)
        for depth, want in enumerate(expected[:max_depth], start=1):
            t0 = time.monotonic()
            got = run(pos, depth)
            elapsed = time.monotonic() - t0
            ok = got == want
            all_ok &= ok
            status = "ok" if ok else "MISMATCH"
            nps = f"{got / elapsed:,.0f} nps" if elapsed > 0 else "n/a"
            print(
                f"{name:12} depth {depth}  got {got:>9,}  want {want:>9,}  "
                f"{status:8} ({elapsed:.2f}s, {nps})"
            )
            if not ok:
                break  # deeper counts from a wrong node are meaningless

    print("\nPASS" if all_ok else "\nFAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
