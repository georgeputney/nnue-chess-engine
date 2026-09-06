"""In-process A/B: the numba agent vs the plain-python-chess reference engine, both colours
from a set of near-level openings on a simulated clock. Prints score, Elo and a 95% band.

Not a substitute for tools/bench.py's fresh-subprocess-per-game isolation - both engines share
this interpreter - but it needs no wiring and is fast (the JIT warms once). Persistent state
(transposition table, history, repetition dicts) is cleared between games.

    uv run python tools/bb_ab.py [--openings N] [--base-ms MS] [--inc-ms MS]
"""

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

import agent  # noqa: E402
import reference  # noqa: E402
from tools.bench import OPENINGS  # noqa: E402

PLY_CAP = 600


def _reset() -> None:
    agent.STATE.tt_depth[:] = -1
    agent.STATE.history[:] = 0
    agent.STATE.killers[:] = agent.NO_MOVE
    agent.STATE.avoid = agent.NO_MOVE
    agent.SEEN.clear()
    agent.PLAYED.clear()
    reference.TT.clear()
    reference.KILLERS[:] = [[None, None] for _ in range(len(reference.KILLERS))]
    for _side in reference.HISTORY:
        for _row in _side:
            _row[:] = [0] * 64
    reference.SEEN.clear()
    reference.PLAYED.clear()


def _play(fen: str, bb_white: bool, base_ms: int, inc_ms: int) -> tuple[float, str]:
    """Returns (points for bb, termination)."""
    board = chess.Board(fen)
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    _reset()

    while True:
        outcome = board.outcome()
        if outcome is not None:
            result = 0.5 if outcome.winner is None else float(outcome.winner == chess.WHITE)
            term = outcome.termination.name.lower()
            break
        if board.is_repetition(3):
            result, term = 0.5, "threefold_repetition"
            break
        if board.is_fifty_moves():
            result, term = 0.5, "fifty_moves"
            break
        if board.ply() >= PLY_CAP:
            result, term = 0.5, "ply_cap"
            break

        stm = board.turn
        use_bb = (stm == chess.WHITE) == bb_white
        mover = agent.get_move if use_bb else reference.get_move

        t0 = time.monotonic()
        uci = mover(board.fen(), int(clock[stm]))
        clock[stm] -= (time.monotonic() - t0) * 1000.0

        if clock[stm] < 0:
            # a flag against a side that cannot mate is a draw, not a loss
            if board.has_insufficient_material(not stm):
                return 0.5, "flag (drawn)"
            return (1.0 if use_bb else 0.0), "flag (opponent)" if use_bb else "flag (bb)"
        clock[stm] += inc_ms

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            move = None
        if move is None or move not in board.legal_moves:
            return (1.0 if use_bb else 0.0), "illegal (opp)" if use_bb else "illegal (bb)"

        board.push(move)

    bb_points = result if bb_white else 1.0 - result
    return bb_points, term


def _elo(score: float) -> float:
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openings", type=int, default=20)
    ap.add_argument("--base-ms", type=int, default=8000)
    ap.add_argument("--inc-ms", type=int, default=80)
    args = ap.parse_args()

    fens = OPENINGS[: args.openings]
    points: list[float] = []
    terms: dict[str, int] = {}
    print(f"{len(fens) * 2} games, {args.base_ms}ms + {args.inc_ms}ms, agent vs reference\n")

    for i, fen in enumerate(fens, 1):
        for bb_white in (True, False):
            t0 = time.monotonic()
            pts, term = _play(fen, bb_white, args.base_ms, args.inc_ms)
            points.append(pts)
            terms[term] = terms.get(term, 0) + 1
            tag = {1.0: "win ", 0.5: "draw", 0.0: "loss"}[pts]
            side = "W" if bb_white else "B"
            dt = time.monotonic() - t0
            print(f"[{len(points):3d}] opening {i:2d} bb-{side}  {tag}  {term:24s} {dt:5.1f}s")

    n = len(points)
    score = sum(points) / n
    wins = points.count(1.0)
    draws = points.count(0.5)
    losses = points.count(0.0)
    var = sum((p - score) ** 2 for p in points) / max(1, n - 1)
    stderr = math.sqrt(var / n)
    lo = _elo(max(0.001, score - 1.96 * stderr))
    hi = _elo(min(0.999, score + 1.96 * stderr))

    print("\nagent vs reference")
    print(f"+{wins} ={draws} -{losses}  score {score * 100:.1f}%")
    print(f"elo {_elo(score):+.0f}  95% ci [{lo:+.0f}, {hi:+.0f}]")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terms.items())))


if __name__ == "__main__":
    main()
