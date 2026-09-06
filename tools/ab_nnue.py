"""In-process A/B: the NNUE engine (nnue/agent.py) vs the current linear engine (agent.py),
both colours from a set of near-level openings on a simulated clock. Prints score, Elo and a
95% band.

Same design and caveats as tools/ab.py - shared interpreter, JIT warms once, persistent state
(transposition table, history, repetition dicts) cleared between games. Not a substitute for
tools/bench.py's fresh-subprocess-per-game isolation; a quick read on whether the net is worth
its slower nodes.

    uv run python tools/ab_nnue.py [--openings N] [--base-ms MS] [--inc-ms MS]
    uv run python tools/ab_nnue.py --openings N --depth D    # fixed depth, no clock

--depth runs both engines to the same fixed search depth per move (via bench_search, clean
table each move). It is deterministic and timing-free, so it isolates one question: is the
NNUE eval stronger per node? The clocked mode is what actually matters for the competition,
where the net's slower nodes cost depth; use --depth to tell "eval too weak" from "eval fine,
just too slow".
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

import agent as linear  # noqa: E402
import nnue.agent as nnue  # noqa: E402
from tools.bench import OPENINGS  # noqa: E402

PLY_CAP = 600


def reset_engine(engine: object) -> None:
    engine.STATE.tt_depth[:] = -1
    engine.STATE.history[:] = 0
    engine.STATE.killers[:] = engine.NO_MOVE
    engine.STATE.avoid = engine.NO_MOVE
    if hasattr(engine.STATE, "tt_eval"):  # nnue engine caches the static eval in the TT
        engine.STATE.tt_eval[:] = engine.NO_EVAL
    engine.SEEN.clear()
    engine.PLAYED.clear()


def play(
    fen: str, nnue_white: bool, base_ms: int, inc_ms: int, depth: int
) -> tuple[float, str]:
    """Returns (points for the NNUE engine, termination). depth > 0 ignores the clock and runs
    both engines to that fixed search depth per move."""
    board = chess.Board(fen)
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    reset_engine(linear)
    reset_engine(nnue)

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
        use_nnue = (stm == chess.WHITE) == nnue_white
        engine = nnue if use_nnue else linear

        if depth > 0:
            uci = engine.bench_search(board.fen(), depth)[0]
        else:
            t0 = time.monotonic()
            uci = engine.get_move(board.fen(), int(clock[stm]))
            clock[stm] -= (time.monotonic() - t0) * 1000.0
            if clock[stm] < 0:
                if board.has_insufficient_material(not stm):
                    return 0.5, "flag (drawn)"
                return (1.0 if use_nnue else 0.0), "flag (linear)" if use_nnue else "flag (nnue)"
            clock[stm] += inc_ms

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            move = None
        if move is None or move not in board.legal_moves:
            return (1.0 if use_nnue else 0.0), "illegal (linear)" if use_nnue else "illegal (nnue)"

        board.push(move)

    nnue_points = result if nnue_white else 1.0 - result
    return nnue_points, term


def elo(score: float) -> float:
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
    ap.add_argument("--depth", type=int, default=0, help="fixed depth per move, ignores the clock")
    args = ap.parse_args()

    fens = OPENINGS[: args.openings]
    points: list[float] = []
    terms: dict[str, int] = {}
    control = f"fixed depth {args.depth}" if args.depth else f"{args.base_ms}ms + {args.inc_ms}ms"
    print(f"{len(fens) * 2} games, {control}, nnue vs linear\n")

    for i, fen in enumerate(fens, 1):
        for nnue_white in (True, False):
            t0 = time.monotonic()
            pts, term = play(fen, nnue_white, args.base_ms, args.inc_ms, args.depth)
            points.append(pts)
            terms[term] = terms.get(term, 0) + 1
            tag = {1.0: "win ", 0.5: "draw", 0.0: "loss"}[pts]
            side = "W" if nnue_white else "B"
            dt = time.monotonic() - t0
            print(f"[{len(points):3d}] opening {i:2d} nnue-{side}  {tag}  {term:24s} {dt:5.1f}s")

    n = len(points)
    score = sum(points) / n
    wins = points.count(1.0)
    draws = points.count(0.5)
    losses = points.count(0.0)
    var = sum((p - score) ** 2 for p in points) / max(1, n - 1)
    stderr = math.sqrt(var / n)
    lo = elo(max(0.001, score - 1.96 * stderr))
    hi = elo(min(0.999, score + 1.96 * stderr))

    print("\nnnue vs linear")
    print(f"+{wins} ={draws} -{losses}  score {score * 100:.1f}%")
    print(f"elo {elo(score):+.0f}  95% ci [{lo:+.0f}, {hi:+.0f}]")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terms.items())))


if __name__ == "__main__":
    main()
