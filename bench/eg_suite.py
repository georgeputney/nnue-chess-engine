"""Endgame suite - where the NNUE static eval and the search fail, by piece count.

The openings A/B and bench/endgame_bench.py (8-12 man rook endings) cannot see it: the net
rates K+R vs K at +57 cp. This generates endgame positions across material templates, labels
each with Stockfish (best move + eval), and reports, per piece band:

  - static error: |our evaluate() - Stockfish|  (the flat-eval problem, measured directly)
  - move cp-loss: Stockfish's eval of its own best move minus its eval of the move our engine
    picks at a fixed depth  (the play-quality problem)

    uv run python bench/eg_suite.py --positions 240 --sf-depth 18 --engine-depth 10
    uv run python bench/eg_suite.py --module engine.agent --positions 400

Judge net / search changes by the per-band numbers here; keep it as a veto (a regression in a
band rejects the change even if the openings A/B likes it).
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.label import find_engine  # noqa: E402

# material templates: (white non-king pieces, black non-king pieces). Pawns are placed on
# ranks 2-7. A spread from trivially won (KRvK) through theoretical draws (KRPvKR) to the
# same-wing majorities the middlegame net should still get roughly right.
TEMPLATES: list[tuple[str, str]] = [
    ("R", ""), ("Q", ""), ("RP", ""), ("QP", ""), ("BN", ""), ("NN", "P"),
    ("R", "R"), ("RP", "R"), ("RPP", "RP"), ("RPPP", "RPP"),
    ("Q", "R"), ("Q", "RP"), ("QP", "Q"), ("R", "BP"), ("RB", "R"),
    ("P", ""), ("PP", "P"), ("PPP", "PPP"),
    ("BPP", "B"), ("BPP", "N"),
]
PIECE = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN}
BANDS = ((2, 4), (5, 6), (7, 9), (10, 12), (13, 16), (17, 32))  # 2-4 = the Syzygy-covered slice
MATE_CP = 20000


def band(n: int) -> tuple[int, int]:
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return lo, hi
    return BANDS[-1]


def score_cp(score: chess.engine.PovScore, pov: chess.Color) -> int:
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate()
        return (MATE_CP - abs(m)) * (1 if m > 0 else -1)
    return max(-MATE_CP, min(MATE_CP, s.score()))


# one random legal position for a template, side to move random, not in check, no piece hanging
# to a capture on move 1 (so the static eval is meaningful). None if placement kept failing.
def make_position(white: str, black: str, rng: random.Random) -> chess.Board | None:
    for _ in range(200):
        board = chess.Board(None)
        squares = rng.sample(range(64), 2 + len(white) + len(black))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        i = 2
        ok = True
        for spec, colour in ((white, chess.WHITE), (black, chess.BLACK)):
            for ch in spec:
                sq = squares[i]
                i += 1
                if ch == "P" and chess.square_rank(sq) in (0, 7):
                    ok = False
                    break
                board.set_piece_at(sq, chess.Piece(PIECE[ch], colour))
            if not ok:
                break
        if not ok:
            continue
        board.turn = rng.choice((chess.WHITE, chess.BLACK))
        if not board.is_valid() or board.is_check() or board.is_game_over():
            continue
        if any(board.is_capture(m) for m in board.legal_moves):
            continue
        return board
    return None


# generate `n` SF-labelled positions (fen, pov, best_cp, best_uci) and cache them to `path` as
# JSON. Run once; every --suite score then hits the identical set, so an A/B on two nets is a
# clean delta instead of two different random samples.
def build_suite(path: Path, n: int, seed: int, sf_depth: int, engine_path: str) -> None:
    import json
    rng = random.Random(seed)
    out: list[dict] = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as sf:
        sf.configure({"Threads": 1, "Hash": 128})
        limit = chess.engine.Limit(depth=sf_depth)
        while len(out) < n:
            board = make_position(*TEMPLATES[rng.randrange(len(TEMPLATES))], rng)
            if board is None:
                continue
            pov = board.turn
            info = sf.analyse(board, limit)
            best_cp = score_cp(info["score"], pov)
            if abs(best_cp) >= MATE_CP - 200:
                continue
            pv = info.get("pv")
            out.append({"fen": board.fen(), "pov": int(pov), "best_cp": best_cp,
                        "best_uci": pv[0].uci() if pv else None})
            if len(out) % 40 == 0:
                print(f"  built {len(out)}/{n}")
    path.write_text(json.dumps(out))
    print(f"wrote {path}  ({len(out)} positions, SF depth {sf_depth})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="engine.agent", help="engine module under test")
    parser.add_argument("--positions", type=int, default=240)
    parser.add_argument("--sf-depth", type=int, default=18, help="Stockfish label depth")
    parser.add_argument("--engine-depth", type=int, default=10, help="our fixed search depth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build", type=Path, help="generate + SF-label positions to this JSON, "
                        "then exit (a fixed suite for reproducible A/B)")
    parser.add_argument("--suite", type=Path, help="score against a cached suite from --build "
                        "instead of generating fresh positions")
    parser.add_argument("--net", help="score against this net.npz instead of engine/net.npz")
    parser.add_argument("--tb", action="store_true",
                        help="take the move from engine.tablebase.best_tb_move when it answers "
                        "(the 2-4 band); measures the Syzygy probe's move quality")
    args = parser.parse_args()

    engine_path = find_engine()
    if not engine_path:
        sys.exit("Stockfish not found - brew install stockfish")

    if args.build:
        build_suite(args.build, args.positions, args.seed, args.sf_depth, engine_path)
        return

    if args.net:
        os.environ["NNUE_NET"] = str(Path(args.net).resolve())  # engine.net reads this at import
        # the njit funcs in engine/accumulator.py freeze the net weights into their cache=True
        # .nbc; point numba at a throwaway dir so this candidate net compiles fresh without
        # clobbering (or reusing) the shipped engine/__pycache__.
        import tempfile
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="egsuite_nb_")
    agent = importlib.import_module(args.module)
    best_tb_move = None
    if args.tb:
        from engine.tablebase import TB_MEN, best_tb_move
        print(f"tablebase probe on: {Path(__file__).parent.parent / 'engine' / 'syzygy'} "
              f"(<= {TB_MEN} men)")

    rng = random.Random(args.seed)
    rows: list[tuple[int, int, int, int]] = []  # (pieces, matbal, static_err, cp_loss)

    cached = None
    if args.suite:
        import json
        cached = iter(json.loads(args.suite.read_text()))

    with chess.engine.SimpleEngine.popen_uci(engine_path) as sf:
        sf.configure({"Threads": 1, "Hash": 128})
        limit = chess.engine.Limit(depth=args.sf_depth)
        made = 0
        target = args.positions
        while made < target:
            if cached is not None:
                entry = next(cached, None)
                if entry is None:
                    break
                board = chess.Board(entry["fen"])
                pov = bool(entry["pov"])
                best_cp = entry["best_cp"]
            else:
                board = make_position(*TEMPLATES[rng.randrange(len(TEMPLATES))], rng)
                if board is None:
                    continue
                pov = board.turn
                best_cp = score_cp(sf.analyse(board, limit)["score"], pov)
                if abs(best_cp) >= MATE_CP - 200:  # already mate-in-a-few: nothing to learn
                    continue

            our_eval = int(agent.evaluate(agent.parse_fen(board.fen())))
            static_err = abs(our_eval - best_cp)

            our_uci = None
            if best_tb_move is not None:
                our_uci = best_tb_move(board.fen())
            if our_uci is None:
                our_uci = agent.bench_search(board.fen(), args.engine_depth)[0]
            child = board.copy()
            try:
                child.push(chess.Move.from_uci(our_uci))
            except ValueError:
                child = None
            if child is None:
                cp_loss = MATE_CP
            elif child.is_game_over():
                res = child.result()
                got = MATE_CP if res == ("1-0" if pov == chess.WHITE else "0-1") else (
                    -MATE_CP if res != "1/2-1/2" else 0)
                cp_loss = max(0, best_cp - got)
            else:
                after = sf.analyse(child, chess.engine.Limit(depth=args.sf_depth - 4))
                got = -score_cp(after["score"], not pov)
                cp_loss = max(0, best_cp - got)

            pieces = chess.popcount(board.occupied)
            matbal = sum(v * (len(board.pieces(p, pov)) - len(board.pieces(p, not pov)))
                         for p, v in ((1, 1), (2, 3), (3, 3), (4, 5), (5, 9)))
            rows.append((pieces, matbal, static_err, min(cp_loss, MATE_CP)))
            made += 1
            if made % 40 == 0:
                print(f"  {made}/{target}")

    print(f"\n{args.module}  (SF depth {args.sf_depth}, our depth {args.engine_depth}, "
          f"{len(rows)} positions)\n")
    print(f"{'piece band':>12} {'n':>4} {'mean static err':>16} {'median':>8} "
          f"{'mean cp-loss':>13} {'blunders>200':>13}")
    for lo, hi in BANDS:
        sub = [r for r in rows if lo <= r[0] <= hi]
        if not sub:
            continue
        se = sorted(r[2] for r in sub)
        cl = [r[3] for r in sub]
        print(f"{f'{lo}-{hi}':>12} {len(sub):>4} {sum(se) / len(se):>16.0f} "
              f"{se[len(se) // 2]:>8.0f} {sum(cl) / len(cl):>13.0f} "
              f"{sum(1 for c in cl if c > 200) / len(cl):>12.0%}")
    won = [r for r in rows if r[1] >= 4]
    if won:
        se = [r[2] for r in won]
        print(f"\n  positions a clean piece+ ahead ({len(won)}): mean static err "
              f"{sum(se) / len(se):.0f} cp  (a healthy eval reads these near the material value)")


if __name__ == "__main__":
    main()
