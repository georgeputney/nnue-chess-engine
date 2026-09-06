"""Build a training set of positions labelled by Stockfish.

Stockfish is a labeller only. Nothing from it ships in the submission, which
is what the rules permit: training data is unrestricted, including positions
annotated by an existing engine, and the ban covers only what goes in the zip.

Each worker generates its own positions and labels them, so nothing is
serialised through the parent. Features are stored bit-packed at 96 bytes per
position rather than 768, which is the difference between a 100 MB file and an
800 MB one at a million positions.

Output .npz:
    packed  uint8 [N, 96]  np.packbits of the 768-bit feature vector
    stm     uint8 [N]      1 if white to move
    cp      int16 [N]      Stockfish score, side-to-move view, centipawns
    wdl     float32 [N]    win probability for the side to move

Unpack at training time with np.unpackbits(packed, axis=1). tools/tune.py fits
its evaluation against `cp` directly (a Huber loss), so keep `--max-cp` wide
enough that a clean extra queen or rook still lands inside it; `wdl` is only
used by that tuner's older `--target wdl` path.

Usage:
    python label.py --games 2000 --workers 8 --out data/smoke.npz

Stockfish is a separate binary rather than a Python package, so it has to be
on PATH (brew install stockfish) or given with --engine.
    python label.py --games 150000 --samples 8 --workers 10 --out data/train.npz
"""

import argparse
import os
import random
import shutil
import struct
import sys
import time
from multiprocessing import Pool

import chess
import chess.engine
import numpy as np

# Centipawns to win probability. 1/400 is the usual logistic scale.
K = 1.0 / 400.0


def features(board: chess.Board) -> np.ndarray:
    """768 bits: six white piece types then six black, from white's view."""
    white, black = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    packed = struct.pack(
        "<12Q",
        board.pawns & white, board.knights & white, board.bishops & white,
        board.rooks & white, board.queens & white, board.kings & white,
        board.pawns & black, board.knights & black, board.bishops & black,
        board.rooks & black, board.queens & black, board.kings & black,
    )
    return np.frombuffer(packed, np.uint8).copy()  # already 96 packed bytes


def random_game_positions(
    rng: random.Random, plies_min: int = 8, plies_max: int = 160, samples: int = 6
) -> list[str]:
    """Weighted-random self play, keeping a few quiet positions per game.

    plies_max is deliberately long: the evaluation's endgame tables need
    endgame positions to fit against, and weighted-random play only sheds
    enough material to reach them after a hundred-odd plies.
    """
    board = chess.Board()
    kept = []
    target = rng.randint(plies_min, plies_max)
    for _ in range(target):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        weights = []
        for move in moves:
            if board.is_capture(move):
                weights.append(1.6)
            elif move.promotion:
                weights.append(2.0)
            else:
                weights.append(1.0)
        board.push(rng.choices(moves, weights)[0])
        # Positions in check are dominated by tactics the search finds anyway.
        if not board.is_check() and len(board.move_stack) >= plies_min:
            kept.append(board.fen())
    if len(kept) > samples:
        kept = rng.sample(kept, samples)
    return kept


Task = tuple[int, int, str, int, int | None, int, int]


def worker(task: Task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate and label one worker's share. Returns packed arrays."""
    games, seed, engine_path, depth, nodes, samples, max_cp = task
    rng = random.Random(seed)

    rows_packed, rows_stm, rows_cp = [], [], []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        limit = (chess.engine.Limit(depth=depth) if nodes is None
                 else chess.engine.Limit(nodes=nodes))
        seen = set()
        for _ in range(games):
            for fen in random_game_positions(rng, samples=samples):
                key = fen.rsplit(" ", 2)[0]  # ignore clocks when de-duplicating
                if key in seen:
                    continue
                seen.add(key)
                board = chess.Board(fen)
                try:
                    info = engine.analyse(board, limit)
                except chess.engine.EngineError:
                    continue
                score = info["score"].relative
                if score.is_mate():
                    continue  # decided; nothing for a static eval to learn

                # Quiet positions only. A static evaluation is being trained to
                # reproduce a depth-N search score, so any position whose score
                # depends on a tactic teaches it the wrong thing: the search
                # sees the hanging queen collected, the static eval never can.
                pv = info.get("pv")
                if pv and (board.is_capture(pv[0]) or board.gives_check(pv[0])):
                    continue

                cp = score.score()
                if cp is None or abs(cp) > max_cp:
                    continue
                rows_packed.append(features(board))
                rows_stm.append(1 if board.turn else 0)
                rows_cp.append(cp)

    if not rows_cp:
        return (np.zeros((0, 96), np.uint8), np.zeros(0, np.uint8),
                np.zeros(0, np.int16))
    return (np.stack(rows_packed), np.array(rows_stm, np.uint8),
            np.array(rows_cp, np.int16))


def find_engine() -> str | None:
    """Locate the Stockfish binary.

    It is a C++ executable, not a Python package, so a venv never provides it.
    PATH covers Homebrew; the extra candidates cover Debian, which installs to
    /usr/games and leaves that off PATH in most shells.
    """
    found = shutil.which("stockfish")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish",
                      "/usr/games/stockfish", "/usr/bin/stockfish"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=None,
                        help="use a node limit instead of a depth limit")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--max-cp", type=int, default=3000,
                        help="drop positions scored beyond this; keep it above a queen")
    parser.add_argument("--engine", default=None,
                        help="path to the Stockfish binary; found on PATH if omitted")
    parser.add_argument("--out", default="data/train.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    engine_path = args.engine or find_engine()
    if not engine_path:
        sys.exit(
            "Stockfish not found on PATH. It is a C++ binary, not a Python "
            "package, so the venv does not provide it.\n"
            "  macOS:  brew install stockfish\n"
            "  Debian: sudo apt install stockfish\n"
            "Then re-run, or pass --engine /path/to/stockfish."
        )

    started = time.time()
    share = max(1, args.games // args.workers)
    tasks = [(share, args.seed * 10_000 + i, engine_path, args.depth,
              args.nodes, args.samples, args.max_cp)
             for i in range(args.workers)]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(worker, tasks)
    else:
        results = [worker(tasks[0])]

    packed = np.concatenate([r[0] for r in results])
    stm = np.concatenate([r[1] for r in results])
    cps = np.concatenate([r[2] for r in results])

    # Positions repeat across workers, mostly early ones.
    _, unique = np.unique(packed, axis=0, return_index=True)
    packed, stm, cps = packed[unique], stm[unique], cps[unique]

    # Train against win probability. Centipawns are effectively unbounded and
    # a squared error on them spends capacity on already-decided positions.
    wdl = (1.0 / (1.0 + np.exp(-K * cps.astype(np.float32)))).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, packed=packed, stm=stm, cp=cps, wdl=wdl)

    elapsed = time.time() - started
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  {len(cps):,} positions  {size:.1f} MB  "
          f"in {elapsed / 60:.1f} min  ({len(cps) / elapsed:.0f} pos/s)")
    print(f"cp: median {np.median(cps):.0f}  sd {cps.std():.0f}  "
          f"5-95 pct {np.percentile(cps, 5):.0f} to {np.percentile(cps, 95):.0f}")
    print(f"wdl: mean {wdl.mean():.3f}  sd {wdl.std():.3f}")


if __name__ == "__main__":
    main()