"""Label a set of endgame positions with Stockfish, in tools/label.py's .npz schema, to mix
into NNUE training. The lichess-eval corpus the net trained on is almost all middlegame and
its decisive scores are clamped, so the net rates K+R vs K at +57 cp - it never learned that a
won endgame is won. tools/eg_suite.py measures the gap; this closes it.

Unlike tools/label.py this keeps decisive positions (a mate is mapped to +/- mate_cp, not
dropped) and does not require the Stockfish PV to be quiet in the check sense - a checking move
in a K+R vs K conversion is technique, not a tactic the static eval cannot represent. It still
drops positions with a hanging piece (pv[0] a capture) and forced mates in <= 3 (too sharp to
teach as "this endgame is won").

    uv run python tools/label_eg.py --positions 3_000_000 --workers 10 --out data/endgame.npz

Then mix into training - either drop it in the shard dir as another shard, or oversample it to
a target fraction (see tools/train_nn.py). Re-run tools/eg_suite.py after: the <= 9-piece band's
static error should fall from ~250 cp toward < 100 without regressing the 13-16 band.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eg_suite import make_position  # noqa: E402
from tools.label import K, features, find_engine  # noqa: E402

# (white pieces, black pieces, weight). Weighted toward decisive material (where the net is
# blind) but with enough drawn-despite-material lines that it does not just learn "count wood".
TEMPLATES: list[tuple[str, str, int]] = [
    ("R", "", 5), ("Q", "", 4), ("RP", "", 4), ("QP", "", 3), ("RR", "", 2),
    ("BN", "", 3), ("BB", "", 2), ("NN", "P", 2), ("RB", "", 2), ("RN", "", 2),
    ("QP", "R", 3), ("Q", "RR", 2), ("RP", "N", 2), ("RP", "B", 2), ("RPP", "R", 3),
    ("R", "R", 4), ("RP", "R", 4), ("RPP", "RP", 4), ("RPPP", "RPP", 3), ("RR", "RR", 2),
    ("Q", "Q", 2), ("Q", "R", 3), ("Q", "RP", 3), ("QP", "Q", 2), ("BP", "B", 2),
    ("BPP", "B", 3), ("BPP", "N", 3), ("NP", "B", 2), ("R", "BP", 2), ("R", "NP", 2),
    ("P", "", 3), ("PP", "", 3), ("PP", "P", 3), ("PPP", "PP", 3), ("PPPP", "PPP", 2),
    ("BPP", "BP", 2), ("RBP", "RB", 2), ("RNP", "RN", 2),
]
# a mate score maps here. Kept close to the material scale on purpose: Stockfish's *eval* of a
# won-but-not-mating ending (K+R vs K at depth 16) is only ~+480 cp - a rook plus technique -
# and a nearby-mate version of the same ending should not train to +5000 while its twin trains
# to +480. +2000 (wdl ~ 0.993) is "clearly won" without a bimodal target or a broken cp scale.
MATE_CP = 2000


# write this worker's rows-so-far to <out>.partNNN.npz (overwrite). Crash insurance for the
# long runs: if the pool dies, tools/label_eg.py --assemble <out> stitches the parts together.
def flush_part(out: str, wid: int, packed: list, stm: list, cp: list) -> None:
    if not cp:
        return
    tmp = f"{out}.part{wid:03d}.tmp.npz"
    np.savez_compressed(tmp, packed=np.stack(packed), stm=np.array(stm, np.uint8),
                        cp=np.array(cp, np.int16))
    os.replace(tmp, f"{out}.part{wid:03d}.npz")


def worker(task: tuple[int, int, str, int, int, str, int]) -> tuple[np.ndarray, ...]:
    count, seed, engine_path, depth, max_cp, out, ckpt_every = task
    wid = seed % 1000
    rng = random.Random(seed)
    weighted = [t for w, b, wt in TEMPLATES for t in [(w, b)] * wt]

    rows_packed: list[np.ndarray] = []
    rows_stm: list[int] = []
    rows_cp: list[int] = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        limit = chess.engine.Limit(depth=depth)
        seen: set[str] = set()
        made = 0
        attempts = 0
        while made < count and attempts < count * 60:
            attempts += 1
            white, black = weighted[rng.randrange(len(weighted))]
            board = make_position(white, black, rng)
            if board is None:
                continue
            key = board.board_fen() + (" w" if board.turn else " b")
            if key in seen:
                continue
            seen.add(key)

            try:
                info = engine.analyse(board, limit)
            except chess.engine.EngineError:
                continue
            score = info["score"].relative

            if score.is_mate():
                m = score.mate()
                if abs(m) <= 3:
                    continue  # a mate-in-3 is a tactic, not "this ending is won"
                cp = MATE_CP if m > 0 else -MATE_CP
            else:
                raw = score.score()
                if raw is None or abs(raw) > max_cp:
                    continue
                cp = raw

            pv = info.get("pv")
            if pv and board.is_capture(pv[0]):
                continue  # a hanging piece teaches the static eval the wrong thing

            rows_packed.append(features(board))
            rows_stm.append(1 if board.turn else 0)
            rows_cp.append(cp)
            made += 1
            if made % 5000 == 0:
                dec = sum(1 for c in rows_cp if abs(c) >= 800) / made
                print(f"  [w{wid:03d}] {made:,}/{count:,} ({dec:.0%} decisive)", flush=True)
            if ckpt_every and made % ckpt_every == 0:
                flush_part(out, wid, rows_packed, rows_stm, rows_cp)

    flush_part(out, wid, rows_packed, rows_stm, rows_cp)
    if not rows_cp:
        return np.zeros((0, 96), np.uint8), np.zeros(0, np.uint8), np.zeros(0, np.int16)
    return np.stack(rows_packed), np.array(rows_stm, np.uint8), np.array(rows_cp, np.int16)


def assemble(out: str) -> None:
    """Stitch <out>.partNNN.npz crash-checkpoints into <out> and delete them."""
    import glob
    parts = sorted(glob.glob(f"{out}.part*.npz"))
    if not parts:
        sys.exit(f"no {out}.part*.npz to assemble")
    packed = np.concatenate([np.load(p)["packed"] for p in parts])
    stm = np.concatenate([np.load(p)["stm"] for p in parts])
    cps = np.concatenate([np.load(p)["cp"] for p in parts])
    _, unique = np.unique(packed, axis=0, return_index=True)
    packed, stm, cps = packed[unique], stm[unique], cps[unique]
    wdl = (1.0 / (1.0 + np.exp(-K * cps.astype(np.float32)))).astype(np.float32)
    np.savez_compressed(out, packed=packed, stm=stm, cp=cps, wdl=wdl)
    for p in parts:
        os.remove(p)
    print(f"assembled {len(parts)} parts -> {out}  {len(cps):,} positions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=1_000_000)
    parser.add_argument("--sf-depth", type=int, default=18)
    parser.add_argument("--max-cp", type=int, default=6000)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--out", default="data/endgame.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--checkpoint-every", type=int, default=20_000,
                        help="each worker flushes <out>.partNNN.npz this often (0 = off)")
    parser.add_argument("--assemble", action="store_true",
                        help="stitch existing <out>.part*.npz into <out> and exit (crash recovery)")
    args = parser.parse_args()

    if args.assemble:
        assemble(args.out)
        return

    engine_path = args.engine or find_engine()
    if not engine_path:
        sys.exit("Stockfish not found - brew install stockfish, or pass --engine")

    started = time.time()
    share = max(1, args.positions // args.workers)
    tasks = [(share, args.seed * 100_000 + i, engine_path, args.sf_depth, args.max_cp,
              args.out, args.checkpoint_every)
             for i in range(args.workers)]
    results = (Pool(args.workers).map(worker, tasks) if args.workers > 1 else [worker(tasks[0])])

    packed = np.concatenate([r[0] for r in results])
    stm = np.concatenate([r[1] for r in results])
    cps = np.concatenate([r[2] for r in results])
    _, unique = np.unique(packed, axis=0, return_index=True)
    packed, stm, cps = packed[unique], stm[unique], cps[unique]
    wdl = (1.0 / (1.0 + np.exp(-K * cps.astype(np.float32)))).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, packed=packed, stm=stm, cp=cps, wdl=wdl)
    for i in range(args.workers):  # the run finished cleanly; drop the crash-checkpoints
        part = f"{args.out}.part{(args.seed * 100_000 + i) % 1000:03d}.npz"
        if os.path.exists(part):
            os.remove(part)

    elapsed = time.time() - started
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  {len(cps):,} positions  {size:.1f} MB  in {elapsed / 60:.1f} min "
          f"({len(cps) / elapsed:.0f} pos/s)")
    decisive = np.mean(np.abs(cps) >= 800)
    print(f"cp: median {np.median(cps):.0f}  |cp|>=800 in {decisive:.0%}  "
          f"5-95 pct {np.percentile(cps, 5):.0f}..{np.percentile(cps, 95):.0f}")
    print(f"wdl: mean {wdl.mean():.3f}  sd {wdl.std():.3f}")


if __name__ == "__main__":
    main()
