"""Convert the Lichess evaluation dump (database.lichess.org/#evals) into the same labelled
.npz schema tools/label.py writes, so tools/tune.py needs no changes to consume it.

Lichess's dump is real-game positions instead of label.py's weighted-random self-play - broader,
more natural coverage of openings and endgame technique, at the cost of not being able to choose
the position distribution the way label.py's samples/plies knobs do. Using both is normal:
label.py for outright coverage of positions self-play reaches, this for coverage of positions
real games reach.

Format: one JSON object per line, e.g.
    {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
     "evals": [{"pvs": [{"cp": 24, "line": "e2e4 e7e5 ..."}], "knodes": 2206765, "depth": 38}]}
The FEN omits the halfmove/fullmove counters; multiple "evals" entries are repeat analyses at
different depths (take the deepest); "cp"/"mate" in a pv is white-relative, unlike label.py's own
output and the UCI convention, so it gets flipped to side-to-move-relative here to match.

Usage (file is a plain path; .zst is decompressed on the fly via the zstd binary, so the 100+ GB
uncompressed dump never touches disk):
    python tools/ingest_lichess.py --in data/lichess_db_eval.jsonl.zst --out data/lichess.npz
    python tools/ingest_lichess.py --in data/lichess_db_eval.jsonl.zst --out data/lichess.npz \
        --keep-every 20 --limit 4000000 --min-depth 20

Then feed it to the tuner exactly like a label.py output:
    uv run python tools/tune.py --npz data/lichess.npz --out tools/tuned_tables.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label import K, features


def open_lines(path: str) -> Iterator[str]:
    """Yield decompressed lines. .zst goes through the zstd binary; anything else is read
    directly, so an already-decompressed .jsonl (or a manual `zstd -dc | ...` pipe via `-`)
    works too."""
    if path == "-":
        yield from sys.stdin
        return
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-dc", path], stdout=subprocess.PIPE, text=True, bufsize=1 << 20
        )
        assert proc.stdout is not None
        try:
            yield from proc.stdout
        finally:
            proc.stdout.close()
            proc.wait()
    else:
        with open(path) as fh:
            yield from fh


# python-chess wants 6 space-separated fields; Lichess's FEN drops the two move counters
def parse_fen(fen: str) -> chess.Board | None:
    try:
        return chess.Board(fen)
    except ValueError:
        pass
    try:
        return chess.Board(fen + " 0 1")
    except ValueError:
        return None


def process_line(line: str, min_depth: int, max_cp: int) -> tuple[chess.Board, int, int] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    evals = obj.get("evals")
    if not evals:
        return None

    best = max(evals, key=lambda e: e.get("depth", 0))
    if best.get("depth", 0) < min_depth:
        return None

    pvs = best.get("pvs")
    if not pvs:
        return None
    pv0 = pvs[0]
    if "mate" in pv0 or "cp" not in pv0:
        return None  # decided, or a malformed entry - nothing for a static eval to learn

    cp_white = pv0["cp"]
    if abs(cp_white) > max_cp:
        return None

    board = parse_fen(obj.get("fen", ""))
    if board is None or board.is_check():
        return None

    # quiet positions only, same reasoning as label.py: a static eval trained to reproduce a
    # tactic-dependent search score learns the wrong thing.
    line_moves = pv0.get("line", "").split()
    if line_moves:
        try:
            first_move = chess.Move.from_uci(line_moves[0])
            if first_move in board.legal_moves and (
                board.is_capture(first_move) or board.gives_check(first_move)
            ):
                return None
        except (ValueError, AssertionError):
            pass

    white_to_move = board.turn == chess.WHITE
    cp_stm = cp_white if white_to_move else -cp_white
    return board, int(white_to_move), cp_stm


def write_shard(path: str, packed: list, stm: list, cp: list) -> None:
    cps = np.array(cp, np.int16)
    wdl = (1.0 / (1.0 + np.exp(-K * cps.astype(np.float32)))).astype(np.float32)
    np.savez_compressed(path, packed=np.stack(packed), stm=np.array(stm, np.uint8),
                        cp=cps, wdl=wdl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True,
                        help="path to the .jsonl or .jsonl.zst dump, or - for stdin")
    parser.add_argument("--out", required=True,
                        help="a .npz file, or (with --shard-size) a directory for shard_NNNN.npz")
    parser.add_argument("--min-depth", type=int, default=20, help="drop shallower analyses")
    parser.add_argument("--max-cp", type=int, default=3000,
                        help="drop positions scored beyond this; keep it above a queen")
    parser.add_argument("--keep-every", type=int, default=1,
                        help="subsample: process only every Nth line, for speed on the full dump")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many kept positions")
    parser.add_argument("--shard-size", type=int, default=0,
                        help="rows per shard; >0 writes --out/shard_NNNN.npz and streams (no OOM "
                             "on the full dump)")
    parser.add_argument("--part", default="0/1",
                        help="'I/N': this worker handles only lines with (line_no %% N == I), so "
                             "N copies over the same dump split the ~35 h single-thread scan")
    parser.add_argument("--report-every", type=int, default=200_000)
    args = parser.parse_args()

    part_i, part_n = (int(x) for x in args.part.split("/"))
    tag = "" if part_n == 1 else f"p{part_i}_"

    sharding = args.shard_size > 0
    if sharding:
        os.makedirs(args.out, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows_packed, rows_stm, rows_cp = [], [], []
    started = time.time()
    seen = 0
    kept = 0
    shard = 0

    for seen, line in enumerate(open_lines(args.input), start=1):
        if seen % part_n != part_i:
            continue
        if args.keep_every > 1 and seen % args.keep_every:
            continue

        result = process_line(line, args.min_depth, args.max_cp)
        if result is None:
            continue

        board, white_to_move, cp_stm = result
        rows_packed.append(features(board))
        rows_stm.append(white_to_move)
        rows_cp.append(cp_stm)
        kept += 1

        if kept % args.report_every == 0:
            elapsed = time.time() - started
            print(f"  {seen:,} lines read, {kept:,} kept  ({seen / elapsed:,.0f} lines/s)",
                  flush=True)

        if sharding and len(rows_cp) >= args.shard_size:
            path = os.path.join(args.out, f"shard_{tag}{shard:04d}.npz")
            write_shard(path, rows_packed, rows_stm, rows_cp)
            print(f"  wrote {path}  ({len(rows_cp):,} rows)", flush=True)
            rows_packed, rows_stm, rows_cp = [], [], []
            shard += 1

        if args.limit and kept >= args.limit:
            break

    if not rows_cp and not (sharding and shard):
        sys.exit("no positions kept - check --in and the depth/cp filters")

    elapsed = time.time() - started
    if sharding:
        if rows_cp:
            path = os.path.join(args.out, f"shard_{tag}{shard:04d}.npz")
            write_shard(path, rows_packed, rows_stm, rows_cp)
            print(f"  wrote {path}  ({len(rows_cp):,} rows)", flush=True)
            shard += 1
        print(f"wrote {shard} shards to {args.out}  {kept:,} / {seen:,} lines  "
              f"in {elapsed / 60:.1f} min")
        return

    write_shard(args.out, rows_packed, rows_stm, rows_cp)
    cps = np.array(rows_cp, np.int16)
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  {kept:,} / {seen:,} lines  {size:.1f} MB  in {elapsed / 60:.1f} min")
    print(f"cp: median {np.median(cps):.0f}  sd {cps.std():.0f}  "
          f"5-95 pct {np.percentile(cps, 5):.0f} to {np.percentile(cps, 95):.0f}")


if __name__ == "__main__":
    main()
