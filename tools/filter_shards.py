"""Keep only the <= MAX_MEN positions of a shard directory, one output shard per input shard.

The endgame net (engine/net_eg.npz) is trained on this subset alone: 14.5% of the Lichess dump
is 12 men or fewer, ~31M positions of the 215M, and a net that only ever sees them spends all
its capacity there. Layout is unchanged (packed / stm / cp / wdl), so tools/train_nn.py reads
the output directory as it reads the full one.

    uv run python tools/filter_shards.py data/lichess_300m data/endgame12 --max-men 12
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def filter_shard(task: tuple[Path, Path, int]) -> tuple[str, int, int]:
    src, dst, max_men = task
    with np.load(src) as d:
        packed = d["packed"]
        men = np.unpackbits(packed, axis=1).sum(axis=1)
        keep = men <= max_men
        np.savez(dst, packed=packed[keep], stm=d["stm"][keep], cp=d["cp"][keep], wdl=d["wdl"][keep])
    return src.name, int(keep.sum()), len(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--max-men", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("shard_*.npz"))
    tasks = [(f, args.dst / f"shard_{i:04d}.npz", args.max_men) for i, f in enumerate(files)]
    kept = total = 0
    with Pool(args.workers) as pool:
        for name, k, n in pool.imap_unordered(filter_shard, tasks):
            kept += k
            total += n
            print(f"{name}: kept {k} of {n}", flush=True)
    print(f"kept {kept} of {total} ({kept / total:.1%}) into {args.dst}")


if __name__ == "__main__":
    main()
