"""Bundle the nnue/ engine into a self-contained agent directory (default candidates/nnue) so
tools/bench.py - and the platform - can run it: agent.py at the root, flat imports, net.npz
beside it. Re-run after a retrain / export or an engine edit.

    uv run python tools/bundle_nnue.py [--out candidates/nnue] [--net nnue/net.npz] [--zip]

harness/runner.py puts only the agent directory on sys.path and does `import agent`, so every
module the engine touches has to live in that one directory with no package prefix. This copies
the nnue/ modules with `from nnue.x` rewritten to `from x`, plus the shared root modules the
engine and its reference fallback import (attacks/bitboard/move/zobrist and reference/tables).
--zip also writes submission.zip with those files at the archive root.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# nnue/ package modules, flattened (the `nnue.` prefix stripped from their imports)
NNUE_MODULES = [
    "agent.py", "accumulator.py", "board.py", "movegen.py", "net.py", "arch.py", "tablebase.py",
]

# prepended to the shipped agent.py, before numpy loads: a referee running games in parallel
# puts many agents on the same cores, and a BLAS that spawns a thread per core in each one
# flags the clock. The search is single-threaded (numba @njit, no parallel=); nothing wants a
# pool. Harmless on the platform's one core, insurance for the local parallel harness.
THREAD_PIN = (
    "# ruff: noqa: E402  (thread pins must precede the numpy import)\n"
)
THREAD_PIN_BODY = (
    "import os as _os\n"
    "for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',\n"
    "           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):\n"
    "    _os.environ.setdefault(_v, '1')\n\n"
)

# shared root modules, copied verbatim. zobrist.py does `from board import Board`, which in the
# bundle resolves to the nnue board above - the only board.py present - which is what we want.
# reference.py (the numba-compile fallback) pulls in tables.py.
ROOT_MODULES = ["attacks.py", "bitboard.py", "move.py", "zobrist.py", "reference.py", "tables.py"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "candidates" / "nnue")
    parser.add_argument("--net", type=Path, default=ROOT / "nnue" / "net.npz")
    parser.add_argument("--net-eg", type=Path, help="endgame net, shipped as net_eg.npz "
                        "(nnue/accumulator.py EG_MEN); omit for a single-net bundle")
    parser.add_argument("--zip", action="store_true", help="also write submission.zip")
    args = parser.parse_args()

    if not args.net.is_file():
        sys.exit(f"no net weights at {args.net} - run tools/export_nn.py first")

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in NNUE_MODULES:
        text = (ROOT / "nnue" / name).read_text()
        text = text.replace("from nnue.", "from ").replace("import nnue.", "import ")
        if name == "agent.py":
            text = THREAD_PIN + text.replace(
                "\nimport time\n", "\n" + THREAD_PIN_BODY + "import time\n", 1
            )
        (out / name).write_text(text)

    for name in ROOT_MODULES:
        shutil.copy2(ROOT / name, out / name)

    shutil.copy2(args.net, out / "net.npz")
    if args.net_eg:
        if not args.net_eg.is_file():
            sys.exit(f"no endgame net at {args.net_eg}")
        shutil.copy2(args.net_eg, out / "net_eg.npz")

    # the Syzygy tables tablebase.py mmaps, at the same relative path (syzygy/ beside the modules)
    syzygy = ROOT / "nnue" / "syzygy"
    if syzygy.is_dir():
        shutil.copytree(syzygy, out / "syzygy")

    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"bundled -> {out}  ({total / 1024:.0f} KB, {len(list(out.iterdir()))} files)")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")

    if args.zip:
        dest = ROOT / "submission.zip"
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(out))
        unzipped = sum(
            p.stat().st_size for p in out.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
        print(f"\nwrote {dest}  ({dest.stat().st_size / 1024:.0f} KB zipped, "
              f"{unzipped / 1024:.0f} KB unzipped)")

    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\n  uv run python tools/bench.py --agent {rel} --opponent candidates/toby")


if __name__ == "__main__":
    main()
