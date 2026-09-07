"""Bundle the nnue/ engine into a self-contained agent directory (default candidates/nnue) so
tools/bench.py - and the platform - can run it: agent.py at the root, flat imports, net.npz
beside it. Re-run after a retrain / export or an engine edit.

    uv run python tools/bundle_nnue.py [--out candidates/nnue] [--net nnue/net.npz]

harness/runner.py puts only the agent directory on sys.path and does `import agent`, so every
module the engine touches has to live in that one directory with no package prefix. This copies
the nnue/ modules with `from nnue.x` rewritten to `from x`, plus the shared root modules the
engine and its reference fallback import (attacks/bitboard/move/zobrist and reference/tables).
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# nnue/ package modules, flattened (the `nnue.` prefix stripped from their imports)
NNUE_MODULES = ["agent.py", "accumulator.py", "board.py", "movegen.py", "net.py", "arch.py"]

# shared root modules, copied verbatim. zobrist.py does `from board import Board`, which in the
# bundle resolves to the nnue board above - the only board.py present - which is what we want.
# reference.py (the numba-compile fallback) pulls in tables.py.
ROOT_MODULES = ["attacks.py", "bitboard.py", "move.py", "zobrist.py", "reference.py", "tables.py"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "candidates" / "nnue")
    parser.add_argument("--net", type=Path, default=ROOT / "nnue" / "net.npz")
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
        (out / name).write_text(text)

    for name in ROOT_MODULES:
        shutil.copy2(ROOT / name, out / name)

    shutil.copy2(args.net, out / "net.npz")

    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"bundled -> {out}  ({total / 1024:.0f} KB, {len(list(out.iterdir()))} files)")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\n  uv run python tools/bench.py --agent {rel} --opponent candidates/toby")


if __name__ == "__main__":
    main()
