"""Flatten engine/ into a self-contained agent directory (default candidates/engine) so
harness/, bench/, and the platform can run it: agent.py at the root, flat imports, the nets
and Syzygy tables beside it. Re-run after a retrain / export or an engine edit.

    uv run python tools/bundle_engine.py [--out candidates/engine] [--net engine/net.npz]
                                          [--net-eg engine/net_eg.npz] [--zip]

harness/runner.py puts only the agent directory on sys.path and does `import agent`, so every
module the engine touches has to live in that one directory with no package prefix - `engine`
itself is not importable from inside a flattened bundle. This copies every engine/*.py module
with `from engine.` rewritten to `from `, plus both nets and syzygy/ alongside. --zip also
writes submission.zip with those files at the archive root.
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"

# every engine/ module except the package marker, flattened (the `engine.` prefix stripped
# from their imports below)
MODULES = sorted(
    p.name for p in ENGINE.glob("*.py") if p.name != "__init__.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "candidates" / "engine")
    parser.add_argument("--net", type=Path, default=ENGINE / "net.npz")
    parser.add_argument("--net-eg", type=Path, default=ENGINE / "net_eg.npz",
                         help="endgame net, shipped as net_eg.npz (engine/accumulator.py "
                         "EG_MEN); pass a missing path to bundle a single-net build")
    parser.add_argument("--zip", action="store_true", help="also write submission.zip")
    args = parser.parse_args()

    if not args.net.is_file():
        sys.exit(f"no net weights at {args.net} - run tools/export_nn.py first")

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in MODULES:
        text = (ENGINE / name).read_text()
        text = re.sub(r"\bfrom engine\.", "from ", text)
        text = re.sub(r"\bimport engine\.", "import ", text)
        text = text.replace("from engine import reference", "import reference")
        (out / name).write_text(text)

    shutil.copy2(args.net, out / "net.npz")
    if args.net_eg.is_file():
        shutil.copy2(args.net_eg, out / "net_eg.npz")

    # tablebase.py mmaps the Syzygy tables at the same relative path (syzygy/ beside the modules)
    syzygy = ENGINE / "syzygy"
    if syzygy.is_dir():
        shutil.copytree(syzygy, out / "syzygy")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"bundled -> {out}  ({total / 1024:.0f} KB, {len(list(out.iterdir()))} entries)")
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
    print(f"\n  uv run python bench/bench.py --agent {rel} --opponent stages/07-numba-classical")


if __name__ == "__main__":
    main()
