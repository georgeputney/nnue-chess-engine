"""Node bench - fixed-depth search over a fixed FEN set; checks a refactor kept move and score."""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

# run from anywhere: put the repo root on the path so `import agent` finds the submission
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# this imports agent.py directly, in-process - off by default so every run doesn't pay the
# opening-search warm-up cost agent.py's own import now does. setdefault, not a flat assignment:
# an explicit AGENT_WARM_UP_S in the environment still wins.
os.environ.setdefault("AGENT_WARM_UP_S", "0")

# one measured position: fen, the move the search picked, its score, and the node count
Row = dict[str, object]

# opening / middlegame / tactical (Win At Chess) / endgame, kept small so a depth-6 run is
# seconds not minutes
FENS: list[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/4PN2/PP3PPP/RNBQKB1R w KQkq - 0 5",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N1BP2/PP1QN1PP/R3KB1R w KQ - 0 10",
    "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 11",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1",
    "r1b1kb1r/3q1ppp/pBp1pn2/8/Np1P4/5Q2/PPP2PPP/R3K2R w KQkq - 0 12",
    "r3r1k1/pp3pbp/1qp1b1p1/2B5/2BP4/Q1n2N2/P4PPP/3R1K1R w - - 0 17",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    "8/2k5/8/8/8/8/2K5/4R3 w - - 0 1",
    "8/8/8/3k4/8/3K4/3P4/8 w - - 0 1",
    "8/5k2/8/8/2P5/8/5K2/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "8/6k1/1p6/p1p5/P1P5/1P6/8/6K1 w - - 0 1",
]


# searches every FEN to `depth` via <module>.bench_search, printing nodes / time / move / score
# per position and a nps total. returns the rows for a snapshot or a compare
def run(depth: int, module: str = "agent") -> list[Row]:
    agent = importlib.import_module(module)
    if not hasattr(agent, "bench_search"):
        raise SystemExit(
            f"{module}.bench_search(fen, depth) -> (uci, score, nodes) is not defined yet. "
            "add it once there is a search (docs/plan.md phase 2)."
        )

    rows: list[Row] = []
    total_nodes = 0
    started = time.monotonic()
    for fen in FENS:
        t0 = time.monotonic()
        uci, score, nodes = agent.bench_search(fen, depth)
        dt = time.monotonic() - t0
        total_nodes += nodes
        rows.append({"fen": fen, "move": uci, "score": int(score), "nodes": int(nodes)})
        print(f"{nodes:>10d}  {dt:6.2f}s  {uci:<6s} {score:>7d}  {fen}")

    wall = time.monotonic() - started
    nps = total_nodes / wall if wall else 0.0
    print(f"\ntotal {total_nodes} nodes in {wall:.2f}s  ({nps:,.0f} nps)  depth {depth}")
    return rows


# compares this run against a saved snapshot. a changed bestmove or score means the refactor
# was not exact; more nodes than before is allowed but flagged. returns the count of changed
# positions, which becomes the exit code
def compare(rows: list[Row], baseline: list[Row]) -> int:
    by_fen = {r["fen"]: r for r in baseline}
    bad = 0
    for r in rows:
        b = by_fen.get(r["fen"])
        if b is None:
            print(f"NEW   {r['fen']}")
            continue
        if r["move"] != b["move"] or r["score"] != b["score"]:
            bad += 1
            print(
                f"DIFF  {r['fen']}\n"
                f"      baseline {b['move']} {b['score']}  ->  now {r['move']} {r['score']}"
            )
        elif r["nodes"] > b["nodes"]:
            print(f"SLOW  {r['fen']}  nodes {b['nodes']} -> {r['nodes']} (allowed, but note it)")

    if bad:
        print(f"\n{bad} position(s) changed bestmove/score: NOT an exact refactor.")
    else:
        print("\nall bestmoves and scores match baseline: exact refactor confirmed.")
    return bad


# usage:
#     uv run python tools/nodebench.py --depth 4
#     uv run python tools/nodebench.py --depth 5 --baseline /tmp/nb.json
#     uv run python tools/nodebench.py --depth 5 --baseline /tmp/nb.json --check
#
# no --baseline just prints the run; --baseline alone saves it; --baseline --check compares
# and exits non-zero if any bestmove or score moved
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--module", default="agent", help="engine module with bench_search")
    parser.add_argument("--baseline", type=Path, help="snapshot to write, or read with --check")
    parser.add_argument("--check", action="store_true", help="compare this run against --baseline")
    args = parser.parse_args()

    rows = run(args.depth, args.module)

    if args.baseline and args.check:
        baseline = json.loads(args.baseline.read_text())
        raise SystemExit(1 if compare(rows, baseline) else 0)
    if args.baseline:
        args.baseline.write_text(json.dumps(rows, indent=2))
        print(f"baseline written to {args.baseline}")


if __name__ == "__main__":
    main()
