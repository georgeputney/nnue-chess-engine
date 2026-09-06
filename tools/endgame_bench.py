"""Endgame A/B - the sibling of tools/bench.py. Same fresh-process-per-game machinery and the
same score / Elo / 95% band, but the start set is roughly level endgames instead of openings.

Our openings hold up better than our endgames (see the game that went 1/2 a rook up), so this
is the suite that should move when endgame play changes - king activity, passed pawns, contempt,
pruning that is too greedy with few pieces on.

    uv run python tools/endgame_bench.py --agent . --opponent snapshots/pre-contempt
    uv run python tools/endgame_bench.py --agent . --opponent baselines/minimax \
        --base-ms 8000 --increment-ms 100 --workers 4
"""

import argparse
import os
import sys
from pathlib import Path

# run from anywhere: put the repo root on the path so `import harness` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# a fresh agent process per game (see tools/bench.py) - keep the import-time warm-up off
os.environ.setdefault("AGENT_WARM_UP_S", "0")

from tools.bench import run_suite

# Endgames that are drawn (or near-drawn) with best play but easy to spoil - what should move
# when endgame play changes. Each was checked at depth 11: all but the three tagged below score
# within ~90 cp of level, and every heavy piece is off an open line so nothing hangs on move 1.
# Played from both colours, so the listed side to move only sets who moves first.
#
# The three "eval reads it high" lines are textbook theoretical draws the static eval currently
# scores as +1 to +2 (4-vs-3 same-wing rook, blocked-pawn king race, opposite bishops two pawns
# up). They are the sharpest regression targets: a healthier endgame eval should hold them.
ENDGAMES: list[str] = [
    # rook endgames - same-wing majorities, rooks off each other's file
    "5rk1/5ppp/8/8/8/8/R4PPP/6K1 w - - 0 1",
    "5rk1/pp3ppp/8/8/8/1P6/P4PPP/3R2K1 w - - 0 1",
    "r5k1/pp3ppp/8/8/8/1P6/P4PPP/3R2K1 w - - 0 1",
    "r3r1k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1",
    "4r1k1/1p3ppp/p7/8/8/1P6/P4PPP/2R3K1 w - - 0 1",
    "8/3r1ppk/7p/8/8/8/1R3PPP/6K1 w - - 0 1",
    "3rr1k1/pp3ppp/8/8/8/8/PP3PPP/3RR1K1 w - - 0 1",
    "8/8/5k2/8/1r6/8/4R1K1/8 b - - 0 1",
    "6k1/5pp1/7p/8/8/5P1P/4R1P1/1r4K1 w - - 0 1",
    "5k2/5ppp/8/8/8/8/r4PPP/3R2K1 w - - 0 1",
    "5rk1/5ppp/8/8/8/8/4PPPP/1R4K1 w - - 0 1",  # 4-v-3 one wing: draw, eval too high
    "3r4/8/8/8/3k4/8/3P4/3KR3 w - - 0 1",       # Philidor R+P vs R: draw
    # king and pawn
    "8/6p1/5k1p/8/5K1P/6P1/8/8 w - - 0 1",
    "8/5pk1/6p1/4P3/5PK1/8/8/8 w - - 0 1",
    "8/8/4kp2/8/4KP2/8/8/8 w - - 0 1",
    "8/2p5/2P1k3/8/4K3/8/8/8 w - - 0 1",        # blocked c-pawns: draw, eval too high
    # minor pieces - opposite bishops, same bishops, bishop vs knight
    "8/8/4kp2/6p1/1b4P1/5P2/4K1B1/8 w - - 0 1",
    "8/5k2/5p2/5Bp1/6P1/8/1b3P2/6K1 w - - 0 1",
    "8/5k2/8/2B1p3/4P3/1b6/5PP1/6K1 w - - 0 1",  # opp bishops +2P: draw, eval too high
    "6k1/5pp1/7p/3b4/3B4/6P1/5P1P/6K1 w - - 0 1",
    "6k1/2n2ppp/8/8/8/2B5/5PPP/6K1 w - - 0 1",
    "6k1/1b3ppp/p7/1p6/1P6/P4NPP/5P2/6K1 w - - 0 1",
    "8/5k2/8/3n4/8/3B4/5K2/8 w - - 0 1",
    "8/8/4k3/8/4n3/4N3/4KP2/8 w - - 0 1",
    # queen endgame - equal pawns, perpetual-prone
    "6k1/5pp1/4q2p/8/8/4Q2P/5PP1/6K1 w - - 0 1",
]


# usage examples at the top of the file
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--endgames", type=int, default=len(ENDGAMES))
    parser.add_argument("--base-ms", type=int, default=6000)
    parser.add_argument("--increment-ms", type=int, default=100)
    # these positions resolve or stay dead-level well inside 200 plies; the platform's own 600
    # cap just makes the drawn ones grind three times as long for no extra signal
    parser.add_argument("--ply-cap", type=int, default=200)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2 - 1))
    args = parser.parse_args()

    run_suite(args.agent, args.opponent, ENDGAMES[:args.endgames],
              args.base_ms, args.increment_ms, args.workers,
              kind="endgame", ply_cap=args.ply_cap)


if __name__ == "__main__":
    main()
