"""Openings A/B - two agents over fixed near-level openings, both colours; score, Elo, 95% band."""

import argparse
import math
import os
import sys
import textwrap
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# run from anywhere: put the repo root on the path so `import harness` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# every game here spawns a fresh agent process (harness/sandbox.py, mirroring the platform), so
# agent.py's own opening-search warm-up would run again and again rather than the once-per-game
# it's meant for - off by default for exactly that reason. setdefault, not a flat assignment: an
# explicit AGENT_WARM_UP_S in the environment (someone deliberately measuring warm-up itself)
# still wins.
os.environ.setdefault("AGENT_WARM_UP_S", "0")

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

# one unit of work: position index, the two agent dirs, the start fen, whether the agent takes
# white, the clock in ms (base, increment), and the ply cap for this game
GameTask = tuple[int, str, str, str, bool, int, int, int]
# what comes back: opening index, agent-was-white, points for the agent, how it ended, and any
# stderr captured when a side broke
GameResult = tuple[int, bool, float, str, str]

# near-level positions 6 to 10 plies in, sampled from an engine's top three moves and kept only
# where it scored them within 40 cp of equal. harness/arena.py starts every game from the
# standard position, so two deterministic engines replay nearly the same game and 120 games
# carry about as much signal as one; varied openings from both colours is what the competition
# itself does
OPENINGS: list[str] = [
    "rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/4P3/PPP2PPP/RN1QKBNR w KQkq - 1 4",
    "rnbqkbnr/pp1p1ppp/4p3/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4",
    "rnbqkb1r/pp2pppp/5n2/2pp4/8/4PN2/PPPPBPPP/RNBQK2R w KQkq - 0 4",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/4PN2/PP3PPP/RNBQKB1R w KQkq - 0 5",
    "r1bqkb1r/pp3ppp/2n1pn2/2pp4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 2 6",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 4 4",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/3P4/4PNP1/PPP2P1P/RNBQKB1R w KQkq - 0 5",
    "rnbqkb1r/ppp1pp1p/5np1/3p4/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 2 4",
    "rnbqkb1r/pp2pppp/5n2/2Pp4/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 1 4",
    "rnbqk2r/pp3ppp/2p2n2/2bpp3/4P3/1BPP4/PP3PPP/RNBQK1NR w KQkq - 0 6",
    "rnbqkb1r/1p3ppp/p2ppn2/2p5/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 6",
    "r1bqkbnr/pp3ppp/2n1p3/1Bpp4/4P3/P4N2/1PPP1PPP/RNBQK2R w KQkq - 0 5",
    "rn1qkb1r/1pp1pppp/p3bn2/8/2p5/5NP1/PPQPPPBP/RNB1K2R w KQkq - 2 6",
    "rnbqkbnr/1pp1pppp/p7/8/2pP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
    "rn1qkb1r/ppp2ppp/3p1n2/4p3/2B1P1b1/3P4/PPP1QPPP/RNB1K1NR w KQkq - 1 5",
    "r1bqk1nr/pppn1ppp/3p4/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 2 5",
    "rnbqk1nr/pp2bppp/4p3/2pp4/3P4/3BPN2/PPP2PPP/RNBQK2R w KQkq - 0 5",
    "rnbqkb1r/pp1ppp1p/5np1/2p5/2P5/5NP1/PP1PPP1P/RNBQKB1R w KQkq - 0 4",
    "r1bqk1nr/pp1p1ppp/2n5/2b1p3/2BpP3/2P2N2/PP3PPP/RNBQK2R w KQkq - 2 6",
    "r1b1kbnr/p3pppp/1qp5/2ppP3/8/2P5/PP1P1PPP/RNBQK1NR w KQkq - 0 6",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPPQPPP/RNB1K1NR w KQkq - 4 4",
    "r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P4/2N1PN2/PPP1BPPP/R1BQK2R w KQkq - 3 6",
    "rn1qkb1r/pp2nppp/2p1p3/3pP3/6b1/2P2N2/PP1PBPPP/RNBQK2R w KQkq - 2 6",
    "rnbqkb1r/pp2pp1p/5np1/2pp4/8/2P2NPP/PP1PPP2/RNBQKB1R w KQkq - 0 5",
    "rnbqk2r/ppp1bppp/4pn2/3p4/3P4/2P2NP1/PP2PP1P/RNBQKB1R w KQkq - 1 5",
    "rnbqkbnr/1p2p1pp/p3p3/2p5/2Bp4/2P5/PP1P1PPP/RNBQK1NR w KQkq - 0 6",
    "rnbqkb1r/pp1p1ppp/4pn2/8/2Pp4/4PN2/PP3PPP/RNBQKB1R w KQkq - 0 5",
    "r1bqk1nr/ppp2ppp/2np4/2b1p3/2B1P3/2PP4/PP3PPP/RNBQK1NR w KQkq - 0 5",
    "rnbqkb1r/pp1pppp1/2p2n1p/8/3P4/5NP1/PPP1PP1P/RNBQKB1R w KQkq - 0 4",
    "r1bqkb1r/pp2pppp/2np1n2/2p5/4P3/2N3P1/PPPPNP1P/R1BQKB1R w KQkq - 1 5",
    "rnbqkb1r/ppp1pppp/5n2/8/2pP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
    "r1bqkb1r/pppnpppp/2n5/8/Q1pP4/P4N2/1P2PPPP/RNB1KB1R w KQkq - 1 6",
    "rnbqkb1r/pp3ppp/2pp1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 6",
    "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/P1N5/1PPP1PPP/R1BQK1NR w KQkq - 0 5",
    "rn1qkbnr/pp2pppp/2p5/3p4/6b1/4PN2/PPPPBPPP/RNBQK2R w KQkq - 2 4",
    "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 4 4",
    "r1bqkbnr/pp3ppp/2n1p3/2pp4/P3P3/2N2N2/1PPPBPPP/R1BQK2R w KQkq - 0 6",
    "rn1qkbnr/pp2pppp/2p5/3p1b2/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 2 4",
    "rnbqkb1r/pp3ppp/3pp3/2pnP3/8/2P2N2/PP1PBPPP/RNBQK2R w KQkq - 0 6",
    "rnbqkb1r/pp1ppp1p/2p3p1/8/3Pn3/5NP1/PPP1PPBP/RNBQK2R w KQkq - 1 5",
]


# plays one game from one opening with one colour assignment and scores it from the agent's
# side. on a failed termination it also pulls back whichever process wrote to stderr, tagged
# agent or opponent, so main() can print it
def one_game(task: GameTask) -> GameResult:
    index, agent, opponent, fen, agent_is_white, base_ms, increment_ms, ply_cap = task
    white, black = (agent, opponent) if agent_is_white else (opponent, agent)
    white_agent, black_agent = local(Path(white)), local(Path(black))
    outcome = play_match(white_agent, black_agent, base_ms, increment_ms,
                         ply_cap=ply_cap, start_fen=fen)

    if outcome.result in ("draw", "void"):
        points = 0.5
    elif (outcome.result == "white") == agent_is_white:
        points = 1.0
    else:
        points = 0.0

    detail = ""
    if outcome.termination in FAILED_TERMINATIONS:
        parts = []
        for side, agent_proc in (("white", white_agent), ("black", black_agent)):
            if agent_proc.stderr_tail:
                who = "agent" if (side == "white") == agent_is_white else "opponent"
                parts.append(f"{side} ({who}):\n"
                             + textwrap.indent(agent_proc.stderr_tail.rstrip(), "    "))
        detail = "\n".join(parts)

    return index, agent_is_white, points, outcome.termination, detail


# logistic score in 0..1 turned into an Elo difference. returns +/-inf at the extremes so a
# clean sweep does not blow up the caller
def elo(score: float) -> float:
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)


# fans the position x colour grid across a process pool, prints each game as it lands, then a
# final score and Elo band. `kind` only labels the per-game line ("opening" / "endgame"). shared
# with bench/endgame_bench.py, which passes its own position set. exits non-zero only via
# one_game failures surfaced at the end
def run_suite(
    agent: str,
    opponent: str,
    positions: list[str],
    base_ms: int,
    increment_ms: int,
    workers: int,
    kind: str = "opening",
    ply_cap: int = PLY_CAP,
) -> None:
    for role, path in (("agent", agent), ("opponent", opponent)):
        if not (Path(path) / "agent.py").is_file():
            raise SystemExit(f"no {role}: {Path(path) / 'agent.py'} does not exist")

    tasks: list[GameTask] = [
        (i + 1, agent, opponent, fen, white, base_ms, increment_ms, ply_cap)
        for i, fen in enumerate(positions)
        for white in (True, False)
    ]
    print(f"{len(tasks)} games over {len(positions)} {kind}s, {workers} in parallel\n")

    # each game runs two agent processes, so half the cores keeps both sides off a contended
    # core. more workers makes the results noisier, not faster
    verdicts = {1.0: "win", 0.5: "draw", 0.0: "loss"}
    results: list[tuple[float, str]] = []
    failures: list[tuple[int, str, str, str]] = []
    with Pool(workers) as pool:
        for done, (index, agent_white, points, termination, detail) in enumerate(
                pool.imap_unordered(one_game, tasks), start=1):
            colour = "white" if agent_white else "black"
            print(f"[{done:3d}/{len(tasks)}] {kind} {index:2d}  agent {colour}  "
                  f"{verdicts[points]:4s}  ({termination})")
            results.append((points, termination))
            if detail:
                failures.append((index, colour, termination, detail))

    points = [p for p, _ in results]
    terminations = Counter(t for _, t in results)
    score = sum(points) / len(points)
    wins = sum(1 for p in points if p == 1.0)
    draws = sum(1 for p in points if p == 0.5)
    losses = sum(1 for p in points if p == 0.0)

    # standard error of the mean score, turned into an Elo band
    variance = sum((p - score) ** 2 for p in points) / max(1, len(points) - 1)
    stderr = math.sqrt(variance / len(points))
    low = elo(min(0.999, max(0.001, score - 1.96 * stderr)))
    high = elo(min(0.999, max(0.001, score + 1.96 * stderr)))

    print(f"\n{agent} vs {opponent}")
    print(f"+{wins} ={draws} -{losses}  score {score:.1%}")
    middle = elo(min(0.999, max(0.001, score)))
    print(f"elo {middle:+.0f}  95% ci [{low:+.0f}, {high:+.0f}]")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in terminations.most_common()))

    broken = {k: v for k, v in terminations.items() if k in FAILED_TERMINATIONS}
    if broken:
        print("FAILURES: " + ", ".join(f"{k} {v}" for k, v in broken.items()))
        for index, colour, termination, detail in failures:
            print(f"\n--- {kind} {index}, agent {colour}, {termination} ---")
            print(detail or "  (no stderr captured)")


# usage:
#     uv run python bench/openings_bench.py --agent stages/08-single-nnue \
#         --opponent stages/07-numba-classical
#     uv run python bench/openings_bench.py --agent stages/08-single-nnue \
#         --opponent stages/07-numba-classical --openings 40 --base-ms 5000 --increment-ms 100 \
#         --workers 4
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--openings", type=int, default=len(OPENINGS))
    parser.add_argument("--base-ms", type=int, default=5000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) // 2 - 1))
    args = parser.parse_args()

    run_suite(args.agent, args.opponent, OPENINGS[:args.openings],
              args.base_ms, args.increment_ms, args.workers)


if __name__ == "__main__":
    main()
