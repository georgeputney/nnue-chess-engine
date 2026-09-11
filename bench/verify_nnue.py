"""Check the numba NNUE engine against the plain-numpy forward in engine/net.py, and check the
accumulator that make_move carries against a from-scratch rebuild.

    uv run python bench/verify_nnue.py [--positions 3000] [--games 60]

Stage 2: make_move rebuilds the accumulator every node, so the accumulator check is trivially
satisfied - it is the scaffold the stage-3 incremental update has to keep passing. The
oracle check is the real one: it proves the int16 transformer + float tail, the feature
orientation, and the centipawn scaling all match engine/net.py, which in turn matches the trainer.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.agent as engine  # noqa: E402
from engine import net as netmod  # noqa: E402
from engine.accumulator import (  # noqa: E402
    ACC_WIDTH,
    EG_MEN,
    EG_WEIGHTS,
    WEIGHTS,
    fill_accumulator,
)
from engine.movegen import legal_moves, make_move  # noqa: E402


# a spread of legal positions from lightly weighted random self-play, a few quiet ones per game
def sample_positions(games: int, rng: random.Random) -> list[str]:
    fens: list[str] = []
    for _ in range(games):
        board = chess.Board()
        for _ply in range(rng.randint(6, 120)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            weights = [1.6 if board.is_capture(m) else 1.0 for m in moves]
            board.push(rng.choices(moves, weights)[0])
            if not board.is_check() and len(board.move_stack) >= 6:
                fens.append(board.fen())
    return fens


def check_oracle(fens: list[str], tolerance: int) -> int:
    worst = 0
    fails = 0
    diffs = []
    endgames = 0
    for fen in fens:
        board = engine.parse_fen(fen)
        got = int(engine.evaluate(board))
        packed, stm = netmod.encode_fen(fen)
        # the twin scores with the net the men count picks, as the engine does
        endgame = chess.popcount(chess.Board(fen).occupied) <= EG_MEN
        endgames += endgame
        want = float(netmod.forward_packed(packed, stm, EG_WEIGHTS if endgame else WEIGHTS)[0])
        diff = abs(got - want)
        diffs.append(diff)
        worst = max(worst, diff)
        if diff > tolerance:
            fails += 1
            if fails <= 5:
                print(f"  MISMATCH  engine {got:+6d}  numpy twin {want:+8.2f}  {fen}")
    print(
        f"oracle check (engine vs numpy twin): {len(fens)} positions ({endgames} with the "
        f"endgame net), mean |diff| {np.mean(diffs):.3f} cp, max {worst:.3f} cp, "
        f"{fails} over {tolerance} cp"
    )
    return fails


# drive an engine board through long random move sequences with its own make_move (the
# incremental path) and assert the carried accumulator still equals a from-scratch rebuild.
# this is the stage-3 guarantee: make_move's column deltas mirror its piece edits exactly.
def check_accumulator(games: int, rng: random.Random) -> int:
    scratch = np.zeros((2, ACC_WIDTH), dtype=np.int32)
    fails = 0
    checked = 0
    for _ in range(games):
        reference_board = chess.Board()
        engine_board = engine.parse_fen(reference_board.fen())
        for _ply in range(rng.randint(20, 200)):
            moves, count = legal_moves(engine_board)
            if count == 0:
                break
            move = int(moves[rng.randrange(count)])
            uci = engine.move_uci(move)
            engine_board = make_move(engine_board, move)
            reference_board.push(chess.Move.from_uci(uci))

            fill_accumulator(engine_board.pieces, scratch)
            carried = np.asarray(engine_board.acc)
            if not np.array_equal(carried, scratch):
                fails += 1
                if fails <= 3:
                    gap = int(np.abs(carried.astype(np.int64) - scratch).max())
                    print(f"  ACC MISMATCH after {uci} (max column gap {gap})  "
                          f"{reference_board.fen()}")
            checked += 1
    print(f"accumulator check: {checked} incremental moves, {fails} disagree with a rebuild")
    return fails


def check_hand_fens() -> int:
    cases = [
        ("startpos white", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", None),
        ("startpos black", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", None),
        ("white up a queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "pos"),
        ("black up a queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", "neg"),
    ]
    fails = 0
    values = {}
    for name, fen, expect in cases:
        value = int(engine.evaluate(engine.parse_fen(fen)))
        values[name] = value
        ok = True
        if expect == "pos" and value <= 50:
            ok = False
        if expect == "neg" and value >= -50:
            ok = False
        print(f"  {name:22s} {value:+6d} cp{'' if ok else '   UNEXPECTED'}")
        fails += not ok
    if abs(values["startpos white"] - values["startpos black"]) > 1:
        print("  startpos not perspective-symmetric   UNEXPECTED")
        fails += 1
    return fails


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=3000)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--tolerance", type=int, default=2, help="allowed engine-vs-oracle cp gap")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fens = sample_positions(args.games, rng)
    if len(fens) > args.positions:
        fens = rng.sample(fens, args.positions)

    fails = 0
    fails += check_hand_fens()
    fails += check_oracle(fens, args.tolerance)
    fails += check_accumulator(args.games, rng)

    print("\nOK" if not fails else f"\n{fails} check(s) failed")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
