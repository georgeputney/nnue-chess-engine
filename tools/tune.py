"""Offline Texel tuner - fits the evaluation weights to game results and writes tables.py."""

import argparse
import importlib
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import chess
import numpy as np

# run from anywhere: put the repo root on the path so `import agent` finds the submission
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
agent = importlib.import_module("agent")
tables = importlib.import_module("tables")

# every eval term is `weight * count`, so the static score is linear in the weights:
#     eval_cp = (phase * sum(midgame[i] * count[i])
#            + (24 - phase) * sum(endgame[i] * count[i])) // 24
# coefficients() reads count[i] (white minus black) off a board; the dataset stacks those into
# a sparse matrix and gradient descent fits the weights to
#     mean((sigmoid(k * eval_cp / 400) - result) ** 2)
# with the game result from white's point of view. only the fitted numbers ship.

# 6 piece types x 64 squares of piece-square table, then one slot per scalar term. every slot
# is fitted as a (midgame, endgame) pair, so there are two weight vectors of this length
PST_PARAMS = 6 * 64
MOBILITY_INDEX = {chess.BISHOP: PST_PARAMS, chess.ROOK: PST_PARAMS + 1, chess.QUEEN: PST_PARAMS + 2}
KING_EXPOSURE_INDEX = PST_PARAMS + 3
TEMPO_INDEX = PST_PARAMS + 4
DOUBLED_INDEX = PST_PARAMS + 5
PARAM_COUNT = PST_PARAMS + 6


# flat index into a weight vector for one piece-square entry. `square` is white's view
# (a1 = 0), already mirrored for black by the caller
def pst_index(piece_type: int, square: int) -> int:
    return (piece_type - 1) * 64 + square


# how many times each eval term applies in `board`, white minus black, plus the game phase.
# mirrors agent.evaluate term for term - --selfcheck fails if this drifts
def coefficients(board: chess.Board) -> tuple[dict[int, int], int]:
    counts: dict[int, int] = defaultdict(int)
    occupied = board.occupied

    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        sign = 1 if piece.color == chess.WHITE else -1
        table_square = square if piece.color == chess.WHITE else square ^ 56

        counts[pst_index(piece_type, table_square)] += sign

        if piece_type in MOBILITY_INDEX:
            reach = chess.popcount(board.attacks_mask(square))
            counts[MOBILITY_INDEX[piece_type]] += sign * reach
        elif piece_type == chess.KING:
            counts[KING_EXPOSURE_INDEX] += sign * agent.slider_scope(square, occupied)

    counts[TEMPO_INDEX] += 1 if board.turn == chess.WHITE else -1

    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]
    counts[DOUBLED_INDEX] += agent.doubled_pawns(white_pawns) - agent.doubled_pawns(black_pawns)

    return counts, agent.game_phase(board)


# weight vectors seeded from tables.py, so a run refines what agent.py already plays. the
# split scalars are already signed the way coefficients() expects (count is white minus
# black, the weight carries the sign), so they copy straight across
def initial_weights() -> tuple[np.ndarray, np.ndarray]:
    midgame = np.zeros(PARAM_COUNT, dtype=np.float64)
    endgame = np.zeros(PARAM_COUNT, dtype=np.float64)

    for piece_type in range(1, 7):
        for square in range(64):
            midgame[pst_index(piece_type, square)] = tables.MIDGAME_TABLE[piece_type][square]
            endgame[pst_index(piece_type, square)] = tables.ENDGAME_TABLE[piece_type][square]

    for piece_type, index in MOBILITY_INDEX.items():
        midgame[index] = tables.MOBILITY_WEIGHT_MG[piece_type]
        endgame[index] = tables.MOBILITY_WEIGHT_EG[piece_type]

    midgame[KING_EXPOSURE_INDEX] = tables.KING_EXPOSURE_MG
    endgame[KING_EXPOSURE_INDEX] = tables.KING_EXPOSURE_EG
    midgame[TEMPO_INDEX] = tables.TEMPO_MG
    endgame[TEMPO_INDEX] = tables.TEMPO_EG
    midgame[DOUBLED_INDEX] = tables.DOUBLED_PAWN_MG
    endgame[DOUBLED_INDEX] = tables.DOUBLED_PAWN_EG

    return midgame, endgame


_RESULT_MARKERS = (("1/2-1/2", 0.5), ("1-0", 1.0), ("0-1", 0.0))
_TRAILING_SCORE = re.compile(r"\[?([01](?:\.\d+)?)\]?\s*$")


# pull a fen and a white-relative result out of one data line, or None if it is a comment,
# blank, or unparseable
def parse_line(line: str) -> tuple[str, float] | None:
    text = line.strip().replace(";", " ")

    if not text or text.startswith("#"):
        return None

    result: float | None = None
    for marker, value in _RESULT_MARKERS:
        if marker in text:
            result = value
            break

    if result is None:
        match = _TRAILING_SCORE.search(text)
        result = float(match.group(1)) if match else None

    if result is None:
        return None

    fields = text.split()
    for field_count in (6, 5, 4):
        candidate = " ".join(fields[:field_count])
        try:
            chess.Board(candidate)
        except ValueError:
            continue
        return candidate, result
    return None


# walk sources.csv (one line each: path, wdl-from-side-to-move 1/0, limit 0=all) and yield
# every (fen, white-relative result) it points at
def load_sources(csv_path: Path) -> Iterator[tuple[str, float]]:
    for raw in csv_path.read_text().splitlines():
        row = raw.strip()

        if not row or row.startswith("#"):
            continue

        path_text, wdl_from_stm, limit_text = (cell.strip() for cell in row.split(","))
        side_relative = wdl_from_stm == "1"
        limit = int(limit_text)

        kept = 0
        for line in Path(path_text).read_text().splitlines():
            parsed = parse_line(line)

            if parsed is None:
                continue

            fen, result = parsed
            if side_relative and fen.split()[1] == "b":
                result = 1.0 - result

            yield fen, result

            kept += 1
            if limit and kept >= limit:
                break


# turn positions into the sparse system: coo triplets (row, column, count) for the design
# matrix, plus per-position phase and result. positions with the side to move in check are
# dropped - the static eval is not meaningful there
def build_dataset(
    samples: Iterator[tuple[str, float]], drop_in_check: bool = True
) -> dict[str, np.ndarray]:
    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    phases: list[int] = []
    results: list[float] = []

    row = 0
    for fen, result in samples:
        board = chess.Board(fen)

        if drop_in_check and board.is_check():
            continue

        counts, phase = coefficients(board)
        for index, count in counts.items():
            if count:
                rows.append(row)
                columns.append(index)
                values.append(count)

        phases.append(phase)
        results.append(result)

        row += 1
        if row % 50_000 == 0:
            print(f"  {row} positions")

    return {
        "rows": np.asarray(rows, dtype=np.int32),
        "columns": np.asarray(columns, dtype=np.int32),
        "values": np.asarray(values, dtype=np.float64),
        "phases": np.asarray(phases, dtype=np.float64),
        "results": np.asarray(results, dtype=np.float64),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# white-relative static eval in centipawns for every position. the two bincounts are the
# sparse matrix-vector products of the design matrix with each weight vector
def eval_cp(data: dict[str, np.ndarray], midgame: np.ndarray, endgame: np.ndarray) -> np.ndarray:
    rows, columns, values = data["rows"], data["columns"], data["values"]
    phases = data["phases"]
    position_count = phases.shape[0]

    midgame_row = values * midgame[columns]
    endgame_row = values * endgame[columns]
    midgame_part = np.bincount(rows, weights=midgame_row, minlength=position_count)
    endgame_part = np.bincount(rows, weights=endgame_row, minlength=position_count)

    return (phases * midgame_part + (24.0 - phases) * endgame_part) / 24.0


def mean_squared_error(evals: np.ndarray, results: np.ndarray, k: float) -> float:
    predicted = _sigmoid(k * evals / 400.0)

    return float(np.mean((predicted - results) ** 2))


# scan for the k that best fits the current weights, so the sigmoid sits on the eval's scale
# and tuning does not have to inflate the weights to compensate. wide range - a k that lands
# on a scan edge means the weights are absorbing the rest of the scale
def best_k(data: dict[str, np.ndarray], midgame: np.ndarray, endgame: np.ndarray) -> float:
    evals = eval_cp(data, midgame, endgame)
    results = data["results"]
    candidates = np.arange(0.5, 8.0, 0.02)
    losses = [mean_squared_error(evals, results, k) for k in candidates]

    return float(candidates[int(np.argmin(losses))])


# closed-form gradient of the loss for each weight vector (no autograd). the chain is
# loss -> sigmoid -> eval_cp, and d eval_cp / d weight is the phase-scaled design matrix, so
# the transpose products are bincounts again
def gradient(
    data: dict[str, np.ndarray], midgame: np.ndarray, endgame: np.ndarray, k: float
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns, values = data["rows"], data["columns"], data["values"]
    phases, results = data["phases"], data["results"]
    position_count = phases.shape[0]

    evals = eval_cp(data, midgame, endgame)
    predicted = _sigmoid(k * evals / 400.0)
    d_loss_d_eval = (predicted - results) * predicted * (1.0 - predicted)
    d_loss_d_eval *= (k / 400.0) * (2.0 / position_count)

    midgame_terms = values * (d_loss_d_eval * phases / 24.0)[rows]
    endgame_terms = values * (d_loss_d_eval * (24.0 - phases) / 24.0)[rows]
    midgame_grad = np.bincount(columns, weights=midgame_terms, minlength=PARAM_COUNT)
    endgame_grad = np.bincount(columns, weights=endgame_terms, minlength=PARAM_COUNT)

    return midgame_grad, endgame_grad


# per-vector Adam optimiser state (https://arxiv.org/abs/1412.6980)
class Adam:
    def __init__(self, size: int) -> None:
        self.first = np.zeros(size)
        self.second = np.zeros(size)

    # one step, updating `weights` in place
    def step(
        self, weights: np.ndarray, grad: np.ndarray, epoch: int, learning_rate: float
    ) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.first[:] = beta1 * self.first + (1.0 - beta1) * grad
        self.second[:] = beta2 * self.second + (1.0 - beta2) * grad * grad
        corrected1 = self.first / (1.0 - beta1**epoch)
        corrected2 = self.second / (1.0 - beta2**epoch)
        weights -= learning_rate * corrected1 / (np.sqrt(corrected2) + eps)


# fit both weight vectors by Adam gradient descent, dropping the learning rate for the last
# third. returns the fitted weights and the k used
def tune(
    data: dict[str, np.ndarray], epochs: int, learning_rate: float
) -> tuple[np.ndarray, np.ndarray, float]:
    midgame, endgame = initial_weights()
    k = best_k(data, midgame, endgame)
    start_loss = mean_squared_error(eval_cp(data, midgame, endgame), data["results"], k)
    print(f"k = {k:.3f}   start loss {start_loss:.6f}")

    midgame_adam = Adam(PARAM_COUNT)
    endgame_adam = Adam(PARAM_COUNT)

    drop_at = epochs * 2 // 3
    for epoch in range(1, epochs + 1):
        rate = learning_rate if epoch < drop_at else learning_rate * 0.1
        midgame_grad, endgame_grad = gradient(data, midgame, endgame, k)
        midgame_adam.step(midgame, midgame_grad, epoch, rate)
        endgame_adam.step(endgame, endgame_grad, epoch, rate)

        if epoch == drop_at:
            # weights have moved off the seed scale; re-fit k before the fine pass
            k = best_k(data, midgame, endgame)
            print(f"  epoch {epoch:5d}   k re-fit to {k:.3f}")

        if epoch % 100 == 0 or epoch == epochs:
            loss = mean_squared_error(eval_cp(data, midgame, endgame), data["results"], k)
            print(f"  epoch {epoch:5d}   loss {loss:.6f}")

    return midgame, endgame, k


def _round(value: float) -> int:
    return round(value)


def _format_table(values: list[int]) -> str:
    lines = []

    for rank in range(8):
        chunk = values[rank * 8 : rank * 8 + 8]
        lines.append("        " + ", ".join(f"{v:4d}" for v in chunk) + ",")

    return "[\n" + "\n".join(lines) + "\n    ]"


# emit the fitted numbers as a tables.py, ready to drop in. piece-square tables are a1-first
# with material folded in; the scalars come out split midgame / endgame
def dump_tables(midgame: np.ndarray, endgame: np.ndarray, out: Path | None) -> None:
    names = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
    blocks = [
        '"""Texel-tuned evaluation weights. Generated by tools/tune.py; do not hand-edit."""',
        "",
        "import chess",
        "",
    ]

    for label, weights in (("MIDGAME", midgame), ("ENDGAME", endgame)):
        blocks.append(f"{label}_TABLE: dict[chess.PieceType, list[int]] = {{")

        for piece_type in range(1, 7):
            row = [_round(weights[pst_index(piece_type, square)]) for square in range(64)]
            blocks.append(f"    {piece_type}: {_format_table(row)},  # {names[piece_type]}")

        blocks.append("}")
        blocks.append("")

    def scalar(index: int) -> tuple[int, int]:
        return _round(midgame[index]), _round(endgame[index])

    bishop_mg, bishop_eg = scalar(MOBILITY_INDEX[chess.BISHOP])
    rook_mg, rook_eg = scalar(MOBILITY_INDEX[chess.ROOK])
    queen_mg, queen_eg = scalar(MOBILITY_INDEX[chess.QUEEN])
    king_mg, king_eg = scalar(KING_EXPOSURE_INDEX)
    tempo_mg, tempo_eg = scalar(TEMPO_INDEX)
    doubled_mg, doubled_eg = scalar(DOUBLED_INDEX)

    blocks += [
        "MOBILITY_WEIGHT_MG: dict[chess.PieceType, int] = {",
        f"    chess.BISHOP: {bishop_mg}, chess.ROOK: {rook_mg}, chess.QUEEN: {queen_mg},",
        "}",
        "MOBILITY_WEIGHT_EG: dict[chess.PieceType, int] = {",
        f"    chess.BISHOP: {bishop_eg}, chess.ROOK: {rook_eg}, chess.QUEEN: {queen_eg},",
        "}",
        "",
        f"KING_EXPOSURE_MG = {king_mg}",
        f"KING_EXPOSURE_EG = {king_eg}",
        f"TEMPO_MG = {tempo_mg}",
        f"TEMPO_EG = {tempo_eg}",
        f"DOUBLED_PAWN_MG = {doubled_mg}",
        f"DOUBLED_PAWN_EG = {doubled_eg}",
        "",
    ]

    text = "\n".join(blocks)

    if out is None:
        print(text)
    else:
        out.write_text(text)
        print(f"wrote {out}")


# opening / middlegame / endgame / black to move - enough spread to catch a sign or mirror bug
_SELFCHECK_FENS: list[str] = [
    chess.STARTING_FEN,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N1BP2/PP1QN1PP/R3KB1R w KQ - 0 10",
    "8/5k2/8/8/2P5/8/5K2/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "8/2k5/2p5/2P5/8/8/6K1/8 b - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]


# prove coefficients() plus initial_weights() reproduce agent.evaluate exactly. a mismatch
# means the extraction has drifted from the eval and any tuning would fit the wrong model
def selfcheck() -> int:
    midgame, endgame = initial_weights()
    mismatches = 0

    for fen in _SELFCHECK_FENS:
        board = chess.Board(fen)
        counts, phase = coefficients(board)

        midgame_total = sum(count * midgame[index] for index, count in counts.items())
        endgame_total = sum(count * endgame[index] for index, count in counts.items())
        mine = int(np.floor((phase * midgame_total + (24 - phase) * endgame_total) / 24.0))
        reference = agent.evaluate(board, chess.WHITE)

        if mine != reference:
            mismatches += 1

        flag = "ok" if mine == reference else "MISMATCH"
        print(f"  {flag:8s} mine {mine:6d}   evaluate {reference:6d}   {fen}")
        
    print("selfcheck passed" if not mismatches else f"{mismatches} mismatch(es)")
    return mismatches


# usage:
#     uv run python tools/tune.py --selfcheck
#     uv run python tools/tune.py tools/sources.csv --out tools/tuned_tables.py
#
# --selfcheck verifies the extraction against agent.evaluate and exits. with a sources file it
# builds (and caches) the dataset, fits the weights, and prints or writes a tables.py
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="?", type=Path, help="sources.csv listing the data files")
    parser.add_argument("--selfcheck", action="store_true", help="verify the extraction, then exit")
    parser.add_argument("--cache", type=Path, default=ROOT / "tools" / "tune_cache.npz")
    parser.add_argument("--rebuild", action="store_true", help="ignore any cached dataset")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="write tables.py here instead of stdout")
    args = parser.parse_args()

    if args.selfcheck:
        raise SystemExit(1 if selfcheck() else 0)

    if args.sources is None:
        parser.error("pass sources.csv, or --selfcheck")

    if args.cache.exists() and not args.rebuild:
        print(f"loading cached dataset {args.cache}")
        loaded = np.load(args.cache)
        data = {key: loaded[key] for key in loaded.files}
    else:
        print("building dataset")
        started = time.monotonic()
        data = build_dataset(load_sources(args.sources))
        np.savez(args.cache, **data)
        print(f"  {data['phases'].shape[0]} positions in {time.monotonic() - started:.1f}s")

    midgame, endgame, _ = tune(data, args.epochs, args.lr)
    dump_tables(midgame, endgame, args.out)


if __name__ == "__main__":
    main()
