"""Offline tuner - fits the evaluation weights to engine scores (or game results) and writes
tables.py. See main()'s comment for the objectives and the regularisation knobs."""

import argparse
import importlib
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import chess
import numpy as np

# run from anywhere: put the repo root on the path so the engine modules resolve
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# coefficients() introspects a python-chess board, so it mirrors reference.evaluate (the plain
# engine) - which tools/verify_eval.py holds bit-identical to the jitted agent.evaluate, so a
# fit against one is a fit against the other.
reference = importlib.import_module("reference")
tables = importlib.import_module("tables")

# every eval term is `weight * count`, so the static score is linear in the weights:
#     eval_cp = (phase * sum(midgame[i] * count[i])
#            + (24 - phase) * sum(endgame[i] * count[i])) // 24
# coefficients() reads count[i] (white minus black) off a board; the dataset stacks those into
# a sparse matrix and gradient descent fits the weights to
#     mean((sigmoid(k * eval_cp / 400) - result) ** 2)
# with the game result from white's point of view. only the fitted numbers ship.

# far pawn (0) + pawn..king (1-6): 64 placement squares each (material is its own slot, not
# folded in - PST_PARAMS is placement only), then one material slot per virtual piece type,
# then the other scalar terms, then one PAWN_AHEAD slot per virtual piece type. every slot is
# fitted as a (midgame, endgame) pair, so there are two weight vectors of this length
PST_PARAMS = 7 * 64
MATERIAL_BASE = PST_PARAMS  # + virtual piece type (0-6) -> 7 slots
MOBILITY_INDEX = {chess.BISHOP: MATERIAL_BASE + 7, chess.ROOK: MATERIAL_BASE + 8,
                  chess.QUEEN: MATERIAL_BASE + 9}
KING_EXPOSURE_INDEX = MATERIAL_BASE + 10
TEMPO_INDEX = MATERIAL_BASE + 11
PAWN_AHEAD_BASE = MATERIAL_BASE + 12  # + virtual piece type (0-6) -> 7 slots
# one passed-pawn slot per relative rank (0-7; a pawn only ever occupies 1-6), then the two
# endgame-only king-activity scalars for an advanced passer
PASSED_PAWN_BASE = PAWN_AHEAD_BASE + 7
KING_PASSER_OWN_INDEX = PASSED_PAWN_BASE + 8
KING_PASSER_ENEMY_INDEX = PASSED_PAWN_BASE + 9
PARAM_COUNT = KING_PASSER_ENEMY_INDEX + 1


# flat index into a weight vector for one piece-square entry. `square` is white's view
# (a1 = 0), already mirrored for black by the caller. `virtual_type` is reference.FAR_PAWN (0)
# or a real chess.PieceType (1-6) - the same indexing reference.evaluate uses for the PST.
def pst_index(virtual_type: int, square: int) -> int:
    return virtual_type * 64 + square


# how many times each eval term applies in `board`, white minus black, plus the game phase.
# mirrors reference.evaluate term for term - --selfcheck fails if this drifts
def coefficients(board: chess.Board) -> tuple[dict[int, int], int]:
    counts: dict[int, int] = defaultdict(int)
    occupied = board.occupied

    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]
    king_sq = {colour: board.king(colour) for colour in (chess.WHITE, chess.BLACK)}
    king_file = {
        colour: (chess.square_file(sq) if (sq := king_sq[colour]) is not None else 4)
        for colour in (chess.WHITE, chess.BLACK)
    }
    kings_known = king_sq[chess.WHITE] is not None and king_sq[chess.BLACK] is not None

    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        colour = piece.color
        sign = 1 if colour == chess.WHITE else -1
        table_square = square if colour == chess.WHITE else square ^ 56

        far = piece_type == chess.PAWN and (chess.square_file(square) ^ king_file[colour]) & 4
        virtual_type = reference.FAR_PAWN if far else piece_type
        counts[pst_index(virtual_type, table_square)] += sign      # placement
        counts[MATERIAL_BASE + virtual_type] += sign                # material

        if piece_type in MOBILITY_INDEX:
            reach = chess.popcount(board.attacks_mask(square))
            counts[MOBILITY_INDEX[piece_type]] += sign * reach
        elif piece_type == chess.KING:
            counts[KING_EXPOSURE_INDEX] += sign * reference.slider_scope(square, occupied)

        own_pawns = white_pawns if colour == chess.WHITE else black_pawns
        stacked = reference.pawns_ahead(square, colour, own_pawns)
        counts[PAWN_AHEAD_BASE + virtual_type] += sign * stacked

        # passed pawn + king activity around an advanced one - mirrors reference.evaluate
        if piece_type == chess.PAWN:
            enemy_pawns = black_pawns if colour == chess.WHITE else white_pawns
            if not enemy_pawns & reference.PASSED_MASK[colour][square]:
                rank = chess.square_rank(square)
                relative_rank = rank if colour == chess.WHITE else 7 - rank
                counts[PASSED_PAWN_BASE + relative_rank] += sign

                if relative_rank >= 4 and kings_known:
                    stop = square + 8 if colour == chess.WHITE else square - 8
                    own_dist = chess.square_distance(king_sq[colour], stop)
                    enemy_dist = chess.square_distance(king_sq[not colour], stop)
                    counts[KING_PASSER_OWN_INDEX] += sign * own_dist
                    counts[KING_PASSER_ENEMY_INDEX] += sign * enemy_dist

    counts[TEMPO_INDEX] += 1 if board.turn == chess.WHITE else -1

    return counts, reference.game_phase(board)


# weight vectors seeded from tables.py, so a run refines what the engine already plays. the
# split scalars are already signed the way coefficients() expects (count is white minus
# black, the weight carries the sign), so they copy straight across
def initial_weights() -> tuple[np.ndarray, np.ndarray]:
    midgame = np.zeros(PARAM_COUNT, dtype=np.float64)
    endgame = np.zeros(PARAM_COUNT, dtype=np.float64)

    for virtual_type in range(0, 7):
        for square in range(64):
            midgame[pst_index(virtual_type, square)] = tables.MIDGAME_PST[virtual_type][square]
            endgame[pst_index(virtual_type, square)] = tables.ENDGAME_PST[virtual_type][square]
        midgame[MATERIAL_BASE + virtual_type] = tables.MATERIAL_MG[virtual_type]
        endgame[MATERIAL_BASE + virtual_type] = tables.MATERIAL_EG[virtual_type]

    for piece_type, index in MOBILITY_INDEX.items():
        midgame[index] = tables.MOBILITY_WEIGHT_MG[piece_type]
        endgame[index] = tables.MOBILITY_WEIGHT_EG[piece_type]

    midgame[KING_EXPOSURE_INDEX] = tables.KING_EXPOSURE_MG
    endgame[KING_EXPOSURE_INDEX] = tables.KING_EXPOSURE_EG
    midgame[TEMPO_INDEX] = tables.TEMPO_MG
    endgame[TEMPO_INDEX] = tables.TEMPO_EG

    for virtual_type in range(0, 7):
        midgame[PAWN_AHEAD_BASE + virtual_type] = tables.PAWN_AHEAD_MG.get(virtual_type, 0)
        endgame[PAWN_AHEAD_BASE + virtual_type] = tables.PAWN_AHEAD_EG.get(virtual_type, 0)

    for relative_rank in range(8):
        midgame[PASSED_PAWN_BASE + relative_rank] = tables.PASSED_PAWN_MG[relative_rank]
        endgame[PASSED_PAWN_BASE + relative_rank] = tables.PASSED_PAWN_EG[relative_rank]
    endgame[KING_PASSER_OWN_INDEX] = tables.KING_PASSER_OWN_EG
    endgame[KING_PASSER_ENEMY_INDEX] = tables.KING_PASSER_ENEMY_EG

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
# every (fen, white-relative result, nan) it points at. text sources carry a game result only,
# never an engine score, so the centipawn slot is nan and --target cp is unavailable for them
def load_sources(csv_path: Path) -> Iterator[tuple[str, float, float]]:
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

            yield fen, result, math.nan

            kept += 1
            if limit and kept >= limit:
                break


# white-to-black order of the 12 bitboards label.py packs per position
_BITBOARD_PIECES: list[tuple[int, chess.Color]] = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]


# rebuild a board from the 12 packed bitboards (a1 = bit 0). castling and en passant are
# not stored and do not matter to the static eval
def _board_from_bitboards(bitboards: np.ndarray, white_to_move: bool) -> chess.Board:
    board = chess.Board(None)
    for (piece_type, color), bitboard in zip(_BITBOARD_PIECES, bitboards, strict=True):
        mask = int(bitboard)
        while mask:
            square = (mask & -mask).bit_length() - 1
            board.set_piece_at(square, chess.Piece(piece_type, color))
            mask &= mask - 1
    board.turn = chess.WHITE if white_to_move else chess.BLACK
    return board


# yield (board, white-relative wdl, white-relative centipawns) from a labelled .npz written by
# tools/label.py. the stored wdl and cp are both from the side to move, so they are flipped
# for black to move
def load_npz(path: Path, limit: int = 0) -> Iterator[tuple[chess.Board, float, float]]:
    blob = np.load(path)
    stm, wdl, cp = blob["stm"], blob["wdl"], blob["cp"]
    bitboards = blob["packed"].view("<u8")  # [N, 12]
    count = bitboards.shape[0] if limit <= 0 else min(limit, bitboards.shape[0])

    for index in range(count):
        white_to_move = bool(stm[index])
        sign = 1.0 if white_to_move else -1.0
        wdl_white = float(wdl[index]) if white_to_move else 1.0 - float(wdl[index])
        board = _board_from_bitboards(bitboards[index], white_to_move)
        yield board, wdl_white, sign * float(cp[index])


# turn positions into the sparse system: coo triplets (row, column, count) for the design
# matrix, plus per-position phase, white-relative wdl and white-relative centipawns. cp is nan
# for text sources. positions with the side to move in check are dropped - the static eval is
# not meaningful there
def build_dataset(
    samples: Iterator[tuple[str | chess.Board, float, float]], drop_in_check: bool = True
) -> dict[str, np.ndarray]:
    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    phases: list[int] = []
    results: list[float] = []
    results_cp: list[float] = []

    row = 0
    for position, result, result_cp in samples:
        board = position if isinstance(position, chess.Board) else chess.Board(position)

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
        results_cp.append(result_cp)

        row += 1
        if row % 50_000 == 0:
            print(f"  {row} positions")

    return {
        "rows": np.asarray(rows, dtype=np.int32),
        "columns": np.asarray(columns, dtype=np.int32),
        "values": np.asarray(values, dtype=np.float64),
        "phases": np.asarray(phases, dtype=np.float64),
        "results": np.asarray(results, dtype=np.float64),
        "results_cp": np.asarray(results_cp, dtype=np.float64),
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


Target = Literal["wdl", "cp"]


# per-position (loss, d_loss/d_eval) for the chosen target. "wdl" compares sigmoid(k*eval/400)
# to the game-result probability with a squared error. "cp" compares the eval straight to the
# engine score with a Huber loss: unlike the sigmoid it does not saturate on lopsided
# positions, so the lopsided positions keep contributing a gradient and piece values stay on
# the engine's scale instead of collapsing
def _loss_terms(
    evals: np.ndarray, data: dict[str, np.ndarray], k: float, target: Target, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    if target == "cp":
        residual = evals - data["results_cp"]
        abs_residual = np.abs(residual)
        loss = np.where(
            abs_residual <= delta, 0.5 * residual**2, delta * (abs_residual - 0.5 * delta)
        )
        return loss, np.clip(residual, -delta, delta)

    predicted = _sigmoid(k * evals / 400.0)
    residual = predicted - data["results"]
    d_eval = 2.0 * residual * predicted * (1.0 - predicted) * (k / 400.0)
    return residual**2, d_eval


# mean objective over the positions picked out by `mask` (all of them if None)
def mean_loss(
    evals: np.ndarray,
    data: dict[str, np.ndarray],
    k: float,
    target: Target,
    delta: float,
    mask: np.ndarray | None = None,
) -> float:
    loss, _ = _loss_terms(evals, data, k, target, delta)
    return float(np.mean(loss if mask is None else loss[mask]))


# root-mean-square eval error in centipawns, for a readable progress line under --target cp
def cp_rmse(evals: np.ndarray, data: dict[str, np.ndarray], mask: np.ndarray) -> float:
    residual = (evals - data["results_cp"])[mask]
    return float(np.sqrt(np.mean(residual**2)))


# scan for the k that best fits the current weights, so the sigmoid sits on the eval's scale
# and tuning does not have to inflate the weights to compensate. wide range - a k that lands
# on a scan edge means the weights are absorbing the rest of the scale. cp tuning has no
# sigmoid, so k stays at 1.0
def best_k(
    data: dict[str, np.ndarray],
    midgame: np.ndarray,
    endgame: np.ndarray,
    target: Target,
    mask: np.ndarray | None = None,
) -> float:
    if target == "cp":
        return 1.0
    evals = eval_cp(data, midgame, endgame)
    candidates = np.arange(0.5, 8.0, 0.02)
    losses = [mean_loss(evals, data, float(k), target, 0.0, mask) for k in candidates]

    return float(candidates[int(np.argmin(losses))])


# closed-form gradient of the loss for each weight vector (no autograd). the chain is
# loss -> (sigmoid or identity) -> eval_cp, and d eval_cp / d weight is the phase-scaled
# design matrix, so the transpose products are bincounts again. only positions in `train_mask`
# contribute; `reg` is (midgame, endgame) L2 pulls back towards `seed`, the starting weights -
# the endgame vector usually needs a firmer pull because a self-play set is short on real
# endgames to constrain it. with `tune_scalars` false the six non-PST terms (mobility, king
# exposure, tempo, doubled pawns) are held at their seed: their features are an order of
# magnitude larger than a PST cell's, so a shared learning rate lets them soak up material
def gradient(
    data: dict[str, np.ndarray],
    midgame: np.ndarray,
    endgame: np.ndarray,
    k: float,
    target: Target,
    delta: float,
    train_mask: np.ndarray,
    seed: tuple[np.ndarray, np.ndarray],
    reg: tuple[float, float],
    tune_scalars: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns, values = data["rows"], data["columns"], data["values"]
    phases = data["phases"]
    train_count = int(train_mask.sum())

    evals = eval_cp(data, midgame, endgame)
    _, d_loss_d_eval = _loss_terms(evals, data, k, target, delta)
    d_loss_d_eval = d_loss_d_eval * train_mask / train_count

    midgame_terms = values * (d_loss_d_eval * phases / 24.0)[rows]
    endgame_terms = values * (d_loss_d_eval * (24.0 - phases) / 24.0)[rows]
    midgame_grad = np.bincount(columns, weights=midgame_terms, minlength=PARAM_COUNT)
    endgame_grad = np.bincount(columns, weights=endgame_terms, minlength=PARAM_COUNT)

    reg_mg, reg_eg = reg
    if reg_mg:
        midgame_grad += (2.0 * reg_mg / PARAM_COUNT) * (midgame - seed[0])
    if reg_eg:
        endgame_grad += (2.0 * reg_eg / PARAM_COUNT) * (endgame - seed[1])

    if not tune_scalars:
        midgame_grad[PST_PARAMS:] = 0.0
        endgame_grad[PST_PARAMS:] = 0.0

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
# third. a held-out split drives early stopping: the weights returned are the ones with the
# lowest validation loss, not the last ones, so a long run cannot overfit the seed away.
# returns the fitted weights and the k used
def tune(
    data: dict[str, np.ndarray],
    epochs: int,
    learning_rate: float,
    *,
    target: Target,
    delta: float,
    val_frac: float,
    patience: int,
    reg: tuple[float, float],
    tune_scalars: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    midgame, endgame = initial_weights()
    seed_weights = (midgame.copy(), endgame.copy())

    position_count = data["phases"].shape[0]
    rng = np.random.default_rng(seed)
    is_val = np.zeros(position_count, dtype=bool)
    val_count = int(val_frac * position_count)
    if val_count:
        is_val[rng.choice(position_count, val_count, replace=False)] = True
    train_mask = ~is_val
    val_mask = is_val if val_count else train_mask

    k = best_k(data, midgame, endgame, target, train_mask)

    def report(mask: np.ndarray) -> str:
        evals = eval_cp(data, midgame, endgame)
        objective = mean_loss(evals, data, k, target, delta, mask)
        if target == "cp":
            return f"{objective:.4f} ({cp_rmse(evals, data, mask):.1f}cp)"
        return f"{objective:.6f}"

    print(
        f"target {target}   k = {k:.3f}   reg mg/eg {reg[0]}/{reg[1]}   "
        f"scalars {'tuned' if tune_scalars else 'frozen'}\n"
        f"  start   train {report(train_mask)}   val {report(val_mask)}   "
        f"({int(train_mask.sum())} train / {int(val_mask.sum())} val)"
    )

    midgame_adam = Adam(PARAM_COUNT)
    endgame_adam = Adam(PARAM_COUNT)

    best_val = mean_loss(eval_cp(data, midgame, endgame), data, k, target, delta, val_mask)
    best_weights = (midgame.copy(), endgame.copy())
    stale = 0

    drop_at = epochs * 2 // 3
    for epoch in range(1, epochs + 1):
        rate = learning_rate if epoch < drop_at else learning_rate * 0.1
        midgame_grad, endgame_grad = gradient(
            data, midgame, endgame, k, target, delta, train_mask, seed_weights, reg, tune_scalars
        )
        midgame_adam.step(midgame, midgame_grad, epoch, rate)
        endgame_adam.step(endgame, endgame_grad, epoch, rate)

        if epoch == drop_at and target == "wdl":
            # weights have moved off the seed scale; re-fit k before the fine pass
            k = best_k(data, midgame, endgame, target, train_mask)
            print(f"  epoch {epoch:5d}   k re-fit to {k:.3f}")

        if epoch % 100 == 0 or epoch == epochs:
            val_loss = mean_loss(
                eval_cp(data, midgame, endgame), data, k, target, delta, val_mask
            )
            # a relative threshold, not an absolute one: an absolute 1e-12 against a loss in
            # the tens of thousands is smaller than floating-point noise, so it never actually
            # detects a plateau - every check reads as "improved" and patience never fires.
            improved = val_loss < best_val * (1.0 - 1e-6)
            if improved:
                best_val, stale = val_loss, 0
                best_weights = (midgame.copy(), endgame.copy())
            else:
                stale += 1
            print(
                f"  epoch {epoch:5d}   train {report(train_mask)}   "
                f"val {report(val_mask)}{'  *' if improved else ''}"
            )
            if patience and stale >= patience:
                print(f"  early stop: no val gain for {patience} checks")
                break

    print(f"best val loss {best_val:.6f}")
    return best_weights[0], best_weights[1], k


def _round(value: float) -> int:
    return round(value)


def _format_table(values: list[int]) -> str:
    lines = []

    for rank in range(8):
        chunk = values[rank * 8 : rank * 8 + 8]
        lines.append("        " + ", ".join(f"{v:4d}" for v in chunk) + ",")

    return "[\n" + "\n".join(lines) + "\n    ]"


# emit the fitted numbers as a tables.py, ready to drop in. material and placement are kept
# as separate tables (not folded together), added at lookup time in evaluate(); the
# scalars come out split midgame / endgame
def dump_tables(midgame: np.ndarray, endgame: np.ndarray, out: Path | None) -> None:
    names = {0: "far_pawn", 1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
    blocks = [
        '"""Texel-tuned evaluation weights. Generated by tools/tune.py; do not hand-edit."""',
        "",
        "import chess",
        "",
    ]

    for label, weights in (("MIDGAME_PST", midgame), ("ENDGAME_PST", endgame)):
        blocks.append(f"{label}: dict[int, list[int]] = {{")

        for virtual_type in range(0, 7):
            row = [_round(weights[pst_index(virtual_type, square)]) for square in range(64)]
            blocks.append(f"    {virtual_type}: {_format_table(row)},  # {names[virtual_type]}")

        blocks.append("}")
        blocks.append("")

    for label, weights in (("MATERIAL_MG", midgame), ("MATERIAL_EG", endgame)):
        entries = ", ".join(
            f"{vt}: {_round(weights[MATERIAL_BASE + vt])}" for vt in range(0, 7)
        )
        blocks.append(f"{label}: dict[int, int] = {{{entries}}}")
    blocks.append("")

    def scalar(index: int) -> tuple[int, int]:
        return _round(midgame[index]), _round(endgame[index])

    bishop_mg, bishop_eg = scalar(MOBILITY_INDEX[chess.BISHOP])
    rook_mg, rook_eg = scalar(MOBILITY_INDEX[chess.ROOK])
    queen_mg, queen_eg = scalar(MOBILITY_INDEX[chess.QUEEN])
    king_mg, king_eg = scalar(KING_EXPOSURE_INDEX)
    tempo_mg, tempo_eg = scalar(TEMPO_INDEX)

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
        "",
    ]

    for label, weights in (("PAWN_AHEAD_MG", midgame), ("PAWN_AHEAD_EG", endgame)):
        entries = ", ".join(
            f"{vt}: {_round(weights[PAWN_AHEAD_BASE + vt])}" for vt in range(0, 7)
        )
        blocks.append(f"{label}: dict[int, int] = {{{entries}}}")
    blocks.append("")

    for label, weights in (("PASSED_PAWN_MG", midgame), ("PASSED_PAWN_EG", endgame)):
        entries = ", ".join(str(_round(weights[PASSED_PAWN_BASE + rr])) for rr in range(8))
        blocks.append(f"{label} = [{entries}]")
    blocks += [
        f"KING_PASSER_OWN_EG = {_round(endgame[KING_PASSER_OWN_INDEX])}",
        f"KING_PASSER_ENEMY_EG = {_round(endgame[KING_PASSER_ENEMY_INDEX])}",
        "",
    ]

    text = "\n".join(blocks)

    if out is None:
        print(text)
    else:
        out.write_text(text)
        print(f"wrote {out}")


# opening / middlegame / endgame / black to move - enough spread to catch a sign or mirror bug.
# the last two carry an advanced passed pawn (own king near / enemy king near) so the passed
# and king-activity terms are exercised, not just the material and placement ones.
_SELFCHECK_FENS: list[str] = [
    chess.STARTING_FEN,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "r2q1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N1BP2/PP1QN1PP/R3KB1R w KQ - 0 10",
    "8/5k2/8/8/2P5/8/5K2/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "8/2k5/2p5/2P5/8/8/6K1/8 b - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2P2k2/2K5/8/8/8/8/8 w - - 0 1",
    "8/6k1/8/8/8/1p6/2p5/6K1 b - - 0 1",
]


# prove coefficients() plus initial_weights() reproduce reference.evaluate exactly. a mismatch
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
        want = reference.evaluate(board, chess.WHITE)

        if mine != want:
            mismatches += 1

        flag = "ok" if mine == want else "MISMATCH"
        print(f"  {flag:8s} mine {mine:6d}   evaluate {want:6d}   {fen}")

    print("selfcheck passed" if not mismatches else f"{mismatches} mismatch(es)")
    return mismatches


# usage:
#     uv run python tools/tune.py --selfcheck
#     uv run python tools/tune.py tools/sources.csv --target wdl --out tools/tuned_tables.py
#     uv run python tools/tune.py --npz data/tune.npz --out tools/tuned_tables.py
#
# --selfcheck verifies the extraction against reference.evaluate and exits. otherwise it builds
# (and caches) the dataset from a sources.csv or a labelled --npz, fits the weights, and
# prints or writes a tables.py.
#
# --target cp (the default whenever the dataset carries engine scores) regresses the eval
# straight onto the Stockfish centipawns with a Huber loss. --target wdl fits the old
# sigmoid-to-result objective and is the only option for a text sources.csv. either way a
# --val-frac slice is held out and the weights kept are the ones at the best validation loss.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="?", type=Path, help="sources.csv listing the data files")
    parser.add_argument("--npz", type=Path, help="labelled .npz from tools/label.py")
    parser.add_argument("--limit", type=int, default=0, help="cap --npz positions, 0 = all")
    parser.add_argument("--selfcheck", action="store_true", help="verify the extraction, then exit")
    parser.add_argument("--cache", type=Path, help="dataset cache path (defaults by input)")
    parser.add_argument("--rebuild", action="store_true", help="ignore any cached dataset")
    parser.add_argument("--target", choices=("cp", "wdl"), help="fit objective (cp if present)")
    parser.add_argument("--huber-delta", type=float, default=500.0, help="cp Huber knee, cp")
    parser.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction, early stop")
    parser.add_argument("--patience", type=int, default=8, help="stop after N flat val checks")
    parser.add_argument("--reg", type=float, default=0.2, help="L2 pull toward seed (midgame)")
    parser.add_argument("--reg-eg", type=float, help="endgame L2 pull (default: 4x --reg)")
    parser.add_argument("--tune-scalars", action="store_true", help="also fit the 6 scalar terms")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for the val split")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="write tables.py here instead of stdout")
    args = parser.parse_args()

    if args.selfcheck:
        raise SystemExit(1 if selfcheck() else 0)

    if (args.sources is None) == (args.npz is None):
        parser.error("pass either sources.csv or --npz (and --selfcheck exits before here)")

    source = args.npz if args.npz else args.sources
    limit_suffix = f"_limit{args.limit}" if args.npz and args.limit else ""
    cache = args.cache or ROOT / "tools" / f"tune_cache_{source.stem}{limit_suffix}.npz"

    # keyed on the source file's name, not just "npz vs sources" - two different --npz
    # datasets used to collide on the same tune_cache_npz.npz and silently serve each
    # other's cached (and possibly stale) design matrix. the mtime check catches the same
    # source path being regenerated with new content under an unchanged name.
    fresh = cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime
    if fresh and not args.rebuild:
        print(f"loading cached dataset {cache}")
        loaded = np.load(cache)
        data = {key: loaded[key] for key in loaded.files}
        if "results_cp" not in data:
            parser.error(f"cached {cache.name} predates --target cp; re-run with --rebuild")
    else:
        print("building dataset")
        started = time.monotonic()
        samples: Iterator[tuple[str | chess.Board, float, float]] = (
            load_npz(args.npz, args.limit) if args.npz else load_sources(args.sources)
        )
        data = build_dataset(samples)
        np.savez(cache, **data)
        print(f"  {data['phases'].shape[0]} positions in {time.monotonic() - started:.1f}s")

    has_scores = bool(np.isfinite(data["results_cp"]).all())
    target: Target = args.target or ("cp" if has_scores else "wdl")
    if target == "cp" and not has_scores:
        parser.error("--target cp needs engine scores; this dataset has none (text source)")

    reg_eg = 4.0 * args.reg if args.reg_eg is None else args.reg_eg

    midgame, endgame, _ = tune(
        data,
        args.epochs,
        args.lr,
        target=target,
        delta=args.huber_delta,
        val_frac=args.val_frac,
        patience=args.patience,
        reg=(args.reg, reg_eg),
        tune_scalars=args.tune_scalars,
        seed=args.seed,
    )
    dump_tables(midgame, endgame, args.out)


if __name__ == "__main__":
    main()
