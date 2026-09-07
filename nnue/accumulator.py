"""The NNUE evaluation, @njit end to end - the drop-in replacement for the linear evaluate().

The feature transformer's int16 weights and int32 bias are loaded from nnue/net.npz at import;
the tail (2*ft_out -> 32 -> 32 -> 1) is float32 (int8 was tried and bought nothing in numba:
the float MAC loop already vectorises, and on this hardware there is no wider int8 path). An
accumulator is one int32 vector per colour: FT_BIAS plus the transformer column of every piece
feature that colour's perspective sees. fill_accumulator builds one from scratch; update_feature
adds or removes a single piece's columns, which is what nnue/movegen.make_move calls so a
child's accumulator costs a handful of column updates rather than a full rebuild.

Everything here takes plain numpy arrays (an accumulator, a piece-bitboard table), never a
Board, so it has no import cycle with nnue/board.py and numba can cache it. See nnue/arch.py
for the feature layout the index maths below implements.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from bitboard import lsb_index
from nnue.net import load

WEIGHTS = load()
FT_WEIGHT_T = WEIGHTS.ft_weight_t   # int16 [768, ft_out], transposed: one feature is a row
FT_BIAS = WEIGHTS.ft_bias           # int32 [ft_out]
FT_SCALE = float(WEIGHTS.ft_scale)  # accumulator_float = clip(accumulator_int * FT_SCALE, 0, 1)
L1_WEIGHT = WEIGHTS.l1_weight       # float32 [32, 2*ft_out]
L1_BIAS = WEIGHTS.l1_bias           # float32 [32]
L2_WEIGHT = WEIGHTS.l2_weight       # float32 [32, 32]
L2_BIAS = WEIGHTS.l2_bias           # float32 [32]
OUT_WEIGHT = WEIGHTS.out_weight     # float32 [out_buckets, 32]
OUT_BIAS = WEIGHTS.out_bias         # float32 [out_buckets]
CP_SCALE = float(WEIGHTS.cp_scale)  # centipawns = raw_output * CP_SCALE

ACC_WIDTH = int(FT_BIAS.shape[0])   # ft_out
OUT_BUCKETS = int(OUT_BIAS.shape[0])  # final-layer heads, one per piece-count band
HALF = 384                          # one perspective-colour block: 6 piece types * 64 squares


# Set-bit count, SWAR - bitboard.popcount is plain-Python (bb.bit_count()), so the jitted eval
# path needs its own. Mirrors agent.popcount.
# ref: https://www.chessprogramming.org/Population_Count#SWAR-Popcount
@njit(cache=True)
def popcount(bb: int) -> int:
    two = np.uint64(0x3333_3333_3333_3333)
    x = np.uint64(bb)
    x -= (x >> np.uint64(1)) & np.uint64(0x5555_5555_5555_5555)
    x = (x & two) + ((x >> np.uint64(2)) & two)
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F_0F0F_0F0F_0F0F)
    return (x * np.uint64(0x0101_0101_0101_0101)) >> np.uint64(56)


# Flat transformer-input index for a piece of `piece_colour` (0 white, 1 black) of `piece_type`
# (0..5) on `square` (a1 = 0), seen from `perspective` (0 white-to-move's own view, 1 black's).
# Mirrors nnue.arch.feature_index and nnue.net's vectorised permutation.
@njit(cache=True)
def feature_index(perspective: int, piece_colour: int, piece_type: int, square: int) -> int:
    relative_colour = 0 if piece_colour == perspective else 1
    relative_square = square if perspective == 0 else square ^ 56
    return relative_colour * HALF + piece_type * 64 + relative_square


# Rebuild both colours' accumulators in `acc` (int32 [2, ft_out]) from the piece bitboards
# `pieces` (uint64 [2, 6]). Used by parse_fen and as the oracle the incremental path is
# checked against.
@njit(cache=True)
def fill_accumulator(pieces: np.ndarray, acc: np.ndarray) -> None:
    for perspective in range(2):
        for i in range(ACC_WIDTH):
            acc[perspective, i] = FT_BIAS[i]

    for piece_colour in range(2):
        for piece_type in range(6):
            bb = pieces[piece_colour, piece_type]
            while bb:
                square = lsb_index(bb)
                bb &= bb - np.uint64(1)
                for perspective in range(2):
                    row = feature_index(perspective, piece_colour, piece_type, square)
                    for i in range(ACC_WIDTH):
                        acc[perspective, i] += FT_WEIGHT_T[row, i]


# Add (`sign` +1) or remove (`sign` -1) one piece's transformer columns from both perspectives
# of `acc`, in place. make_move calls this once per square that a piece enters or leaves - the
# per-node hot path. A scalar loop with fastmath beat every array-expression form tried here
# (numba materialises a promoted int32 temporary for `acc[p, :] += int16_row`).
@njit(cache=True, fastmath=True)
def update_feature(
    acc: np.ndarray, piece_colour: int, piece_type: int, square: int, sign: int
) -> None:
    for perspective in range(2):
        row = feature_index(perspective, piece_colour, piece_type, square)
        if sign > 0:
            for i in range(ACC_WIDTH):
                acc[perspective, i] += FT_WEIGHT_T[row, i]
        else:
            for i in range(ACC_WIDTH):
                acc[perspective, i] -= FT_WEIGHT_T[row, i]


# Centipawns, side-to-move relative, from the accumulators `acc` (int32 [2, ft_out]) with
# `side` to move and `piece_count` men on the board (picks the output-bucket head). Dequantise
# both perspectives (own first) to [0, 1] with a clipped ReLU, then the float tail. This runs at
# every search leaf - the hot path. fastmath lets numba vectorise the MAC loops.
@njit(cache=True, fastmath=True)
def evaluate_accumulator(acc: np.ndarray, side: int, piece_count: int) -> int:
    own = side
    other = 1 - side

    bucket = (piece_count - 2) // 4  # nnue.arch.output_bucket, rederived
    if bucket < 0:
        bucket = 0
    elif bucket >= OUT_BUCKETS:
        bucket = OUT_BUCKETS - 1

    activated = np.empty(2 * ACC_WIDTH, dtype=np.float32)
    for i in range(ACC_WIDTH):
        v = acc[own, i] * FT_SCALE
        activated[i] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        w = acc[other, i] * FT_SCALE
        activated[ACC_WIDTH + i] = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

    hidden1 = np.empty(32, dtype=np.float32)
    for j in range(32):
        total = L1_BIAS[j]
        for i in range(2 * ACC_WIDTH):
            total += L1_WEIGHT[j, i] * activated[i]
        hidden1[j] = 0.0 if total < 0.0 else (1.0 if total > 1.0 else total)

    hidden2 = np.empty(32, dtype=np.float32)
    for j in range(32):
        total = L2_BIAS[j]
        for i in range(32):
            total += L2_WEIGHT[j, i] * hidden1[i]
        hidden2[j] = 0.0 if total < 0.0 else (1.0 if total > 1.0 else total)

    raw = OUT_BIAS[bucket]
    for i in range(32):
        raw += OUT_WEIGHT[bucket, i] * hidden2[i]

    return np.int64(round(raw * CP_SCALE))


# compile every jitted function here inside the platform's init budget
def warm_up() -> None:
    pieces = np.zeros((2, 6), dtype=np.uint64)
    pieces[0, 0] = np.uint64(0x000000000000FF00)  # a couple of pawns so the fill loop runs
    pieces[1, 0] = np.uint64(0x00FF000000000000)
    acc = np.zeros((2, ACC_WIDTH), dtype=np.int32)
    fill_accumulator(pieces, acc)
    update_feature(acc, 0, 0, 8, 1)
    update_feature(acc, 0, 0, 8, -1)
    evaluate_accumulator(acc, 0, 4)
    evaluate_accumulator(acc, 1, 24)
    popcount(np.uint64(0xFF00))


warm_up()
