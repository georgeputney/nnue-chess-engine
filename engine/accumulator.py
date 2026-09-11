"""The NNUE evaluation, @njit end to end - the drop-in replacement for the linear evaluate().

The feature transformer's int16 weights and int32 bias are loaded from engine/net.npz at import;
the tail (2*ft_out -> 32 -> 32 -> 1) is float32 (int8 was tried and bought nothing in numba:
the float MAC loop already vectorises, and on this hardware there is no wider int8 path). An
accumulator is one int32 vector per colour: FT_BIAS plus the transformer column of every piece
feature that colour's perspective sees. fill_accumulator builds one from scratch; update_feature
adds or removes a single piece's columns, which is what engine/movegen.make_move calls so a
child's accumulator costs a handful of column updates rather than a full rebuild.

Everything here takes plain numpy arrays (an accumulator, a piece-bitboard table), never a
Board, so it has no import cycle with engine/board.py and numba can cache it. See engine/arch.py
for the feature layout the index maths below implements.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from engine.bitboard import lsb_index
from engine.net import EG_PATH, load

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

# The endgame net: the same architecture trained on the <= EG_MEN slice of the corpus alone
# (tools/filter_shards.py), used for every position with that few men. The post-mortem of the
# rated games (docs/nnue-plan.md) put every endgame loss in the 8-12 men band, where the shared
# net scores drawn and won positions alike. A second net cannot touch the middlegame, so the A/B
# isolates the band. With no engine/net_eg.npz beside this module the endgame net is the main net
# and the engine is bit-identical to a single-net build (tools/nodebench.py proves that).
# numba freezes these arrays into its on-disk cache (cache=True) and does not notice when the
# file behind them changes: after swapping either net, delete the .nbi/.nbc files in
# engine/__pycache__ (a fresh bundle dir has none) or tools/verify_nnue.py fails its oracle check.
EG_MEN = 12
EG_WEIGHTS = load(EG_PATH) if EG_PATH.exists() else WEIGHTS
EG_FT_WEIGHT_T = EG_WEIGHTS.ft_weight_t
EG_FT_BIAS = EG_WEIGHTS.ft_bias
EG_FT_SCALE = float(EG_WEIGHTS.ft_scale)
EG_L1_WEIGHT = EG_WEIGHTS.l1_weight
EG_L1_BIAS = EG_WEIGHTS.l1_bias
EG_L2_WEIGHT = EG_WEIGHTS.l2_weight
EG_L2_BIAS = EG_WEIGHTS.l2_bias
EG_OUT_WEIGHT = EG_WEIGHTS.out_weight
EG_OUT_BIAS = EG_WEIGHTS.out_bias
EG_CP_SCALE = float(EG_WEIGHTS.cp_scale)
if EG_FT_BIAS.shape[0] != ACC_WIDTH or EG_OUT_BIAS.shape[0] != OUT_BUCKETS:
    raise ValueError("engine/net_eg.npz must have the same width and output buckets as net.npz")


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


# Men on the board from the piece-bitboard table `pieces` (uint64 [2, 6]) - which net a
# position belongs to.
@njit(cache=True)
def men_of(pieces: np.ndarray) -> int:
    men = 0
    for piece_colour in range(2):
        for piece_type in range(6):
            men += popcount(pieces[piece_colour, piece_type])
    return men


# Flat transformer-input index for a piece of `piece_colour` (0 white, 1 black) of `piece_type`
# (0..5) on `square` (a1 = 0), seen from `perspective` (0 white-to-move's own view, 1 black's).
# Mirrors engine.arch.feature_index and engine.net's vectorised permutation.
@njit(cache=True)
def feature_index(perspective: int, piece_colour: int, piece_type: int, square: int) -> int:
    relative_colour = 0 if piece_colour == perspective else 1
    relative_square = square if perspective == 0 else square ^ 56
    return relative_colour * HALF + piece_type * 64 + relative_square


# Rebuild both colours' accumulators in `acc` (int32 [2, ft_out]) from the piece bitboards
# `pieces` (uint64 [2, 6]), with the net the men count picks. Used by parse_fen, by make_move
# when a capture crosses into the endgame net, and as the oracle the incremental path is
# checked against.
@njit(cache=True)
def fill_accumulator(pieces: np.ndarray, acc: np.ndarray) -> None:
    eg = men_of(pieces) <= EG_MEN
    for perspective in range(2):
        for i in range(ACC_WIDTH):
            acc[perspective, i] = EG_FT_BIAS[i] if eg else FT_BIAS[i]

    for piece_colour in range(2):
        for piece_type in range(6):
            bb = pieces[piece_colour, piece_type]
            while bb:
                square = lsb_index(bb)
                bb &= bb - np.uint64(1)
                update_feature(acc, piece_colour, piece_type, square, 1, eg)


# Add (`sign` +1) or remove (`sign` -1) one piece's transformer columns from both perspectives
# of `acc`, in place, from the endgame net's table when `eg` is set. make_move calls this once
# per square that a piece enters or leaves - the per-node hot path. A scalar loop with fastmath
# beat every array-expression form tried here (numba materialises a promoted int32 temporary
# for `acc[p, :] += int16_row`).
@njit(cache=True, fastmath=True)
def update_feature(
    acc: np.ndarray, piece_colour: int, piece_type: int, square: int, sign: int, eg: bool
) -> None:
    for perspective in range(2):
        row = feature_index(perspective, piece_colour, piece_type, square)
        if eg:
            if sign > 0:
                for i in range(ACC_WIDTH):
                    acc[perspective, i] += EG_FT_WEIGHT_T[row, i]
            else:
                for i in range(ACC_WIDTH):
                    acc[perspective, i] -= EG_FT_WEIGHT_T[row, i]
        elif sign > 0:
            for i in range(ACC_WIDTH):
                acc[perspective, i] += FT_WEIGHT_T[row, i]
        else:
            for i in range(ACC_WIDTH):
                acc[perspective, i] -= FT_WEIGHT_T[row, i]


# The float tail over an activated accumulator pair: 2*ft_out -> 32 -> 32 -> the output head
# `bucket`. Takes the weight set as arguments so the main and endgame nets share one compiled
# body; fastmath lets numba vectorise the MAC loops.
@njit(cache=True, fastmath=True)
def tail(
    activated: np.ndarray, bucket: int,
    l1_weight: np.ndarray, l1_bias: np.ndarray, l2_weight: np.ndarray, l2_bias: np.ndarray,
    out_weight: np.ndarray, out_bias: np.ndarray, cp_scale: float,
) -> int:
    hidden1 = np.empty(32, dtype=np.float32)
    for j in range(32):
        total = l1_bias[j]
        for i in range(2 * ACC_WIDTH):
            total += l1_weight[j, i] * activated[i]
        hidden1[j] = 0.0 if total < 0.0 else (1.0 if total > 1.0 else total)

    hidden2 = np.empty(32, dtype=np.float32)
    for j in range(32):
        total = l2_bias[j]
        for i in range(32):
            total += l2_weight[j, i] * hidden1[i]
        hidden2[j] = 0.0 if total < 0.0 else (1.0 if total > 1.0 else total)

    raw = out_bias[bucket]
    for i in range(32):
        raw += out_weight[bucket, i] * hidden2[i]

    return np.int64(round(raw * cp_scale))


# Centipawns, side-to-move relative, from the accumulators `acc` (int32 [2, ft_out]) with
# `side` to move and `piece_count` men on the board (picks the net, then the output-bucket
# head). Dequantise both perspectives (own first) to [0, 1] with a clipped ReLU, then the float
# tail. This runs at every search leaf - the hot path. The accumulator was built with the net
# `piece_count` picks (make_move / fill_accumulator keep the two in step).
@njit(cache=True, fastmath=True)
def evaluate_accumulator(acc: np.ndarray, side: int, piece_count: int) -> int:
    own = side
    other = 1 - side

    bucket = (piece_count - 2) // 4  # engine.arch.output_bucket, rederived
    if bucket < 0:
        bucket = 0
    elif bucket >= OUT_BUCKETS:
        bucket = OUT_BUCKETS - 1

    eg = piece_count <= EG_MEN
    scale = EG_FT_SCALE if eg else FT_SCALE
    activated = np.empty(2 * ACC_WIDTH, dtype=np.float32)
    for i in range(ACC_WIDTH):
        v = acc[own, i] * scale
        activated[i] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
        w = acc[other, i] * scale
        activated[ACC_WIDTH + i] = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

    if eg:
        return tail(activated, bucket, EG_L1_WEIGHT, EG_L1_BIAS, EG_L2_WEIGHT, EG_L2_BIAS,
                    EG_OUT_WEIGHT, EG_OUT_BIAS, EG_CP_SCALE)
    return tail(activated, bucket, L1_WEIGHT, L1_BIAS, L2_WEIGHT, L2_BIAS,
                OUT_WEIGHT, OUT_BIAS, CP_SCALE)


# compile every jitted function here inside the platform's init budget - both nets' paths
def warm_up() -> None:
    pieces = np.zeros((2, 6), dtype=np.uint64)
    pieces[0, 0] = np.uint64(0x000000000000FF00)  # sixteen pawns: the main net's fill
    pieces[1, 0] = np.uint64(0x00FF000000000000)
    acc = np.zeros((2, ACC_WIDTH), dtype=np.int32)
    fill_accumulator(pieces, acc)
    update_feature(acc, 0, 0, 8, 1, False)
    update_feature(acc, 0, 0, 8, -1, False)
    evaluate_accumulator(acc, 1, 24)
    pieces[1, 0] = np.uint64(0)  # eight men: the endgame net's fill
    fill_accumulator(pieces, acc)
    update_feature(acc, 0, 0, 8, 1, True)
    update_feature(acc, 0, 0, 8, -1, True)
    evaluate_accumulator(acc, 0, 4)
    men_of(pieces)
    popcount(np.uint64(0xFF00))


warm_up()
