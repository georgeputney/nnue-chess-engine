"""Runtime-side NNUE: load the quantised weights tools/export_nn.py writes, and a plain-numpy
forward used as the correctness oracle for the numba engine and by the training pipeline to
build feature planes. No torch, no numba here - just numpy - so the engine can import it inside
the platform's init budget and the verify tools can call it freely.

net.npz layout (all little-endian):
    ft_weight_t   int16   [768, ft_out]  transformer weights, transposed so one feature's
                                         contribution is a contiguous row (cheap column add)
    ft_bias       int32   [ft_out]       transformer bias, on the accumulator's integer scale
    ft_scale      float32 scalar         accumulator_float = clip(accumulator_int * ft_scale, 0, 1)
    l1_weight     float32 [32, 2*ft_out]  l1_bias float32 [32]   tail (int8 was tried, no numba win)
    l2_weight     float32 [32, 32]        l2_bias float32 [32]
    out_weight    float32 [1, 32]         out_bias float32 [1]
    cp_scale      float32 scalar          centipawns = round(raw_output * cp_scale)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nnue.arch import BLACK, FEATURES, WHITE, black_perspective_perm

# built once: black_plane[:, i] == white_plane[:, PERM[i]]
PERM = np.asarray(black_perspective_perm(), dtype=np.int64)

DEFAULT_PATH = Path(__file__).resolve().parent / "net.npz"


class Weights:
    """The loaded arrays, named. Kept as a small class rather than a dict so the numba agent's
    module-level unpacking reads cleanly."""

    def __init__(self, blob: dict[str, np.ndarray]) -> None:
        self.ft_weight_t = blob["ft_weight_t"].astype(np.int16)
        self.ft_bias = blob["ft_bias"].astype(np.int32)
        self.ft_scale = float(blob["ft_scale"])
        self.l1_weight = blob["l1_weight"].astype(np.float32)
        self.l1_bias = blob["l1_bias"].astype(np.float32)
        self.l2_weight = blob["l2_weight"].astype(np.float32)
        self.l2_bias = blob["l2_bias"].astype(np.float32)
        self.out_weight = blob["out_weight"].astype(np.float32)
        self.out_bias = blob["out_bias"].astype(np.float32)
        self.cp_scale = float(blob["cp_scale"])


def load(path: str | Path = DEFAULT_PATH) -> Weights:
    with np.load(path) as blob:
        return Weights({key: blob[key] for key in blob.files})


# (own_plane, other_plane) float32 [N, 768] from packed bitboards and side-to-move, own view
# first. stm is 1 where white is to move (tools/label.py's convention). This is the one place
# the perspective orientation is implemented; the numba refresh loop mirrors it.
def perspective_planes(packed: np.ndarray, stm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    white_plane = np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32)
    white_plane = white_plane[:, :FEATURES]  # 96 bytes -> 768 bits exactly, guard anyway
    black_plane = white_plane[:, PERM]

    white_to_move = stm.astype(bool)[:, None]
    own = np.where(white_to_move, white_plane, black_plane)
    other = np.where(white_to_move, black_plane, white_plane)
    return own, other


# integer accumulators [N, 2, 256] (index 0 own, 1 other) the way the engine holds them:
# ft_bias plus the int16 transformer columns for every active feature.
def accumulators(packed: np.ndarray, stm: np.ndarray, weights: Weights) -> np.ndarray:
    own, other = perspective_planes(packed, stm)
    ft = weights.ft_weight_t.astype(np.int32)  # [768, 256]
    acc_own = own.astype(np.int32) @ ft + weights.ft_bias
    acc_other = other.astype(np.int32) @ ft + weights.ft_bias
    return np.stack((acc_own, acc_other), axis=1).astype(np.int32)


def clipped_relu(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# centipawns, side-to-move relative, from integer accumulators [N, 2, ft_out]. The same
# computation the numba evaluate_accumulator() runs: dequantise the accumulator, clipped ReLU,
# then the float tail. tools/verify_nnue.py holds the engine to this.
def forward_from_accumulators(acc: np.ndarray, weights: Weights) -> np.ndarray:
    activated = clipped_relu(acc.astype(np.float32) * weights.ft_scale)  # [N, 2, ft_out]
    x = activated.reshape(acc.shape[0], 2 * acc.shape[-1])
    x = clipped_relu(x @ weights.l1_weight.T + weights.l1_bias)
    x = clipped_relu(x @ weights.l2_weight.T + weights.l2_bias)
    raw = x @ weights.out_weight.T + weights.out_bias
    return raw[:, 0] * weights.cp_scale


# convenience: centipawns straight from packed bitboards + stm.
def forward_packed(packed: np.ndarray, stm: np.ndarray, weights: Weights) -> np.ndarray:
    return forward_from_accumulators(accumulators(packed, stm, weights), weights)


# 96-byte packed feature vector from a python-chess board, matching tools/label.py:features so
# a FEN can be scored through this module without pulling label.py in.
def pack_board(board: object) -> np.ndarray:
    import chess  # local: keep module import cheap for the engine

    assert isinstance(board, chess.Board)
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    bitboards = [
        board.pawns & white, board.knights & white, board.bishops & white,
        board.rooks & white, board.queens & white, board.kings & white,
        board.pawns & black, board.knights & black, board.bishops & black,
        board.rooks & black, board.queens & black, board.kings & black,
    ]
    return np.frombuffer(np.asarray(bitboards, dtype="<u8").tobytes(), dtype=np.uint8).copy()


# (packed, stm) for a single FEN, ready for forward_packed.
def encode_fen(fen: str) -> tuple[np.ndarray, np.ndarray]:
    import chess

    board = chess.Board(fen)
    packed = pack_board(board).reshape(1, 96)
    stm = np.array([1 if board.turn == chess.WHITE else 0], dtype=np.uint8)
    return packed, stm


_ = (WHITE, BLACK)  # re-exported for callers that import orientation constants from here
