"""Zobrist hashing for a Board: a 64-bit incremental-style hash used for the transposition
table and for repetition detection, neither of which chess.Board's own machinery (transposition
key, is_repetition) is available for any more once the search stops using chess.Board.

Not incremental yet - zobrist_hash rebuilds the hash from scratch by XORing in a random key per
(colour, piece type, square) plus side/castling/en-passant keys, same approach as
board.recompute_occupancy. A real engine updates the hash move by move instead of
recomputing it every node; that's a speed detail to add once this is on the hot path, not a
correctness one now.

ref: https://www.chessprogramming.org/Zobrist_Hashing
"""

import numpy as np
from numba import njit

from bitboard import NO_SQUARE, lsb_index
from board import Board

_RNG = np.random.default_rng(0xC0FFEE)  # fixed seed: the hash must be the same across runs of
# the same process (and, since games are separate processes, that's all that's ever needed)

PIECE_SQUARE_KEYS = _RNG.integers(0, 2**64, size=(2, 6, 64), dtype=np.uint64)
# indexed directly by the 4-bit castling-rights value (bitboard.WHITE_KINGSIDE etc.) - no
# per-flag decomposition needed, since that value already is the position's whole rights state.
CASTLING_KEYS = _RNG.integers(0, 2**64, size=16, dtype=np.uint64)
EP_FILE_KEYS = _RNG.integers(0, 2**64, size=8, dtype=np.uint64)  # only the file matters
SIDE_KEY = np.uint64(_RNG.integers(0, 2**64, dtype=np.uint64))


@njit(cache=True)
def zobrist_hash(pos: Board) -> int:
    h = np.uint64(0)
    for colour in range(2):
        for pt in range(6):
            bb = pos.pieces[colour, pt]
            while bb:
                sq = lsb_index(bb)
                h ^= PIECE_SQUARE_KEYS[colour, pt, sq]
                bb &= bb - np.uint64(1)

    h ^= CASTLING_KEYS[pos.castling]

    if pos.ep_square != NO_SQUARE:
        h ^= EP_FILE_KEYS[pos.ep_square & 7]

    if pos.side == 1:
        h ^= SIDE_KEY

    return h  # type: ignore[no-any-return]  # numba unboxes uint64 back to plain int


# compile now, inside the init budget
zobrist_hash(Board())
