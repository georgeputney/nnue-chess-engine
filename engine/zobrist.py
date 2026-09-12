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

from engine.bitboard import NO_SQUARE, lsb_index
from engine.board import Board

RNG = np.random.default_rng(0xC0FFEE)  # fixed seed: the hash must be the same across runs of
# the same process (and, since games are separate processes, that's all that's ever needed)

PIECE_SQUARE_KEYS = RNG.integers(0, 2**64, size=(2, 6, 64), dtype=np.uint64)
# indexed directly by the 4-bit castling-rights value (bitboard.WHITE_KINGSIDE etc.) - no
# per-flag decomposition needed, since that value already is the position's whole rights state.
CASTLING_KEYS = RNG.integers(0, 2**64, size=16, dtype=np.uint64)
EP_FILE_KEYS = RNG.integers(0, 2**64, size=8, dtype=np.uint64)  # only the file matters
SIDE_KEY = np.uint64(RNG.integers(0, 2**64, dtype=np.uint64))


# The 64-bit hash of a position - chess.polyglot.zobrist_hash, on a Board. Rebuilt from scratch
# every call: XOR one random key per (colour, piece type, square) actually occupied, then the
# castling / en-passant / side keys. make_move keeps an incremental copy in step for the hot path
# (verify_zobrist checks the two match); this is the ground truth they are checked against.
# ref: https://www.chessprogramming.org/Zobrist_Hashing
@njit(cache=True)
def zobrist_hash(board: Board) -> int:
    key = np.uint64(0)

    for colour in range(2):
        for piece_type in range(6):
            bitboard = board.pieces[colour, piece_type]
            
            while bitboard:  # walk the set bits, clearing the lowest each pass
                square = lsb_index(bitboard)
                key ^= PIECE_SQUARE_KEYS[colour, piece_type, square]
                bitboard &= bitboard - np.uint64(1)

    # one key per whole 4-bit rights value, so no per-flag mixing is needed
    key ^= CASTLING_KEYS[board.castling]

    # only the file of the en-passant target matters to the hash, never the rank
    if board.ep_square != NO_SQUARE:
        key ^= EP_FILE_KEYS[board.ep_square & 7]

    # side to move: mixed in for black only, so the two turns of one position hash differently
    if board.side == 1:
        key ^= SIDE_KEY

    return key  # type: ignore[no-any-return]  # numba unboxes uint64 back to plain int


# compile now, inside the init budget, not on the first real call
def warm_up() -> None:
    zobrist_hash(Board())


warm_up()
