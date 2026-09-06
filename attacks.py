"""Attack generation: which squares a piece on `square` could move to or capture on, given
what is occupied. Pawn / knight / king attacks never depend on occupancy, so they are
precomputed once per square at import; sliding-piece attacks do depend on occupancy, so they
are computed on the fly by stepping outward one square at a time and stopping at the first
blocker (included, since a blocker can be captured) - the classical approach, not magic
bitboards. It is slower, but a lot harder to get subtly wrong, and correctness comes first
here (see docs/numba-rewrite.md).

This is the bitboard layer's stand-in for python-chess's attack tables (`chess.BB_DIAG_ATTACKS`
and friends, `chess.Board.attacks_mask`): bishop / rook / queen attacks - the most call-heavy
part of move generation - are numba-jitted. `ray_attacks_reference` / `bishop_attacks_reference`
/ `rook_attacks_reference` are the plain-Python originals, kept only so tools/verify_attacks.py
can check the jitted versions against them; nothing else calls them any more.
"""

import numba as nb
import numpy as np
from numba import njit

from bitboard import (
    BLACK,
    FULL_BB,
    NOT_FILE_A,
    NOT_FILE_H,
    WHITE,
    bit,
    square_file,
    square_rank,
)

# each direction as (file_step, rank_step); first four are bishop-like, last four rook-like
BISHOP_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ROOK_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))


# The squares a pawn of `colour` on `square` attacks - the two forward diagonals, off-board
# files dropped. Built for every square, so the white table is masked back to 64 bits: a pawn
# on rank 8 never attacks from there, but bit 63 << 9 would overflow, and a numpy uint64 slot
# for a jitted caller cannot hold an out-of-range value.
def mask_pawn_attacks(colour: int, square: int) -> int:
    origin = bit(square)

    if colour == WHITE:
        return ((origin & NOT_FILE_A) << 7 | (origin & NOT_FILE_H) << 9) & FULL_BB
    
    return ((origin & NOT_FILE_H) >> 7) | ((origin & NOT_FILE_A) >> 9)


# The squares a knight on `square` reaches - each of the eight L-steps that stays on the board.
def mask_knight_attacks(square: int) -> int:
    attacks = 0
    file, rank = square_file(square), square_rank(square)

    for file_step, rank_step in KNIGHT_STEPS:
        next_file, next_rank = file + file_step, rank + rank_step

        if 0 <= next_file < 8 and 0 <= next_rank < 8:
            attacks |= bit(next_rank * 8 + next_file)

    return attacks


# The squares a king on `square` reaches - the eight neighbours that stay on the board.
def mask_king_attacks(square: int) -> int:
    attacks = 0
    file, rank = square_file(square), square_rank(square)

    for file_step in (-1, 0, 1):
        for rank_step in (-1, 0, 1):
            if file_step == 0 and rank_step == 0:
                continue

            next_file, next_rank = file + file_step, rank + rank_step

            if 0 <= next_file < 8 and 0 <= next_rank < 8:
                attacks |= bit(next_rank * 8 + next_file)

    return attacks


# Plain-Python sliding attacks: walk each direction from `square` one step at a time, adding
# every square passed and stopping on (but including) the first occupied one. bishop_attacks /
# rook_attacks below are the jitted twins verify_attacks.py checks against this.
def ray_attacks_reference(
    square: int, directions: tuple[tuple[int, int], ...], occupied: int
) -> int:
    
    attacks = 0
    start_file, start_rank = square_file(square), square_rank(square)

    for file_step, rank_step in directions:
        file, rank = start_file + file_step, start_rank + rank_step

        while 0 <= file < 8 and 0 <= rank < 8:
            target = rank * 8 + file
            attacks |= bit(target)

            if occupied & bit(target):
                break  # blocker: as far as the ray goes, but still a legal target

            file, rank = file + file_step, rank + rank_step
            
    return attacks


def bishop_attacks_reference(square: int, occupied: int) -> int:
    return ray_attacks_reference(square, BISHOP_DIRECTIONS, occupied)


def rook_attacks_reference(square: int, occupied: int) -> int:
    return ray_attacks_reference(square, ROOK_DIRECTIONS, occupied)


PAWN_ATTACKS = [[mask_pawn_attacks(colour, sq) for sq in range(64)] for colour in (WHITE, BLACK)]
KNIGHT_ATTACKS = [mask_knight_attacks(sq) for sq in range(64)]
KING_ATTACKS = [mask_king_attacks(sq) for sq in range(64)]

# numpy-array copies of the three tables above, for jitted callers - a plain Python list is not
# something numba's nopython mode can use as a captured global (it is untyped as far as numba is
# concerned), but a numpy array is. Filled one element at a time rather than via
# np.array(nested_list, dtype=np.uint64): numpy infers a signed intermediate type from a nested
# Python list first and only casts to the requested dtype after, and a bitboard with bit 63 set
# (any position with a piece on the a8-h8 rank) overflows that intermediate type.
PAWN_ATTACKS_NB = np.zeros((2, 64), dtype=np.uint64)
KNIGHT_ATTACKS_NB = np.zeros(64, dtype=np.uint64)
KING_ATTACKS_NB = np.zeros(64, dtype=np.uint64)
for colour in (WHITE, BLACK):
    for square in range(64):
        PAWN_ATTACKS_NB[colour, square] = np.uint64(PAWN_ATTACKS[colour][square])
for square in range(64):
    KNIGHT_ATTACKS_NB[square] = np.uint64(KNIGHT_ATTACKS[square])
    KING_ATTACKS_NB[square] = np.uint64(KING_ATTACKS[square])


# Same rule as ray_attacks_reference above, typed for nopython mode: `square` and `occupied`
# have to be explicitly unsigned (nb.uint64), or a bitboard with bit 63 set - any position with
# a piece on a8 - would be read as a negative number and every shift / compare on it would be
# wrong.
# ref: https://numba.readthedocs.io/en/stable/reference/types.html
@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def bishop_attacks(square: int, occupied: int) -> int:
    attacks = nb.uint64(0)
    start_file = square & 7
    start_rank = square >> 3

    for file_step, rank_step in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        file, rank = start_file + file_step, start_rank + rank_step

        while 0 <= file < 8 and 0 <= rank < 8:
            target_bit = nb.uint64(1) << nb.uint8(rank * 8 + file)
            attacks |= target_bit

            if occupied & target_bit:
                break

            file, rank = file + file_step, rank + rank_step

    return attacks  # type: ignore[return-value]  # numba unboxes uint64 back to plain int


@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def rook_attacks(square: int, occupied: int) -> int:
    attacks = nb.uint64(0)
    start_file = square & 7
    start_rank = square >> 3

    for file_step, rank_step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        file, rank = start_file + file_step, start_rank + rank_step

        while 0 <= file < 8 and 0 <= rank < 8:
            target_bit = nb.uint64(1) << nb.uint8(rank * 8 + file)
            attacks |= target_bit

            if occupied & target_bit:
                break

            file, rank = file + file_step, rank + rank_step
            
    return attacks  # type: ignore[return-value]  # numba unboxes uint64 back to plain int


# A queen reaches everything a bishop and a rook on the same square would.
@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def queen_attacks(square: int, occupied: int) -> int:
    return bishop_attacks(square, occupied) | rook_attacks(square, occupied)


# compile now, inside the init budget, not on the first real call
def warm_up() -> None:
    bishop_attacks(0, 0)
    rook_attacks(0, 0)
    queen_attacks(0, 0)


warm_up()
