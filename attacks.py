"""Attack generation: which squares a piece on `sq` could move to or capture on, given what's
occupied. Pawn/knight/king attacks never depend on occupancy, so they're precomputed once per
square at import; sliding-piece attacks do depend on occupancy, so they're computed on the fly
by stepping outward one square at a time and stopping at the first blocker (included, since a
blocker can be captured) - the classical approach, not magic bitboards. It's slower, but a lot
harder to get subtly wrong, and correctness comes first here (see docs/numba-rewrite.md).

Phase 2 starts here: bishop/rook/queen attacks - the most call-heavy part of move generation -
are now numba-jitted. `_bishop_attacks_reference` / `_rook_attacks_reference` are the plain-
Python originals, kept only so tools/verify_attacks.py can check the jitted versions against
them; nothing else calls them any more.
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


def _mask_pawn_attacks(colour: int, sq: int) -> int:
    bb = bit(sq)
    if colour == WHITE:
        # masked to 64 bits: a pawn on rank 8 (sq >= 56) never really attacks from there, but
        # this table is precomputed for every square regardless, and left-shifting a high bit
        # (e.g. bit 63 << 9) overflows 64 bits. Harmless as an unbounded Python int - every use
        # ANDs it with a real, bounded bitboard anyway - but not a valid bitboard on its own,
        # and it has to be one to fit in a fixed-width numpy array for a jitted caller.
        return ((bb & NOT_FILE_A) << 7 | (bb & NOT_FILE_H) << 9) & FULL_BB
    return ((bb & NOT_FILE_H) >> 7) | ((bb & NOT_FILE_A) >> 9)


def _mask_knight_attacks(sq: int) -> int:
    attacks = 0
    f, r = square_file(sq), square_rank(sq)
    for df, dr in KNIGHT_STEPS:
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            attacks |= bit(nr * 8 + nf)
    return attacks


def _mask_king_attacks(sq: int) -> int:
    attacks = 0
    f, r = square_file(sq), square_rank(sq)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                attacks |= bit(nr * 8 + nf)
    return attacks


def _ray_attacks_reference(
    sq: int, directions: tuple[tuple[int, int], ...], occupied: int
) -> int:
    attacks = 0
    f0, r0 = square_file(sq), square_rank(sq)
    for df, dr in directions:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            s = r * 8 + f
            attacks |= bit(s)
            if occupied & bit(s):
                break  # blocker: this is as far as the ray goes, but it's a legal target
            f, r = f + df, r + dr
    return attacks


def _bishop_attacks_reference(sq: int, occupied: int) -> int:
    return _ray_attacks_reference(sq, BISHOP_DIRECTIONS, occupied)


def _rook_attacks_reference(sq: int, occupied: int) -> int:
    return _ray_attacks_reference(sq, ROOK_DIRECTIONS, occupied)


PAWN_ATTACKS = [[_mask_pawn_attacks(c, s) for s in range(64)] for c in (WHITE, BLACK)]
KNIGHT_ATTACKS = [_mask_knight_attacks(s) for s in range(64)]
KING_ATTACKS = [_mask_king_attacks(s) for s in range(64)]

# numpy-array copies of the three tables above, for jitted callers - a plain Python list isn't
# something numba's nopython mode can use as a captured global (it's untyped as far as numba is
# concerned), but a numpy array is. Filled one element at a time rather than via
# np.array(nested_list, dtype=np.uint64): numpy infers a signed intermediate type from a nested
# Python list first and only casts to the requested dtype after, and a bitboard with bit 63 set
# (any position with a piece on the a8-h8 rank) overflows that intermediate type.
PAWN_ATTACKS_NB = np.zeros((2, 64), dtype=np.uint64)
KNIGHT_ATTACKS_NB = np.zeros(64, dtype=np.uint64)
KING_ATTACKS_NB = np.zeros(64, dtype=np.uint64)
for _c in (WHITE, BLACK):
    for _s in range(64):
        PAWN_ATTACKS_NB[_c, _s] = np.uint64(PAWN_ATTACKS[_c][_s])
for _s in range(64):
    KNIGHT_ATTACKS_NB[_s] = np.uint64(KNIGHT_ATTACKS[_s])
    KING_ATTACKS_NB[_s] = np.uint64(KING_ATTACKS[_s])


# Same rule as _ray_attacks_reference above, typed for nopython mode: sq and occupied have to
# be explicitly unsigned (nb.uint64), or a bitboard with bit 63 set - any position with a piece
# on a8 - would be read as a negative number and every shift/compare on it would be wrong.
# ref: https://numba.readthedocs.io/en/stable/reference/types.html
@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def bishop_attacks(sq: int, occupied: int) -> int:
    attacks = nb.uint64(0)
    f0 = sq & 7
    r0 = sq >> 3
    for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            s = r * 8 + f
            b = nb.uint64(1) << nb.uint8(s)
            attacks |= b
            if occupied & b:
                break
            f, r = f + df, r + dr
    return attacks  # type: ignore[return-value]  # numba unboxes uint64 back to plain int


@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def rook_attacks(sq: int, occupied: int) -> int:
    attacks = nb.uint64(0)
    f0 = sq & 7
    r0 = sq >> 3
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            s = r * 8 + f
            b = nb.uint64(1) << nb.uint8(s)
            attacks |= b
            if occupied & b:
                break
            f, r = f + df, r + dr
    return attacks  # type: ignore[return-value]  # numba unboxes uint64 back to plain int


@njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def queen_attacks(sq: int, occupied: int) -> int:
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied)


# compile now, inside the init budget, not on the first real call
bishop_attacks(0, 0)
rook_attacks(0, 0)
queen_attacks(0, 0)
