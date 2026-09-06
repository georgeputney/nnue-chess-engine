"""Square numbering, piece/colour ids, and bitboard masks shared by every other module here.

Square numbering matches python-chess: a1=0, b1=1, ... h1=7, a2=8, ... h8=63. So square s has
file s & 7 and rank s >> 3, and a bitboard's bit `s` is set iff a piece sits on that square.
The helpers mirror the python-chess names too - square_file / square_rank / square / square_name
/ popcount / scan_forward all behave like `chess.square_file` and friends; lsb_index is the
nopython-safe form of `lsb_square`.
"""

from collections.abc import Iterator

import numba as nb
from numba import njit

FULL_BB = (1 << 64) - 1

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
PIECE_LETTERS = "PNBRQK"

# castling-rights bit flags, stored together as one 4-bit int on a Board
WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE = 1, 2, 4, 8

NO_SQUARE = 64  # sentinel: no en-passant target this position


# File (0..7, a..h) and rank (0..7, 1..8) of a square, and the inverse. chess.square_file /
# chess.square_rank / chess.square, on plain ints.
@njit(cache=True)
def square_file(sq: int) -> int:
    return sq & 7


@njit(cache=True)
def square_rank(sq: int) -> int:
    return sq >> 3


def square(file: int, rank: int) -> int:
    return rank * 8 + file


# Algebraic name, e.g. 27 -> "d4" - for fen() and move_uci(), never the hot path.
def square_name(sq: int) -> str:
    return "abcdefgh"[square_file(sq)] + str(square_rank(sq) + 1)


FILE_A = 0x0101_0101_0101_0101
FILE_H = FILE_A << 7
FILE_MASKS = [FILE_A << f for f in range(8)]

RANK_1 = 0xFF
RANK_8 = RANK_1 << 56
RANK_MASKS = [RANK_1 << (8 * r) for r in range(8)]

NOT_FILE_A = FULL_BB ^ FILE_A
NOT_FILE_H = FULL_BB ^ FILE_H
NOT_FILE_AB = FULL_BB ^ (FILE_A | FILE_MASKS[1])
NOT_FILE_GH = FULL_BB ^ (FILE_MASKS[6] | FILE_H)


# A bitboard with only `sq`'s bit set - the building block of every mask below.
@njit(nb.uint64(nb.uint64), cache=True)
def bit(sq: int) -> int:
    # nb.uint64(1), not the bare literal: numba would otherwise infer `1` as a signed int64,
    # and shifting a signed type left by 63 sets its sign bit instead of just its top bit.
    return nb.uint64(1) << sq  # type: ignore[return-value]  # numba unboxes to plain int


# A bounded (0..FULL_BB) complement of bit(sq) - `bb & clear_mask(sq)` clears that one bit.
# Not `~bit(sq)`: Python's `~` on a plain int is arbitrary-precision and goes negative, and a
# negative Python int can't be combined into a numpy uint64 array element (Board.pieces is one) -
# numpy raises OverflowError outright rather than silently doing the wrong thing.
@njit(nb.uint64(nb.uint64), cache=True)
def clear_mask(sq: int) -> int:
    return nb.uint64(FULL_BB) ^ bit(sq)  # type: ignore[return-value]


# Set-bit count - chess.popcount. bb.bit_count() is Python 3.10+, so this is plain-Python only;
# jitted callers use agent.popcount, which loops.
def popcount(bb: int) -> int:
    return bb.bit_count()


# Index of the least-significant set bit. Caller must ensure bb != 0.
def lsb_square(bb: int) -> int:
    return (bb & -bb).bit_length() - 1


# Yield every set square, low to high, without mutating the caller's bitboard - chess.scan_forward.
def scan_forward(bb: int) -> Iterator[int]:
    while bb:
        s = lsb_square(bb)
        yield s
        bb &= bb - 1


# lsb_square's numba-callable twin: nopython mode can't call .bit_length() (that's a Python int
# method, not something numba's native integer types support), so this counts trailing zero
# bits with a plain loop instead - correct, if not the fastest possible way to do it. Every
# jitted function elsewhere that needs to walk set bits one at a time calls this directly rather
# than trying to use the scan_forward generator above, which nopython mode can't compile either.
@njit(nb.uint8(nb.uint64), cache=True)
def lsb_index(bb: int) -> int:
    idx = nb.uint8(0)
    one = nb.uint64(1)

    while (bb & one) == 0:
        bb >>= one  # type: ignore[assignment]  # numba's own uint64, not numpy's
        idx += nb.uint8(1)
        
    return idx  # type: ignore[return-value]  # numba unboxes uint8 back to plain int


# compile now, inside the init budget, not on the first real call
def warm_up() -> None:
    square_file(0)
    square_rank(0)
    bit(0)
    clear_mask(0)
    lsb_index(1)


warm_up()
