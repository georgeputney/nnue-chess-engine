"""Board state: piece bitboards, occupancy, side to move, castling rights, en-passant target.

`Board` is a numba jitclass - the shape that lets it be built, read, and copied from inside
another jitted function (make_move_reference in movegen.py, once that's jitted too), not just from
plain Python. Its own `__init__` is the only thing that has to be jit-compilable; everything
else here (`piece_at`, `fen`, `parse_fen`) stays a plain function taking a Board as an
argument, since a jitclass method has to compile in nopython mode and string building doesn't.
`recompute_occupancy` is the one exception - it's needed from both the (jitted) hot path and the
(plain) cold one, so it's `@njit` and called from both.

`from_chess_board` builds a Board straight from a chess.Board, which is what lets every other
module here be checked directly against python-chess during development instead of only against
itself.

A value read out of `pieces`/`occupancy` is a numpy scalar, not a plain int, and that matters:
it has no `.bit_length()` (`bitboard.scan_forward` needs one), and combining a negative
Python int into one raises `OverflowError` outright rather than doing the wrong thing quietly
(numpy 2.x). Anywhere outside a jitted function that feeds one of these values into plain-Python
bit-trick code casts it to `int(...)` first, and clears a bit with `clear_mask(sq)` rather than
`~bit(sq)` for the same reason - see `bitboard.clear_mask`'s docstring. Inside a jitted
function this isn't an issue: numba's own uint64 arithmetic wraps the way C's does, with none of
numpy's Python-level scalar overrides.
"""

from __future__ import annotations

import chess
import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass  # type: ignore[attr-defined]

from bitboard import (
    BLACK,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    NO_SQUARE,
    PIECE_LETTERS,
    WHITE,
    WHITE_KINGSIDE,
    WHITE_QUEENSIDE,
    bit,
    square,
    square_name,
)

_PIECE_INDEX = {letter: i for i, letter in enumerate(PIECE_LETTERS)}
_CASTLE_CHAR_FLAG = {
    "K": WHITE_KINGSIDE, "Q": WHITE_QUEENSIDE, "k": BLACK_KINGSIDE, "q": BLACK_QUEENSIDE,
}

_BOARD_SPEC = [
    ("pieces", nb.uint64[:, :]),      # pieces[colour][piece_type]
    ("occupancy", nb.uint64[:]),      # [white, black, both]
    ("side", nb.uint8),
    ("castling", nb.uint8),
    ("ep_square", nb.uint8),
    ("halfmove_clock", nb.int32),
    ("fullmove_number", nb.int32),
    ("zobrist", nb.uint64),          # incremental hash; make_move keeps it in step
]


@jitclass(_BOARD_SPEC)  # type: ignore[no-untyped-call]  # mypy can't see through this
class Board:
    def __init__(self) -> None:
        self.pieces = np.zeros((2, 6), dtype=np.uint64)
        self.occupancy = np.zeros(3, dtype=np.uint64)
        self.side = nb.uint8(WHITE)
        self.castling = nb.uint8(0)
        self.ep_square = nb.uint8(NO_SQUARE)
        self.halfmove_clock = nb.int32(0)
        self.fullmove_number = nb.int32(1)
        self.zobrist = nb.uint64(0)


@njit(cache=True)
def recompute_occupancy(pos: Board) -> None:
    pos.occupancy[WHITE] = 0
    for pt in range(6):
        pos.occupancy[WHITE] |= pos.pieces[WHITE][pt]
    pos.occupancy[BLACK] = 0
    for pt in range(6):
        pos.occupancy[BLACK] |= pos.pieces[BLACK][pt]
    pos.occupancy[2] = pos.occupancy[WHITE] | pos.occupancy[BLACK]


@njit  # not cache=True: constructing a Board inside here counts as a "dynamic global" to
# numba's caching (it can't serialize the jitclass constructor to disk), so caching would just
# warn on every import instead of doing anything - moot anyway, since each game gets a fresh
# process with nothing on disk to reuse.
def copy_board(pos: Board) -> Board:
    new = Board()
    new.pieces = pos.pieces.copy()
    new.occupancy = pos.occupancy.copy()
    new.side = pos.side
    new.castling = pos.castling
    new.ep_square = pos.ep_square
    new.halfmove_clock = pos.halfmove_clock
    new.fullmove_number = pos.fullmove_number
    new.zobrist = pos.zobrist
    return new


def piece_at(pos: Board, sq: int) -> tuple[int, int] | None:
    """(colour, piece_type) on `sq`, or None if it's empty."""
    b = bit(sq)
    if not int(pos.occupancy[2]) & b:
        return None
    for colour in (WHITE, BLACK):
        if int(pos.occupancy[colour]) & b:
            for pt in range(6):
                if int(pos.pieces[colour][pt]) & b:
                    return colour, pt
    return None  # pragma: no cover - occupancy says a piece is there; this is unreachable


def fen(pos: Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            found = piece_at(pos, rank * 8 + file)
            if found is None:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            colour, pt = found
            letter = PIECE_LETTERS[pt]
            row += letter if colour == WHITE else letter.lower()
        if empty:
            row += str(empty)
        rows.append(row)
    board_part = "/".join(rows)

    side_part = "w" if pos.side == WHITE else "b"

    castling = int(pos.castling)
    castle_part = (
        ("K" if castling & WHITE_KINGSIDE else "")
        + ("Q" if castling & WHITE_QUEENSIDE else "")
        + ("k" if castling & BLACK_KINGSIDE else "")
        + ("q" if castling & BLACK_QUEENSIDE else "")
    ) or "-"

    ep_square = int(pos.ep_square)
    ep_part = square_name(ep_square) if ep_square != NO_SQUARE else "-"

    return (
        f"{board_part} {side_part} {castle_part} {ep_part} "
        f"{pos.halfmove_clock} {pos.fullmove_number}"
    )


def parse_fen(fen_str: str) -> Board:
    pos = Board()
    board_part, side_part, castle_part, ep_part, halfmove_part, fullmove_part = fen_str.split()

    for rank, row in enumerate(board_part.split("/")):
        file = 0
        r = 7 - rank
        for ch in row:
            if ch.isdigit():
                file += int(ch)
                continue
            colour = WHITE if ch.isupper() else BLACK
            pt = _PIECE_INDEX[ch.upper()]
            pos.pieces[colour][pt] |= bit(square(file, r))
            file += 1

    pos.side = WHITE if side_part == "w" else BLACK  # type: ignore[assignment]  # jitclass field

    castling = 0
    for ch in castle_part:
        if ch in _CASTLE_CHAR_FLAG:
            castling |= _CASTLE_CHAR_FLAG[ch]
    pos.castling = castling  # type: ignore[assignment]

    ep = NO_SQUARE if ep_part == "-" else chess.parse_square(ep_part)
    pos.ep_square = ep  # type: ignore[assignment]
    pos.halfmove_clock = int(halfmove_part)  # type: ignore[assignment]
    pos.fullmove_number = int(fullmove_part)  # type: ignore[assignment]

    recompute_occupancy(pos)
    return pos


def from_chess_board(board: chess.Board) -> Board:
    """Build a Board straight from a python-chess Board - the shortcut that makes every
    other module in this package checkable against python-chess during development."""
    return parse_fen(board.fen())


# compile Board/recompute_occupancy/copy_board now, inside the init budget
_warm = Board()
recompute_occupancy(_warm)
copy_board(_warm)  # type: ignore[type-var]  # mypy can't unify the jitclass return type
