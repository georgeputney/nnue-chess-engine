"""Board state: piece bitboards, occupancy, side to move, castling rights, en-passant target.

`Board` stands in for `chess.Board`: it carries the same state and every other module reads it
the same way, but as a numba jitclass - the shape that lets it be built, read, and copied from
inside another jitted function (movegen.make_move), not just from plain Python. Its own
`__init__` is the only thing that has to be jit-compilable; everything else here (`piece_at` ~
`chess.Board.piece_at`, `fen` ~ `chess.Board.fen`, `parse_fen` ~ `chess.Board(fen)`) stays a
plain function taking a Board as an argument, since a jitclass method has to compile in nopython
mode and string building doesn't. `recompute_occupancy` is the one exception - it is needed from
both the (jitted) hot path and the (plain) cold one, so it is `@njit` and called from both.

`from_chess_board` builds a Board straight from a chess.Board, which is what lets every other
module here be checked directly against python-chess during development instead of only against
itself.

A value read out of `pieces` / `occupancy` is a numpy scalar, not a plain int, and that matters:
it has no `.bit_length()` (`bitboard.scan_forward` needs one), and combining a negative Python
int into one raises `OverflowError` outright rather than doing the wrong thing quietly (numpy
2.x). Anywhere outside a jitted function that feeds one of these values into plain-Python
bit-trick code casts it to `int(...)` first, and clears a bit with `clear_mask(square)` rather
than `~bit(square)` for the same reason - see the comment on `bitboard.clear_mask`. Inside a
jitted function this is not an issue: numba's own uint64 arithmetic wraps the way C's does, with
none of numpy's Python-level scalar overrides.
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
from nnue.accumulator import ACC_WIDTH, fill_accumulator

# fen character <-> piece-type id, and fen castling character -> rights flag
PIECE_INDEX = {letter: index for index, letter in enumerate(PIECE_LETTERS)}
CASTLE_CHAR_FLAG = {
    "K": WHITE_KINGSIDE, "Q": WHITE_QUEENSIDE, "k": BLACK_KINGSIDE, "q": BLACK_QUEENSIDE,
}

BOARD_SPEC = [
    ("pieces", nb.uint64[:, :]),      # pieces[colour][piece_type]
    ("occupancy", nb.uint64[:]),      # [white, black, both]
    ("side", nb.uint8),
    ("castling", nb.uint8),
    ("ep_square", nb.uint8),
    ("halfmove_clock", nb.int32),
    ("fullmove_number", nb.int32),
    ("zobrist", nb.uint64),          # incremental hash; make_move keeps it in step
    ("acc", nb.int32[:, :]),         # NNUE accumulator [colour][ACC_WIDTH]; make_move keeps it
]                                    # in step, parse_fen fills it (nnue/accumulator.py)


@jitclass(BOARD_SPEC)  # type: ignore[no-untyped-call]  # mypy can't see through this
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
        self.acc = np.zeros((2, ACC_WIDTH), dtype=np.int32)


# Rebuild the three occupancy bitboards (white, black, both) by OR-ing the six piece bitboards
# per side. make_move updates occupancy incrementally on the hot path; this is for parse_fen and
# anything that edits `pieces` directly.
@njit(cache=True)
def recompute_occupancy(board: Board) -> None:
    board.occupancy[WHITE] = 0
    for piece_type in range(6):
        board.occupancy[WHITE] |= board.pieces[WHITE][piece_type]

    board.occupancy[BLACK] = 0
    for piece_type in range(6):
        board.occupancy[BLACK] |= board.pieces[BLACK][piece_type]

    board.occupancy[2] = board.occupancy[WHITE] | board.occupancy[BLACK]


# A deep copy - make_move / null_move mutate a copy and never pop, so every child needs its own
# `pieces` / `occupancy` arrays.
@njit  # not cache=True: constructing a Board inside here counts as a "dynamic global" to numba's
# caching (it can't serialize the jitclass constructor to disk), so caching would just warn on
# every import instead of doing anything - moot anyway, since each game gets a fresh process with
# nothing on disk to reuse.
def copy_board(board: Board) -> Board:
    copy = Board()
    copy.pieces = board.pieces.copy()
    copy.occupancy = board.occupancy.copy()
    copy.side = board.side
    copy.castling = board.castling
    copy.ep_square = board.ep_square
    copy.halfmove_clock = board.halfmove_clock
    copy.fullmove_number = board.fullmove_number
    copy.zobrist = board.zobrist
    copy.acc = board.acc.copy()

    return copy


# (colour, piece_type) on `square`, or None if it is empty - chess.Board.piece_at without the
# Piece object. Plain Python (it returns an Optional tuple); the search uses agent.piece_on.
def piece_at(board: Board, square: int) -> tuple[int, int] | None:
    square_bit = bit(square)
    
    if not int(board.occupancy[2]) & square_bit:
        return None
    
    for colour in (WHITE, BLACK):
        if int(board.occupancy[colour]) & square_bit:
            for piece_type in range(6):
                if int(board.pieces[colour][piece_type]) & square_bit:
                    return colour, piece_type
                
    return None  # pragma: no cover - occupancy says a piece is there; this is unreachable


# The position as a FEN string - chess.Board.fen. Cold path only (string building), used by
# from_chess_board's round trip and by the tools.
def fen(board: Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            found = piece_at(board, rank * 8 + file)

            if found is None:
                empty += 1
                continue

            if empty:
                row += str(empty)
                empty = 0

            colour, piece_type = found
            letter = PIECE_LETTERS[piece_type]
            row += letter if colour == WHITE else letter.lower()

        if empty:
            row += str(empty)

        rows.append(row)

    board_part = "/".join(rows)

    side_part = "w" if board.side == WHITE else "b"

    castling = int(board.castling)
    castle_part = (
        ("K" if castling & WHITE_KINGSIDE else "")
        + ("Q" if castling & WHITE_QUEENSIDE else "")
        + ("k" if castling & BLACK_KINGSIDE else "")
        + ("q" if castling & BLACK_QUEENSIDE else "")
    ) or "-"

    ep_square = int(board.ep_square)
    ep_part = square_name(ep_square) if ep_square != NO_SQUARE else "-"

    return (
        f"{board_part} {side_part} {castle_part} {ep_part} "
        f"{board.halfmove_clock} {board.fullmove_number}"
    )


# Parse a FEN string into a Board - chess.Board(fen). The one constructor the rest of the engine
# calls; get_move starts every move here.
def parse_fen(fen_str: str) -> Board:
    board = Board()
    board_part, side_part, castle_part, ep_part, halfmove_part, fullmove_part = fen_str.split()

    for row_index, row in enumerate(board_part.split("/")):
        file = 0
        rank = 7 - row_index  # fen lists rank 8 first, our squares number rank 1 first

        for char in row:

            if char.isdigit():
                file += int(char)
                continue

            colour = WHITE if char.isupper() else BLACK
            piece_type = PIECE_INDEX[char.upper()]
            board.pieces[colour][piece_type] |= bit(square(file, rank))
            file += 1

    board.side = WHITE if side_part == "w" else BLACK  # type: ignore[assignment]  # jitclass field

    castling = 0
    for char in castle_part:
        if char in CASTLE_CHAR_FLAG:
            castling |= CASTLE_CHAR_FLAG[char]

    board.castling = castling  # type: ignore[assignment]

    ep = NO_SQUARE if ep_part == "-" else chess.parse_square(ep_part)
    board.ep_square = ep  # type: ignore[assignment]
    board.halfmove_clock = int(halfmove_part)  # type: ignore[assignment]
    board.fullmove_number = int(fullmove_part)  # type: ignore[assignment]

    recompute_occupancy(board)
    fill_accumulator(board.pieces, board.acc)
    return board


# Build a Board straight from a python-chess Board - the shortcut that makes every other module
# in this package checkable against python-chess during development.
def from_chess_board(board: chess.Board) -> Board:
    return parse_fen(board.fen())


# compile Board / recompute_occupancy / copy_board now, inside the init budget
def warm_up() -> None:
    warm = Board()
    recompute_occupancy(warm)
    fill_accumulator(warm.pieces, warm.acc)
    copy_board(warm)  # type: ignore[type-var]  # mypy can't unify the jitclass return type


warm_up()
