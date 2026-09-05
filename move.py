"""A move packed into a single int, instead of a NamedTuple.

This only matters for one reason: numba's nopython mode does not deal well with NamedTuples
being built and returned from a growing Python list the way _pseudo_legal_moves_reference does -
jitting that function means it has to produce plain ints (or write into a preallocated numpy
array of them) instead. Packing now, before movegen itself is jitted, means that conversion
doesn't have to happen at the same time as everything else.

Bit layout (22 bits used, fits comfortably in a plain int or nb.int32):

    0-5    from_sq       (0-63)
    6-11   to_sq         (0-63)
    12-14  piece         (0-5, the piece type ids in bitboard)
    15-17  promotion     (0-5, or PROMOTION_NONE if this isn't a promotion)
    18     is_capture
    19     is_en_passant
    20     is_castle
    21     is_double_push

move_from_square/move_to_square/move_piece/move_is_* are plain bit arithmetic with nothing
numba-unfriendly in them, so they're `@njit` directly - callable from both jitted and ordinary
Python code, same as attacks.bishop_attacks. move_promotion (returns None for "no
promotion") and encode_move (keyword arguments, a None default) aren't nopython-compatible as
written, so jitted code uses move_promotion_raw / encode_move_nb instead - same bits, no
Optional type or keyword arguments crossing the jit boundary.
"""

from numba import njit

from bitboard import square_name

PROMOTION_NONE = 7  # fits in the 3 promotion bits (0-5 are real piece types) alongside them

_FROM_SHIFT, _FROM_MASK = 0, 0x3F
_TO_SHIFT, _TO_MASK = 6, 0x3F
_PIECE_SHIFT, _PIECE_MASK = 12, 0x7
_PROMO_SHIFT, _PROMO_MASK = 15, 0x7
_CAPTURE_BIT = 1 << 18
_EN_PASSANT_BIT = 1 << 19
_CASTLE_BIT = 1 << 20
_DOUBLE_PUSH_BIT = 1 << 21


def encode_move(
    from_sq: int,
    to_sq: int,
    piece: int,
    promotion: int | None = None,
    is_capture: bool = False,
    is_en_passant: bool = False,
    is_castle: bool = False,
    is_double_push: bool = False,
) -> int:
    promo = PROMOTION_NONE if promotion is None else promotion
    return encode_move_nb(
        from_sq, to_sq, piece, promo, is_capture, is_en_passant, is_castle, is_double_push
    )


@njit(cache=True)
def encode_move_nb(
    from_sq: int,
    to_sq: int,
    piece: int,
    promotion: int,
    is_capture: bool,
    is_en_passant: bool,
    is_castle: bool,
    is_double_push: bool,
) -> int:
    """Same as encode_move, but for jitted callers: no keyword arguments, no None - pass
    PROMOTION_NONE explicitly for a non-promoting move."""
    m = from_sq | (to_sq << _TO_SHIFT) | (piece << _PIECE_SHIFT) | (promotion << _PROMO_SHIFT)
    if is_capture:
        m |= _CAPTURE_BIT
    if is_en_passant:
        m |= _EN_PASSANT_BIT
    if is_castle:
        m |= _CASTLE_BIT
    if is_double_push:
        m |= _DOUBLE_PUSH_BIT
    return m


@njit(cache=True)
def move_from_square(m: int) -> int:
    return (m >> _FROM_SHIFT) & _FROM_MASK


@njit(cache=True)
def move_to_square(m: int) -> int:
    return (m >> _TO_SHIFT) & _TO_MASK


@njit(cache=True)
def move_piece(m: int) -> int:
    return (m >> _PIECE_SHIFT) & _PIECE_MASK


def move_promotion(m: int) -> int | None:
    promo = move_promotion_raw(m)
    return None if promo == PROMOTION_NONE else promo


@njit(cache=True)
def move_promotion_raw(m: int) -> int:
    """Same as move_promotion, but PROMOTION_NONE instead of None - for jitted callers."""
    return (m >> _PROMO_SHIFT) & _PROMO_MASK


@njit(cache=True)
def move_is_capture(m: int) -> bool:
    return bool(m & _CAPTURE_BIT)


@njit(cache=True)
def move_is_en_passant(m: int) -> bool:
    return bool(m & _EN_PASSANT_BIT)


@njit(cache=True)
def move_is_castle(m: int) -> bool:
    return bool(m & _CASTLE_BIT)


@njit(cache=True)
def move_is_double_push(m: int) -> bool:
    return bool(m & _DOUBLE_PUSH_BIT)


def move_uci(m: int) -> str:
    """UCI notation, e.g. e2e4 or e7e8q - for debugging/printing only."""
    # piece ids: PAWN=0 KNIGHT=1 BISHOP=2 ROOK=3 QUEEN=4 KING=5 - a promotion is always one of
    # knight/bishop/rook/queen (1-4), so it indexes this string directly.
    promo = move_promotion(m)
    suffix = "?nbrq"[promo] if promo is not None else ""
    return square_name(move_from_square(m)) + square_name(move_to_square(m)) + suffix


# compile the jitted accessors now, inside the init budget
_warm_move = encode_move_nb(0, 0, 0, PROMOTION_NONE, False, False, False, False)
move_from_square(_warm_move)
move_to_square(_warm_move)
move_piece(_warm_move)
move_promotion_raw(_warm_move)
move_is_capture(_warm_move)
move_is_en_passant(_warm_move)
move_is_castle(_warm_move)
move_is_double_push(_warm_move)
