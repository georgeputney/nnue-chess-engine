"""A move packed into a single int, instead of a NamedTuple.

This only matters for one reason: numba's nopython mode does not deal well with NamedTuples
being built and returned from a growing Python list the way pseudo_legal_moves_reference does -
jitting that function means it has to produce plain ints (or write into a preallocated numpy
array of them) instead. Packing now, before movegen itself is jitted, means that conversion
doesn't have to happen at the same time as everything else.

Bit layout (22 bits used, fits comfortably in a plain int or nb.int32):

    0-5    from_square    (0-63)
    6-11   to_square      (0-63)
    12-14  piece          (0-5, the piece type ids in bitboard)
    15-17  promotion      (0-5, or PROMOTION_NONE if this isn't a promotion)
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

FROM_SHIFT, FROM_MASK = 0, 0x3F
TO_SHIFT, TO_MASK = 6, 0x3F
PIECE_SHIFT, PIECE_MASK = 12, 0x7
PROMO_SHIFT, PROMO_MASK = 15, 0x7
CAPTURE_BIT = 1 << 18
EN_PASSANT_BIT = 1 << 19
CASTLE_BIT = 1 << 20
DOUBLE_PUSH_BIT = 1 << 21


# Pack the fields above into one int - the ergonomic form, for plain-Python callers (movegen's
# pseudo_legal_moves_reference and the verify tools): `promotion=None` for a non-promoting move,
# the rest keyword flags. Jitted callers can't take keyword arguments or a None default, so they
# call encode_move_nb directly.
def encode_move(
    from_square: int,
    to_square: int,
    piece: int,
    promotion: int | None = None,
    is_capture: bool = False,
    is_en_passant: bool = False,
    is_castle: bool = False,
    is_double_push: bool = False,
) -> int:
    
    promotion_id = PROMOTION_NONE if promotion is None else promotion
    return encode_move_nb(
        from_square, to_square, piece, promotion_id,
        is_capture, is_en_passant, is_castle, is_double_push,
    )


# Same as encode_move, but for jitted callers: positional only, PROMOTION_NONE (not None) for a
# non-promoting move. This is the one that actually lays out the bits.
@njit(cache=True)
def encode_move_nb(
    from_square: int,
    to_square: int,
    piece: int,
    promotion: int,
    is_capture: bool,
    is_en_passant: bool,
    is_castle: bool,
    is_double_push: bool,
) -> int:
    
    packed = (
        from_square
        | (to_square << TO_SHIFT)
        | (piece << PIECE_SHIFT)
        | (promotion << PROMO_SHIFT)
    )
    if is_capture:
        packed |= CAPTURE_BIT

    if is_en_passant:
        packed |= EN_PASSANT_BIT

    if is_castle:
        packed |= CASTLE_BIT

    if is_double_push:
        packed |= DOUBLE_PUSH_BIT

    return packed


# Field accessors - the packed-int equivalents of chess.Move.from_square / .to_square, plus the
# piece and flags this encoding folds in that chess.Move keeps on the board instead. All plain
# shift-and-mask, so `@njit` and callable from jitted and ordinary code alike.
@njit(cache=True)
def move_from_square(move: int) -> int:
    return (move >> FROM_SHIFT) & FROM_MASK


@njit(cache=True)
def move_to_square(move: int) -> int:
    return (move >> TO_SHIFT) & TO_MASK


@njit(cache=True)
def move_piece(move: int) -> int:
    return (move >> PIECE_SHIFT) & PIECE_MASK


# Promotion piece type, or None for a non-promoting move - chess.Move.promotion. Not
# nopython-safe (the Optional crosses the jit boundary); jitted code calls move_promotion_raw.
def move_promotion(move: int) -> int | None:
    promotion = move_promotion_raw(move)
    return None if promotion == PROMOTION_NONE else promotion


# Same as move_promotion, but PROMOTION_NONE instead of None - for jitted callers.
@njit(cache=True)
def move_promotion_raw(move: int) -> int:
    return (move >> PROMO_SHIFT) & PROMO_MASK


# The four flag bits. python-chess re-derives these from the board each time; here they are
# baked into the move at generation, so the search never needs the parent position to read them.
@njit(cache=True)
def move_is_capture(move: int) -> bool:
    return bool(move & CAPTURE_BIT)


@njit(cache=True)
def move_is_en_passant(move: int) -> bool:
    return bool(move & EN_PASSANT_BIT)


@njit(cache=True)
def move_is_castle(move: int) -> bool:
    return bool(move & CASTLE_BIT)


@njit(cache=True)
def move_is_double_push(move: int) -> bool:
    return bool(move & DOUBLE_PUSH_BIT)


# UCI notation, e.g. e2e4 or e7e8q - chess.Move.uci. get_move returns this; otherwise it is for
# printing only, never the hot path.
def move_uci(move: int) -> str:
    # piece ids: PAWN=0 KNIGHT=1 BISHOP=2 ROOK=3 QUEEN=4 KING=5 - a promotion is always one of
    # knight/bishop/rook/queen (1-4), so it indexes this string directly.
    promotion = move_promotion(move)
    suffix = "?nbrq"[promotion] if promotion is not None else ""
    
    return square_name(move_from_square(move)) + square_name(move_to_square(move)) + suffix


# compile the jitted accessors now, inside the init budget, not on the first real call
def warm_up() -> None:
    warm = encode_move_nb(0, 0, 0, PROMOTION_NONE, False, False, False, False)
    move_from_square(warm)
    move_to_square(warm)
    move_piece(warm)
    move_promotion_raw(warm)
    move_is_capture(warm)
    move_is_en_passant(warm)
    move_is_castle(warm)
    move_is_double_push(warm)


warm_up()
