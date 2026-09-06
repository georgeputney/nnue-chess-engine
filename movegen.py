"""Pseudo-legal move generation, make_move, and legality filtering.

Legality is checked the simple, slow way on purpose: generate every pseudo-legal move (piece
attack pattern, ignoring pins and existing checks), make it on a fresh copy, and keep it only if
the mover's own king is safe afterward. That's one king-safety check per pseudo-legal move
instead of the pin-aware generation a fast engine would use, but it can't miss a pin or a
check-evasion case by construction, which matters more than speed until perft holds.

Moves are packed ints (move.py), not objects - see that module's docstring for why.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numba import njit

from attacks import (
    KING_ATTACKS,
    KING_ATTACKS_NB,
    KNIGHT_ATTACKS,
    KNIGHT_ATTACKS_NB,
    PAWN_ATTACKS,
    PAWN_ATTACKS_NB,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)
from bitboard import (
    BISHOP,
    BLACK,
    BLACK_KINGSIDE,
    BLACK_QUEENSIDE,
    FULL_BB,
    KING,
    KNIGHT,
    NO_SQUARE,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    WHITE_KINGSIDE,
    WHITE_QUEENSIDE,
    bit,
    clear_mask,
    lsb_index,
    scan_forward,
    square_rank,
)
from board import Board, copy_board, parse_fen, recompute_occupancy
from move import (
    PROMOTION_NONE,
    encode_move,
    encode_move_nb,
    move_from_square,
    move_is_capture,
    move_is_castle,
    move_is_double_push,
    move_is_en_passant,
    move_piece,
    move_promotion,
    move_promotion_raw,
    move_to_square,
)
from zobrist import CASTLING_KEYS, EP_FILE_KEYS, PIECE_SQUARE_KEYS, SIDE_KEY

MAX_MOVES = 256  # a real legal chess position never has more than 218 (a published upper
# bound); pseudo-legal generation can't exceed that by much since it's the same piece-attack
# patterns without the king-safety filter, so this leaves a comfortable margin.

_ROOK_HOME_TO_CASTLE_FLAG = {
    0: WHITE_QUEENSIDE, 7: WHITE_KINGSIDE, 56: BLACK_QUEENSIDE, 63: BLACK_KINGSIDE,
}

# Squares strictly between two collinear squares, and the whole board line running through
# them - both 0 for a pair that shares no rank, file, or diagonal. Built once here in plain
# Python (a 64x64 pair of tables, 64 KB) and frozen into numpy arrays for the jitted legality
# check. They are what let legal_moves() rule on a move without making it: a pinned piece may
# only travel along _LINE[king][piece], and a single check is answered only on the checker's
# square or _BETWEEN[king][checker].
_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

_BETWEEN = [[0] * 64 for _ in range(64)]
_LINE = [[0] * 64 for _ in range(64)]
for _from in range(64):
    _ff, _fr = _from & 7, _from >> 3
    for _df, _dr in _DIRECTIONS:
        _ray: list[int] = []
        _f, _r = _ff, _fr
        while 0 <= _f < 8 and 0 <= _r < 8:
            _ray.append(_r * 8 + _f)
            _f, _r = _f + _df, _r + _dr
        _line_bb = 0
        for _s in _ray:
            _line_bb |= 1 << _s
        _f, _r = _ff - _df, _fr - _dr
        while 0 <= _f < 8 and 0 <= _r < 8:
            _line_bb |= 1 << (_r * 8 + _f)
            _f, _r = _f - _df, _r - _dr
        for _i in range(1, len(_ray)):
            _between_bb = 0
            for _j in range(1, _i):
                _between_bb |= 1 << _ray[_j]
            _BETWEEN[_from][_ray[_i]] = _between_bb
            _LINE[_from][_ray[_i]] = _line_bb

_BETWEEN_NB = np.zeros((64, 64), dtype=np.uint64)
_LINE_NB = np.zeros((64, 64), dtype=np.uint64)
for _a in range(64):
    for _b in range(64):
        _BETWEEN_NB[_a, _b] = np.uint64(_BETWEEN[_a][_b])
        _LINE_NB[_a, _b] = np.uint64(_LINE[_a][_b])


def is_attacked_reference(pos: Board, sq: int, by_colour: int) -> bool:
    """Is `sq` attacked by any of `by_colour`'s pieces, on the position as it stands?"""
    occ = pos.occupancy[2]
    opp = 1 - by_colour

    if PAWN_ATTACKS[opp][sq] & pos.pieces[by_colour][PAWN]:
        return True
    if KNIGHT_ATTACKS[sq] & pos.pieces[by_colour][KNIGHT]:
        return True
    if KING_ATTACKS[sq] & pos.pieces[by_colour][KING]:
        return True

    bishops_queens = pos.pieces[by_colour][BISHOP] | pos.pieces[by_colour][QUEEN]
    if bishop_attacks(sq, occ) & bishops_queens:
        return True

    rooks_queens = pos.pieces[by_colour][ROOK] | pos.pieces[by_colour][QUEEN]
    return bool(rook_attacks(sq, occ) & rooks_queens)


def is_check_reference(pos: Board, colour: int) -> bool:
    king_sq = next(scan_forward(int(pos.pieces[colour][KING])))
    return is_attacked_reference(pos, king_sq, 1 - colour)


def _castling_moves(pos: Board, colour: int, king_sq: int, occ: int) -> list[int]:
    moves = []
    opp = 1 - colour

    if colour == WHITE and king_sq == 4:
        clear_k = not occ & (bit(5) | bit(6))
        if pos.castling & WHITE_KINGSIDE and clear_k and not any(
            is_attacked_reference(pos, s, opp) for s in (4, 5, 6)
        ):
            moves.append(encode_move(4, 6, KING, is_castle=True))
        clear_q = not occ & (bit(1) | bit(2) | bit(3))
        if pos.castling & WHITE_QUEENSIDE and clear_q and not any(
            is_attacked_reference(pos, s, opp) for s in (4, 3, 2)
        ):
            moves.append(encode_move(4, 2, KING, is_castle=True))
    elif colour == BLACK and king_sq == 60:
        clear_k = not occ & (bit(61) | bit(62))
        if pos.castling & BLACK_KINGSIDE and clear_k and not any(
            is_attacked_reference(pos, s, opp) for s in (60, 61, 62)
        ):
            moves.append(encode_move(60, 62, KING, is_castle=True))
        clear_q = not occ & (bit(57) | bit(58) | bit(59))
        if pos.castling & BLACK_QUEENSIDE and clear_q and not any(
            is_attacked_reference(pos, s, opp) for s in (60, 59, 58)
        ):
            moves.append(encode_move(60, 58, KING, is_castle=True))

    return moves


def _pseudo_legal_moves_reference(pos: Board) -> list[int]:
    moves: list[int] = []
    colour = int(pos.side)
    opp = 1 - colour
    ep_square = int(pos.ep_square)
    own = int(pos.occupancy[colour])
    enemy = int(pos.occupancy[opp])
    occ = int(pos.occupancy[2])
    not_own = FULL_BB ^ own

    forward = 8 if colour == WHITE else -8
    start_rank = 1 if colour == WHITE else 6
    promo_rank = 7 if colour == WHITE else 0

    for frm in scan_forward(int(pos.pieces[colour][PAWN])):
        one = frm + forward
        if 0 <= one < 64 and not occ & bit(one):
            if square_rank(one) == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append(encode_move(frm, one, PAWN, promotion=promo))
            else:
                moves.append(encode_move(frm, one, PAWN))
                if square_rank(frm) == start_rank:
                    two = one + forward
                    if not occ & bit(two):
                        moves.append(encode_move(frm, two, PAWN, is_double_push=True))

        for to in scan_forward(PAWN_ATTACKS[colour][frm] & enemy):
            if square_rank(to) == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append(encode_move(frm, to, PAWN, promotion=promo, is_capture=True))
            else:
                moves.append(encode_move(frm, to, PAWN, is_capture=True))

        if ep_square != NO_SQUARE and PAWN_ATTACKS[colour][frm] & bit(ep_square):
            moves.append(encode_move(frm, ep_square, PAWN, is_capture=True, is_en_passant=True))

    for frm in scan_forward(int(pos.pieces[colour][KNIGHT])):
        for to in scan_forward(KNIGHT_ATTACKS[frm] & not_own):
            moves.append(encode_move(frm, to, KNIGHT, is_capture=bool(enemy & bit(to))))

    for frm in scan_forward(int(pos.pieces[colour][BISHOP])):
        for to in scan_forward(bishop_attacks(frm, occ) & not_own):
            moves.append(encode_move(frm, to, BISHOP, is_capture=bool(enemy & bit(to))))

    for frm in scan_forward(int(pos.pieces[colour][ROOK])):
        for to in scan_forward(rook_attacks(frm, occ) & not_own):
            moves.append(encode_move(frm, to, ROOK, is_capture=bool(enemy & bit(to))))

    for frm in scan_forward(int(pos.pieces[colour][QUEEN])):
        for to in scan_forward(queen_attacks(frm, occ) & not_own):
            moves.append(encode_move(frm, to, QUEEN, is_capture=bool(enemy & bit(to))))

    for frm in scan_forward(int(pos.pieces[colour][KING])):
        for to in scan_forward(KING_ATTACKS[frm] & not_own):
            moves.append(encode_move(frm, to, KING, is_capture=bool(enemy & bit(to))))
        moves.extend(_castling_moves(pos, colour, frm, occ))

    return moves


def _clear_castling_right(pos: Board, sq: int) -> None:
    flag = _ROOK_HOME_TO_CASTLE_FLAG.get(sq)
    if flag is not None:
        pos.castling &= 0xFF ^ flag  # not ~flag: pos.castling is a jitclass uint8 field, and a
        # negative Python int can't combine into one any more than into a uint64 array element
        # (see bitboard.clear_mask) - same fix, bounded to a byte instead of 64 bits.


def make_move_reference(pos: Board, move: int) -> Board:
    new: Board = copy_board(pos)  # type: ignore[assignment, type-var]  # njit return type
    colour = int(pos.side)
    opp = 1 - colour

    frm, to, piece = move_from_square(move), move_to_square(move), move_piece(move)
    promotion = move_promotion(move)
    is_capture = move_is_capture(move)
    is_castle = move_is_castle(move)

    new.pieces[colour][piece] &= clear_mask(frm)

    if move_is_en_passant(move):
        captured_sq = to + (-8 if colour == WHITE else 8)
        new.pieces[opp][PAWN] &= clear_mask(captured_sq)
    elif is_capture:
        for pt in range(6):
            if new.pieces[opp][pt] & bit(to):
                new.pieces[opp][pt] &= clear_mask(to)
                break

    placed = promotion if promotion is not None else piece
    new.pieces[colour][placed] |= bit(to)

    if is_castle:
        if to == frm + 2:  # kingside: rook a-side of the king moves in
            rook_from, rook_to = frm + 3, frm + 1
        else:  # queenside
            rook_from, rook_to = frm - 4, frm - 1
        new.pieces[colour][ROOK] &= clear_mask(rook_from)
        new.pieces[colour][ROOK] |= bit(rook_to)

    if piece == KING:
        if colour == WHITE:
            new.castling &= 0xFF ^ (WHITE_KINGSIDE | WHITE_QUEENSIDE)
        else:
            new.castling &= 0xFF ^ (BLACK_KINGSIDE | BLACK_QUEENSIDE)
    _clear_castling_right(new, frm)
    _clear_castling_right(new, to)

    ep = (frm + to) // 2 if move_is_double_push(move) else NO_SQUARE
    new.ep_square = ep  # type: ignore[assignment]  # jitclass field, see board.py
    new.halfmove_clock = 0 if piece == PAWN or is_capture else pos.halfmove_clock + 1  # type: ignore[assignment]
    new.fullmove_number = pos.fullmove_number + (1 if colour == BLACK else 0)
    new.side = opp  # type: ignore[assignment]

    recompute_occupancy(new)
    return new


def legal_moves_reference(pos: Board) -> list[int]:
    mover = int(pos.side)
    legal = []
    for move in _pseudo_legal_moves_reference(pos):
        if not is_check_reference(make_move_reference(pos, move), mover):
            legal.append(move)
    return legal


# Below: jitted equivalents of everything above. Same algorithms, but nopython-compatible: no
# Python lists (moves fill a preallocated array up to MAX_MOVES instead), no dict lookups
# (castling rights use a small if/elif chain on the square instead of _ROOK_HOME_TO_CASTLE_FLAG),
# no scan_forward generator (lsb_index plus an explicit "clear the lowest bit and loop"
# instead), no any()-over-a-generator (an explicit loop with an early exit). The plain versions
# above stay as the reference tools/verify_movegen.py checks these against - move for move, not
# just perft's leaf counts, since two different bugs could cancel out into the same node count.


@njit(cache=True)
def is_attacked(pos: Board, sq: int, by_colour: int) -> bool:
    occ = pos.occupancy[2]
    opp = 1 - by_colour

    if PAWN_ATTACKS_NB[opp, sq] & pos.pieces[by_colour, PAWN]:
        return True
    if KNIGHT_ATTACKS_NB[sq] & pos.pieces[by_colour, KNIGHT]:
        return True
    if KING_ATTACKS_NB[sq] & pos.pieces[by_colour, KING]:
        return True

    bishops_queens = pos.pieces[by_colour, BISHOP] | pos.pieces[by_colour, QUEEN]
    if bishop_attacks(nb.uint8(sq), occ) & bishops_queens:  # type: ignore[arg-type]
        return True

    rooks_queens = pos.pieces[by_colour, ROOK] | pos.pieces[by_colour, QUEEN]
    return bool(rook_attacks(nb.uint8(sq), occ) & rooks_queens)  # type: ignore[arg-type]


@njit(cache=True)
def is_check(pos: Board, colour: int) -> bool:
    king_sq = lsb_index(pos.pieces[colour, KING])
    return is_attacked(pos, king_sq, 1 - colour)


@njit(cache=True)
def _squares_attacked_nb(pos: Board, by_colour: int, a: int, b: int, c: int) -> bool:
    """Is any of squares a/b/c attacked by by_colour? (castling's "king doesn't pass through
    check" rule always checks exactly three squares - its start, middle, and end.)"""
    return (
        is_attacked(pos, a, by_colour)
        or is_attacked(pos, b, by_colour)
        or is_attacked(pos, c, by_colour)
    )


@njit(cache=True)
def _castling_moves_nb(
    pos: Board, colour: int, king_sq: int, occ: int, moves: np.ndarray, count: int
) -> int:
    opp = 1 - colour

    if colour == WHITE and king_sq == 4:
        clear_k = occ & (bit(5) | bit(6)) == 0
        safe_k = not _squares_attacked_nb(pos, opp, 4, 5, 6)
        if pos.castling & WHITE_KINGSIDE and clear_k and safe_k:
            moves[count] = encode_move_nb(4, 6, KING, PROMOTION_NONE, False, False, True, False)
            count += 1
        clear_q = occ & (bit(1) | bit(2) | bit(3)) == 0
        safe_q = not _squares_attacked_nb(pos, opp, 4, 3, 2)
        if pos.castling & WHITE_QUEENSIDE and clear_q and safe_q:
            moves[count] = encode_move_nb(4, 2, KING, PROMOTION_NONE, False, False, True, False)
            count += 1
    elif colour == BLACK and king_sq == 60:
        clear_k = occ & (bit(61) | bit(62)) == 0
        safe_k = not _squares_attacked_nb(pos, opp, 60, 61, 62)
        if pos.castling & BLACK_KINGSIDE and clear_k and safe_k:
            moves[count] = encode_move_nb(60, 62, KING, PROMOTION_NONE, False, False, True, False)
            count += 1
        clear_q = occ & (bit(57) | bit(58) | bit(59)) == 0
        safe_q = not _squares_attacked_nb(pos, opp, 60, 59, 58)
        if pos.castling & BLACK_QUEENSIDE and clear_q and safe_q:
            moves[count] = encode_move_nb(60, 58, KING, PROMOTION_NONE, False, False, True, False)
            count += 1

    return count


@njit(cache=True)
def _pseudo_legal_moves(pos: Board) -> tuple[np.ndarray, int]:
    moves = np.empty(MAX_MOVES, dtype=np.int64)
    count = 0

    colour = pos.side
    opp = 1 - colour
    ep_square = pos.ep_square
    own = pos.occupancy[colour]
    enemy = pos.occupancy[opp]
    occ = pos.occupancy[2]
    not_own = FULL_BB ^ own

    forward = 8 if colour == WHITE else -8
    start_rank = 1 if colour == WHITE else 6
    promo_rank = 7 if colour == WHITE else 0

    pawns = pos.pieces[colour, PAWN]
    while pawns:
        frm = lsb_index(pawns)
        pawns &= pawns - nb.uint64(1)

        one = frm + forward
        if 0 <= one < 64 and occ & bit(one) == 0:
            if square_rank(one) == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode_move_nb(frm, one, PAWN, promo, False, False, False, False)
                    count += 1
            else:
                moves[count] = encode_move_nb(
                    frm, one, PAWN, PROMOTION_NONE, False, False, False, False
                )
                count += 1
                if square_rank(frm) == start_rank:
                    two = one + forward
                    if occ & bit(two) == 0:
                        moves[count] = encode_move_nb(
                            frm, two, PAWN, PROMOTION_NONE, False, False, False, True
                        )
                        count += 1

        pawn_targets = PAWN_ATTACKS_NB[colour, frm] & enemy
        while pawn_targets:
            to = lsb_index(pawn_targets)
            pawn_targets &= pawn_targets - nb.uint64(1)
            if square_rank(to) == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode_move_nb(frm, to, PAWN, promo, True, False, False, False)
                    count += 1
            else:
                moves[count] = encode_move_nb(
                    frm, to, PAWN, PROMOTION_NONE, True, False, False, False
                )
                count += 1

        if ep_square != NO_SQUARE and PAWN_ATTACKS_NB[colour, frm] & bit(ep_square):  # type: ignore[arg-type]
            moves[count] = encode_move_nb(
                frm, ep_square, PAWN, PROMOTION_NONE, True, True, False, False  # type: ignore[arg-type]
            )
            count += 1

    knights = pos.pieces[colour, KNIGHT]
    while knights:
        frm = lsb_index(knights)
        knights &= knights - nb.uint64(1)
        targets = KNIGHT_ATTACKS_NB[frm] & not_own
        while targets:
            to = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_cap = (enemy & bit(to)) != 0
            moves[count] = encode_move_nb(
                frm, to, KNIGHT, PROMOTION_NONE, is_cap, False, False, False
            )
            count += 1

    bishops = pos.pieces[colour, BISHOP]
    while bishops:
        frm = lsb_index(bishops)
        bishops &= bishops - nb.uint64(1)
        targets = bishop_attacks(frm, occ) & not_own
        while targets:
            to = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_cap = (enemy & bit(to)) != 0
            moves[count] = encode_move_nb(
                frm, to, BISHOP, PROMOTION_NONE, is_cap, False, False, False
            )
            count += 1

    rooks = pos.pieces[colour, ROOK]
    while rooks:
        frm = lsb_index(rooks)
        rooks &= rooks - nb.uint64(1)
        targets = rook_attacks(frm, occ) & not_own
        while targets:
            to = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_cap = (enemy & bit(to)) != 0
            moves[count] = encode_move_nb(
                frm, to, ROOK, PROMOTION_NONE, is_cap, False, False, False
            )
            count += 1

    queens = pos.pieces[colour, QUEEN]
    while queens:
        frm = lsb_index(queens)
        queens &= queens - nb.uint64(1)
        targets = queen_attacks(frm, occ) & not_own
        while targets:
            to = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_cap = (enemy & bit(to)) != 0
            moves[count] = encode_move_nb(
                frm, to, QUEEN, PROMOTION_NONE, is_cap, False, False, False
            )
            count += 1

    kings = pos.pieces[colour, KING]
    while kings:
        frm = lsb_index(kings)
        kings &= kings - nb.uint64(1)
        targets = KING_ATTACKS_NB[frm] & not_own
        while targets:
            to = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_cap = (enemy & bit(to)) != 0
            moves[count] = encode_move_nb(
                frm, to, KING, PROMOTION_NONE, is_cap, False, False, False
            )
            count += 1
        count = _castling_moves_nb(pos, colour, frm, occ, moves, count)  # type: ignore[arg-type]

    return moves, count


@njit  # not cache=True: constructs a Board (via copy_board) internally, which numba's
# caching can't serialize to disk - see copy_board's own comment in board.py.
def make_move(pos: Board, move: int) -> Board:
    new: Board = copy_board(pos)  # type: ignore[assignment, type-var]  # njit return type
    colour = pos.side
    opp = 1 - colour

    frm = move_from_square(move)
    to = move_to_square(move)
    piece = move_piece(move)
    promotion = move_promotion_raw(move)
    is_capture = move_is_capture(move)
    is_castle = move_is_castle(move)
    is_ep = move_is_en_passant(move)
    captured_sq = to + (-8 if colour == WHITE else 8)  # only meaningful when is_ep
    if to == frm + 2:  # kingside: the rook a-side of the king hops in; only used when is_castle
        rook_from, rook_to = frm + 3, frm + 1
    else:  # queenside
        rook_from, rook_to = frm - 4, frm - 1

    # incremental zobrist: start from the parent's hash and XOR in every change below. it must
    # track zobrist.zobrist_hash exactly - tools/verify_zobrist.py checks that after every
    # move. side flips every move; the old en-passant file, if any, comes back out.
    h = pos.zobrist ^ SIDE_KEY
    if pos.ep_square != NO_SQUARE:
        h ^= EP_FILE_KEYS[pos.ep_square & 7]

    new.pieces[colour, piece] &= clear_mask(frm)
    h ^= PIECE_SQUARE_KEYS[colour, piece, frm]

    if is_ep:
        new.pieces[opp, PAWN] &= clear_mask(captured_sq)
        h ^= PIECE_SQUARE_KEYS[opp, PAWN, captured_sq]
    elif is_capture:
        for pt in range(6):
            if new.pieces[opp, pt] & bit(to):
                new.pieces[opp, pt] &= clear_mask(to)
                h ^= PIECE_SQUARE_KEYS[opp, pt, to]
                break

    placed = piece if promotion == PROMOTION_NONE else promotion
    new.pieces[colour, placed] |= bit(to)
    h ^= PIECE_SQUARE_KEYS[colour, placed, to]

    if is_castle:
        new.pieces[colour, ROOK] &= clear_mask(rook_from)
        new.pieces[colour, ROOK] |= bit(rook_to)
        h ^= PIECE_SQUARE_KEYS[colour, ROOK, rook_from] ^ PIECE_SQUARE_KEYS[colour, ROOK, rook_to]

    if piece == KING:
        if colour == WHITE:
            new.castling &= 0xFF ^ (WHITE_KINGSIDE | WHITE_QUEENSIDE)
        else:
            new.castling &= 0xFF ^ (BLACK_KINGSIDE | BLACK_QUEENSIDE)
    if frm == 0 or to == 0:
        new.castling &= 0xFF ^ WHITE_QUEENSIDE
    if frm == 7 or to == 7:
        new.castling &= 0xFF ^ WHITE_KINGSIDE
    if frm == 56 or to == 56:
        new.castling &= 0xFF ^ BLACK_QUEENSIDE
    if frm == 63 or to == 63:
        new.castling &= 0xFF ^ BLACK_KINGSIDE
    # zobrist_hash always mixes CASTLING_KEYS[rights]; swap old for new (a no-op XOR if equal).
    h ^= CASTLING_KEYS[pos.castling] ^ CASTLING_KEYS[new.castling]

    new.ep_square = (frm + to) // 2 if move_is_double_push(move) else NO_SQUARE  # type: ignore[assignment]
    if new.ep_square != NO_SQUARE:
        h ^= EP_FILE_KEYS[new.ep_square & 7]

    new.halfmove_clock = 0 if piece == PAWN or is_capture else pos.halfmove_clock + 1  # type: ignore[assignment]
    new.fullmove_number = pos.fullmove_number + (1 if colour == BLACK else 0)
    new.side = opp
    new.zobrist = h

    # occupancy, updated from the parent's rather than rebuilt from twelve bitboards: the mover
    # leaves frm and lands on to (a castling rook moves with it), and a captured man leaves the
    # board - the en-passant victim from its own square, any other from the to-square.
    own_occ = (pos.occupancy[colour] & clear_mask(frm)) | bit(to)
    if is_castle:
        own_occ = (own_occ & clear_mask(rook_from)) | bit(rook_to)
    opp_occ = pos.occupancy[opp]
    if is_ep:
        opp_occ &= clear_mask(captured_sq)
    elif is_capture:
        opp_occ &= clear_mask(to)
    new.occupancy[colour] = own_occ
    new.occupancy[opp] = opp_occ
    new.occupancy[2] = own_occ | opp_occ
    return new


@njit(cache=True)
def _attacked_by_with_occ(pos: Board, sq: int, by_colour: int, occ: int) -> bool:
    """is_attacked, but on a caller-supplied occupancy. Used to test a king's destination with
    the king lifted off the board, so it cannot shield the square it is fleeing along."""
    opp = 1 - by_colour
    if PAWN_ATTACKS_NB[opp, sq] & pos.pieces[by_colour, PAWN]:
        return True
    if KNIGHT_ATTACKS_NB[sq] & pos.pieces[by_colour, KNIGHT]:
        return True
    if KING_ATTACKS_NB[sq] & pos.pieces[by_colour, KING]:
        return True
    bishops_queens = pos.pieces[by_colour, BISHOP] | pos.pieces[by_colour, QUEEN]
    if bishop_attacks(nb.uint8(sq), occ) & bishops_queens:  # type: ignore[arg-type]
        return True
    rooks_queens = pos.pieces[by_colour, ROOK] | pos.pieces[by_colour, QUEEN]
    return bool(rook_attacks(nb.uint8(sq), occ) & rooks_queens)  # type: ignore[arg-type]


@njit(cache=True)
def legal_moves(pos: Board) -> tuple[np.ndarray, int]:
    """Legal moves, decided without playing them: compute once per position which enemy pieces
    check the king (so non-king moves must capture the checker or block between it and the king)
    and which of our pieces are pinned (so they may only slide along the pin line). Every move
    that isn't a king move or an en passant then passes with two bitboard tests and no make_move.
    En passant - rare, and the one case a rank-skewer pin can hide - keeps the make-and-test."""
    pseudo, pseudo_count = _pseudo_legal_moves(pos)

    us = pos.side
    them = 1 - us
    occ = pos.occupancy[2]
    own_occ = pos.occupancy[us]
    enemy_occ = pos.occupancy[them]
    ksq = lsb_index(pos.pieces[us, KING])

    bishops_queens = pos.pieces[them, BISHOP] | pos.pieces[them, QUEEN]
    rooks_queens = pos.pieces[them, ROOK] | pos.pieces[them, QUEEN]

    checkers = KNIGHT_ATTACKS_NB[ksq] & pos.pieces[them, KNIGHT]
    checkers |= PAWN_ATTACKS_NB[us, ksq] & pos.pieces[them, PAWN]
    checkers |= bishop_attacks(nb.uint8(ksq), occ) & bishops_queens  # type: ignore[arg-type]
    checkers |= rook_attacks(nb.uint8(ksq), occ) & rooks_queens  # type: ignore[arg-type]

    double_check = False
    if checkers == 0:
        block_mask = nb.uint64(FULL_BB)
    elif checkers & (checkers - nb.uint64(1)) == 0:
        block_mask = checkers | _BETWEEN_NB[ksq, lsb_index(checkers)]
    else:
        block_mask = nb.uint64(0)
        double_check = True

    # a piece is pinned when an enemy slider's path to the king is blocked by it alone
    pinned = nb.uint64(0)
    snipers = rook_attacks(nb.uint8(ksq), enemy_occ) & rooks_queens  # type: ignore[arg-type]
    snipers |= bishop_attacks(nb.uint8(ksq), enemy_occ) & bishops_queens  # type: ignore[arg-type]
    while snipers:
        sniper_sq = lsb_index(snipers)
        snipers &= snipers - nb.uint64(1)
        blockers = _BETWEEN_NB[ksq, sniper_sq] & occ
        if blockers != 0 and blockers & (blockers - nb.uint64(1)) == 0 and blockers & own_occ:
            pinned |= blockers

    # legal moves are compacted back into the pseudo array in place - the write index never
    # overtakes the read index - so this costs no second MAX_MOVES allocation per node.
    legal_count = 0
    for i in range(pseudo_count):
        move = pseudo[i]
        piece = move_piece(move)
        frm = move_from_square(move)
        to = move_to_square(move)

        if piece == KING:
            if move_is_castle(move):
                pseudo[legal_count] = move  # _castling_moves_nb proved every square safe already
                legal_count += 1
                continue
            test_occ = occ ^ bit(ksq)
            if move_is_capture(move):
                test_occ &= clear_mask(to)
            if not _attacked_by_with_occ(pos, to, them, test_occ):  # type: ignore[arg-type]
                pseudo[legal_count] = move
                legal_count += 1
            continue

        if double_check:
            continue

        if move_is_en_passant(move):
            if not is_check(make_move(pos, move), us):  # type: ignore[arg-type, type-var, call-arg]
                pseudo[legal_count] = move
                legal_count += 1
            continue

        if bit(to) & block_mask == 0:
            continue
        if bit(frm) & pinned and bit(to) & _LINE_NB[ksq, frm] == 0:
            continue
        pseudo[legal_count] = move
        legal_count += 1

    return pseudo, legal_count


# compile every jitted function above now, inside the init budget - a real position, not an
# empty Board(): is_check (and anything that calls it) calls lsb_index on the king
# bitboard, which loops forever on an empty board with no king to find.
_warm_pos = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
is_attacked(_warm_pos, 4, WHITE)
is_check(_warm_pos, WHITE)
_attacked_by_with_occ(_warm_pos, 4, BLACK, int(_warm_pos.occupancy[2]))
_warm_moves, _warm_count = _pseudo_legal_moves(_warm_pos)
_warm_legal, _warm_legal_count = legal_moves(_warm_pos)
make_move(_warm_pos, int(_warm_moves[0]))  # type: ignore[type-var, call-arg]
# a position with our king in check and a friendly piece pinned, so legal_moves compiles its
# check-evasion and pin branches now rather than on the first such position in a real game
legal_moves(parse_fen("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1"))
legal_moves(parse_fen("rnbqkbnr/ppp1p1pp/3p4/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"))
