"""Pseudo-legal move generation, make_move, and legality filtering - the bitboard-layer stand-in
for python-chess's own: `legal_moves` ~ `chess.Board.legal_moves`, `make_move` ~
`chess.Board.push` (on a copy, so there is no pop), `is_check` ~ `chess.Board.is_check`.

The `*_reference` functions (is_attacked_reference, is_check_reference, make_move_reference,
legal_moves_reference, pseudo_legal_moves_reference) are plain-Python originals kept only so
bench/perft.py and bench/verify_movegen.py can check the jitted versions against them move for
move; nothing on the hot path calls them.

legal_moves_reference checks legality the simple, slow way: generate every pseudo-legal move
(piece attack pattern, ignoring pins and existing checks), make it on a fresh copy, and keep it
only if the mover's own king is safe afterward. The jitted legal_moves does better - it decides
each move with two bitboard tests and no make_move - but the reference is what proves it right.

Moves are packed ints (move.py), not objects - see that module's docstring for why.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numba import njit

from accumulator import EG_MEN, fill_accumulator, men_of, update_feature
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

ROOK_HOME_TO_CASTLE_FLAG = {
    0: WHITE_QUEENSIDE, 7: WHITE_KINGSIDE, 56: BLACK_QUEENSIDE, 63: BLACK_KINGSIDE,
}


# Squares strictly between two collinear squares, and the whole board line running through
# them - both 0 for a pair that shares no rank, file, or diagonal. Built once in plain Python
# (a 64x64 pair of tables, 64 KB) and frozen into numpy arrays for the jitted legality check.
# They are what let legal_moves() rule on a move without making it: a pinned piece may only
# travel along LINE_NB[king][piece], and a single check is answered only on the checker's
# square or BETWEEN_NB[king][checker].
def build_line_tables() -> tuple[np.ndarray, np.ndarray]:
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    between = [[0] * 64 for _ in range(64)]
    line = [[0] * 64 for _ in range(64)]

    for from_square in range(64):
        from_file, from_rank = from_square & 7, from_square >> 3

        for file_step, rank_step in directions:
            # walk out from the square in this direction, collecting the squares passed
            ray: list[int] = []
            file, rank = from_file, from_rank

            while 0 <= file < 8 and 0 <= rank < 8:
                ray.append(rank * 8 + file)
                file, rank = file + file_step, rank + rank_step
            # the full line is the forward ray plus the same direction walked backward
            line_bits = 0

            for square in ray:
                line_bits |= 1 << square

            file, rank = from_file - file_step, from_rank - rank_step

            while 0 <= file < 8 and 0 <= rank < 8:
                line_bits |= 1 << (rank * 8 + file)
                file, rank = file - file_step, rank - rank_step

            # for each square further along the ray, "between" is the squares strictly before it
            for target_index in range(1, len(ray)):
                between_bits = 0

                for inner in range(1, target_index):
                    between_bits |= 1 << ray[inner]

                between[from_square][ray[target_index]] = between_bits
                line[from_square][ray[target_index]] = line_bits

    between_nb = np.zeros((64, 64), dtype=np.uint64)
    line_nb = np.zeros((64, 64), dtype=np.uint64)

    for from_square in range(64):
        for to_square in range(64):
            between_nb[from_square, to_square] = np.uint64(between[from_square][to_square])
            line_nb[from_square, to_square] = np.uint64(line[from_square][to_square])

    return between_nb, line_nb


BETWEEN_NB, LINE_NB = build_line_tables()


# Is `square` attacked by any of `by_colour`'s pieces, on the position as it stands? Reads each
# attacker type from the target square outward - chess.Board.is_attacked_by, plain-Python.
def is_attacked_reference(board: Board, square: int, by_colour: int) -> bool:
    occupied = board.occupancy[2]
    enemy_colour = 1 - by_colour

    if PAWN_ATTACKS[enemy_colour][square] & board.pieces[by_colour][PAWN]:
        return True
    
    if KNIGHT_ATTACKS[square] & board.pieces[by_colour][KNIGHT]:
        return True
    
    if KING_ATTACKS[square] & board.pieces[by_colour][KING]:
        return True
    

    bishops_queens = board.pieces[by_colour][BISHOP] | board.pieces[by_colour][QUEEN]
    if bishop_attacks(square, occupied) & bishops_queens:
        return True

    rooks_queens = board.pieces[by_colour][ROOK] | board.pieces[by_colour][QUEEN]
    return bool(rook_attacks(square, occupied) & rooks_queens)


# Is `colour`'s king attacked? chess.Board.is_check, plain-Python.
def is_check_reference(board: Board, colour: int) -> bool:
    king_square = next(scan_forward(int(board.pieces[colour][KING])))
    return is_attacked_reference(board, king_square, 1 - colour)


# The castling moves available to `colour` (king still on its home square, rook still home, the
# squares between empty, and the king not passing through check). Plain-Python half of
# pseudo_legal_moves_reference.
def castling_moves(board: Board, colour: int, king_square: int, occupied: int) -> list[int]:
    moves = []
    enemy_colour = 1 - colour

    if colour == WHITE and king_square == 4:
        kingside_clear = not occupied & (bit(5) | bit(6))

        if board.castling & WHITE_KINGSIDE and kingside_clear and not any(
            is_attacked_reference(board, sq, enemy_colour) for sq in (4, 5, 6)
        ):
            moves.append(encode_move(4, 6, KING, is_castle=True))

        queenside_clear = not occupied & (bit(1) | bit(2) | bit(3))

        if board.castling & WHITE_QUEENSIDE and queenside_clear and not any(
            is_attacked_reference(board, sq, enemy_colour) for sq in (4, 3, 2)
        ):
            moves.append(encode_move(4, 2, KING, is_castle=True))

    elif colour == BLACK and king_square == 60:
        kingside_clear = not occupied & (bit(61) | bit(62))

        if board.castling & BLACK_KINGSIDE and kingside_clear and not any(
            is_attacked_reference(board, sq, enemy_colour) for sq in (60, 61, 62)
        ):
            moves.append(encode_move(60, 62, KING, is_castle=True))

        queenside_clear = not occupied & (bit(57) | bit(58) | bit(59))

        if board.castling & BLACK_QUEENSIDE and queenside_clear and not any(
            is_attacked_reference(board, sq, enemy_colour) for sq in (60, 59, 58)
        ):
            moves.append(encode_move(60, 58, KING, is_castle=True))

    return moves


# Every pseudo-legal move for the side to move - piece attack patterns only, pins and existing
# checks ignored. legal_moves_reference filters these; the jitted pseudo_legal_moves below is
# the twin verify_movegen.py checks it against.
def pseudo_legal_moves_reference(board: Board) -> list[int]:
    moves: list[int] = []
    colour = int(board.side)
    enemy_colour = 1 - colour
    ep_square = int(board.ep_square)
    own = int(board.occupancy[colour])
    enemy_occ = int(board.occupancy[enemy_colour])
    occupied = int(board.occupancy[2])
    not_own = FULL_BB ^ own

    forward = 8 if colour == WHITE else -8
    start_rank = 1 if colour == WHITE else 6
    promo_rank = 7 if colour == WHITE else 0

    for from_square in scan_forward(int(board.pieces[colour][PAWN])):
        one_ahead = from_square + forward

        if 0 <= one_ahead < 64 and not occupied & bit(one_ahead):

            if square_rank(one_ahead) == promo_rank:
                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append(encode_move(from_square, one_ahead, PAWN, promotion=promotion))
            else:
                moves.append(encode_move(from_square, one_ahead, PAWN))

                if square_rank(from_square) == start_rank:
                    two_ahead = one_ahead + forward

                    if not occupied & bit(two_ahead):
                        moves.append(
                            encode_move(from_square, two_ahead, PAWN, is_double_push=True)
                        )

        for to_square in scan_forward(PAWN_ATTACKS[colour][from_square] & enemy_occ):

            if square_rank(to_square) == promo_rank:

                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append(
                        encode_move(
                            from_square, to_square, PAWN, promotion=promotion, is_capture=True
                        )
                    )
            else:
                moves.append(encode_move(from_square, to_square, PAWN, is_capture=True))

        if ep_square != NO_SQUARE and PAWN_ATTACKS[colour][from_square] & bit(ep_square):
            moves.append(
                encode_move(from_square, ep_square, PAWN, is_capture=True, is_en_passant=True)
            )

    for from_square in scan_forward(int(board.pieces[colour][KNIGHT])):
        for to_square in scan_forward(KNIGHT_ATTACKS[from_square] & not_own):
            is_capture = bool(enemy_occ & bit(to_square))
            moves.append(encode_move(from_square, to_square, KNIGHT, is_capture=is_capture))

    for from_square in scan_forward(int(board.pieces[colour][BISHOP])):
        for to_square in scan_forward(bishop_attacks(from_square, occupied) & not_own):
            is_capture = bool(enemy_occ & bit(to_square))
            moves.append(encode_move(from_square, to_square, BISHOP, is_capture=is_capture))

    for from_square in scan_forward(int(board.pieces[colour][ROOK])):
        for to_square in scan_forward(rook_attacks(from_square, occupied) & not_own):
            is_capture = bool(enemy_occ & bit(to_square))
            moves.append(encode_move(from_square, to_square, ROOK, is_capture=is_capture))

    for from_square in scan_forward(int(board.pieces[colour][QUEEN])):
        for to_square in scan_forward(queen_attacks(from_square, occupied) & not_own):
            is_capture = bool(enemy_occ & bit(to_square))
            moves.append(encode_move(from_square, to_square, QUEEN, is_capture=is_capture))

    for from_square in scan_forward(int(board.pieces[colour][KING])):
        for to_square in scan_forward(KING_ATTACKS[from_square] & not_own):
            is_capture = bool(enemy_occ & bit(to_square))
            moves.append(encode_move(from_square, to_square, KING, is_capture=is_capture))

        moves.extend(castling_moves(board, colour, from_square, occupied))

    return moves


# Clear the castling right that a rook leaving (or being captured on) `square` forfeits.
def clear_castling_right(board: Board, square: int) -> None:
    flag = ROOK_HOME_TO_CASTLE_FLAG.get(square)
    if flag is not None:
        board.castling &= 0xFF ^ flag  # not ~flag: board.castling is a jitclass uint8 field, and
        # a negative Python int can't combine into one any more than into a uint64 array element
        # (see bitboard.clear_mask) - same fix, bounded to a byte instead of 64 bits.


# Apply `move` to a copy of `board` and return it - chess.Board.push, plain-Python. make_move
# below is the jitted twin (and also keeps the zobrist hash in step, which this does not).
def make_move_reference(board: Board, move: int) -> Board:
    new: Board = copy_board(board)  # type: ignore[assignment, type-var]  # njit return type
    colour = int(board.side)
    enemy_colour = 1 - colour

    from_square = move_from_square(move)
    to_square = move_to_square(move)
    piece = move_piece(move)
    promotion = move_promotion(move)
    is_capture = move_is_capture(move)
    is_castle = move_is_castle(move)

    new.pieces[colour][piece] &= clear_mask(from_square)

    if move_is_en_passant(move):
        captured_square = to_square + (-8 if colour == WHITE else 8)
        new.pieces[enemy_colour][PAWN] &= clear_mask(captured_square)
    elif is_capture:
        for piece_type in range(6):
        
            if new.pieces[enemy_colour][piece_type] & bit(to_square):
                new.pieces[enemy_colour][piece_type] &= clear_mask(to_square)
                break

    placed = promotion if promotion is not None else piece
    new.pieces[colour][placed] |= bit(to_square)

    if is_castle:

        if to_square == from_square + 2:  # kingside: rook a-side of the king moves in
            rook_from, rook_to = from_square + 3, from_square + 1
        else:  # queenside
            rook_from, rook_to = from_square - 4, from_square - 1

        new.pieces[colour][ROOK] &= clear_mask(rook_from)
        new.pieces[colour][ROOK] |= bit(rook_to)

    if piece == KING:
        if colour == WHITE:
            new.castling &= 0xFF ^ (WHITE_KINGSIDE | WHITE_QUEENSIDE)
        else:
            new.castling &= 0xFF ^ (BLACK_KINGSIDE | BLACK_QUEENSIDE)

    clear_castling_right(new, from_square)
    clear_castling_right(new, to_square)

    ep = (from_square + to_square) // 2 if move_is_double_push(move) else NO_SQUARE
    new.ep_square = ep  # type: ignore[assignment]  # jitclass field, see board.py
    new.halfmove_clock = 0 if piece == PAWN or is_capture else board.halfmove_clock + 1  # type: ignore[assignment]
    new.fullmove_number = board.fullmove_number + (1 if colour == BLACK else 0)
    new.side = enemy_colour  # type: ignore[assignment]

    recompute_occupancy(new)
    fill_accumulator(new.pieces, new.acc)
    return new


# Legal moves the slow, obviously-correct way: play each pseudo-legal move and keep it only if
# the mover's king is safe after. chess.Board.legal_moves, plain-Python.
def legal_moves_reference(board: Board) -> list[int]:
    mover = int(board.side)
    legal = []

    for move in pseudo_legal_moves_reference(board):
        if not is_check_reference(make_move_reference(board, move), mover):
            legal.append(move)

    return legal


# Below: jitted equivalents of everything above. Same algorithms, but nopython-compatible: no
# Python lists (moves fill a preallocated array up to MAX_MOVES instead), no dict lookups
# (castling rights use a small if/elif chain on the square instead of ROOK_HOME_TO_CASTLE_FLAG),
# no scan_forward generator (lsb_index plus an explicit "clear the lowest bit and loop"
# instead), no any()-over-a-generator (an explicit loop with an early exit). The plain versions
# above stay as the reference bench/verify_movegen.py checks these against - move for move, not
# just perft's leaf counts, since two different bugs could cancel out into the same node count.


# is_attacked_reference, jitted.
@njit(cache=True)
def is_attacked(board: Board, square: int, by_colour: int) -> bool:
    occupied = board.occupancy[2]
    enemy_colour = 1 - by_colour

    if PAWN_ATTACKS_NB[enemy_colour, square] & board.pieces[by_colour, PAWN]:
        return True
    if KNIGHT_ATTACKS_NB[square] & board.pieces[by_colour, KNIGHT]:
        return True
    if KING_ATTACKS_NB[square] & board.pieces[by_colour, KING]:
        return True

    bishops_queens = board.pieces[by_colour, BISHOP] | board.pieces[by_colour, QUEEN]
    if bishop_attacks(nb.uint8(square), occupied) & bishops_queens:
        return True

    rooks_queens = board.pieces[by_colour, ROOK] | board.pieces[by_colour, QUEEN]
    return bool(rook_attacks(nb.uint8(square), occupied) & rooks_queens)


# is_check_reference, jitted - the hot-path check test the search calls every node.
@njit(cache=True)
def is_check(board: Board, colour: int) -> bool:
    king_square = lsb_index(board.pieces[colour, KING])
    return is_attacked(board, king_square, 1 - colour)


# Is any of `start` / `middle` / `end` attacked by by_colour? Castling's "king does not pass
# through check" rule always tests exactly those three squares.
@njit(cache=True)
def squares_attacked_nb(
    board: Board, by_colour: int, start: int, middle: int, end: int
) -> bool:
    return (
        is_attacked(board, start, by_colour)
        or is_attacked(board, middle, by_colour)
        or is_attacked(board, end, by_colour)
    )


# castling_moves, jitted: append any available castling move into `moves` and return the new
# count.
@njit(cache=True)
def castling_moves_nb(
    board: Board, colour: int, king_square: int, occupied: int, moves: np.ndarray, count: int
) -> int:
    enemy_colour = 1 - colour

    if colour == WHITE and king_square == 4:
        kingside_clear = occupied & (bit(5) | bit(6)) == 0
        kingside_safe = not squares_attacked_nb(board, enemy_colour, 4, 5, 6)

        if board.castling & WHITE_KINGSIDE and kingside_clear and kingside_safe:
            moves[count] = encode_move_nb(4, 6, KING, PROMOTION_NONE, False, False, True, False)
            count += 1

        queenside_clear = occupied & (bit(1) | bit(2) | bit(3)) == 0
        queenside_safe = not squares_attacked_nb(board, enemy_colour, 4, 3, 2)

        if board.castling & WHITE_QUEENSIDE and queenside_clear and queenside_safe:
            moves[count] = encode_move_nb(4, 2, KING, PROMOTION_NONE, False, False, True, False)
            count += 1

    elif colour == BLACK and king_square == 60:
        kingside_clear = occupied & (bit(61) | bit(62)) == 0
        kingside_safe = not squares_attacked_nb(board, enemy_colour, 60, 61, 62)

        if board.castling & BLACK_KINGSIDE and kingside_clear and kingside_safe:
            moves[count] = encode_move_nb(60, 62, KING, PROMOTION_NONE, False, False, True, False)
            count += 1

        queenside_clear = occupied & (bit(57) | bit(58) | bit(59)) == 0
        queenside_safe = not squares_attacked_nb(board, enemy_colour, 60, 59, 58)

        if board.castling & BLACK_QUEENSIDE and queenside_clear and queenside_safe:
            moves[count] = encode_move_nb(60, 58, KING, PROMOTION_NONE, False, False, True, False)
            count += 1

    return count


# pseudo_legal_moves_reference, jitted: fills a preallocated (moves, count) instead of a list.
@njit(cache=True)
def pseudo_legal_moves(board: Board) -> tuple[np.ndarray, int]:
    moves = np.empty(MAX_MOVES, dtype=np.int64)
    count = 0

    colour = board.side
    enemy_colour = 1 - colour
    ep_square = board.ep_square
    own = board.occupancy[colour]
    enemy_occ = board.occupancy[enemy_colour]
    occupied = board.occupancy[2]
    not_own = FULL_BB ^ own

    forward = 8 if colour == WHITE else -8
    start_rank = 1 if colour == WHITE else 6
    promo_rank = 7 if colour == WHITE else 0

    pawns = board.pieces[colour, PAWN]
    while pawns:
        from_square = lsb_index(pawns)
        pawns &= pawns - nb.uint64(1)

        one_ahead = from_square + forward
        if 0 <= one_ahead < 64 and occupied & bit(one_ahead) == 0:
            if square_rank(one_ahead) == promo_rank:
                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode_move_nb(
                        from_square, one_ahead, PAWN, promotion, False, False, False, False
                    )
                    count += 1
            else:
                moves[count] = encode_move_nb(
                    from_square, one_ahead, PAWN, PROMOTION_NONE, False, False, False, False
                )
                count += 1
                if square_rank(from_square) == start_rank:
                    two_ahead = one_ahead + forward

                    if occupied & bit(two_ahead) == 0:
                        moves[count] = encode_move_nb(
                            from_square, two_ahead, PAWN, PROMOTION_NONE, False, False, False, True
                        )
                        count += 1

        pawn_targets = PAWN_ATTACKS_NB[colour, from_square] & enemy_occ
        while pawn_targets:

            to_square = lsb_index(pawn_targets)
            pawn_targets &= pawn_targets - nb.uint64(1)

            if square_rank(to_square) == promo_rank:
                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode_move_nb(
                        from_square, to_square, PAWN, promotion, True, False, False, False
                    )
                    count += 1
            else:
                moves[count] = encode_move_nb(
                    from_square, to_square, PAWN, PROMOTION_NONE, True, False, False, False
                )
                count += 1

        if ep_square != NO_SQUARE and PAWN_ATTACKS_NB[colour, from_square] & bit(ep_square):
            moves[count] = encode_move_nb(
                from_square, ep_square, PAWN, PROMOTION_NONE, True, True, False, False
            )
            count += 1

    knights = board.pieces[colour, KNIGHT]
    while knights:

        from_square = lsb_index(knights)
        knights &= knights - nb.uint64(1)
        targets = KNIGHT_ATTACKS_NB[from_square] & not_own

        while targets:
            to_square = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_capture = (enemy_occ & bit(to_square)) != 0
            moves[count] = encode_move_nb(
                from_square, to_square, KNIGHT, PROMOTION_NONE, is_capture, False, False, False
            )
            count += 1

    bishops = board.pieces[colour, BISHOP]
    while bishops:

        from_square = lsb_index(bishops)
        bishops &= bishops - nb.uint64(1)
        targets = bishop_attacks(from_square, occupied) & not_own

        while targets:
            to_square = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_capture = (enemy_occ & bit(to_square)) != 0
            moves[count] = encode_move_nb(
                from_square, to_square, BISHOP, PROMOTION_NONE, is_capture, False, False, False
            )
            count += 1

    rooks = board.pieces[colour, ROOK]
    while rooks:

        from_square = lsb_index(rooks)
        rooks &= rooks - nb.uint64(1)
        targets = rook_attacks(from_square, occupied) & not_own

        while targets:
            to_square = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_capture = (enemy_occ & bit(to_square)) != 0
            moves[count] = encode_move_nb(
                from_square, to_square, ROOK, PROMOTION_NONE, is_capture, False, False, False
            )
            count += 1

    queens = board.pieces[colour, QUEEN]
    while queens:

        from_square = lsb_index(queens)
        queens &= queens - nb.uint64(1)
        targets = queen_attacks(from_square, occupied) & not_own

        while targets:
            to_square = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_capture = (enemy_occ & bit(to_square)) != 0
            moves[count] = encode_move_nb(
                from_square, to_square, QUEEN, PROMOTION_NONE, is_capture, False, False, False
            )
            count += 1

    kings = board.pieces[colour, KING]
    while kings:

        from_square = lsb_index(kings)
        kings &= kings - nb.uint64(1)
        targets = KING_ATTACKS_NB[from_square] & not_own

        while targets:
            to_square = lsb_index(targets)
            targets &= targets - nb.uint64(1)
            is_capture = (enemy_occ & bit(to_square)) != 0
            moves[count] = encode_move_nb(
                from_square, to_square, KING, PROMOTION_NONE, is_capture, False, False, False
            )
            count += 1
        count = castling_moves_nb(board, colour, from_square, occupied, moves, count)

    return moves, count


# make_move_reference, jitted, plus the incremental zobrist hash the reference does not keep.
@njit  # not cache=True: constructs a Board (via copy_board) internally, which numba's
# caching can't serialize to disk - see copy_board's own comment in board.py.
def make_move(board: Board, move: int) -> Board:
    new: Board = copy_board(board)  # type: ignore[assignment, type-var]  # njit return type
    colour = board.side
    enemy_colour = 1 - colour

    from_square = move_from_square(move)
    to_square = move_to_square(move)
    piece = move_piece(move)
    promotion = move_promotion_raw(move)
    is_capture = move_is_capture(move)
    is_castle = move_is_castle(move)
    is_en_passant = move_is_en_passant(move)
    # only meaningful when the corresponding flag is set
    captured_square = to_square + (-8 if colour == WHITE else 8)

    if to_square == from_square + 2:  # kingside; rook_from / rook_to only used when is_castle
        rook_from, rook_to = from_square + 3, from_square + 1
    else:  # queenside
        rook_from, rook_to = from_square - 4, from_square - 1

    # incremental zobrist: start from the parent's hash and XOR in every change below. it must
    # track zobrist.zobrist_hash exactly - bench/verify_zobrist.py checks that after every
    # move. side flips every move; the old en-passant file, if any, comes back out.
    key = board.zobrist ^ SIDE_KEY
    if board.ep_square != NO_SQUARE:
        key ^= EP_FILE_KEYS[board.ep_square & 7]

    # every piece bitboard edit below is mirrored into new.acc by an update_feature call, so the
    # child accumulator is the parent's (copy_board carried it over) plus a handful of column
    # deltas instead of a full transformer pass. bench/verify_nnue.py checks this stays equal to
    # a from-scratch fill after every move. The net is the one the child's men count picks: a
    # capture that takes the board down to EG_MEN men crosses into the endgame net, whose
    # accumulator has nothing in common with the parent's, so that child is filled from scratch
    # once the pieces are placed (a few column adds - the board is small by then).
    parent_men = men_of(board.pieces)
    child_men = parent_men - 1 if (is_capture or is_en_passant) else parent_men
    eg = child_men <= EG_MEN
    crossing = eg and parent_men > EG_MEN

    new.pieces[colour, piece] &= clear_mask(from_square)
    key ^= PIECE_SQUARE_KEYS[colour, piece, from_square]
    if not crossing:
        update_feature(new.acc, colour, piece, from_square, -1, eg)

    if is_en_passant:
        new.pieces[enemy_colour, PAWN] &= clear_mask(captured_square)
        key ^= PIECE_SQUARE_KEYS[enemy_colour, PAWN, captured_square]
        if not crossing:
            update_feature(new.acc, enemy_colour, PAWN, captured_square, -1, eg)
    elif is_capture:
        for piece_type in range(6):

            if new.pieces[enemy_colour, piece_type] & bit(to_square):
                new.pieces[enemy_colour, piece_type] &= clear_mask(to_square)
                key ^= PIECE_SQUARE_KEYS[enemy_colour, piece_type, to_square]
                if not crossing:
                    update_feature(new.acc, enemy_colour, piece_type, to_square, -1, eg)
                break

    placed = piece if promotion == PROMOTION_NONE else promotion
    new.pieces[colour, placed] |= bit(to_square)
    key ^= PIECE_SQUARE_KEYS[colour, placed, to_square]
    if not crossing:
        update_feature(new.acc, colour, placed, to_square, 1, eg)

    if is_castle:
        new.pieces[colour, ROOK] &= clear_mask(rook_from)
        new.pieces[colour, ROOK] |= bit(rook_to)
        key ^= PIECE_SQUARE_KEYS[colour, ROOK, rook_from] ^ PIECE_SQUARE_KEYS[colour, ROOK, rook_to]
        if not crossing:
            update_feature(new.acc, colour, ROOK, rook_from, -1, eg)
            update_feature(new.acc, colour, ROOK, rook_to, 1, eg)

    if crossing:
        fill_accumulator(new.pieces, new.acc)

    if piece == KING:
        if colour == WHITE:
            new.castling &= 0xFF ^ (WHITE_KINGSIDE | WHITE_QUEENSIDE)
        else:
            new.castling &= 0xFF ^ (BLACK_KINGSIDE | BLACK_QUEENSIDE)

    if from_square == 0 or to_square == 0:
        new.castling &= 0xFF ^ WHITE_QUEENSIDE
    if from_square == 7 or to_square == 7:
        new.castling &= 0xFF ^ WHITE_KINGSIDE
    if from_square == 56 or to_square == 56:
        new.castling &= 0xFF ^ BLACK_QUEENSIDE
    if from_square == 63 or to_square == 63:
        new.castling &= 0xFF ^ BLACK_KINGSIDE
    # zobrist_hash always mixes CASTLING_KEYS[rights]; swap old for new (a no-op XOR if equal).
    key ^= CASTLING_KEYS[board.castling] ^ CASTLING_KEYS[new.castling]

    new.ep_square = (from_square + to_square) // 2 if move_is_double_push(move) else NO_SQUARE  # type: ignore[assignment]
    if new.ep_square != NO_SQUARE:
        key ^= EP_FILE_KEYS[new.ep_square & 7]

    new.halfmove_clock = 0 if piece == PAWN or is_capture else board.halfmove_clock + 1  # type: ignore[assignment]
    new.fullmove_number = board.fullmove_number + (1 if colour == BLACK else 0)
    new.side = enemy_colour
    new.zobrist = key

    # occupancy, updated from the parent's rather than rebuilt from twelve bitboards: the mover
    # leaves from_square and lands on to_square (a castling rook moves with it), and a captured
    # man leaves the board - the en-passant victim from its own square, any other from to_square.
    own_occ = (board.occupancy[colour] & clear_mask(from_square)) | bit(to_square)

    if is_castle:
        own_occ = (own_occ & clear_mask(rook_from)) | bit(rook_to)

    enemy_occ = board.occupancy[enemy_colour]
    if is_en_passant:
        enemy_occ &= clear_mask(captured_square)

    elif is_capture:
        enemy_occ &= clear_mask(to_square)

    new.occupancy[colour] = own_occ
    new.occupancy[enemy_colour] = enemy_occ
    new.occupancy[2] = own_occ | enemy_occ
    return new


# is_attacked, but on a caller-supplied occupancy. Used to test a king's destination with the
# king lifted off the board, so it cannot shield the square it is fleeing along.
@njit(cache=True)
def attacked_by_with_occ(board: Board, square: int, by_colour: int, occupied: int) -> bool:
    enemy_colour = 1 - by_colour

    if PAWN_ATTACKS_NB[enemy_colour, square] & board.pieces[by_colour, PAWN]:
        return True
    if KNIGHT_ATTACKS_NB[square] & board.pieces[by_colour, KNIGHT]:
        return True
    if KING_ATTACKS_NB[square] & board.pieces[by_colour, KING]:
        return True
    
    bishops_queens = board.pieces[by_colour, BISHOP] | board.pieces[by_colour, QUEEN]
    if bishop_attacks(nb.uint8(square), occupied) & bishops_queens:
        return True
    
    rooks_queens = board.pieces[by_colour, ROOK] | board.pieces[by_colour, QUEEN]
    return bool(rook_attacks(nb.uint8(square), occupied) & rooks_queens)


# Legal moves, decided without playing them: compute once per position which enemy pieces check
# the king (so non-king moves must capture the checker or block between it and the king) and
# which of our pieces are pinned (so they may only slide along the pin line). Every move that
# isn't a king move or an en passant then passes with two bitboard tests and no make_move. En
# passant - rare, and the one case a rank-skewer pin can hide - keeps the make-and-test.
# The jitted, hot-path form of legal_moves_reference; chess.Board.legal_moves.
@njit(cache=True)
def legal_moves(board: Board) -> tuple[np.ndarray, int]:
    pseudo, pseudo_count = pseudo_legal_moves(board)

    us = board.side
    them = 1 - us
    occupied = board.occupancy[2]
    own_occ = board.occupancy[us]
    enemy_occ = board.occupancy[them]
    king_square = lsb_index(board.pieces[us, KING])

    bishops_queens = board.pieces[them, BISHOP] | board.pieces[them, QUEEN]
    rooks_queens = board.pieces[them, ROOK] | board.pieces[them, QUEEN]

    # every enemy piece giving check right now
    checkers = KNIGHT_ATTACKS_NB[king_square] & board.pieces[them, KNIGHT]
    checkers |= PAWN_ATTACKS_NB[us, king_square] & board.pieces[them, PAWN]
    checkers |= bishop_attacks(nb.uint8(king_square), occupied) & bishops_queens
    checkers |= rook_attacks(nb.uint8(king_square), occupied) & rooks_queens

    # squares a non-king move is allowed to land on: anywhere if not in check, the checker or a
    # blocking square if singly checked, nowhere (king must move) if doubly checked
    double_check = False

    if checkers == 0:
        block_mask = nb.uint64(FULL_BB)
    elif checkers & (checkers - nb.uint64(1)) == 0:
        block_mask = checkers | BETWEEN_NB[king_square, lsb_index(checkers)]
    else:
        block_mask = nb.uint64(0)
        double_check = True

    # a piece is pinned when an enemy slider's path to the king is blocked by it alone
    pinned = nb.uint64(0)
    snipers = rook_attacks(nb.uint8(king_square), enemy_occ) & rooks_queens
    snipers |= bishop_attacks(nb.uint8(king_square), enemy_occ) & bishops_queens
    while snipers:

        sniper_square = lsb_index(snipers)
        snipers &= snipers - nb.uint64(1)
        blockers = BETWEEN_NB[king_square, sniper_square] & occupied

        if blockers != 0 and blockers & (blockers - nb.uint64(1)) == 0 and blockers & own_occ:
            pinned |= blockers

    # legal moves are compacted back into the pseudo array in place - the write index never
    # overtakes the read index - so this costs no second MAX_MOVES allocation per node.
    legal_count = 0
    for i in range(pseudo_count):

        move = pseudo[i]
        piece = move_piece(move)
        from_square = move_from_square(move)
        to_square = move_to_square(move)

        if piece == KING:
            if move_is_castle(move):
                pseudo[legal_count] = move  # castling_moves_nb proved every square safe already
                legal_count += 1
                continue

            test_occupied = occupied ^ bit(king_square)  # king can't shield the square it leaves
            if move_is_capture(move):
                test_occupied &= clear_mask(to_square)

            if not attacked_by_with_occ(board, to_square, them, test_occupied):
                pseudo[legal_count] = move
                legal_count += 1

            continue

        if double_check:
            continue

        if move_is_en_passant(move):
            if not is_check(make_move(board, move), us):  # type: ignore[type-var, call-arg]
                pseudo[legal_count] = move
                legal_count += 1

            continue

        if bit(to_square) & block_mask == 0:
            continue
        if bit(from_square) & pinned and bit(to_square) & LINE_NB[king_square, from_square] == 0:
            continue
        
        pseudo[legal_count] = move
        legal_count += 1

    return pseudo, legal_count


# compile every jitted function above now, inside the init budget - a real position, not an
# empty Board(): is_check (and anything that calls it) calls lsb_index on the king bitboard,
# which loops forever on an empty board with no king to find.
def warm_up() -> None:
    board = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    is_attacked(board, 4, WHITE)
    is_check(board, WHITE)
    attacked_by_with_occ(board, 4, BLACK, int(board.occupancy[2]))
    moves, _count = pseudo_legal_moves(board)
    legal_moves(board)
    make_move(board, int(moves[0]))  # type: ignore[type-var, call-arg]
    # a position with our king in check and a friendly piece pinned, so legal_moves compiles its
    # check-evasion and pin branches now rather than on the first such position in a real game
    legal_moves(parse_fen("4r3/8/8/8/8/8/4N3/4K3 w - - 0 1"))
    legal_moves(parse_fen("rnbqkbnr/ppp1p1pp/3p4/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"))


warm_up()
