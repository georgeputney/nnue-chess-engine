"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state lasts until that game ends, then resets.
Import runs first, inside a 60 s budget, before our clock starts.

An alpha-beta search over a piece-square evaluation, built up in the phases in docs/plan.md.
"""

import time

import chess
import chess.polyglot

popcount = chess.popcount  # aliased once; called per slider in the eval loop

# centipawn value per piece type. no king entry: both sides always have one, so it never
# affects the difference.
# ref: https://www.chessprogramming.org/Point_Value
PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# fmt: off
# PeSTO piece-square tables (public domain), midgame and endgame, pawn..king. published
# a8-first; _a1() below flips to python-chess order (a1 = 0). these are placement only -
# material is added in where MIDGAME_TABLE / ENDGAME_TABLE are built. not yet tuned.
# ref: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function

# base piece value per phase, pawn..king, indexed [piece_type - 1]. king is 0: both sides
# always have one.
MIDGAME_VALUE = [82, 337, 365, 477, 1025, 0]
ENDGAME_VALUE = [94, 281, 297, 512,  936, 0]

MIDGAME_PAWN   = [ 
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
      ]   # mg_pawn_table

ENDGAME_PAWN   = [ 
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
 ]   # eg_pawn_table

MIDGAME_KNIGHT = [ 
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
 ]

ENDGAME_KNIGHT = [ 
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
 ]

MIDGAME_BISHOP = [ 
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
 ]

ENDGAME_BISHOP = [ 
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
 ]

MIDGAME_ROOK   = [ 
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
 ]

ENDGAME_ROOK   = [ 
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
 ]

MIDGAME_QUEEN  = [ 
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
 ]

ENDGAME_QUEEN  = [ 
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
 ]

MIDGAME_KING   = [ 
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
 ]

ENDGAME_KING   = [ 
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
 ]
# fmt: on


# PeSTO tables are published a8-first; return them in a1-first (python-chess) order.
def _a1(table: list[int]) -> list[int]:
    return [table[sq ^ 56] for sq in range(64)]


MIDGAME_RAW = [MIDGAME_PAWN, MIDGAME_KNIGHT, MIDGAME_BISHOP,
                MIDGAME_ROOK, MIDGAME_QUEEN, MIDGAME_KING]
ENDGAME_RAW = [ENDGAME_PAWN, ENDGAME_KNIGHT, ENDGAME_BISHOP,
                ENDGAME_ROOK, ENDGAME_QUEEN, ENDGAME_KING]

# [piece_type][square] -> material + placement, white's view, a1-first
MIDGAME_TABLE: dict[chess.PieceType, list[int]] = {
    pt: [MIDGAME_VALUE[pt - 1] + v for v in _a1(MIDGAME_RAW[pt - 1])] for pt in range(1, 7)
}
ENDGAME_TABLE: dict[chess.PieceType, list[int]] = {
    pt: [ENDGAME_VALUE[pt - 1] + v for v in _a1(ENDGAME_RAW[pt - 1])] for pt in range(1, 7)
}

# tapering weight per piece type; a full board sums to 24, bare kings to 0.
PHASE_WEIGHT: dict[chess.PieceType, int] = {
    chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4,
}

# sentinel beyond any reachable material total. +MATE = we deliver mate, -MATE = we are
# mated (also the "no move yet" starting value).
MATE_SCORE = 1_000_000

# hard ceiling on search depth, so a position of only forced moves cannot deepen forever.
MAX_DEPTH = 64

# nodes visited by the current search. read by tools/nodebench.py; bench_search resets it.
NODES = 0

# time budget as clock fractions: at most 1/HARD_LIMIT per move, and no new depth once
# 1/SOFT_LIMIT of the clock has been spent. rough, worth tuning.
HARD_LIMIT = 4
SOFT_LIMIT = 40

CHECK_EVERY = 1024  # poll the clock every this many nodes; smaller = less overshoot past DEADLINE
DEADLINE: float | None = None  # time to stop at, or None when there is no clock

DELTA_PRUNING_MARGIN = 200  # delta-pruning cushion in centipawns; a guess, tune later

TT_MASK = (1 << 20) - 1  # ~1M entries; power-of-two so `key & TT_MASK` indexes it
UPPER, EXACT, LOWER = 0, 1, 2  # is the stored score a ceiling, exact, or a floor?

# transposition table: results of positions already searched, keyed by zobrist hash.
# index -> (zobrist key, depth searched, score, bound kind, best move), or None.
TT: list[tuple[int, int, int, int, chess.Move] | None] = [None] * (TT_MASK + 1)

# quiet-move ordering signals, both learned during the search (like the TT), both persistent
# across iterations and moves in a game.
MAX_HISTORY = 1 << 14  # history score saturates toward +/- this

# the last two quiet moves that caused a cutoff, per remaining-depth slot
KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_DEPTH + 1)]
# [piece_type][to_square] -> running tally of how often that quiet move has cut
HISTORY: list[list[int]] = [[0] * 64 for _ in range(7)]

# ref: https://www.chessprogramming.org/Tempo
TEMPO = 10  # centipawns; a small edge just for having the move. a split midgame/endgame
            # tempo is left for phase 19 tuning

# centipawns per square a slider attacks. queen weighted low - its scope is already huge.
MOBILITY_WEIGHT: dict[chess.PieceType, int] = {chess.BISHOP: 4, chess.ROOK: 2, chess.QUEEN: 1}

# penalty per square a phantom queen on our own king's square would see. midgame term only,
# so it tapers out of the endgame where the king wants to be active.
KING_EXPOSURE = 2

DOUBLED_PAWN = 12  # centipawns per extra pawn stacked on a file


# Thrown when the search hits the time cap; get_move catches it.
class Timeout(Exception):
    pass


# Count "extra" pawns across all files: 0 for a healthy structure, +1 per doubled file,
# +2 for a tripled one. Same routine for either colour's pawn bitboard.
def doubled_pawns(pawns: chess.Bitboard) -> int:
    extra = 0
    for file_bb in chess.BB_FILES:
        count = chess.popcount(pawns & file_bb)
        if count > 1:
            extra += count - 1
    return extra


# Squares a queen on `square` would attack through `occupied` - used from the king's square
# as a king-exposure proxy. board.attacks() only does the piece actually on the square.
def slider_scope(square: chess.Square, occupied: chess.Bitboard) -> int:
    attacks = (
        chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
        | chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied]
        | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied]
    )
    return popcount(attacks)


# Static score from `side`'s point of view, in centipawns. Per piece: material + placement
# from the piece-square tables, blended between a midgame and an endgame set by game phase
# (kings included). Sliders add a mobility term; the king is scored as a phantom queen, so
# open lines around it read as danger. After the loop, a flat tempo bonus for the side to
# move and a doubled-pawn penalty.
# ref: https://www.chessprogramming.org/Piece-Square_Tables
# ref: https://www.chessprogramming.org/Tapered_Eval
# ref: https://www.chessprogramming.org/Mobility
# ref: https://www.chessprogramming.org/King_Safety
# ref: https://www.chessprogramming.org/Tempo
# ref: https://www.chessprogramming.org/Doubled_Pawn
def evaluate(board: chess.Board, side: chess.Color) -> int:
    phase = game_phase(board)
    midgame = endgame = 0  # white's point of view

    occupied = board.occupied
    for square, piece in board.piece_map().items():
        i = square if piece.color == chess.WHITE else square ^ 56
        pt = piece.piece_type

        mg = MIDGAME_TABLE[pt][i]
        eg = ENDGAME_TABLE[pt][i]

        if pt in MOBILITY_WEIGHT:
            reach = MOBILITY_WEIGHT[pt] * popcount(board.attacks_mask(square))
            mg += reach
            eg += reach
        elif pt == chess.KING:
            mg -= KING_EXPOSURE * slider_scope(square, occupied)

        if piece.color == chess.WHITE:
            midgame += mg
            endgame += eg
        else:
            midgame -= mg
            endgame -= eg

    score = (midgame * phase + endgame * (24 - phase)) // 24
    score += TEMPO if board.turn == chess.WHITE else -TEMPO

    white_pawns = board.pawns & board.occupied_co[chess.WHITE]
    black_pawns = board.pawns & board.occupied_co[chess.BLACK]
    score += DOUBLED_PAWN * (doubled_pawns(black_pawns) - doubled_pawns(white_pawns))

    return score if side == chess.WHITE else -score


# How far into the endgame the position is: 24 = every piece on, 0 = only kings and pawns.
def game_phase(board: chess.Board) -> int:
    phase = sum(
        weight * len(board.pieces(pt, colour))
        for pt, weight in PHASE_WEIGHT.items()
        for colour in (chess.WHITE, chess.BLACK)
    )
    return min(phase, 24)


# Adjust a quiet move's history score: positive `bonus` when it caused a cutoff, negative
# when it was tried and did not. The gravity term shrinks the effect as the score nears the
# cap, so entries saturate instead of running away.
# ref: https://www.chessprogramming.org/History_Heuristic
def update_history(board: chess.Board, move: chess.Move, bonus: int) -> None:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return

    h = HISTORY[piece.piece_type][move.to_square]
    HISTORY[piece.piece_type][move.to_square] = h + bonus - h * abs(bonus) // MAX_HISTORY


# Ordering score so the likely-best moves are tried first: captures before quiet moves, and
# among captures the most valuable victim taken by the least valuable attacker (MVV-LVA).
# ref: https://www.chessprogramming.org/MVV-LVA
def move_ordering_score(board: chess.Board, move: chess.Move) -> int:
    score = 0

    # value of the captured piece
    if board.is_en_passant(move):
        score = PIECE_VALUE[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        if victim is not None:
            score = PIECE_VALUE[victim.piece_type]

    # most valuable victim, least valuable attacker. the * 16 keeps every capture ranked
    # above every quiet move.
    if score:
        attacker = board.piece_at(move.from_square)
        score = score * 16 - (PIECE_VALUE.get(attacker.piece_type, 0) if attacker else 0)

    # promotions are usually worth trying early
    if move.promotion:
        score += PIECE_VALUE[move.promotion] - PIECE_VALUE[chess.PAWN]

    return score


# Best score for the side to move over `depth` plies of best play. alpha/beta is the score
# window still in contention; a result outside it cannot change the chosen move, so the
# branch is cut.
# ref: https://www.chessprogramming.org/Negamax
# ref: https://www.chessprogramming.org/Alpha-Beta
# ref: https://www.chessprogramming.org/Transposition_Table
# ref: https://www.chessprogramming.org/Principal_Variation_Search
# ref: https://www.chessprogramming.org/Killer_Heuristic
# ref: https://www.chessprogramming.org/History_Heuristic
def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    global NODES
    NODES += 1

    # abort once the time cap is reached
    if DEADLINE is not None and NODES % CHECK_EVERY == 0 and time.monotonic() >= DEADLINE:
        raise Timeout

    # transposition table probe: have we searched this exact position before?
    key = chess.polyglot.zobrist_hash(board)
    entry = TT[key & TT_MASK]
    tt_move = None

    # entry[0] == key rules out a different position that landed on the same slot
    if entry is not None and entry[0] == key:
        _, tt_depth, tt_score, tt_flag, tt_move = entry

        # trust it only if searched at least as deep. an exact score stands; a bound only
        # when it already proves a cutoff
        if tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_score
            if tt_flag == LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == UPPER and tt_score <= alpha:
                return tt_score

    # --- null-move pruning (deferred, see below) ---
    # Pass our turn and search shallow: if that still beats beta, our real move does too, so
    # cut. Guards: not in check, non-PV (zero window), depth >= 3, and non-pawn material for
    # the side to move (in a pawn ending, being forced to move usually hurts, so the logic
    # inverts).
    # Implemented and correct, but A/B vs the phase 9.5 snapshot showed no gain - eval-gated
    # +17 [-21, +56], ungated -17 [-54, +19], and it lost games to a version without it. NMP
    # earns its keep at higher depth with a trustworthy eval; revisit after phase 18. On
    # re-enable, move this below the `not moves` / `depth <= 0` checks.
    # ref: https://www.chessprogramming.org/Null_Move_Pruning
    #
    # R = 2 + depth // 6                                # reduction; deeper -> larger cut
    # if (
    #     not board.is_check()
    #     and beta - alpha == 1                         # zero window = non-PV node
    #     and depth >= 3
    #     and board.occupied_co[board.turn] & ~board.pawns & ~board.kings  # has a piece to lose
    # ):
    #     board.push(chess.Move.null())                 # skip our turn
    #     score = -negamax(board, depth - 1 - R, -beta, -beta + 1)  # shallow, zero-window
    #     board.pop()
    #     if score >= beta:
    #         return beta                               # too good even after passing - prune

    moves = list(board.legal_moves)
    # no legal moves: checkmate if in check, else stalemate (a draw)
    if not moves:
        return -MATE_SCORE if board.is_check() else 0
    # out of depth: hand off to a captures-only search so we don't judge a half-finished trade
    if depth <= 0:
        return quiescence_search(board, alpha, beta)

    # best-first ordering: TT move, then captures by MVV-LVA, then killers, then history
    moves.sort(
        key=lambda m: (
            m == tt_move,
            move_ordering_score(board, m),
            m in KILLERS[depth],
            HISTORY[p.piece_type][m.to_square] if (p := board.piece_at(m.from_square)) else 0,
        ),
        reverse=True,
    )

    alpha_original = alpha  # incoming window, kept to tag the stored score below
    best = -MATE_SCORE
    best_move = moves[0]  # always have a move to store, even if none improves on -MATE_SCORE
    tried_quiets: list[chess.Move] = []  # quiets that didn't cut here - they take the malus

    for i, move in enumerate(moves):

        is_quiet = not board.is_capture(move) and not move.promotion
        board.push(move)

        if i == 0:
            # first move: trust the ordering, search it at full width
            score = -negamax(board, depth - 1, -beta, -alpha)
        else:
            # rest: cheap null-window check for "does this beat alpha?"
            score = -negamax(board, depth - 1, -alpha - 1, -alpha)

            if alpha < score < beta:
                # it does - re-search at full width for the real score
                score = -negamax(board, depth - 1, -beta, -alpha)

        board.pop()

        if score > best:
            best = score
            best_move = move

        alpha = max(alpha, score)
        # opponent already has a better option earlier; this branch cannot matter
        if alpha >= beta:
            if is_quiet:
                # a quiet move cut: remember it as a killer, reward it, penalise the quiets
                # that were tried first and failed
                if move not in KILLERS[depth]:
                    KILLERS[depth] = [move, KILLERS[depth][0]]

                bonus = depth * depth
                update_history(board, move, bonus)
                for q in tried_quiets:
                    update_history(board, q, -bonus)

            break

        if is_quiet:
            tried_quiets.append(move)  # this move didn't cut

    # record what we learned: an exact value, or which side of the window it fell on
    if best >= beta:
        # cut early - best is only a floor
        flag = LOWER  
    elif best > alpha_original:
        # raised alpha without cutting - best is exact
        flag = EXACT  
    else:
        # nothing beat alpha - best is a ceiling
        flag = UPPER  

    TT[key & TT_MASK] = (key, depth, best, flag, best_move)  # always replace; fine at this size

    return best


# Past the search horizon: keep going through captures only (and every reply when in check)
# until the position is quiet, then score it - so the search never trusts an eval taken
# mid-trade. Delta pruning skips captures too small to reach alpha.
# ref: https://www.chessprogramming.org/Quiescence_Search
# ref: https://www.chessprogramming.org/Delta_Pruning
def quiescence_search(board: chess.Board, alpha: int, beta: int) -> int:
    global NODES
    NODES += 1

    # abort once the time cap is reached
    if DEADLINE is not None and NODES % CHECK_EVERY == 0 and time.monotonic() >= DEADLINE:
        raise Timeout

    in_check = board.is_check()
    if in_check:
        # can't standing pat out of check: search every reply
        moves = list(board.legal_moves)

        if not moves:
            return -MATE_SCORE
    else:
        # the side to move can decline to capture, so the static evaluation is a floor
        standing_pat = evaluate(board, board.turn)

        if standing_pat >= beta:
            return standing_pat

        alpha = max(alpha, standing_pat)
        moves = list(board.generate_legal_captures())

    moves.sort(key=lambda m: move_ordering_score(board, m), reverse=True)

    for move in moves:
        # delta pruning: if winning this piece plus a margin still falls short of alpha,
        # so does every smaller capture after it (list is biggest-victim first)
        if not in_check and not move.promotion:
            if board.is_en_passant(move):
                victim = PIECE_VALUE[chess.PAWN]
            else:
                piece = board.piece_at(move.to_square)
                victim = PIECE_VALUE[piece.piece_type] if piece else 0

            if standing_pat + victim + DELTA_PRUNING_MARGIN < alpha:
                break

        board.push(move)
        score = -quiescence_search(board, -beta, -alpha)
        board.pop()

        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return alpha


# Root of the search: negamax's move loop, but it keeps the best move, not just the score.
# Seeded with the first legal move so it always returns one.
def search_root(board: chess.Board, depth: int) -> tuple[chess.Move, int]:
    best_move = next(iter(board.legal_moves))
    best_score = -MATE_SCORE

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: move_ordering_score(board, m), reverse=True)

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -MATE_SCORE, MATE_SCORE)
        board.pop()

        if score > best_score:  # strict, so ties keep the earlier (better-ordered) move
            best_score = score
            best_move = move

    return best_move, best_score


# The platform's per-move entry point. Deepens one ply at a time until the clock stops it,
# returning the best move from the last depth that completed.
def get_move(fen: str, time_left_ms: int) -> str:
    global DEADLINE

    board = chess.Board(fen)
    start = time.monotonic()

    # hard deadline to abort at - the 50 ms is slack for the node batch that runs past the
    # last clock check plus move-gen and the reply; soft cap past which no new depth starts
    DEADLINE = start + time_left_ms / HARD_LIMIT / 1000 - 0.05
    soft_cap = time_left_ms / SOFT_LIMIT / 1000

    best = next(iter(board.legal_moves))  # fallback if depth 1 itself times out
    try:
        for depth in range(1, MAX_DEPTH):
            if time.monotonic() - start >= soft_cap:
                break

            move, _ = search_root(board, depth)
            best = move  # depth completed, adopt its move

    except Timeout:
        pass
    finally:
        DEADLINE = None

    return best.uci()


# tools/nodebench.py hook, not used in games: fixed-depth search returning (move, score, nodes).
def bench_search(fen: str, depth: int) -> tuple[str, int, int]:
    global NODES
    NODES = 0

    TT[:] = [None] * len(TT)  # empty table per position so node counts stay comparable
    KILLERS[:] = [[None, None] for _ in range(len(KILLERS))]
    for row in HISTORY:
        row[:] = [0] * 64

    move, score = search_root(chess.Board(fen), depth)

    return move.uci(), score, NODES
