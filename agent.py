"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state lasts until that game ends, then resets.
Import runs first, inside a 60 s budget, before our clock starts.

An alpha-beta search over a piece-square evaluation, built up in the phases in docs/plan.md.
"""

import time

import chess
import chess.polyglot

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
# piece-square tables: positional bonus (centipawns) per square, white's view. index is the
# square number (a1 = 0); a black piece mirrors with `sq ^ 56`. seeded, not yet tuned.
# ref: https://www.chessprogramming.org/Simplified_Evaluation_Function
_PST_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_PST_ROOK = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
_PST_KING_MG = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
KING_EG = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]
# fmt: on

PST: dict[chess.PieceType, list[int]] = {
    chess.PAWN: _PST_PAWN,
    chess.KNIGHT: _PST_KNIGHT,
    chess.BISHOP: _PST_BISHOP,
    chess.ROOK: _PST_ROOK,
    chess.QUEEN: _PST_QUEEN,
    chess.KING: _PST_KING_MG,
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


# Thrown when the search hits the time cap; get_move catches it.
class Timeout(Exception):
    pass


# Material plus piece-square tables, from `side`'s point of view. The king's table is
# tapered between a midgame set (stay tucked away) and an endgame set (come to the centre).
# ref: https://www.chessprogramming.org/Piece-Square_Tables
# ref: https://www.chessprogramming.org/Tapered_Eval
def evaluate(board: chess.Board, side: chess.Color) -> int:
    phase = game_phase(board)
    score = 0  # white's point of view

    for square, piece in board.piece_map().items():
        i = square if piece.color == chess.WHITE else square ^ 56

        if piece.piece_type == chess.KING:
            positional = (_PST_KING_MG[i] * phase + KING_EG[i] * (24 - phase)) // 24
        else:
            positional = PST[piece.piece_type][i]

        value = PIECE_VALUE.get(piece.piece_type, 0) + positional
        score += value if piece.color == chess.WHITE else -value

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
