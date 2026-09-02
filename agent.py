"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state lasts until that game ends, then resets.
Import runs first, inside a 60 s budget, before our clock starts.

An alpha-beta search over a material evaluation, built up in the phases in docs/plan.md.
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

CHECK_EVERY = 2048  # poll the clock once per this many nodes, not every node
DEADLINE: float | None = None  # time to stop at, or None when there is no clock

DELTA_PRUNING_MARGIN = 200  # delta-pruning cushion in centipawns; a guess, tune later

TT_MASK = (1 << 20) - 1  # ~1M entries; power-of-two so `key & TT_MASK` indexes it
UPPER, EXACT, LOWER = 0, 1, 2  # is the stored score a ceiling, exact, or a floor?

# transposition table: results of positions already searched, keyed by zobrist hash.
# index -> (zobrist key, depth searched, score, bound kind, best move), or None.
TT: list[tuple[int, int, int, int, chess.Move] | None] = [None] * (TT_MASK + 1)


# Thrown when the search hits the time cap; get_move catches it.
class Timeout(Exception):
    pass


# Static score for `side` from material count alone, no search. Positive means `side` is
# ahead.
# ref: https://www.chessprogramming.org/Evaluation
def evaluate(board: chess.Board, side: chess.Color) -> int:
    return sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUE.items()
    )


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

    # TT's move first (best guess from a past search), then captures by MVV-LVA
    moves.sort(key=lambda m: (m == tt_move, move_ordering_score(board, m)), reverse=True)

    alpha_original = alpha  # incoming window, kept to tag the stored score below
    best = -MATE_SCORE
    best_move = moves[0]  # always have a move to store, even if none improves on -MATE_SCORE

    for i, move in enumerate(moves):
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
            break

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

    # a hard deadline to abort at, and a soft cap past which no new depth is started
    DEADLINE = start + time_left_ms / HARD_LIMIT / 1000
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

    move, score = search_root(chess.Board(fen), depth)

    return move.uci(), score, NODES
