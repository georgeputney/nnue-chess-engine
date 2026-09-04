"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state lasts until that game ends, then resets.
Import runs first, inside a 60 s budget, before our clock starts.

An alpha-beta search over a Texel-tuned tapered evaluation, built up in the phases in docs/plan.md.
"""

import math
import time

import chess
import chess.polyglot

# Texel-tuned evaluation weights (tools/tune.py). The piece-square tables have material
# folded in and are a1-first; the scalars are split midgame / endgame.
from tables import (
    DOUBLED_PAWN_EG,
    DOUBLED_PAWN_MG,
    ENDGAME_TABLE,
    KING_EXPOSURE_EG,
    KING_EXPOSURE_MG,
    MIDGAME_TABLE,
    MOBILITY_WEIGHT_EG,
    MOBILITY_WEIGHT_MG,
    TEMPO_EG,
    TEMPO_MG,
)

popcount = chess.popcount  # aliased once; called per slider in the eval loop

# rough centipawn values for move ordering and delta pruning only - the evaluation itself
# works off the tuned tables. no king entry: both sides always have one.
# ref: https://www.chessprogramming.org/Point_Value
PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
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

# a score at or past this magnitude encodes a forced mate; MATE_SCORE - |score| is its
# distance in plies. no real evaluation comes near it. the 2x leaves headroom for the extra
# plies quiescence can add past MAX_DEPTH on a checking sequence.
MATE_THRESHOLD = MATE_SCORE - 2 * MAX_DEPTH

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

RFP_MAX_DEPTH = 6   # only prune at shallow depth
RFP_MARGIN = 90     # centipawns of allowed decline per ply; plan says 70-120, tune

LMR_MIN_DEPTH = 3   # don't reduce within a few plies of the horizon
LMR_MIN_MOVE = 3    # the first few moves at a node are searched at full depth

# reduction amount indexed [depth][move index] - the widely used log-formula shape. built
# once at import so the search never calls math.log per node.
# ref: https://www.chessprogramming.org/Late_Move_Reductions
_LMR = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _i in range(1, 64):
        _LMR[_d][_i] = int(0.75 + math.log(_d) * math.log(_i) / 2.25)

LMP_MAX_DEPTH = 6      # move-count and futility pruning only this shallow
FUTILITY_MARGIN = 100  # cp per ply; a quiet this far below alpha won't rescue the node

# LMP quiet-move cap by depth: 3 + d*d -> 4, 7, 12, 19, 28, 39 for depths 1-6
_LMP = [3 + d * d for d in range(LMP_MAX_DEPTH + 1)]


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
# open lines around it read as danger. A doubled-pawn penalty and a tempo bonus for the side
# to move fold into both phase scores before the blend. Every weight is tuned, from tables.py.
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

        if pt in MOBILITY_WEIGHT_MG:
            reach = popcount(board.attacks_mask(square))
            mg += MOBILITY_WEIGHT_MG[pt] * reach
            eg += MOBILITY_WEIGHT_EG[pt] * reach

        elif pt == chess.KING:
            scope = slider_scope(square, occupied)
            mg += KING_EXPOSURE_MG * scope
            eg += KING_EXPOSURE_EG * scope

        if piece.color == chess.WHITE:
            midgame += mg
            endgame += eg
        else:
            midgame -= mg
            endgame -= eg

    doubled = doubled_pawns(board.pawns & board.occupied_co[chess.WHITE]) - doubled_pawns(
        board.pawns & board.occupied_co[chess.BLACK]
    )

    midgame += DOUBLED_PAWN_MG * doubled
    endgame += DOUBLED_PAWN_EG * doubled

    tempo = 1 if board.turn == chess.WHITE else -1
    midgame += TEMPO_MG * tempo
    endgame += TEMPO_EG * tempo

    score = (midgame * phase + endgame * (24 - phase)) // 24
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


# A mate score carries its distance from the root as MATE_SCORE - plies, so a faster mate
# scores higher. That distance is root-relative, so a score stored in the TT at one ply
# cannot be read back verbatim at another: re-root it to the storing node on the way in and
# back to the current node on the way out.
# ref: https://www.chessprogramming.org/Score_Bounds_in_the_TT#Mate_Scores
def score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


# Best score for the side to move over `depth` plies of best play. alpha/beta is the score
# window still in contention; a result outside it cannot change the chosen move, so the
# branch is cut. `ply` is the distance from the root - bounds check extensions and scales
# mate scores.
# ref: https://www.chessprogramming.org/Negamax
# ref: https://www.chessprogramming.org/Alpha-Beta
# ref: https://www.chessprogramming.org/Transposition_Table
# ref: https://www.chessprogramming.org/Principal_Variation_Search
# ref: https://www.chessprogramming.org/Killer_Heuristic
# ref: https://www.chessprogramming.org/History_Heuristic
# ref: https://www.chessprogramming.org/Check_Extensions
def negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    global NODES
    NODES += 1

    # abort once the time cap is reached
    if DEADLINE is not None and NODES % CHECK_EVERY == 0 and time.monotonic() >= DEADLINE:
        raise Timeout

    # threefold repetition is a draw. checking for the third occurrence (not the second)
    # keeps this from firing on a position the game has only reached once for real.
    if board.is_repetition(3):
        return 0

    # recursion floor: a checking sequence extends every ply, so `depth` never falls -
    # this stops it running away. don't stand pat out of check here: if it's mate the
    # static eval would badly misjudge it, so resolve no-legal-moves first.
    if ply >= MAX_DEPTH:
        if board.is_check() and not any(board.generate_legal_moves()):
            return -MATE_SCORE + ply
        return evaluate(board, board.turn)

    # transposition table probe: have we searched this exact position before?
    key = chess.polyglot.zobrist_hash(board)
    entry = TT[key & TT_MASK]
    tt_move = None

    # entry[0] == key rules out a different position that landed on the same slot
    if entry is not None and entry[0] == key:
        _, tt_depth, tt_score, tt_flag, tt_move = entry
        tt_score = score_from_tt(tt_score, ply)  # re-root a stored mate score to this node

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
    # no legal moves: checkmate if in check, else stalemate (a draw). the +ply makes a mate
    # found sooner score higher once negated back up the tree.
    if not moves:
        return -MATE_SCORE + ply if board.is_check() else 0
    
    # out of depth: hand off to a captures-only search so we don't judge a half-finished trade
    if depth <= 0:
        return quiescence_search(board, alpha, beta, ply)

    # static eval, shared by RFP here and futility pruning in the move loop. only at shallow
    # depth (where they prune) and never in check (the eval would badly misjudge it).
    shallow = depth <= LMP_MAX_DEPTH and not board.is_check()
    static_eval = evaluate(board, board.turn) if shallow else 0

    # reverse futility pruning
    # ref: https://www.chessprogramming.org/Reverse_Futility_Pruning
    if (
        shallow
        and beta - alpha == 1
        and abs(beta) < MATE_THRESHOLD
        and static_eval - RFP_MARGIN * depth >= beta
    ):
            return static_eval

    # null-move pruning: hand the opponent a free move and search shallow. if we still beat
    # beta after passing, the real move almost certainly cuts too - prune. guards: not in
    # check, non-PV (zero window), depth to spare, and non-pawn material for the side to move
    # (in a pawn ending, being forced to move often helps the opponent - zugzwang - so the
    # "passing only hurts me" assumption breaks).
    # ref: https://www.chessprogramming.org/Null_Move_Pruning
    reduction = 2 + depth // 6  # deeper -> cut more
    if (
        not board.is_check()
        and beta - alpha == 1  # zero window = a non-PV node
        and depth >= 3
        and board.occupied_co[board.turn] & ~board.pawns & ~board.kings  # a piece to lose
    ):
        board.push(chess.Move.null())  # skip our turn
        score = -negamax(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1)
        board.pop()

        if score >= beta:
            return score  # too good even after passing

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
    quiets_seen = 0                        # for late move pruning

    for i, move in enumerate(moves):
        # check extension: a checking move forces the reply, so search that line a ply deeper.
        # only extend while the child stays inside the ceiling, so the extension itself never
        # drives `ply` past MAX_DEPTH - the floor above is just the backstop.
        extension = 1 if board.gives_check(move) and ply + 1 < MAX_DEPTH else 0
        is_quiet = not board.is_capture(move) and not move.promotion

        # shallow non-PV quiet-move pruning, once we have a real score to fall back on
        if is_quiet:
            if not extension and shallow and beta - alpha == 1 and best > -MATE_THRESHOLD:
                # late move pruning: enough quiets tried without a cut, skip the rest
                # ref: https://www.chessprogramming.org/Futility_Pruning
                if quiets_seen >= _LMP[depth]:
                    break
                # futility: this quiet can't lift a position already far below alpha
                if static_eval + FUTILITY_MARGIN * depth <= alpha:
                    quiets_seen += 1
                    continue
            quiets_seen += 1

        board.push(move)

        new_depth = depth - 1 + extension  # the check extension, if any, folds in here

        if i == 0:
            # the move the ordering trusts most - search it at full width
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1)
        else:
            # late move reductions: a quiet move past the first few is unlikely to be best,
            # so probe it shallower and only pay full depth if it beats alpha anyway.
            reduction = 0
            if is_quiet and extension == 0 and depth >= LMR_MIN_DEPTH and i >= LMR_MIN_MOVE:
                reduction = min(_LMR[min(depth, 63)][min(i, 63)], new_depth - 1)

            # reduced, null window: is this move even worth a closer look?
            score = -negamax(board, new_depth - reduction, -alpha - 1, -alpha, ply + 1)

            # it cleared alpha despite the cut - redo at full depth, still null window
            if reduction and score > alpha:
                score = -negamax(board, new_depth, -alpha - 1, -alpha, ply + 1)

            # a full-depth score inside the window needs the real, full-window search
            if alpha < score < beta:
                score = -negamax(board, new_depth, -beta, -alpha, ply + 1)

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

    # store the mate distance relative to this node, not the root, so it reads back correctly
    TT[key & TT_MASK] = (key, depth, score_to_tt(best, ply), flag, best_move)  # always replace

    return best


# Past the search horizon: keep going through captures only (and every reply when in check)
# until the position is quiet, then score it - so the search never trusts an eval taken
# mid-trade. Delta pruning skips captures too small to reach alpha.
# ref: https://www.chessprogramming.org/Quiescence_Search
# ref: https://www.chessprogramming.org/Delta_Pruning
def quiescence_search(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
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
            return -MATE_SCORE + ply  # checkmate; +ply so a nearer one scores higher
    else:
        # the side to move can decline to capture, so the static evaluation is a floor
        standing_pat = evaluate(board, board.turn)

        if standing_pat >= beta:
            return standing_pat

        alpha = max(alpha, standing_pat)

        # captures, plus quiet promotions: a pawn pushing to the back rank swings the eval
        # as much as any capture, so the quiet search has to see e7e8q too. restricting the
        # generator to pawns landing on empty back-rank squares keeps this off the hot path.
        back_rank = chess.BB_RANK_8 if board.turn == chess.WHITE else chess.BB_RANK_1
        moves = list(board.generate_legal_captures())
        moves += board.generate_legal_moves(board.pawns, back_rank & ~board.occupied)

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
        score = -quiescence_search(board, -beta, -alpha, ply + 1)
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
        # ply 1: the child is one move from the root, so a mate there is mate in 1
        score = -negamax(board, depth - 1, -MATE_SCORE, MATE_SCORE, 1)
        board.pop()

        if score > best_score:  # strict, so ties keep the earlier (better-ordered) move
            best_score = score
            best_move = move

    return best_move, best_score


# The platform's per-move entry point. Deepens one ply at a time until the clock stops it,
# returning the best move from the last depth that completed.
def get_move(fen: str, time_left_ms: int) -> str:
    global DEADLINE, NODES
    NODES = 0  # per-move, so the CHECK_EVERY clock poll doesn't inherit the last move's phase

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
