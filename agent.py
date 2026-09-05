"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state lasts until that game ends, then resets.
Import runs first, inside a 90 s budget, before our clock starts.

An alpha-beta search over a Texel-tuned tapered evaluation, built up in the phases in docs/plan.md.
"""

import os
import time
from collections.abc import Callable, Hashable

import chess
import chess.polyglot

# Texel-tuned evaluation weights (tools/tune.py). Material and placement are two separate
# tables, added together at lookup time; the piece-square tables are a1-first; every term is
# split midgame / endgame.
from tables import (
    ENDGAME_PST,
    KING_EXPOSURE_EG,
    KING_EXPOSURE_MG,
    MATERIAL_EG,
    MATERIAL_MG,
    MIDGAME_PST,
    MOBILITY_WEIGHT_EG,
    MOBILITY_WEIGHT_MG,
    PAWN_AHEAD_EG,
    PAWN_AHEAD_MG,
    TEMPO_EG,
    TEMPO_MG,
)

popcount = chess.popcount  # aliased once; called per slider in the eval loop
FULL_BB = (1 << 64) - 1

# virtual piece type: a pawn on the far side of the board from its own king behaves
# differently (storms, weak shelter) and scores on its own material/PST/pawn-ahead row
# instead of PAWN's. MATERIAL_MG/EG, MIDGAME_PST/ENDGAME_PST and PAWN_AHEAD_MG/EG all carry
# a row for it, indexed by this sentinel.
# ref: https://www.chessprogramming.org/Pawn_Structure
FAR_PAWN = 0

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

# time budget as clock fractions: at most 1/HARD_LIMIT per move, and no new depth once
# 1/SOFT_LIMIT of the clock has been spent. rough, worth tuning.
HARD_LIMIT = 4
SOFT_LIMIT = 40
CHECK_EVERY = 1024  # poll the wall clock every this many nodes; smaller = less overshoot
PANIC_TIME_MS = 300  # below this much left, skip the search rather than risk a clock loss

# transposition-table shape. an entry is (depth, score, bound, best move); the bound is one
# of these three. TT_MAX_ENTRIES bounds the dict's memory over a long game.
UPPER, EXACT, LOWER = 0, 1, 2  # stored score is a ceiling / exact / a floor
MAX_HISTORY = 1 << 14          # quiet-move history score saturates toward +/- this
TT_MAX_ENTRIES = 1_500_000     # get_move clears the table past this many entries

# module state the search mutates; get_move / bench_search reset what needs it per move.
NODES = 0                      # nodes visited this search; read by tools/nodebench.py
DEADLINE: float | None = None  # wall-clock time to abort at, or None when off the clock

# keyed by board._transposition_key() directly, not a hashed-and-masked int: that key is a
# plain tuple (pieces, turn, castling rights, ep square), ~22x cheaper to compute per node
# than chess.polyglot.zobrist_hash, and a dict lookup on it can never collide with a
# different position the way a fixed-size array indexed by `hash & mask` can.
# ref: https://www.chessprogramming.org/Transposition_Table
TT: dict[Hashable, tuple[int, int, int, chess.Move]] = {}
# KILLERS[depth] holds up to two quiet moves that caused a cutoff there; HISTORY is a
# [side to move][piece_type][to_square] cutoff tally - a quiet that's good for white on a
# square says nothing about the same piece landing there for black, so each side keeps its
# own table. both persist across iterations and moves in a game.
KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_DEPTH + 1)]
HISTORY: list[list[list[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
# per-game anti-repetition (see get_move / search_root), keyed by zobrist hash: how many
# times we've been asked to move in a position, and what we chose there last. AVOID is that
# move for the current call, or None.
SEEN: dict[int, int] = {}
PLAYED: dict[int, chess.Move] = {}
AVOID: chess.Move | None = None

# search-tuning knobs; each one's reasoning and reference live at its use site in negamax /
# get_move. LMP_MAX_DEPTH doubles as the "shallow" gate shared by RFP, LMP and futility.
LMP_MAX_DEPTH = 6           # RFP / late-move / futility pruning only this shallow
RFP_MARGIN = 90             # reverse futility: cp of allowed decline per ply (plan: 70-120)
FUTILITY_MARGIN = 100       # move-loop futility: cp per ply a quiet must be within of alpha
DELTA_PRUNING_MARGIN = 200  # quiescence delta pruning: cp cushion on a capture's value
NULL_MOVE_MIN_DEPTH = 3     # null-move pruning: none within this many plies of the horizon
NULL_MOVE_DEPTH_WEIGHT = 100  # null-move pruning: depth's pull on the probe's reduced depth
NULL_MOVE_SCALE = 200         # null-move pruning: divisor for the weight above and the margin
LMR_MIN_DEPTH = 3           # late-move reduction: none within this many plies of the horizon
LMR_MOVE_WEIGHT = 100       # late-move reduction: pull per move index, thousandths of a ply
LMR_DEPTH_WEIGHT = 150      # late-move reduction: pull per remaining depth, thousandths of a ply
LMR_SCALE = 1000            # late-move reduction: divisor for the two weights above
LMR_HISTORY_SCALE = MAX_HISTORY // 4  # late-move reduction: history's pull, capped near +/-4 ply
LMR_CAPTURE_FLOOR = MAX_HISTORY * 2   # late-move reduction: keeps a capture/promotion above history
IIR_MIN_DEPTH = 7           # internal iterative reduction: fires only at least this deep
ASPIRATION_MIN_DEPTH = 4    # full-width search up to here, a thin window after
ASPIRATION_WINDOW = 25      # aspiration half-width in cp, doubled on each miss

# cp docked at the root for replaying the move we chose last time we were asked to move in
# this exact position. get_move builds a fresh chess.Board(fen) with no move history every
# call, so is_repetition() inside the search can only catch a repeat invented within its own
# lookahead - it never sees that the real game already visited this position. This penalty is
# the only thing that does, so it has to be big: a search blind to the real history can score
# a move that only loops as if it were a genuine, decisive try (measured gap in one real game:
# ~360 cp - see the commit). MATE_THRESHOLD still exempts a real forced mate.
REPETITION_PENALTY = 400

# lookup table built once at import so the search never recomputes it per node
_LMP = [3 + d * d for d in range(LMP_MAX_DEPTH + 1)]  # quiet-move cap by depth: 4, 7, 12, 19, ...


# Thrown when the search hits the time cap; get_move catches it.
class Timeout(Exception):
    pass


# How many of `colour`'s pawns stand on `square`'s file, strictly ahead of it toward the far
# rank. For a pawn this doubles as a doubled-pawn count (each pawn behind another on its file
# counts the ones in front of it); for any other piece it is a free structural signal, e.g. a
# rook parked behind its own pawns. The shift-and-mask is a single-file "ahead of this square"
# mask: shifting the file-A pattern left by `square` re-bases it onto square's own file and
# rank for white; the mirrored pattern and a right-shift do the same for black.
# ref: https://www.chessprogramming.org/Doubled_Pawn
def pawns_ahead(square: chess.Square, colour: chess.Color, own_pawns: chess.Bitboard) -> int:
    if colour == chess.WHITE:
        ahead = (0x0101_0101_0101_0100 << square) & FULL_BB
    else:
        ahead = 0x0080_8080_8080_8080 >> (63 - square)
    return popcount(ahead & own_pawns)


# Squares a queen on `square` would attack through `occupied` - used from the king's square
# as a king-exposure proxy. board.attacks() only does the piece actually on the square.
def slider_scope(square: chess.Square, occupied: chess.Bitboard) -> int:
    attacks = (
        chess.BB_DIAG_ATTACKS[square][chess.BB_DIAG_MASKS[square] & occupied]
        | chess.BB_RANK_ATTACKS[square][chess.BB_RANK_MASKS[square] & occupied]
        | chess.BB_FILE_ATTACKS[square][chess.BB_FILE_MASKS[square] & occupied]
    )
    return popcount(attacks)


# Static score from `side`'s point of view, in centipawns. Each colour's terms are summed
# from white's side and negated at the end for black. Every weight is tuned, from tables.py.
def evaluate(board: chess.Board, side: chess.Color) -> int:
    midgame = endgame = phase = 0  # white's point of view; phase kept in step with game_phase
    occupied = board.occupied
    pawns_by_colour = {
        chess.WHITE: board.pawns & board.occupied_co[chess.WHITE],
        chess.BLACK: board.pawns & board.occupied_co[chess.BLACK],
    }

    for colour in (chess.WHITE, chess.BLACK):
        sign = 1 if colour == chess.WHITE else -1
        own_pawns = pawns_by_colour[colour]

        king_sq = board.king(colour)
        king_file = chess.square_file(king_sq) if king_sq is not None else 4

        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            for square in chess.scan_forward(board.pieces_mask(pt, colour)):
                # far pawn: on the opposite half of the board from its own king's file, so
                # it scores on FAR_PAWN's row instead of PAWN's.
                far = pt == chess.PAWN and (chess.square_file(square) ^ king_file) & 4
                vt = FAR_PAWN if far else pt

                i = square if colour == chess.WHITE else square ^ 56

                # material - one flat number per virtual piece type - plus placement, looked
                # up per square. kept as two separate tables (not folded together) so each
                # can be read, tuned and regularised on its own.
                # ref: https://www.chessprogramming.org/Piece-Square_Tables
                mg = MATERIAL_MG[vt] + MIDGAME_PST[vt][i]
                eg = MATERIAL_EG[vt] + ENDGAME_PST[vt][i]

                # slider mobility: more reachable squares is better
                # ref: https://www.chessprogramming.org/Mobility
                if pt in MOBILITY_WEIGHT_MG:
                    reach = popcount(board.attacks_mask(square))
                    mg += MOBILITY_WEIGHT_MG[pt] * reach
                    eg += MOBILITY_WEIGHT_EG[pt] * reach

                # king safety as a phantom queen: open lines from the king square read as danger
                # ref: https://www.chessprogramming.org/King_Safety
                elif pt == chess.KING:
                    scope = slider_scope(square, occupied)
                    mg += KING_EXPOSURE_MG * scope
                    eg += KING_EXPOSURE_EG * scope

                # friendly pawns on this file, ahead of this piece toward the far rank - a
                # doubled-pawn penalty for a pawn, a free per-type term for anything else
                # ref: https://www.chessprogramming.org/Doubled_Pawn
                stacked = pawns_ahead(square, colour, own_pawns)
                mg += PAWN_AHEAD_MG.get(vt, 0) * stacked
                eg += PAWN_AHEAD_EG.get(vt, 0) * stacked

                midgame += sign * mg
                endgame += sign * eg
                phase += PHASE_WEIGHT.get(pt, 0)

    phase = min(phase, 24)

    # a small bonus for having the move
    # ref: https://www.chessprogramming.org/Tempo
    tempo = 1 if board.turn == chess.WHITE else -1
    midgame += TEMPO_MG * tempo
    endgame += TEMPO_EG * tempo

    # tapered blend: all midgame with every piece on the board, all endgame with none
    # ref: https://www.chessprogramming.org/Tapered_Eval
    score = (midgame * phase + endgame * (24 - phase)) // 24
    return score if side == chess.WHITE else -score


# How far into the endgame the position is: 24 = every piece on, 0 = only kings and pawns.
# evaluate() computes this itself in its single pass over the board; this copy is kept for
# tools/tune.py, which extracts eval terms independently of evaluate() and needs the phase
# to line up. ref: https://www.chessprogramming.org/Tapered_Eval
def game_phase(board: chess.Board) -> int:
    phase = sum(
        weight * len(board.pieces(pt, colour))
        for pt, weight in PHASE_WEIGHT.items()
        for colour in (chess.WHITE, chess.BLACK)
    )
    return min(phase, 24)


# A quiet move's cutoff tally for the side about to play it, or 0 for a move whose from-square
# is somehow empty (never true in a real search - just keeps this total).
def history_score(board: chess.Board, move: chess.Move) -> int:
    piece = board.piece_at(move.from_square)

    return HISTORY[board.turn][piece.piece_type][move.to_square] if piece else 0


# Adjust a quiet move's history score: positive `bonus` when it caused a cutoff, negative
# when it was tried and did not. The gravity term shrinks the effect as the score nears the
# cap, so entries saturate instead of running away.
# ref: https://www.chessprogramming.org/History_Heuristic
def update_history(board: chess.Board, move: chess.Move, bonus: int) -> None:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return

    row = HISTORY[board.turn][piece.piece_type]
    h = row[move.to_square]
    row[move.to_square] = h + bonus - h * abs(bonus) // MAX_HISTORY


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


# Emergency move for get_move when there isn't time to search at all: the best-looking capture
# or promotion by the same ordering the real search uses, or - since every quiet move scores 0
# and max() then just keeps the first one it sees - whatever legal move comes first if there is
# no capture. No search, no board mutation: safe to call with next to no time left.
def panic_move(board: chess.Board) -> chess.Move:
    return max(board.legal_moves, key=lambda m: move_ordering_score(board, m))


# How much negamax's late-move reduction should trust a move: a quiet's key is its history
# score (can be negative - a quiet that keeps failing low pulls its own reduction deeper), and
# a capture or promotion's key is its ordering score pushed above history's ceiling. That floor
# guarantees a real capture's key always dwarfs even a maxed-out history entry, so the formula
# in negamax reduces it back to nothing - the same "don't reduce captures" outcome as gating on
# is_quiet, without needing the gate.
def lmr_key(board: chess.Board, move: chess.Move, is_quiet: bool) -> int:
    if not is_quiet:
        return LMR_CAPTURE_FLOOR + move_ordering_score(board, move)
    
    return history_score(board, move)


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
# branch is cut. `ply` is the distance from the root - it bounds check extensions and scales
# mate scores. Each pruning step below carries its own reference.
# ref: https://www.chessprogramming.org/Negamax
# ref: https://www.chessprogramming.org/Alpha-Beta
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

    # mate-distance pruning: we cannot be mated sooner than `ply` from here, nor deliver
    # mate sooner than `ply + 1`. clamp the window to that band; if it collapses, no move
    # at this node can change the result.
    # ref: https://www.chessprogramming.org/Score#Mate_Distance_Pruning
    alpha = max(alpha, ply - MATE_SCORE)
    beta = min(beta, MATE_SCORE - ply - 1)
    if alpha >= beta:
        return alpha

    # recursion floor: a checking sequence extends every ply, so `depth` never falls -
    # this stops it running away. don't stand pat out of check here: if it's mate the
    # static eval would badly misjudge it, so resolve no-legal-moves first.
    if ply >= MAX_DEPTH:
        if board.is_check() and not any(board.generate_legal_moves()):
            return -MATE_SCORE + ply
        return evaluate(board, board.turn)

    # transposition table probe: have we searched this exact position before?
    # ref: https://www.chessprogramming.org/Transposition_Table
    key = board._transposition_key()
    entry = TT.get(key)
    tt_move = None

    if entry is not None:
        tt_depth, tt_score, tt_flag, tt_move = entry
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

    # internal iterative reduction: on a non-PV node deep enough to be worth ordering, with no
    # TT move to order by, shave a ply rather than grind every move unordered. PV nodes keep
    # full depth so a mate on the principal variation is not missed by an iteration. every
    # step below reads the reduced depth.
    # ref: https://www.chessprogramming.org/Internal_Iterative_Reductions
    if tt_move is None and depth >= IIR_MIN_DEPTH and beta - alpha == 1:
        depth -= 1

    # static eval, shared by reverse futility and null-move here and late-move / futility
    # pruning in the move loop. computed whenever not in check - the eval badly misjudges a
    # position in check - regardless of depth, since null-move pruning below needs it past
    # the shallow cutoff RFP and futility stop at.
    in_check = board.is_check()
    shallow = depth <= LMP_MAX_DEPTH and not in_check
    static_eval = evaluate(board, board.turn) if not in_check else 0

    # reverse futility pruning: so far ahead that even conceding RFP_MARGIN per remaining
    # ply still clears beta, so assume the real search fails high too
    # ref: https://www.chessprogramming.org/Reverse_Futility_Pruning
    if (
        shallow
        and beta - alpha == 1
        and abs(beta) < MATE_THRESHOLD
        and static_eval - RFP_MARGIN * depth >= beta
    ):
        return static_eval

    # null-move pruning: hand the opponent a free move and search shallow - if the position
    # already looks at least as good as beta, the free move is worth trusting, and if it
    # still clears beta after passing a whole turn the real move almost certainly cuts too.
    # guards: not in check, non-PV (zero window), a real edge to spend, depth to spare, and
    # non-pawn material for the side to move (in a pawn ending, being forced to move often
    # helps the opponent - zugzwang - so the "passing only hurts me" assumption breaks).
    # ref: https://www.chessprogramming.org/Null_Move_Pruning
    if (
        not in_check
        and beta - alpha == 1  # zero window = a non-PV node
        and depth >= NULL_MOVE_MIN_DEPTH
        and static_eval >= beta
        and board.occupied_co[board.turn] & ~board.pawns & ~board.kings  # a piece to lose
    ):
        # reduce harder the deeper the search and the more static already clears beta by -
        # the bigger that margin, the more the free-move probe can be trusted to have found
        # the same thing the real move would, so it can search less to prove it.
        margin = static_eval - beta
        reduced_depth = max((depth * NULL_MOVE_DEPTH_WEIGHT - margin) // NULL_MOVE_SCALE - 1, 0)

        board.push(chess.Move.null())  # skip our turn
        score = -negamax(board, reduced_depth, -beta, -beta + 1, ply + 1)
        board.pop()

        if score >= beta:
            return score  # too good even after passing

    # best-first ordering: TT move, then captures by MVV-LVA, then killers, then history
    # ref: https://www.chessprogramming.org/Killer_Heuristic
    # ref: https://www.chessprogramming.org/History_Heuristic
    moves.sort(
        key=lambda m: (
            m == tt_move,
            move_ordering_score(board, m),
            m in KILLERS[depth],
            history_score(board, m),
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
        # ref: https://www.chessprogramming.org/Check_Extensions
        extension = 1 if board.gives_check(move) and ply + 1 < MAX_DEPTH else 0
        is_quiet = not board.is_capture(move) and not move.promotion

        # late-move reduction trust score, read below - has to be taken before board.push(move)
        # moves the piece being scored. move 0 always gets the full-window PVS search below and
        # never reaches the reduction, so it never needs one.
        lmr_score = lmr_key(board, move, is_quiet) if i else 0

        # shallow non-PV quiet-move pruning, once we have a real score to fall back on
        # ref: https://www.chessprogramming.org/Futility_Pruning
        if is_quiet:
            if not extension and shallow and beta - alpha == 1 and best > -MATE_THRESHOLD:
                # late move pruning: enough quiets tried without a cut, skip the rest
                if quiets_seen >= _LMP[depth]:
                    break
                # futility: this quiet can't lift a position already far below alpha
                if static_eval + FUTILITY_MARGIN * depth <= alpha:
                    quiets_seen += 1
                    continue
                
            quiets_seen += 1

        board.push(move)

        new_depth = depth - 1 + extension  # the check extension, if any, folds in here

        # first repetition scores as a draw and the line is not searched on - a move before
        # the threefold rule, so the search can still steer into the draw when worse or away
        # from it when better.
        # ref: https://www.chessprogramming.org/Repetitions
        if board.halfmove_clock >= 4 and board.is_repetition(2):
            score = 0
        elif i == 0:
            # the move the ordering trusts most - full-window principal variation search
            # ref: https://www.chessprogramming.org/Principal_Variation_Search
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1)
        else:
            # later moves: probe with a reduced depth and a null window, and only pay for a
            # wider / deeper search if the probe beats alpha. almost every move here gets some
            # reduction - it grows with the move's index and the remaining depth, and eases off
            # for a move with a good trust score (lmr_key); a real capture's score is built to
            # cancel the reduction back to zero rather than being gated out separately.
            # ref: https://www.chessprogramming.org/Late_Move_Reductions
            reduction = 0
            if extension == 0 and depth >= LMR_MIN_DEPTH:
                pull = i * LMR_MOVE_WEIGHT + depth * LMR_DEPTH_WEIGHT
                raw = pull // LMR_SCALE - lmr_score // LMR_HISTORY_SCALE
                reduction = min(max(raw, 0), new_depth - 1)

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
    TT[key] = (depth, score_to_tt(best, ply), flag, best_move)  # always replace

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

    # threefold repetition is a draw - same guard as negamax. a check sequence that runs
    # into the qsearch tail (every reply is searched here, not just captures) can still
    # cycle; without this, quiescence has no way to notice and just keeps searching it.
    if board.is_repetition(3):
        return 0

    # recursion floor, matching negamax's - the only thing that bounds a checking sequence
    # here, since the in-check branch below searches every reply rather than spending depth.
    if ply >= MAX_DEPTH:
        if board.is_check() and not any(board.generate_legal_moves()):
            return -MATE_SCORE + ply
        return evaluate(board, board.turn)

    # transposition table probe: shared with negamax, so a capture sequence reached by two
    # move orders (or a position negamax already resolved before handing off here) is priced
    # once. stored at depth 0 below, so any negamax entry (depth >= 1) is trusted here too.
    # ref: https://www.chessprogramming.org/Transposition_Table
    key = board._transposition_key()
    entry = TT.get(key)
    tt_move = None

    if entry is not None:
        _, tt_score, tt_flag, tt_move = entry
        tt_score = score_from_tt(tt_score, ply)

        if tt_flag == EXACT:
            return tt_score
        if tt_flag == LOWER and tt_score >= beta:
            return tt_score
        if tt_flag == UPPER and tt_score <= alpha:
            return tt_score

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

    moves.sort(key=lambda m: (m == tt_move, move_ordering_score(board, m)), reverse=True)

    alpha_original = alpha
    best_move: chess.Move | None = None

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

        if score > alpha:
            alpha = score
            best_move = move
        if alpha >= beta:
            break

    # same bookkeeping as negamax's store, at a flat depth 0 - qsearch has no depth of its own
    flag = LOWER if alpha >= beta else (EXACT if alpha > alpha_original else UPPER)
    TT[key] = (0, score_to_tt(alpha, ply), flag, best_move or chess.Move.null())

    return alpha


# Root of the search: negamax's move loop, but it keeps the best move, not just the score,
# and runs inside the caller's aspiration window. `prev_best` (last iteration's choice) is
# tried first, the rest fall back to MVV-LVA. A returned score that reached alpha or beta is
# only a bound - get_move widens the window and calls again.
def search_root(
    board: chess.Board, depth: int, alpha: int, beta: int, prev_best: chess.Move | None
) -> tuple[chess.Move, int]:
    
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: move_ordering_score(board, m), reverse=True)

    if prev_best is not None and prev_best in moves:
        moves.remove(prev_best)
        moves.insert(0, prev_best)

    best_move = moves[0]
    best_score = -MATE_SCORE

    for i, move in enumerate(moves):
        board.push(move)
        if board.halfmove_clock >= 4 and board.is_repetition(2):
            # this move brings a position up for the second time - a draw (see negamax)
            score = 0
        elif i == 0:
            # first move: full window; children search from ply 1 so a mate there is mate-in-1
            score = -negamax(board, depth - 1, -beta, -alpha, 1)
        else:
            # scout the rest with a null window; re-search only the ones that beat alpha
            score = -negamax(board, depth - 1, -alpha - 1, -alpha, 1)
            if alpha < score < beta:
                score = -negamax(board, depth - 1, -beta, -alpha, 1)

        board.pop()

        # dock the move we chose here last time we were asked to move in this exact position
        # (unless it is a real mate) so a level game isn't rubber-stamped into a repetition
        if move == AVOID and score < MATE_THRESHOLD:
            score -= REPETITION_PENALTY

        if score > best_score:  # strict, so ties keep the earlier (better-ordered) move
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)
        if alpha >= beta:
            break  # fail-high: this move beats the window; get_move widens and re-searches

    return best_move, best_score


# The platform's per-move entry point. Deepens one ply at a time until the clock stops it,
# returning the best move from the last depth that completed.
# Iterative deepening with aspiration windows: deepens from ply 1 until DEADLINE cuts the search
# off mid-depth (Timeout) or soft_cap elapses with no depth in flight, returning the best move/
# score from the last depth that completed inside the window - `fallback` if none did. Pulled
# out of get_move so _warm_up (below) can reuse the exact same driver on a position with no real
# clock, rather than keeping a second, slightly-different copy of this loop in sync by hand.
# `extra_stop`, if given, is checked alongside the time budget - _warm_up uses it to bail out
# once the transposition table has grown enough, without changing get_move's own behaviour at
# all (its calls never pass one).
def deepen(
    board: chess.Board,
    start: float,
    deadline: float,
    soft_cap: float,
    fallback: chess.Move,
    extra_stop: Callable[[], bool] | None = None,
) -> tuple[chess.Move, int]:
    
    global DEADLINE
    DEADLINE = deadline

    best = fallback
    score = 0

    try:
        for depth in range(1, MAX_DEPTH):
            out_of_time = time.monotonic() - start >= soft_cap
            
            if depth > 1 and (out_of_time or (extra_stop and extra_stop())):
                break

            # aspiration: a thin band around the last score past the opening plies, full
            # width before that. widen geometrically on whichever side failed and re-search
            # with the widened window - not the one that just failed.
            # ref: https://www.chessprogramming.org/Aspiration_Windows
            if depth <= ASPIRATION_MIN_DEPTH:
                alpha, beta = -MATE_SCORE, MATE_SCORE
            else:
                alpha, beta = score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW

            delta = ASPIRATION_WINDOW

            while True:
                move, value = search_root(board, depth, alpha, beta, best)

                if value <= alpha and alpha > -MATE_SCORE:
                    alpha = max(alpha - 2 * delta, -MATE_SCORE)  # fail low: drop the floor
                    delta *= 2
                    continue

                if value >= beta and beta < MATE_SCORE:
                    beta = min(beta + 2 * delta, MATE_SCORE)     # fail high: raise the roof
                    delta *= 2
                    best = move  # a fail-high move is still the best guess we have
                    continue

                break

            best = move       # depth completed inside the window - adopt its move
            score = value

    except Timeout:
        pass
    except Exception as exc:
        # a bug anywhere in the search must not cost the game the way an uncaught crash would
        # (see the contract) - keep whatever depth already completed, or the depth-1 fallback
        # if none did, and move on. logged so it still shows up in the validation log.
        print(f"agent: search crashed, falling back: {exc!r}")
    finally:
        DEADLINE = None

    return best, score


def get_move(fen: str, time_left_ms: int) -> str:
    global NODES, AVOID
    NODES = 0  # per-move, so the CHECK_EVERY clock poll doesn't inherit the last move's phase

    board = chess.Board(fen)

    # if we've been asked to move in this exact position before, tell search_root which move
    # we played last time so it can shy away from it. SEEN / PLAYED persist for the game.
    # ref: https://www.chessprogramming.org/Repetitions
    key = chess.polyglot.zobrist_hash(board)
    seen = SEEN.get(key, 0)
    SEEN[key] = seen + 1
    AVOID = PLAYED.get(key) if seen else None

    # critically low on time: don't search at all. the clock is only checked every CHECK_EVERY
    # nodes, so even a single depth-1 iteration could cost more than this move has left and
    # lose the game on time instead of the position - grab the best-looking move and return.
    if time_left_ms < PANIC_TIME_MS:
        best = panic_move(board)
        PLAYED[key] = best
        return best.uci()

    start = time.monotonic()

    # hard deadline to abort at - the 50 ms is slack for the node batch that runs past the
    # last clock check plus move-gen and the reply; soft cap past which no new depth starts
    deadline = start + time_left_ms / HARD_LIMIT / 1000 - 0.05
    soft_cap = time_left_ms / SOFT_LIMIT / 1000
    fallback = next(iter(board.legal_moves))  # in case depth 1 itself times out

    best, _score = deepen(board, start, deadline, soft_cap, fallback)

    PLAYED[key] = best

    # the TT persists all game as a plain dict now, with nothing bounding its size the way
    # the old fixed array did - clear it if it has grown large enough to be a memory concern.
    if len(TT) > TT_MAX_ENTRIES:
        TT.clear()

    return best.uci()


# Free head start on the position space most openings live near, spent during the platform's
# init budget rather than the game clock: the harness (harness/runner.py mirrors this) imports
# agent.py strictly before it ever tells us the fen - "there is no other input" - so this can't
# target the real opening, and rated games start from curated, unpublished positions anyway. But
# TT/KILLERS/HISTORY are already persistent, already read by the real search regardless of which
# position it's asked about, so whatever this reaches before the clock starts is free.
#
# WARM_UP_BUDGET_S is well inside the platform's 90 s init budget (see AGENTS.md) - the
# difference is slack for interpreter/python-chess import and for the platform's own overhead
# around the subprocess, which the 90 s is measured across, not just the time inside this file.
# Overridable via AGENT_WARM_UP_S: every local verification tool spawns a fresh process per
# game (harness/sandbox.py, mirroring the platform), so a real 70 s here would add that much to
# every single game - tools/bench.py and tools/nodebench.py default it to 0 for exactly that
# reason. Reading an env var isn't a network or filesystem access, so it's fine under the
# contract; the platform itself never sets this, so a real game always gets the full budget.
WARM_UP_BUDGET_S = float(os.environ.get("AGENT_WARM_UP_S", "70.0"))
WARM_UP_TT_CAP = TT_MAX_ENTRIES // 2  # get_move's own TT_MAX_ENTRIES clears the *whole* table
# the moment total size crosses that line - stop warm-up earlier so a fast first real move
# doesn't immediately wipe everything this just built.


def _warm_up() -> None:
    if WARM_UP_BUDGET_S <= 0:  # AGENT_WARM_UP_S=0: skip entirely, not even a depth-1 search -
        return                 # every local verification tool spawns a process per game
    start = time.monotonic()
    board = chess.Board()
    deepen(
        board,
        start,
        start + WARM_UP_BUDGET_S,
        WARM_UP_BUDGET_S,
        next(iter(board.legal_moves)),
        extra_stop=lambda: len(TT) >= WARM_UP_TT_CAP,
    )


_warm_up()


# tools/nodebench.py hook, not used in games: fixed-depth search returning (move, score, nodes).
def bench_search(fen: str, depth: int) -> tuple[str, int, int]:
    global NODES
    NODES = 0

    TT.clear()  # empty table per position so node counts stay comparable
    KILLERS[:] = [[None, None] for _ in range(len(KILLERS))]
    
    for side in HISTORY:
        for row in side:
            row[:] = [0] * 64

    move, score = search_root(chess.Board(fen), depth, -MATE_SCORE, MATE_SCORE, None)

    return move.uci(), score, NODES
