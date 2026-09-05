"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state lasts until that game ends, then resets.
Import runs first, inside a 90 s budget, before our clock starts.

An alpha-beta search over a Texel-tuned tapered evaluation. The board, move generator,
make-move and hashing are the numba-compiled bitboard layer (board.py, movegen.py, attacks.py,
move.py, zobrist.py, bitboard.py) that stands in for python-chess; the search and evaluation
here are @njit too, so a node never leaves compiled code. reference.py is the plain
python-chess version this was ported from - the verify_* tools check against it, and get_move
falls back to it if numba fails to compile.
"""

import os
import time

import numba as nb
import numpy as np
from numba import njit, objmode
from numba.experimental import jitclass

import reference
from attacks import bishop_attacks, queen_attacks, rook_attacks
from bitboard import BISHOP, KING, NO_SQUARE, QUEEN, ROOK, WHITE
from board import Board, copy_board, parse_fen
from move import (
    PROMOTION_NONE,
    move_is_capture,
    move_is_en_passant,
    move_piece,
    move_to_square,
    move_uci,
)
from move import (
    move_promotion_raw as move_promotion,
)
from movegen import is_check, legal_moves, make_move
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
from zobrist import EP_FILE_KEYS, SIDE_KEY, zobrist_hash

# virtual piece type: a pawn on the far side of the board from its own king scores on its own
# material / PST / pawn-ahead row instead of PAWN's - a cheap pawn-storm / weak-shelter signal.
# tables.py numbers real pieces the python-chess way (pawn 1 .. king 6); the bitboard layer
# numbers them 0..5, so a real piece's virtual type is its bitboard id + 1.
# ref: https://www.chessprogramming.org/Pawn_Structure
FAR_PAWN = 0

# rough centipawn values for move ordering and delta pruning only; indexed by bitboard piece id
# (pawn 0 .. king 5), king 0 since it is never a victim.
# ref: https://www.chessprogramming.org/Point_Value
PIECE_VALUE = np.array((100, 320, 330, 500, 900, 0), dtype=np.int64)

# sentinel beyond any reachable material total; +MATE_SCORE = we deliver mate, -MATE_SCORE = we
# are mated. a score past MATE_THRESHOLD in magnitude encodes a forced mate MATE_SCORE - |score|
# plies away.
MATE_SCORE = 1_000_000
MAX_DEPTH = 64
MATE_THRESHOLD = MATE_SCORE - 2 * MAX_DEPTH

# time budget as clock fractions: at most 1/HARD_LIMIT of the clock on a move, and no new
# deepening iteration once 1/SOFT_LIMIT of it is spent.
HARD_LIMIT = 4
SOFT_LIMIT = 40
CHECK_EVERY = 1024        # poll the wall clock every this many nodes
PANIC_TIME_MS = 300       # below this much left, skip the search and grab a move

# transposition table: one slot per (hash & mask), always-replace, the full key stored so an
# index collision with a different position simply misses. ~4M slots ~ 170 MB.
UPPER, EXACT, LOWER = 0, 1, 2
_TT_BITS = 22
_TT_MASK = (1 << _TT_BITS) - 1

MAX_HISTORY = 1 << 14        # quiet-move history score saturates toward +/- this
TT_MAX_ENTRIES = 1_500_000  # after a game grows the table past this, get_move clears it

# search-tuning knobs; the reasoning for each lives at its use site in negamax / deepen.
LMP_MAX_DEPTH = 6         # reverse-futility / late-move / futility pruning only this shallow
RFP_MARGIN = 90           # reverse futility: cp of allowed decline per remaining ply
FUTILITY_MARGIN = 100     # move-loop futility: cp per ply a quiet must be within of alpha
DELTA_PRUNING_MARGIN = 200  # quiescence delta pruning: cp cushion on a capture's value
NULL_MOVE_MIN_DEPTH = 3   # null-move pruning: none within this many plies of the horizon
NULL_MOVE_DEPTH_WEIGHT = 100  # null-move: depth's pull on the probe's reduced depth
NULL_MOVE_SCALE = 200     # null-move: divisor for the weight above and the static margin
LMR_MIN_DEPTH = 3         # late-move reduction: none within this many plies of the horizon
LMR_MOVE_WEIGHT = 100     # late-move reduction: pull per move index, thousandths of a ply
LMR_DEPTH_WEIGHT = 150    # late-move reduction: pull per remaining depth, thousandths of a ply
LMR_SCALE = 1000          # late-move reduction: divisor for the two weights above
LMR_HISTORY_SCALE = MAX_HISTORY // 4  # late-move reduction: history's pull, capped near +/-4 ply
LMR_CAPTURE_FLOOR = MAX_HISTORY * 2   # late-move reduction: keeps a capture / promo above history
IIR_MIN_DEPTH = 7         # internal iterative reduction: fires only at least this deep
ASPIRATION_MIN_DEPTH = 4  # full-width search up to here, a thin window after
ASPIRATION_WINDOW = 25    # aspiration half-width in cp, doubled on each miss
REPETITION_PENALTY = 400  # cp docked at the root for replaying a move that loops the game

_LMP = np.array([3 + d * d for d in range(LMP_MAX_DEPTH + 1)], dtype=np.int64)  # quiet cap by depth
NO_MOVE = -1

# tables.py's dicts as virtual-type-indexed (0..6) numpy arrays for the jitted evaluate().
_MATERIAL_MG = np.array([MATERIAL_MG[vt] for vt in range(7)], dtype=np.int64)
_MATERIAL_EG = np.array([MATERIAL_EG[vt] for vt in range(7)], dtype=np.int64)
_MIDGAME_PST = np.array([MIDGAME_PST[vt] for vt in range(7)], dtype=np.int64)   # (7, 64)
_ENDGAME_PST = np.array([ENDGAME_PST[vt] for vt in range(7)], dtype=np.int64)
_PAWN_AHEAD_MG = np.array([PAWN_AHEAD_MG.get(vt, 0) for vt in range(7)], dtype=np.int64)
_PAWN_AHEAD_EG = np.array([PAWN_AHEAD_EG.get(vt, 0) for vt in range(7)], dtype=np.int64)
# mobility weight per virtual type (only bishop/rook/queen non-zero); phase weight per bitboard
# piece id (knight/bishop 1, rook 2, queen 4).
_MOBILITY_MG = np.zeros(7, dtype=np.int64)
_MOBILITY_EG = np.zeros(7, dtype=np.int64)
for _pt, _w in MOBILITY_WEIGHT_MG.items():
    _MOBILITY_MG[_pt] = _w
for _pt, _w in MOBILITY_WEIGHT_EG.items():
    _MOBILITY_EG[_pt] = _w
_PHASE_WEIGHT = np.array((0, 1, 1, 2, 4, 0), dtype=np.int64)

_FULL_BB = np.uint64((1 << 64) - 1)
_SEEN: dict[int, int] = {}
_PLAYED: dict[int, int] = {}


_STATE_SPEC = [
    ("tt_key", nb.uint64[:]),
    ("tt_move", nb.int64[:]),
    ("tt_score", nb.int64[:]),
    ("tt_depth", nb.int64[:]),        # -1 = empty slot
    ("tt_flag", nb.int64[:]),
    ("history", nb.int64[:, :, :]),   # [side][bitboard piece id][to-square] cutoff tally
    ("killers", nb.int64[:, :]),      # [depth][0..1] quiet move that cut there, or NO_MOVE
    ("path", nb.uint64[:]),           # node hash per ply, for repetition detection
    ("nodes", nb.int64),
    ("root_best", nb.int64),
    ("aborted", nb.int64),
    ("deadline", nb.float64),
    ("avoid", nb.int64),
]


@jitclass(_STATE_SPEC)  # type: ignore[no-untyped-call]
class SearchState:
    """All the mutable search state, threaded through the recursion - numba freezes module
    globals read-only, so this cannot be plain arrays. The transposition table and history
    persist across a game; get_move / bench_search reset the per-move fields."""

    def __init__(self) -> None:
        size = 1 << _TT_BITS
        self.tt_key = np.zeros(size, dtype=np.uint64)
        self.tt_move = np.zeros(size, dtype=np.int64)
        self.tt_score = np.zeros(size, dtype=np.int64)
        self.tt_depth = np.full(size, -1, dtype=np.int64)
        self.tt_flag = np.zeros(size, dtype=np.int64)
        self.history = np.zeros((2, 7, 64), dtype=np.int64)
        self.killers = np.full((MAX_DEPTH + 1, 2), NO_MOVE, dtype=np.int64)
        self.path = np.zeros(MAX_DEPTH * 2 + 8, dtype=np.uint64)
        self.nodes = 0
        self.root_best = NO_MOVE
        self.aborted = 0
        self.deadline = 0.0
        self.avoid = NO_MOVE


_STATE = SearchState()


@njit(cache=True)
def _popcount(bb: int) -> int:
    count = 0
    while bb:
        count += 1
        bb &= bb - np.uint64(1)
    return count


@njit(cache=True)
def _lsb(bb: int) -> int:
    idx = 0
    while (bb & np.uint64(1)) == 0:
        bb >>= np.uint64(1)
        idx += 1
    return idx


# How many of `colour`'s pawns stand on `square`'s file, strictly ahead of it toward the far
# rank - a doubled-pawn count for a pawn, a free structural signal for anything else. Same
# shift-and-mask as reference.pawns_ahead.
# ref: https://www.chessprogramming.org/Doubled_Pawn
@njit(cache=True)
def pawns_ahead(square: int, colour: int, own_pawns: int) -> int:
    if colour == WHITE:
        ahead = (np.uint64(0x0101_0101_0101_0100) << np.uint8(square)) & _FULL_BB
    else:
        ahead = np.uint64(0x0080_8080_8080_8080) >> np.uint8(63 - square)
    return _popcount(ahead & own_pawns)


# Squares a queen on `square` would reach through `occupied` - used from the king's square as a
# king-exposure proxy (reference.slider_scope, bitboard-native here).
# ref: https://www.chessprogramming.org/Mobility
@njit(cache=True)
def slider_scope(square: int, occupied: int) -> int:
    return _popcount(queen_attacks(np.uint8(square), occupied))


# Static score in centipawns from the side to move's point of view - reference.evaluate off
# bitboards, jitted end to end. Every weight is Texel-tuned, from tables.py.
@njit(cache=True)
def evaluate(board: Board) -> int:
    side_to_move = board.side
    occupied = board.occupancy[2]
    midgame = 0
    endgame = 0
    phase = 0

    for colour in range(2):
        sign = 1 if colour == WHITE else -1
        own_pawns = board.pieces[colour, 0]

        king_square = _lsb(board.pieces[colour, KING])
        king_file = king_square & 7

        for piece_type in range(6):
            bb = board.pieces[colour, piece_type]
            while bb:
                square = _lsb(bb)
                bb &= bb - np.uint64(1)

                far = piece_type == 0 and ((square & 7) ^ king_file) & 4
                virtual_type = FAR_PAWN if far else piece_type + 1

                i = square if colour == WHITE else square ^ 56
                mg = _MATERIAL_MG[virtual_type] + _MIDGAME_PST[virtual_type, i]
                eg = _MATERIAL_EG[virtual_type] + _ENDGAME_PST[virtual_type, i]

                if piece_type == BISHOP:
                    reach = _popcount(bishop_attacks(np.uint8(square), occupied))
                    mg += _MOBILITY_MG[virtual_type] * reach
                    eg += _MOBILITY_EG[virtual_type] * reach
                elif piece_type == ROOK:
                    reach = _popcount(rook_attacks(np.uint8(square), occupied))
                    mg += _MOBILITY_MG[virtual_type] * reach
                    eg += _MOBILITY_EG[virtual_type] * reach
                elif piece_type == QUEEN:
                    reach = _popcount(queen_attacks(np.uint8(square), occupied))
                    mg += _MOBILITY_MG[virtual_type] * reach
                    eg += _MOBILITY_EG[virtual_type] * reach
                elif piece_type == KING:
                    scope = slider_scope(square, occupied)
                    mg += KING_EXPOSURE_MG * scope
                    eg += KING_EXPOSURE_EG * scope

                stacked = pawns_ahead(square, colour, own_pawns)
                mg += _PAWN_AHEAD_MG[virtual_type] * stacked
                eg += _PAWN_AHEAD_EG[virtual_type] * stacked

                midgame += sign * mg
                endgame += sign * eg
                phase += _PHASE_WEIGHT[piece_type]

    if phase > 24:
        phase = 24

    # a small bonus for having the move.  ref: https://www.chessprogramming.org/Tempo
    tempo = 1 if side_to_move == WHITE else -1
    midgame += TEMPO_MG * tempo
    endgame += TEMPO_EG * tempo

    # tapered blend, floor division to match reference.evaluate.
    # ref: https://www.chessprogramming.org/Tapered_Eval
    score = (midgame * phase + endgame * (24 - phase)) // 24
    return score if side_to_move == WHITE else -score


# Piece type (0..5) on `square`, either colour, or -1 if empty.
@njit(cache=True)
def _piece_on(board: Board, square: int) -> int:
    b = np.uint64(1) << np.uint8(square)
    for colour in range(2):
        if board.occupancy[colour] & b:
            for pt in range(6):
                if board.pieces[colour, pt] & b:
                    return pt
    return -1


# A quiet move's cutoff tally for the side about to play it.
# ref: https://www.chessprogramming.org/History_Heuristic
@njit(cache=True)
def history_score(state: SearchState, board: Board, move: int) -> int:
    return state.history[board.side, move_piece(move), move_to_square(move)]


# Reward a quiet that cut (positive bonus) or punish one that was tried and did not (negative);
# the gravity term shrinks the effect near the cap so entries saturate instead of running away.
@njit(cache=True)
def update_history(state: SearchState, board: Board, move: int, bonus: int) -> None:
    side = board.side
    pt = move_piece(move)
    to = move_to_square(move)
    h = state.history[side, pt, to]
    state.history[side, pt, to] = h + bonus - h * abs(bonus) // MAX_HISTORY


# Ordering score, highest first: captures before quiets, and among captures the most valuable
# victim taken by the least valuable attacker (MVV-LVA), plus a promotion bonus.
# ref: https://www.chessprogramming.org/MVV-LVA
@njit(cache=True)
def move_ordering_score(board: Board, move: int) -> int:
    if move_is_en_passant(move):
        score = 100
    else:
        victim = _piece_on(board, move_to_square(move))
        score = PIECE_VALUE[victim] if victim >= 0 else 0

    if score:
        score = score * 16 - PIECE_VALUE[move_piece(move)]

    promo = move_promotion(move)
    if promo != PROMOTION_NONE:
        score += PIECE_VALUE[promo] - 100

    return score


# How much negamax's late-move reduction should trust a move: a quiet's key is its history
# score, a capture / promotion's is its ordering score pushed above history's ceiling so the
# reduction formula cancels back to zero for it - "don't reduce captures" without a gate.
@njit(cache=True)
def lmr_key(state: SearchState, board: Board, move: int, is_quiet: bool) -> int:
    if not is_quiet:
        return LMR_CAPTURE_FLOOR + move_ordering_score(board, move)
    return history_score(state, board, move)


# A mate score carries its distance from the root; re-root it to the storing node on the way
# into the TT and back to the current node on the way out.
# ref: https://www.chessprogramming.org/Score_Bounds_in_the_TT#Mate_Scores
@njit(cache=True)
def score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


@njit(cache=True)
def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


@njit  # not cache=True: builds a Board, which numba caching cannot serialize
def _null_move(board: Board) -> Board:
    child: Board = copy_board(board)
    child.side = np.uint8(1 - board.side)
    h = board.zobrist ^ SIDE_KEY
    if board.ep_square != NO_SQUARE:
        h ^= EP_FILE_KEYS[board.ep_square & 7]
    child.ep_square = np.uint8(NO_SQUARE)
    child.zobrist = h
    return child


# Occurrences of `node_hash` among path[0 .. upto-1].
@njit(cache=True)
def _repetitions(path: np.ndarray, upto: int, node_hash: int) -> int:
    n = 0
    for i in range(upto):
        if path[i] == node_hash:
            n += 1
    return n


# The four-tier ordering key (tt move, MVV-LVA, killer, history) packed into one int so an
# argsort reproduces reference.negamax's tuple sort exactly: each shift clears the full range
# of every lower tier (history is +/- MAX_HISTORY, lifted non-negative by the trailing 1<<16).
@njit(cache=True)
def _order_key(is_tt: int, mvv: int, is_killer: int, hist: int) -> int:
    return (is_tt << 44) + (mvv << 20) + (is_killer << 17) + hist + (1 << 16)


@njit  # not cache=True: recursive + objmode
def negamax(state: SearchState, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    state.nodes += 1

    if state.nodes % CHECK_EVERY == 0:
        with objmode(now="float64"):
            now = time.monotonic()
        if now >= state.deadline:
            state.aborted = 1
            return 0

    key = board.zobrist
    state.path[ply] = key

    # threefold repetition inside the search is a draw.
    # ref: https://www.chessprogramming.org/Repetitions
    if _repetitions(state.path, ply + 1, key) >= 3:
        return 0

    # mate-distance pruning: clamp the window to the mate band still reachable from here.
    # ref: https://www.chessprogramming.org/Score#Mate_Distance_Pruning
    if alpha < ply - MATE_SCORE:
        alpha = ply - MATE_SCORE
    if beta > MATE_SCORE - ply - 1:
        beta = MATE_SCORE - ply - 1
    if alpha >= beta:
        return alpha

    in_check = is_check(board, board.side)

    if ply >= MAX_DEPTH:
        _mv, count = legal_moves(board)
        if in_check and count == 0:
            return -MATE_SCORE + ply
        return evaluate(board)

    # transposition-table probe.  ref: https://www.chessprogramming.org/Transposition_Table
    slot = key & _TT_MASK
    tt_move = NO_MOVE
    tt_hit = state.tt_depth[slot] >= 0 and state.tt_key[slot] == key
    if tt_hit:
        tt_move = state.tt_move[slot]
        tt_score = score_from_tt(state.tt_score[slot], ply)
        if state.tt_depth[slot] >= depth:
            flag = state.tt_flag[slot]
            if flag == EXACT:
                return tt_score
            if flag == LOWER and tt_score >= beta:
                return tt_score
            if flag == UPPER and tt_score <= alpha:
                return tt_score

    moves, count = legal_moves(board)
    if count == 0:
        return -MATE_SCORE + ply if in_check else 0

    if depth <= 0:
        return quiescence_search(state, board, alpha, beta, ply)

    # internal iterative reduction: a non-PV node deep enough to be worth ordering with no TT
    # move loses a ply rather than grind every move unordered.
    # ref: https://www.chessprogramming.org/Internal_Iterative_Reductions
    if tt_move == NO_MOVE and depth >= IIR_MIN_DEPTH and beta - alpha == 1:
        depth -= 1

    shallow = depth <= LMP_MAX_DEPTH and not in_check
    static_eval = evaluate(board) if not in_check else 0

    # reverse futility pruning: so far ahead that conceding RFP_MARGIN per remaining ply still
    # clears beta.  ref: https://www.chessprogramming.org/Reverse_Futility_Pruning
    if (
        shallow
        and beta - alpha == 1
        and abs(beta) < MATE_THRESHOLD
        and static_eval - RFP_MARGIN * depth >= beta
    ):
        return static_eval

    # null-move pruning: pass the turn, search shallow; if the position still clears beta the
    # real move almost certainly cuts. guards: not in check, non-PV, an edge to spend, depth to
    # spare, non-pawn material (else zugzwang breaks the assumption).
    # ref: https://www.chessprogramming.org/Null_Move_Pruning
    own = board.occupancy[board.side]
    has_piece = own & ~board.pieces[board.side, 0] & ~board.pieces[board.side, KING]
    if (
        not in_check
        and beta - alpha == 1
        and depth >= NULL_MOVE_MIN_DEPTH
        and static_eval >= beta
        and has_piece
    ):
        margin = static_eval - beta
        reduced = (depth * NULL_MOVE_DEPTH_WEIGHT - margin) // NULL_MOVE_SCALE - 1
        if reduced < 0:
            reduced = 0
        score = -negamax(state, _null_move(board), reduced, -beta, -beta + 1, ply + 1)
        if state.aborted:
            return 0
        if score >= beta:
            return score

    # best-first ordering: TT move, then MVV-LVA, then killers, then history.
    # ref: https://www.chessprogramming.org/Killer_Heuristic
    order = np.empty(count, dtype=np.int64)
    k0 = state.killers[depth, 0]
    k1 = state.killers[depth, 1]
    for j in range(count):
        m = moves[j]
        is_killer = 1 if m in (k0, k1) else 0
        order[j] = _order_key(
            1 if m == tt_move else 0,
            move_ordering_score(board, m),
            is_killer,
            history_score(state, board, m),
        )
    ranked = np.argsort(-order, kind="mergesort")

    alpha_original = alpha
    best = -MATE_SCORE
    best_move = moves[ranked[0]]
    tried_quiets = np.empty(count, dtype=np.int64)
    tried = 0
    quiets_seen = 0

    for r in range(count):
        move = moves[ranked[r]]
        child = make_move(board, move)
        gives_check = is_check(child, child.side)
        extension = 1 if gives_check and ply + 1 < MAX_DEPTH else 0
        is_quiet = not move_is_capture(move) and move_promotion(move) == PROMOTION_NONE
        lmr_score = lmr_key(state, board, move, is_quiet) if r else 0

        if is_quiet:
            if not extension and shallow and beta - alpha == 1 and best > -MATE_THRESHOLD:
                if quiets_seen >= _LMP[depth]:
                    break
                if static_eval + FUTILITY_MARGIN * depth <= alpha:
                    quiets_seen += 1
                    continue
            quiets_seen += 1

        new_depth = depth - 1 + extension

        if child.halfmove_clock >= 4 and _repetitions(state.path, ply + 1, child.zobrist) >= 1:
            score = 0
        elif r == 0:
            score = -negamax(state, child, new_depth, -beta, -alpha, ply + 1)
        else:
            reduction = 0
            if extension == 0 and depth >= LMR_MIN_DEPTH:
                pull = r * LMR_MOVE_WEIGHT + depth * LMR_DEPTH_WEIGHT
                raw = pull // LMR_SCALE - lmr_score // LMR_HISTORY_SCALE
                reduction = raw
                if reduction < 0:
                    reduction = 0
                if reduction > new_depth - 1:
                    reduction = new_depth - 1
            score = -negamax(state, child, new_depth - reduction, -alpha - 1, -alpha, ply + 1)
            if reduction and score > alpha:
                score = -negamax(state, child, new_depth, -alpha - 1, -alpha, ply + 1)
            if alpha < score < beta:
                score = -negamax(state, child, new_depth, -beta, -alpha, ply + 1)

        if state.aborted:
            return 0

        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score

        if alpha >= beta:
            if is_quiet:
                if move != state.killers[depth, 0] and move != state.killers[depth, 1]:
                    state.killers[depth, 1] = state.killers[depth, 0]
                    state.killers[depth, 0] = move
                bonus = depth * depth
                update_history(state, board, move, bonus)
                for q in range(tried):
                    update_history(state, board, tried_quiets[q], -bonus)
            break

        if is_quiet:
            tried_quiets[tried] = move
            tried += 1

    if best >= beta:
        flag = LOWER
    elif best > alpha_original:
        flag = EXACT
    else:
        flag = UPPER

    state.tt_key[slot] = key
    state.tt_move[slot] = best_move
    state.tt_score[slot] = score_to_tt(best, ply)
    state.tt_depth[slot] = depth
    state.tt_flag[slot] = flag
    return best


# Past the horizon: keep going through captures only (every reply when in check) until the
# position is quiet, then score it. Delta pruning skips captures too small to reach alpha.
# ref: https://www.chessprogramming.org/Quiescence_Search
# ref: https://www.chessprogramming.org/Delta_Pruning
@njit  # not cache=True: recursive + objmode
def quiescence_search(state: SearchState, board: Board, alpha: int, beta: int, ply: int) -> int:
    state.nodes += 1

    if state.nodes % CHECK_EVERY == 0:
        with objmode(now="float64"):
            now = time.monotonic()
        if now >= state.deadline:
            state.aborted = 1
            return 0

    key = board.zobrist
    state.path[ply] = key
    if _repetitions(state.path, ply + 1, key) >= 3:
        return 0

    in_check = is_check(board, board.side)

    if ply >= MAX_DEPTH:
        _mv, count = legal_moves(board)
        if in_check and count == 0:
            return -MATE_SCORE + ply
        return evaluate(board)

    slot = key & _TT_MASK
    tt_move = NO_MOVE
    if state.tt_depth[slot] >= 0 and state.tt_key[slot] == key:
        tt_move = state.tt_move[slot]
        tt_score = score_from_tt(state.tt_score[slot], ply)
        flag = state.tt_flag[slot]
        if flag == EXACT:
            return tt_score
        if flag == LOWER and tt_score >= beta:
            return tt_score
        if flag == UPPER and tt_score <= alpha:
            return tt_score

    all_moves, count = legal_moves(board)

    if in_check:
        if count == 0:
            return -MATE_SCORE + ply
        moves = all_moves
        n = count
        standing_pat = -MATE_SCORE
    else:
        standing_pat = evaluate(board)
        if standing_pat >= beta:
            return standing_pat
        if standing_pat > alpha:
            alpha = standing_pat
        moves = np.empty(count, dtype=np.int64)
        n = 0
        for j in range(count):
            m = all_moves[j]
            if move_is_capture(m) or move_promotion(m) != PROMOTION_NONE:
                moves[n] = m
                n += 1

    order = np.empty(n, dtype=np.int64)
    for j in range(n):
        m = moves[j]
        tt_bit = 1 << 40 if m == tt_move else 0
        order[j] = tt_bit + move_ordering_score(board, m)
    ranked = np.argsort(-order[:n], kind="mergesort")

    alpha_original = alpha
    best_move = NO_MOVE

    for r in range(n):
        move = moves[ranked[r]]
        if not in_check and move_promotion(move) == PROMOTION_NONE:
            if move_is_en_passant(move):
                victim = 100
            else:
                v = _piece_on(board, move_to_square(move))
                victim = PIECE_VALUE[v] if v >= 0 else 0
            if standing_pat + victim + DELTA_PRUNING_MARGIN < alpha:
                break

        score = -quiescence_search(state, make_move(board, move), -beta, -alpha, ply + 1)
        if state.aborted:
            return 0

        if score > alpha:
            alpha = score
            best_move = move
        if alpha >= beta:
            break

    flag = LOWER if alpha >= beta else (EXACT if alpha > alpha_original else UPPER)
    state.tt_key[slot] = key
    state.tt_move[slot] = best_move
    state.tt_score[slot] = score_to_tt(alpha, ply)
    state.tt_depth[slot] = 0
    state.tt_flag[slot] = flag
    return alpha


# Root of the search: negamax's move loop keeping the best move, inside the caller's aspiration
# window. `prev_best` (last iteration's choice) is tried first. Returns (move, score).
@njit  # not cache=True: calls negamax
def search_root(
    state: SearchState, board: Board, depth: int, alpha: int, beta: int, prev_best: int
) -> tuple[int, int]:
    moves, count = legal_moves(board)
    state.path[0] = board.zobrist

    order = np.empty(count, dtype=np.int64)
    for j in range(count):
        order[j] = move_ordering_score(board, moves[j])
    ranked = list(np.argsort(-order, kind="mergesort"))

    if prev_best != NO_MOVE:
        for j in range(count):
            if moves[j] == prev_best:
                ranked.remove(j)
                ranked.insert(0, j)
                break

    best_move = moves[ranked[0]]
    best_score = -MATE_SCORE

    for r in range(count):
        move = moves[ranked[r]]
        child = make_move(board, move)
        if child.halfmove_clock >= 4 and _repetitions(state.path, 1, child.zobrist) >= 1:
            score = 0
        elif r == 0:
            score = -negamax(state, child, depth - 1, -beta, -alpha, 1)
        else:
            score = -negamax(state, child, depth - 1, -alpha - 1, -alpha, 1)
            if alpha < score < beta:
                score = -negamax(state, child, depth - 1, -beta, -alpha, 1)

        if state.aborted:
            return best_move, best_score

        if move == state.avoid and score < MATE_THRESHOLD:
            score -= REPETITION_PENALTY

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    return best_move, best_score


# Iterative deepening from the root with aspiration windows: deepen a ply at a time until
# soft_cap elapses with no depth in flight or the deadline aborts one mid-search, returning the
# best move / score from the last depth that finished inside the window.
# ref: https://www.chessprogramming.org/Iterative_Deepening
# ref: https://www.chessprogramming.org/Aspiration_Windows
def deepen(
    board: Board, start: float, deadline: float, soft_cap: float, fallback: int
) -> tuple[int, int]:
    _STATE.deadline = deadline
    _STATE.aborted = 0

    best = fallback
    score = 0
    move, value = fallback, 0

    for depth in range(1, MAX_DEPTH):
        if depth > 1 and time.monotonic() - start >= soft_cap:
            break

        if depth <= ASPIRATION_MIN_DEPTH:
            alpha, beta = -MATE_SCORE, MATE_SCORE
        else:
            alpha, beta = score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW
        delta = ASPIRATION_WINDOW

        while True:
            move, value = search_root(_STATE, board, depth, alpha, beta, best)
            if _STATE.aborted:
                break
            if value <= alpha and alpha > -MATE_SCORE:
                alpha = max(alpha - 2 * delta, -MATE_SCORE)
                delta *= 2
                continue
            if value >= beta and beta < MATE_SCORE:
                beta = min(beta + 2 * delta, MATE_SCORE)
                delta *= 2
                best = move
                continue
            break

        if _STATE.aborted:
            break
        best, score = move, value

    return best, score


# Emergency move when there is no time to search: the best-looking capture / promotion by the
# search's own ordering, else the first legal move.
def panic_move(board: Board) -> int:
    moves, count = legal_moves(board)
    best, best_score = int(moves[0]), -1
    for j in range(count):
        s = move_ordering_score(board, int(moves[j]))
        if s > best_score:
            best, best_score = int(moves[j]), s
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    board = parse_fen(fen)
    moves, count = legal_moves(board)
    if count == 0:
        return "0000"

    # anti-repetition: if we have been asked to move in this exact position before, tell
    # search_root the move we chose last time so it can shy away from a repeat.
    key = int(zobrist_hash(board))
    seen = _SEEN.get(key, 0)
    _SEEN[key] = seen + 1
    _STATE.avoid = _PLAYED.get(key, NO_MOVE) if seen else NO_MOVE

    if time_left_ms < PANIC_TIME_MS:
        best = panic_move(board)
        _PLAYED[key] = best
        return move_uci(best)

    try:
        _STATE.nodes = 0
        board.zobrist = np.uint64(key)
        start = time.monotonic()
        deadline = start + time_left_ms / HARD_LIMIT / 1000 - 0.05
        soft_cap = time_left_ms / SOFT_LIMIT / 1000
        best, _score = deepen(board, start, deadline, soft_cap, int(moves[0]))
        _PLAYED[key] = int(best)

        if _tt_used() > TT_MAX_ENTRIES:
            _STATE.tt_depth[:] = -1
        return move_uci(int(best))
    except Exception as exc:  # numba failed to compile, or a bug in the port - hand off
        print(f"agent: numba search unavailable ({exc!r}); using the reference engine")
        return reference.get_move(fen, time_left_ms)


def _tt_used() -> int:
    return int(np.count_nonzero(_STATE.tt_depth >= 0))


# tools/nodebench.py / tools/verify_search.py hook: fixed-depth search from a clean table.
def bench_search(fen: str, depth: int) -> tuple[str, int, int]:
    _STATE.nodes = 0
    _STATE.aborted = 0
    _STATE.deadline = time.monotonic() + 3600.0
    _STATE.avoid = NO_MOVE
    _STATE.tt_depth[:] = -1
    _STATE.history[:] = 0
    _STATE.killers[:] = NO_MOVE

    board = parse_fen(fen)
    board.zobrist = np.uint64(zobrist_hash(board))
    move, score = search_root(_STATE, board, depth, -MATE_SCORE, MATE_SCORE, NO_MOVE)
    return move_uci(int(move)), int(score), int(_STATE.nodes)


# Spend the platform's init budget searching the opening position, filling the transposition
# table before the game clock starts (it persists across a game). AGENT_WARM_UP_S overrides the
# seconds; tools that spawn a process per game set it to 0.
WARM_UP_S = float(os.environ.get("AGENT_WARM_UP_S", "40.0"))


def _warm_up() -> None:
    board = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    board.zobrist = np.uint64(zobrist_hash(board))
    moves, _count = legal_moves(board)
    search_root(_STATE, board, 2, -MATE_SCORE, MATE_SCORE, NO_MOVE)  # compile the search
    if WARM_UP_S > 0:
        now = time.monotonic()
        _STATE.nodes = 0
        _STATE.aborted = 0
        deepen(board, now, now + WARM_UP_S, WARM_UP_S, int(moves[0]))


_warm_up()
