"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state lasts until that game ends, then resets.
Import runs first, inside a 90 s budget, before our clock starts.

An alpha-beta search over a Texel-tuned tapered evaluation, built up in the phases in
docs/plan.md. The board, move generator, make-move and hashing are the numba-compiled bitboard
layer (board.py, movegen.py, attacks.py, move.py, zobrist.py, bitboard.py) standing in for
python-chess; the search and evaluation here are @njit too, so a node never leaves compiled
code. reference.py is the plain python-chess engine this was ported from - it keeps the same
function names, the verify_* tools check the two against each other, and get_move falls back to
it if numba fails to compile on the platform.

Features (each carries a chessprogramming.org reference at its use site):

  Search framework
    - iterative deepening (deepen)
    - aspiration windows, geometric widening on a fail high / low
    - negamax alpha-beta with a fail-soft window
    - principal variation search: full window on move 0, null-window scout + re-search after
    - transposition table: one slot per hash & mask, always-replace, full key stored so an
      index collision misses instead of returning a wrong score; mate scores re-rooted in / out
    - internal iterative reduction when a deep non-PV node has no TT move to order by

  Selectivity / pruning
    - mate-distance pruning
    - reverse futility pruning (static null move)
    - null-move pruning, reduction scaled by depth and the margin over beta, zugzwang guard
    - late move reductions, scaled by move index and depth, eased by history, captures floored
      out rather than gated
    - late move pruning (quiet-move count cap by depth)
    - move-loop futility pruning
    - check extensions
    - quiescence search over captures and quiet promotions (every reply when in check)
    - delta pruning in quiescence

  Move ordering
    - TT move first, then MVV-LVA captures, then killers, then the history heuristic
    - killer moves, two per ply
    - history heuristic with a gravity term so scores saturate; a separate table per side

  Repetition handling
    - threefold repetition detected inside the search off a per-ply hash trail
    - the fifty-move rule scored as the draw the referee claims, checkmate first
    - a first repetition scored as a draw so the search can steer into or away from it
    - a cross-call penalty on replaying the move chosen last time in this exact position, since
      each get_move rebuilds the board with no history and cannot otherwise see a real repeat

  Evaluation (NNUE - engine/accumulator.py, weights engine/net.npz from tools/train_nn.py)
    - dual-perspective piece-placement net: 768 -> 256 feature transformer per colour, then a
      512 -> 32 -> 32 -> 1 tail over [own-to-move, other] with clipped-ReLU activations
    - the transformer output is an int32 accumulator carried on the Board; make_move keeps it
      in step so a leaf eval is just the small float tail (stage 3), rebuilt from scratch here
      in stage 2
    - trained on Stockfish-labelled Lichess positions; only the net we trained ships, no engine
      or borrowed net

  Time management and safety
    - hard cap per move plus a soft cap past which no new depth starts
    - wall clock polled every CHECK_EVERY nodes, abort via a flag threaded through the recursion
    - a panic path that returns a move without searching when almost out of time
    - the whole search wrapped so a bug keeps the last completed depth instead of losing on a
      crash; get_move itself falls back to reference.py if the numba layer will not compile

  Infrastructure
    - numba @njit end to end; a jitclass for the board and for the mutable search state
    - a pinned / checker-aware move generator that decides most moves with two bitboard tests
      and no make-move
    - incremental Zobrist hashing kept in step by make_move
    - import-time JIT warm-up so compilation lands in the platform's 90 s init budget
"""

# ruff: noqa: E402  (the thread pin has to precede the numpy import)
import os as _os

# a referee running games in parallel puts many agents on the same cores; a BLAS that spawns a
# thread per core in each one flags the clock. The search is single-threaded (numba @njit, no
# parallel=) - nothing here wants a pool. Harmless on the platform's one core, insurance for a
# local parallel harness and for BLAS calls chess.syzygy / numpy might make underneath.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import time

import numba as nb
import numpy as np
from numba import njit, objmode
from numba.experimental import jitclass

from engine import reference
from engine.accumulator import evaluate_accumulator, popcount
from engine.attacks import (
    KING_ATTACKS_NB,
    KNIGHT_ATTACKS_NB,
    PAWN_ATTACKS_NB,
    bishop_attacks,
    rook_attacks,
)
from engine.bitboard import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    NO_SQUARE,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    lsb_index,
)
from engine.board import Board, copy_board, parse_fen
from engine.move import (
    PROMOTION_NONE,
    move_from_square,
    move_is_capture,
    move_is_en_passant,
    move_piece,
    move_to_square,
    move_uci,
)
from engine.move import (
    move_promotion_raw as move_promotion,
)
from engine.movegen import is_check, legal_moves, make_move
from engine.tablebase import COVERED_MATERIAL, TB_MEN, TB_NONE, best_tb_move, tb_score
from engine.zobrist import EP_FILE_KEYS, SIDE_KEY, zobrist_hash

# sorted material keys we have a Syzygy table for (engine.tablebase._material_key layout), frozen
# into the jitted probe so an uncovered <= TB_MEN node skips the objmode call.
TB_COVERED = np.array(COVERED_MATERIAL, dtype=np.int64)

# rough centipawn values for move ordering and delta pruning only; indexed by bitboard piece id
# (pawn 0 .. king 5), king 0 since it is never a victim.
# ref: https://www.chessprogramming.org/Point_Value
PIECE_VALUE = np.array((100, 320, 330, 500, 900, 0), dtype=np.int64)

# flat values for the SEE capture swap, indexed the same way. the king is huge, not zero: it can
# only ever be the last capturer (the swap folds out any line where the far side could recapture
# it), and the sentinel is what makes that fall out.  reference.SEE_VALUE.
SEE_VALUE = np.array((100, 320, 330, 500, 900, 30_000), dtype=np.int64)

# sentinel beyond any reachable material total; +MATE_SCORE = we deliver mate, -MATE_SCORE = we
# are mated. a score past MATE_THRESHOLD in magnitude encodes a forced mate MATE_SCORE - |score|
# plies away.
MATE_SCORE = 1_000_000
MAX_DEPTH = 64
MATE_THRESHOLD = MATE_SCORE - 2 * MAX_DEPTH

# a Syzygy win/loss in the search: far beyond any NNUE eval so the search always prefers it, but
# below MATE_THRESHOLD so a real forced mate still outranks it and the mate-score plumbing leaves
# it alone. NO_TB is outside every real score - tb_probe_score's "no tablebase answer" sentinel.
TB_WIN_SCORE = MATE_SCORE - 2000
NO_TB = MATE_SCORE + 1

# time budget as clock fractions: at most 1/HARD_LIMIT of the clock on a move, and no new
# deepening iteration once 1/SOFT_LIMIT of it is spent. A flat fraction starves the endgame:
# in the rated games (docs/nnue-plan.md post-mortem) the median spend fell from 2.85 s above 20
# men to 0.47 s at 8 men or fewer while 15-40 s sat unused at the end, and every endgame loss
# was played on sub-second moves. With EG_MEN men or fewer the soft cap loosens to
# 1/SOFT_LIMIT_ENDGAME; the hard cap is unchanged.
HARD_LIMIT = 4
SOFT_LIMIT = 40
SOFT_LIMIT_ENDGAME = 20
# Men at or below which the endgame budget applies. Deliberately separate from EG_MEN, which
# only says which net evaluates: round 97's thirty-move grind was played at 13-14 men, above the
# net's threshold, on moves that moved the evaluation by single centipawns.
ENDGAME_TIME_MEN = 16
# The endgame hard cap. The soft cap only gates STARTING an iteration, so one begun just under it
# runs on to the hard cap - in round 97 three consecutive moves each took 98% of a quarter of the
# remaining clock and changed the evaluation by 3 cp, with 11 s left.
HARD_LIMIT_ENDGAME = 4
CHECK_EVERY = 1024        # poll the wall clock every this many nodes
PANIC_TIME_MS = 300       # below this much left, skip the search and grab a move

# transposition table: one slot per (hash & TT_MASK), always-replace, the full key stored so an
# index collision with a different position simply misses. ~4M slots ~ 170 MB.
# ref: https://www.chessprogramming.org/Transposition_Table
UPPER, EXACT, LOWER = 0, 1, 2
TT_BITS = 22
TT_MASK = (1 << TT_BITS) - 1

MAX_HISTORY = 1 << 14        # quiet-move history score saturates toward +/- this
TT_MAX_ENTRIES = 1_500_000  # after a game grows the table past this, get_move clears it

# search-tuning knobs; the reasoning for each lives at its use site in negamax / deepen. Mirrors
# reference.py's block of the same constants.
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
CONTEMPT = 30             # cp a draw is worth to us, negated: the search plays on from a level
                          # position rather than settle for a repetition. Mirrored in
                          # reference.py as CONTEMPT - keep the two equal.

LMP = np.array([3 + d * d for d in range(LMP_MAX_DEPTH + 1)], dtype=np.int64)  # quiet cap by depth
NO_MOVE = -1

# sentinel in the TT's static-eval slot for "not computed here". Out of band: real evals sit
# near +/- a few thousand cp, mate scores at +/- MATE_SCORE (1e6).
NO_EVAL = 1 << 30

# kill switch for the TT static-eval cache. On (1) the search reuses a TT key match's stored
# NNUE eval instead of recomputing; evaluate is pure in (acc, side) so this is exact - move,
# score and node count are identical to the recompute path, only wall time drops. Kept as a
# constant so tools/nodebench.py can prove the equivalence by flipping it.
EVAL_CACHE = 1

# The static evaluation is the NNUE forward pass over the Board's accumulator - see
# engine/accumulator.py (evaluate below is a one-line wrapper). None of the linear eval's tuned
# tables, phase blend, or hand-built pawn masks survive the switch. The NNUE forward is heavier
# than the old linear eval, so its result is cached in the TT (tt_eval) and reused whenever the
# search revisits a position - evaluate is pure in (acc, side), so a cache hit is exact.

# per-game anti-repetition, keyed by zobrist hash: how many times we have been asked to move in
# a position, and what we chose there last (reference.py keeps SEEN / PLAYED the same way).
SEEN: dict[int, int] = {}
PLAYED: dict[int, int] = {}

# every position the real game has reached, oldest first, trimmed to the reversible window (only
# positions since the last pawn move or capture can ever recur). get_move appends the position
# it was last handed and loads this into STATE.game_hashes before each search, so the search's
# repetition checks count the whole game, not just the current tree. _LAST_KEY is the previous
# get_move's position hash, 0 when there is nothing to carry (first move, or a non-search reply).
GAME_HASHES: list[int] = []
_LAST_KEY: int = 0
MASK64 = (1 << 64) - 1


STATE_SPEC = [
    ("tt_key", nb.uint64[:]),
    ("tt_move", nb.int64[:]),
    ("tt_score", nb.int64[:]),
    ("tt_depth", nb.int64[:]),        # -1 = empty slot
    ("tt_flag", nb.int64[:]),
    ("tt_eval", nb.int64[:]),         # cached NNUE static eval for this slot's position, or NO_EVAL
    ("history", nb.int64[:, :, :]),   # [side][bitboard piece id][to-square] cutoff tally
    ("killers", nb.int64[:, :]),      # [depth][0..1] quiet move that cut there, or NO_MOVE
    ("path", nb.uint64[:]),           # node hash per ply, for repetition detection
    ("game_hashes", nb.uint64[:]),    # positions from the real game before this search (the
    ("game_n", nb.int64),             # reversible window), so repetition checks see the whole game
    ("nodes", nb.int64),
    ("root_best", nb.int64),
    ("aborted", nb.int64),
    ("deadline", nb.float64),
    ("avoid", nb.int64),
]


@jitclass(STATE_SPEC)  # type: ignore[no-untyped-call]
class SearchState:
    """All the mutable search state, threaded through the recursion - numba freezes module
    globals read-only, so this cannot be plain arrays. The transposition table and history
    persist across a game; get_move / bench_search reset the per-move fields. In reference.py
    this is just module globals (TT, KILLERS, HISTORY, NODES, DEADLINE, AVOID)."""

    def __init__(self) -> None:
        size = 1 << TT_BITS
        self.tt_key = np.zeros(size, dtype=np.uint64)
        self.tt_move = np.zeros(size, dtype=np.int64)
        self.tt_score = np.zeros(size, dtype=np.int64)
        self.tt_depth = np.full(size, -1, dtype=np.int64)
        self.tt_flag = np.zeros(size, dtype=np.int64)
        self.tt_eval = np.full(size, NO_EVAL, dtype=np.int64)
        self.history = np.zeros((2, 7, 64), dtype=np.int64)
        self.killers = np.full((MAX_DEPTH + 1, 2), NO_MOVE, dtype=np.int64)
        self.path = np.zeros(MAX_DEPTH * 2 + 8, dtype=np.uint64)
        self.game_hashes = np.zeros(128, dtype=np.uint64)
        self.game_n = 0
        self.nodes = 0
        self.root_best = NO_MOVE
        self.aborted = 0
        self.deadline = 0.0
        self.avoid = NO_MOVE


STATE = SearchState()


# Static score in centipawns from the side to move's point of view - the NNUE forward pass over
# the Board's accumulator. make_move / parse_fen keep board.acc in step, so this is just the
# dequantise-and-run-the-float-tail in engine/accumulator.py; no board scan here.
@njit(cache=True)
def evaluate(board: Board) -> int:
    return evaluate_accumulator(board.acc, board.side, popcount(board.occupancy[2]))


# Piece type (0..5) on `square`, either colour, or -1 if empty - python-chess's board.piece_at
# without the Piece object.
@njit(cache=True)
def piece_on(board: Board, square: int) -> int:
    b = np.uint64(1) << np.uint8(square)
    for colour in range(2):
        if board.occupancy[colour] & b:

            for pt in range(6):
                if board.pieces[colour, pt] & b:
                    return pt

    return -1


# Every piece of either colour attacking `square`, given occupancy `occ`. is_attacked widened to
# the whole attacker set and taking an explicit occupancy, so clearing a piece's bit uncovers
# the x-ray slider behind it. reference.attackers_to (chess.Board.attackers).
# ref: https://www.chessprogramming.org/Square_Attacked_By
@njit(cache=True)
def attackers_to(board: Board, square: int, occ: int) -> int:
    sq = np.uint8(square)
    knights = board.pieces[0, KNIGHT] | board.pieces[1, KNIGHT]
    kings = board.pieces[0, KING] | board.pieces[1, KING]
    bishops_queens = board.pieces[0, BISHOP] | board.pieces[1, BISHOP] \
        | board.pieces[0, QUEEN] | board.pieces[1, QUEEN]
    rooks_queens = board.pieces[0, ROOK] | board.pieces[1, ROOK] \
        | board.pieces[0, QUEEN] | board.pieces[1, QUEEN]

    result = KNIGHT_ATTACKS_NB[square] & knights
    result |= KING_ATTACKS_NB[square] & kings
    result |= PAWN_ATTACKS_NB[BLACK, square] & board.pieces[0, PAWN]  # white pawns hitting here
    result |= PAWN_ATTACKS_NB[WHITE, square] & board.pieces[1, PAWN]  # black pawns hitting here
    result |= bishop_attacks(sq, occ) & bishops_queens
    result |= rook_attacks(sq, occ) & rooks_queens
    return result & occ


# Static exchange evaluation: the material `board.side` wins or loses if the capture `move` is
# met by the best sequence of recaptures on its square, cheapest attacker first, either side
# free to stop once continuing would lose. No search, no make-move. The swap algorithm, with the
# attacker set recomputed each step so a revealed x-ray slider joins in. reference.see.
# ref: https://www.chessprogramming.org/Static_Exchange_Evaluation
@njit(cache=True)
def see(board: Board, move: int) -> int:
    to = move_to_square(move)
    one = np.uint64(1)
    occ = board.occupancy[2] ^ (one << np.uint8(move_from_square(move)))

    if move_is_en_passant(move):
        captured = to - 8 if board.side == WHITE else to + 8  # the pawn is behind the ep square
        occ ^= one << np.uint8(captured)
        captured_value = SEE_VALUE[PAWN]
    else:
        victim = piece_on(board, to)
        captured_value = SEE_VALUE[victim] if victim >= 0 else 0

    gain = np.empty(34, dtype=np.int64)
    gain[0] = captured_value
    last_value = SEE_VALUE[move_piece(move)]  # our piece now standing on `to`
    side = 1 - board.side                     # the side to recapture next
    atk = attackers_to(board, to, occ)
    d = 0

    while True:
        found = -1
        found_pt = 0
        for pt in range(6):                   # this side's least valuable attacker still standing
            subset = atk & board.pieces[side, pt] & occ
            if subset:
                found = lsb_index(subset)
                found_pt = pt
                break
        if found < 0:
            break
        if found_pt == KING:
            # a king can only recapture onto a square the far side no longer attacks
            answered = False
            for pt in range(6):
                if atk & board.pieces[1 - side, pt] & occ:
                    answered = True
                    break
            if answered:
                break

        d += 1
        gain[d] = last_value - gain[d - 1]     # material for `side` if it recaptures here
        if max(-gain[d - 1], gain[d]) < 0:     # recapturing and standing pat both lose: stop
            break

        last_value = SEE_VALUE[found_pt]
        occ ^= one << np.uint8(found)
        atk = attackers_to(board, to, occ)     # a slider behind `found` may now bear on `to`
        side = 1 - side
        if d >= 32:
            break

    while d > 0:
        d -= 1
        gain[d] = -max(-gain[d], gain[d + 1])
    return gain[0]


# Is the static exchange evaluation of `move` at least `threshold`? Same swap as see(), carried
# as one running balance the Stockfish way so there is no gain[] array to allocate - see() is
# called once per capture in quiescence, the hottest loop in the engine. The two agree exactly:
# see_ge(m, t) == (see(m) >= t), which tools/verify_see.py checks. reference.see_ge.
# ref: https://www.chessprogramming.org/Static_Exchange_Evaluation
@njit(cache=True)
def see_ge(board: Board, move: int, threshold: int) -> bool:
    to = move_to_square(move)
    one = np.uint64(1)
    occ = board.occupancy[2] ^ (one << np.uint8(move_from_square(move)))

    if move_is_en_passant(move):
        captured = to - 8 if board.side == WHITE else to + 8
        occ ^= one << np.uint8(captured)
        captured_value = SEE_VALUE[PAWN]
    else:
        victim = piece_on(board, to)
        captured_value = SEE_VALUE[victim] if victim >= 0 else 0

    # balance = what the side to move is up if the exchange stops here, less the threshold
    balance = captured_value - threshold
    if balance < 0:
        return False  # winning the victim outright still falls short

    balance = SEE_VALUE[move_piece(move)] - balance
    if balance <= 0:
        return True  # losing the moved piece to the first recapture still clears the bar

    side = 1 - board.side  # side to recapture next
    result = 1             # 1 while the exchange stands in the original mover's favour
    atk = attackers_to(board, to, occ)

    while True:
        atk &= occ
        found = -1
        found_pt = 0
        for pt in range(6):  # this side's least valuable attacker still standing
            subset = atk & board.pieces[side, pt] & occ
            if subset:
                found = lsb_index(subset)
                found_pt = pt
                break
        if found < 0:
            break

        result ^= 1
        if found_pt == KING:
            # a king recaptures only onto a square the far side no longer attacks; if it still
            # does, this capture never happens and the exchange stopped one move earlier
            if atk & board.occupancy[1 - side] & occ:
                result ^= 1
            break

        balance = SEE_VALUE[found_pt] - balance
        if balance < result:
            break

        occ ^= one << np.uint8(found)
        atk = attackers_to(board, to, occ)
        side = 1 - side

    return result == 1


# A quiet move's cutoff tally for the side about to play it (reference.history_score).
# ref: https://www.chessprogramming.org/History_Heuristic
@njit(cache=True)
def history_score(state: SearchState, board: Board, move: int) -> int:
    return state.history[board.side, move_piece(move), move_to_square(move)]


# Reward a quiet that cut (positive bonus) or punish one that was tried and did not (negative);
# the gravity term shrinks the effect near the cap so entries saturate instead of running away.
# reference.update_history.  ref: https://www.chessprogramming.org/History_Heuristic
@njit(cache=True)
def update_history(state: SearchState, board: Board, move: int, bonus: int) -> None:
    side = board.side
    pt = move_piece(move)
    to = move_to_square(move)
    h = state.history[side, pt, to]

    state.history[side, pt, to] = h + bonus - h * abs(bonus) // MAX_HISTORY


# Ordering score, highest first: captures before quiets, and among captures the most valuable
# victim taken by the least valuable attacker (MVV-LVA), plus a promotion bonus. A capture that
# loses material to the recapture sequence (negative SEE) is scored by that SEE instead, which
# drops it below every quiet. reference.move_ordering_score.
# ref: https://www.chessprogramming.org/MVV-LVA
@njit(cache=True)
def move_ordering_score(board: Board, move: int) -> int:
    if move_is_en_passant(move):
        victim = PAWN
        score = 100
    else:
        victim = piece_on(board, move_to_square(move))
        score = PIECE_VALUE[victim] if victim >= 0 else 0

    if score:
        attacker = move_piece(move)
        # only the "wrong way round" captures can lose material; SEE the rest so an even or
        # winning trade keeps its MVV-LVA rank without paying for the swap.
        if PIECE_VALUE[attacker] > PIECE_VALUE[victim]:
            exchange = see(board, move)
            if exchange < 0:
                return exchange
        # most valuable victim, least valuable attacker. the * 16 keeps every capture above
        # every quiet move.
        score = score * 16 - PIECE_VALUE[attacker]

    promo = move_promotion(move)
    if promo != PROMOTION_NONE:
        score += PIECE_VALUE[promo] - 100

    return score


# How much negamax's late-move reduction should trust a move: a quiet's key is its history
# score, a capture / promotion's is its ordering score pushed above history's ceiling so the
# reduction formula cancels back to zero for it - "don't reduce captures" without a gate.
# reference.lmr_key.
@njit(cache=True)
def lmr_key(state: SearchState, board: Board, move: int, is_quiet: bool) -> int:
    if not is_quiet:
        return LMR_CAPTURE_FLOOR + move_ordering_score(board, move)
    
    return history_score(state, board, move)


# A mate score carries its distance from the root; re-root it to the storing node on the way
# into the TT and back to the current node on the way out. reference.score_to_tt / score_from_tt.
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


# The child position after passing the turn, for null-move pruning: chess.Move.null() pushed
# onto a copy, with the zobrist side / en-passant keys mixed the same way make_move does.
@njit  # not cache=True: builds a Board, which numba caching cannot serialize
def null_move(board: Board) -> Board:
    child: Board = copy_board(board)
    child.side = np.uint8(1 - board.side)

    key = board.zobrist ^ SIDE_KEY

    if board.ep_square != NO_SQUARE:
        key ^= EP_FILE_KEYS[board.ep_square & 7]

    child.ep_square = np.uint8(NO_SQUARE)
    child.zobrist = key

    return child


# Occurrences of `node_hash` among path[0 .. upto-1] - the search's own board.is_repetition,
# walking a per-ply hash trail instead of a move stack.
# ref: https://www.chessprogramming.org/Repetitions
@njit(cache=True)
def repetitions(path: np.ndarray, upto: int, node_hash: int) -> int:
    n = 0
    for i in range(upto):
        if path[i] == node_hash:
            n += 1

    return n


# Occurrences of `node_hash` among the positions the real game reached before this search
# (get_move seeds state.game_hashes with the reversible window). Added to `repetitions` at every
# repetition check so a line that returns to a position the game has already visited is scored
# as the draw the referee would claim - the search cannot see the game history any other way.
@njit(cache=True)
def game_reps(state: SearchState, node_hash: int) -> int:
    n = 0
    for i in range(state.game_n):
        if state.game_hashes[i] == node_hash:
            n += 1

    return n


# A draw scored from the side-to-move's view at `ply`: negative when it is our move (an even ply
# from the root), positive when it is the opponent's. A repetition or stalemate then only wins
# the search when every real try is worse than conceding CONTEMPT, so a level game is played on
# rather than settled. reference._contempt_draw.
# ref: https://www.chessprogramming.org/Contempt_Factor
@njit(cache=True)
def contempt_draw(ply: int) -> int:
    return -CONTEMPT if ply % 2 == 0 else CONTEMPT


# The four-tier ordering key (tt move, MVV-LVA, killer, history) packed into one int so an
# argsort reproduces reference.negamax's tuple sort exactly: each shift clears the full range
# of every lower tier (history is +/- MAX_HISTORY, lifted non-negative by the trailing 1<<16).
@njit(cache=True)
def order_key(is_tt: int, mvv: int, is_killer: int, hist: int) -> int:
    return (is_tt << 44) + (mvv << 20) + (is_killer << 17) + hist + (1 << 16)


# Best score for the side to move over `depth` plies of best play, within the alpha/beta window.
# `ply` is the distance from the root - it bounds check extensions and scales mate scores. The
# jitted twin of reference.negamax; each pruning step carries its own reference there and here.
# ref: https://www.chessprogramming.org/Negamax
# ref: https://www.chessprogramming.org/Alpha-Beta
# Exact Syzygy score for a position with <= TB_MEN men, or NO_TB when the tables cannot answer
# (too many men, or the child table is absent). A win/loss is ply-adjusted so the search still
# prefers the faster one. objmode drops to engine.tablebase.tb_score with the raw piece bitboards.
@njit  # not cache=True: objmode
def tb_probe_score(board: Board, ply: int) -> int:
    if popcount(board.occupancy[2]) > TB_MEN:
        return NO_TB

    # material key (counts of P,N,B,R,Q per side, 3 bits each; colour-canonicalised) - skip the
    # objmode probe entirely when we have no table for this ending.
    white_key = np.int64(0)
    black_key = np.int64(0)
    for piece_type in range(5):  # pawn..queen, king excluded
        white_key |= np.int64(popcount(board.pieces[0, piece_type])) << np.int64(3 * piece_type)
        black_key |= np.int64(popcount(board.pieces[1, piece_type])) << np.int64(3 * piece_type)
    key = white_key | (black_key << np.int64(15))
    swapped = black_key | (white_key << np.int64(15))
    if swapped < key:
        key = swapped
    lo = np.searchsorted(TB_COVERED, key)
    if lo >= TB_COVERED.size or TB_COVERED[lo] != key:
        return NO_TB

    pieces = board.pieces
    side = board.side
    outcome = 0
    with objmode(outcome="int64"):
        outcome = tb_score(pieces, side)
    if outcome == TB_NONE:
        return NO_TB
    if outcome > 0:
        return TB_WIN_SCORE - ply
    if outcome < 0:
        return -TB_WIN_SCORE + ply
    return 0  # exact draw; caller applies contempt


@njit  # not cache=True: recursive + objmode
def negamax(state: SearchState, board: Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    state.nodes += 1

    # abort once the time cap is reached
    if state.nodes % CHECK_EVERY == 0:
        with objmode(now="float64"):
            now = time.monotonic()

        if now >= state.deadline:
            state.aborted = 1
            return 0

    key = board.zobrist
    state.path[ply] = key

    # threefold repetition is a draw - counting the search path plus what the real game already
    # reached (game_reps), so a position the game has seen twice needs only one more here.
    # ref: https://www.chessprogramming.org/Repetitions
    if repetitions(state.path, ply + 1, key) + game_reps(state, key) >= 3:
        return contempt_draw(ply)

    # mate-distance pruning: clamp the window to the mate band still reachable from here.
    # ref: https://www.chessprogramming.org/Score#Mate_Distance_Pruning
    if alpha < ply - MATE_SCORE:
        alpha = ply - MATE_SCORE

    if beta > MATE_SCORE - ply - 1:
        beta = MATE_SCORE - ply - 1

    if alpha >= beta:
        return alpha

    in_check = is_check(board, board.side)

    # fifty-move rule: the referee claims it, so a hundred halfmoves with no capture and no pawn
    # move is a draw however won the position looks - and a tablebase probe answers as if the
    # clock were zeroed, so this has to come first. Checkmate outranks the clock (a mate on the
    # hundredth halfmove is a win), which is why the rare branch pays for a move generation.
    # ref: https://www.chessprogramming.org/Fifty-move_Rule
    if board.halfmove_clock >= 100:
        _fifty_moves, fifty_count = legal_moves(board)

        if in_check and fifty_count == 0:
            return -MATE_SCORE + ply

        return contempt_draw(ply)

    # tablebase: an exact WDL for a <= TB_MEN position - ground truth in place of the net's
    # saturated "winning-ish" number, so the search actually converts won endings and steers
    # clear of tablebase draws. before the TT probe so a stale blind entry can't shadow it.
    if popcount(board.occupancy[2]) <= TB_MEN:
        tb = tb_probe_score(board, ply)
        if tb != NO_TB:
            return contempt_draw(ply) if tb == 0 else tb

    # recursion floor: a checking sequence extends every ply, so `depth` never falls - this
    # stops it running away. resolve no-legal-moves first so a mate isn't misjudged by the eval.
    if ply >= MAX_DEPTH:
        _mv, count = legal_moves(board)
        if in_check and count == 0:
            return -MATE_SCORE + ply
        return evaluate(board)

    # transposition-table probe: have we searched this exact position before?
    # ref: https://www.chessprogramming.org/Transposition_Table
    slot = key & TT_MASK
    tt_move = NO_MOVE
    tt_hit = state.tt_depth[slot] >= 0 and state.tt_key[slot] == key

    if tt_hit:
        tt_move = state.tt_move[slot]
        tt_score = score_from_tt(state.tt_score[slot], ply)  # re-root a stored mate score here
        # trust it only if searched at least as deep. an exact score stands; a bound only when
        # it already proves a cutoff.
        if state.tt_depth[slot] >= depth:
            flag = state.tt_flag[slot]
            if flag == EXACT:
                return tt_score
            if flag == LOWER and tt_score >= beta:
                return tt_score
            if flag == UPPER and tt_score <= alpha:
                return tt_score

    moves, count = legal_moves(board)
    # no legal moves: checkmate if in check, else stalemate (a draw). the +ply makes a mate
    # found sooner score higher once negated back up the tree.
    if count == 0:
        return -MATE_SCORE + ply if in_check else contempt_draw(ply)

    # out of depth: hand off to a captures-only search so we don't judge a half-finished trade
    if depth <= 0:
        return quiescence_search(state, board, alpha, beta, ply)

    # internal iterative reduction: a non-PV node deep enough to be worth ordering with no TT
    # move loses a ply rather than grind every move unordered.
    # ref: https://www.chessprogramming.org/Internal_Iterative_Reductions
    if tt_move == NO_MOVE and depth >= IIR_MIN_DEPTH and beta - alpha == 1:
        depth -= 1

    # static eval, shared by reverse futility and null-move here and late-move / futility
    # pruning in the move loop. computed whenever not in check - the eval misjudges a position
    # in check - regardless of depth, since null-move pruning needs it past the shallow cutoff.
    # reuse the TT's cached eval on a key match: evaluate is pure in (acc, side), so it is the
    # exact value evaluate(board) would return, minus the NNUE forward.
    shallow = depth <= LMP_MAX_DEPTH and not in_check
    if in_check:
        static_eval = 0
    elif EVAL_CACHE and tt_hit and state.tt_eval[slot] != NO_EVAL:
        static_eval = state.tt_eval[slot]
    else:
        static_eval = evaluate(board)

    # reverse futility pruning: so far ahead that conceding RFP_MARGIN per remaining ply still
    # clears beta, so assume the real search fails high too.
    # ref: https://www.chessprogramming.org/Reverse_Futility_Pruning
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
        # reduce harder the deeper the search and the more static already clears beta by.
        margin = static_eval - beta
        reduced = (depth * NULL_MOVE_DEPTH_WEIGHT - margin) // NULL_MOVE_SCALE - 1
        if reduced < 0:
            reduced = 0

        score = -negamax(state, null_move(board), reduced, -beta, -beta + 1, ply + 1)

        if state.aborted:
            return 0
        
        if score >= beta:
            return score  # too good even after passing

    # best-first ordering: TT move, then MVV-LVA, then killers, then history.
    # ref: https://www.chessprogramming.org/Killer_Heuristic
    # ref: https://www.chessprogramming.org/History_Heuristic
    order = np.empty(count, dtype=np.int64)
    k0 = state.killers[depth, 0]
    k1 = state.killers[depth, 1]

    for j in range(count):
        m = moves[j]
        is_killer = 1 if m in (k0, k1) else 0
        order[j] = order_key(
            1 if m == tt_move else 0,
            move_ordering_score(board, m),
            is_killer,
            history_score(state, board, m),
        )

    ranked = np.argsort(-order, kind="mergesort")

    alpha_original = alpha  # incoming window, kept to tag the stored score below
    best = -MATE_SCORE
    best_move = moves[ranked[0]]  # always have a move to store
    tried_quiets = np.empty(count, dtype=np.int64)  # quiets that didn't cut - they take the malus
    tried = 0
    quiets_seen = 0

    for r in range(count):
        move = moves[ranked[r]]
        child = make_move(board, move)
        gives_check = is_check(child, child.side)
        # check extension: a checking move forces the reply, so search that line a ply deeper,
        # while the child stays inside the ceiling.
        # ref: https://www.chessprogramming.org/Check_Extensions
        extension = 1 if gives_check and ply + 1 < MAX_DEPTH else 0
        is_quiet = not move_is_capture(move) and move_promotion(move) == PROMOTION_NONE
        # LMR trust score, read below. move 0 always gets the full-window search and never
        # reaches the reduction, so it never needs one.
        lmr_score = lmr_key(state, board, move, is_quiet) if r else 0

        # shallow non-PV quiet-move pruning, once we have a real score to fall back on.
        # ref: https://www.chessprogramming.org/Futility_Pruning
        if is_quiet:
            if not extension and shallow and beta - alpha == 1 and best > -MATE_THRESHOLD:
                # late move pruning: enough quiets tried without a cut, skip the rest
                if quiets_seen >= LMP[depth]:
                    break
                # futility: this quiet can't lift a position already far below alpha
                if static_eval + FUTILITY_MARGIN * depth <= alpha:
                    quiets_seen += 1
                    continue

            quiets_seen += 1

        new_depth = depth - 1 + extension  # the check extension, if any, folds in here

        # first repetition scores as a draw and the line is not searched on - a move before the
        # threefold rule, so the search can still steer into or away from the draw. counts the
        # real game too: while winning we then never bring a position up for a second time.
        # ref: https://www.chessprogramming.org/Repetitions
        if child.halfmove_clock >= 4 and (
            repetitions(state.path, ply + 1, child.zobrist)
            + game_reps(state, child.zobrist) >= 1
        ):
            score = contempt_draw(ply)
        elif r == 0:
            # the move the ordering trusts most - full-window principal variation search.
            # ref: https://www.chessprogramming.org/Principal_Variation_Search
            score = -negamax(state, child, new_depth, -beta, -alpha, ply + 1)
        else:
            # later moves: probe with a reduced depth and a null window, and only pay for a
            # wider / deeper search if the probe beats alpha.
            # ref: https://www.chessprogramming.org/Late_Move_Reductions
            reduction = 0
            if extension == 0 and depth >= LMR_MIN_DEPTH:
                pull = r * LMR_MOVE_WEIGHT + depth * LMR_DEPTH_WEIGHT
                raw = pull // LMR_SCALE - lmr_score // LMR_HISTORY_SCALE
                reduction = raw

                if reduction < 0:
                    reduction = 0

                if reduction > new_depth - 1:
                    reduction = new_depth - 1
            # reduced, null window: is this move even worth a closer look?
            score = -negamax(state, child, new_depth - reduction, -alpha - 1, -alpha, ply + 1)
            # it cleared alpha despite the cut - redo at full depth, still null window
            if reduction and score > alpha:
                score = -negamax(state, child, new_depth, -alpha - 1, -alpha, ply + 1)
            # a full-depth score inside the window needs the real, full-window search
            if alpha < score < beta:
                score = -negamax(state, child, new_depth, -beta, -alpha, ply + 1)

        if state.aborted:
            return 0

        if score > best:
            best = score
            best_move = move

        if score > alpha:
            alpha = score

        # opponent already has a better option earlier; this branch cannot matter
        if alpha >= beta:
            if is_quiet:
                # a quiet move cut: remember it as a killer, reward it, penalise the quiets
                # that were tried first and failed
                if move != state.killers[depth, 0] and move != state.killers[depth, 1]:
                    state.killers[depth, 1] = state.killers[depth, 0]
                    state.killers[depth, 0] = move

                bonus = depth * depth
                update_history(state, board, move, bonus)

                for q in range(tried):
                    update_history(state, board, tried_quiets[q], -bonus)
            break

        if is_quiet:
            tried_quiets[tried] = move  # this move didn't cut
            tried += 1

    # record what we learned: an exact value, or which side of the window it fell on
    if best >= beta:
        flag = LOWER
    elif best > alpha_original:
        flag = EXACT
    else:
        flag = UPPER

    # store the mate distance relative to this node, not the root, so it reads back correctly
    state.tt_key[slot] = key
    state.tt_move[slot] = best_move
    state.tt_score[slot] = score_to_tt(best, ply)
    state.tt_depth[slot] = depth
    state.tt_flag[slot] = flag
    state.tt_eval[slot] = NO_EVAL if in_check else static_eval  # cache the NNUE eval for revisits

    return best


# Past the horizon: keep going through captures only (every reply when in check) until the
# position is quiet, then score it - so the search never trusts an eval taken mid-trade. Delta
# pruning skips captures too small to reach alpha. The jitted twin of reference.quiescence_search.
# ref: https://www.chessprogramming.org/Quiescence_Search
# ref: https://www.chessprogramming.org/Delta_Pruning
@njit  # not cache=True: recursive + objmode
def quiescence_search(state: SearchState, board: Board, alpha: int, beta: int, ply: int) -> int:
    state.nodes += 1

    # abort once the time cap is reached
    if state.nodes % CHECK_EVERY == 0:
        with objmode(now="float64"):
            now = time.monotonic()

        if now >= state.deadline:
            state.aborted = 1
            return 0

    key = board.zobrist
    state.path[ply] = key
    # threefold repetition is a draw - same guard as negamax, real game included.
    if repetitions(state.path, ply + 1, key) + game_reps(state, key) >= 3:
        return contempt_draw(ply)

    in_check = is_check(board, board.side)

    # fifty-move rule: the referee claims it, so a hundred halfmoves with no capture and no pawn
    # move is a draw however won the position looks - and a tablebase probe answers as if the
    # clock were zeroed, so this has to come first. Checkmate outranks the clock (a mate on the
    # hundredth halfmove is a win), which is why the rare branch pays for a move generation.
    # ref: https://www.chessprogramming.org/Fifty-move_Rule
    if board.halfmove_clock >= 100:
        _fifty_moves, fifty_count = legal_moves(board)

        if in_check and fifty_count == 0:
            return -MATE_SCORE + ply

        return contempt_draw(ply)

    # tablebase: exact WDL for a <= TB_MEN position, same as negamax - qsearch is where a capture
    # sequence collapses into a bare-piece ending, so it is the node that most needs the truth.
    if popcount(board.occupancy[2]) <= TB_MEN:
        tb = tb_probe_score(board, ply)
        if tb != NO_TB:
            return contempt_draw(ply) if tb == 0 else tb

    # recursion floor, matching negamax's - the only thing that bounds a checking sequence here.
    if ply >= MAX_DEPTH:
        _mv, count = legal_moves(board)

        if in_check and count == 0:
            return -MATE_SCORE + ply

        return evaluate(board)

    # transposition-table probe: shared with negamax, so a capture sequence reached by two move
    # orders is priced once. stored at depth 0 below, so any negamax entry is trusted here too.
    # ref: https://www.chessprogramming.org/Transposition_Table
    slot = key & TT_MASK
    tt_move = NO_MOVE
    tt_eval_cached = NO_EVAL
    if state.tt_depth[slot] >= 0 and state.tt_key[slot] == key:
        tt_move = state.tt_move[slot]
        tt_eval_cached = state.tt_eval[slot]
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
        # can't stand pat out of check: search every reply
        if count == 0:
            return -MATE_SCORE + ply  # checkmate; +ply so a nearer one scores higher
        
        moves = all_moves
        n = count
        standing_pat = -MATE_SCORE
    else:
        # the side to move can decline to capture, so the static evaluation is a floor. a TT
        # key match carries the exact eval (evaluate is pure in acc/side) - reuse it.
        if EVAL_CACHE and tt_eval_cached != NO_EVAL:
            standing_pat = tt_eval_cached
        else:
            standing_pat = evaluate(board)

        if standing_pat >= beta:
            return standing_pat
        
        if standing_pat > alpha:
            alpha = standing_pat
        # captures, plus quiet promotions: a pawn pushing to the back rank swings the eval as
        # much as any capture.
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
                v = piece_on(board, move_to_square(move))
                victim = PIECE_VALUE[v] if v >= 0 else 0

            # skip a capture that loses material once the recaptures are counted: it isn't worth
            # a qsearch node and it only adds noise to the leaf score. only the captures that can
            # lose (attacker worth more than victim) pay for the swap.
            # ref: https://www.chessprogramming.org/Static_Exchange_Evaluation
            if PIECE_VALUE[move_piece(move)] > victim and not see_ge(board, move, 0):
                continue

            # delta pruning: if winning this piece plus a margin still falls short of alpha, so
            # does every smaller capture after it (list is biggest-victim first)
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

    # same bookkeeping as negamax's store, at a flat depth 0 - qsearch has no depth of its own
    flag = LOWER if alpha >= beta else (EXACT if alpha > alpha_original else UPPER)
    state.tt_key[slot] = key
    state.tt_move[slot] = best_move
    state.tt_score[slot] = score_to_tt(alpha, ply)
    state.tt_depth[slot] = 0
    state.tt_flag[slot] = flag
    state.tt_eval[slot] = NO_EVAL if in_check else standing_pat  # cache the NNUE eval for revisits

    return alpha


# Root of the search: negamax's move loop keeping the best move, inside the caller's aspiration
# window. `prev_best` (last iteration's choice) is tried first, the rest fall back to MVV-LVA.
# The jitted twin of reference.search_root; returns (move, score).
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
        
        if child.halfmove_clock >= 4 and (
            repetitions(state.path, 1, child.zobrist) + game_reps(state, child.zobrist) >= 1
        ):
            # this move brings a position up for the second time (search path or real game) - a
            # draw (see negamax). the root is our move, so contempt_draw(0) docks it: don't
            # repeat unless all else is worse.
            score = contempt_draw(0)
        elif r == 0:
            # first move: full window; children search from ply 1 so a mate there is mate-in-1
            score = -negamax(state, child, depth - 1, -beta, -alpha, 1)
        else:
            # scout the rest with a null window; re-search only the ones that beat alpha
            score = -negamax(state, child, depth - 1, -alpha - 1, -alpha, 1)
            if alpha < score < beta:
                score = -negamax(state, child, depth - 1, -beta, -alpha, 1)

        if state.aborted:
            return best_move, best_score

        # dock the move we chose here last time we were asked to move in this exact position
        # (unless it is a real mate) so a level game isn't rubber-stamped into a repetition
        if move == state.avoid and score < MATE_THRESHOLD:
            score -= REPETITION_PENALTY

        if score > best_score:  # strict, so ties keep the earlier (better-ordered) move
            best_score = score
            best_move = move

        if best_score > alpha:
            alpha = best_score

        if alpha >= beta:
            break  # fail-high: this move beats the window; deepen widens and re-searches

    return best_move, best_score


# Iterative deepening from the root with aspiration windows: deepen a ply at a time until
# soft_cap elapses with no depth in flight or the deadline aborts one mid-search, returning the
# best move / score from the last depth that finished inside the window. reference.deepen.
# ref: https://www.chessprogramming.org/Iterative_Deepening
# ref: https://www.chessprogramming.org/Aspiration_Windows
def deepen(
    board: Board, start: float, deadline: float, soft_cap: float, fallback: int
) -> tuple[int, int]:
    STATE.deadline = deadline
    STATE.aborted = 0

    best = fallback
    score = 0
    move, value = fallback, 0

    for depth in range(1, MAX_DEPTH):
        if depth > 1 and time.monotonic() - start >= soft_cap:
            break

        # aspiration: a thin band around the last score past the opening plies, full width
        # before that. widen geometrically on whichever side failed and re-search.
        if depth <= ASPIRATION_MIN_DEPTH:
            alpha, beta = -MATE_SCORE, MATE_SCORE
        else:
            alpha, beta = score - ASPIRATION_WINDOW, score + ASPIRATION_WINDOW

        delta = ASPIRATION_WINDOW

        while True:
            move, value = search_root(STATE, board, depth, alpha, beta, best)

            if STATE.aborted:
                break

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

        if STATE.aborted:
            break

        best, score = move, value  # depth completed inside the window - adopt its move

    return best, score


# Emergency move when there is no time to search: the best-looking capture / promotion by the
# search's own ordering, else the first legal move. reference.panic_move.
def panic_move(board: Board) -> int:
    moves, count = legal_moves(board)
    best, best_score = int(moves[0]), -1
    for j in range(count):

        s = move_ordering_score(board, int(moves[j]))
        if s > best_score:
            best, best_score = int(moves[j]), s

    return best


# Copy the tail of GAME_HASHES into STATE.game_hashes so the jitted repetition checks can read
# it. The window is already small (bounded by the fifty-move clock); the cap is a safety net.
def load_game_hashes() -> None:
    cap = STATE.game_hashes.shape[0]
    n = min(len(GAME_HASHES), cap)
    for i in range(n):
        STATE.game_hashes[i] = np.uint64(GAME_HASHES[len(GAME_HASHES) - n + i] & MASK64)
    STATE.game_n = n


def get_move(fen: str, time_left_ms: int) -> str:
    board = parse_fen(fen)
    moves, count = legal_moves(board)

    if count == 0:
        return "0000"

    # if we've been asked to move in this exact position before, tell search_root the move we
    # chose last time so it can shy away from a repeat. SEEN / PLAYED persist for the game.
    # ref: https://www.chessprogramming.org/Repetitions
    key = int(zobrist_hash(board))
    seen = SEEN.get(key, 0)
    SEEN[key] = seen + 1
    STATE.avoid = PLAYED.get(key, NO_MOVE) if seen else NO_MOVE

    # fold the position we were handed last move into the game trail, trim it to the reversible
    # window (halfmove_clock plies precede this one with no pawn move or capture, so nothing
    # older can recur), and load it for the search. game_reps() then counts real-game
    # occurrences alongside the search path.
    global _LAST_KEY
    if _LAST_KEY:
        GAME_HASHES.append(_LAST_KEY)
    window = int(board.halfmove_clock)
    if len(GAME_HASHES) > window:
        del GAME_HASHES[: len(GAME_HASHES) - window]
    load_game_hashes()
    _LAST_KEY = 0  # set again only on a real-move return; a non-search reply carries nothing

    # Syzygy: with few enough men the tablebase gives the exact best move, so the endgame is
    # played perfectly instead of trusting a net that never learned to convert (KRvK evals at
    # +57 cp). Optimal and deterministic - no anti-repetition bookkeeping needed - but keep the
    # game trail continuous in case a later position falls back to the search. A probe failure
    # must never cost the game: fall through.
    try:
        tb_uci = best_tb_move(fen)
    except Exception:  # any tablebase trouble just means "search instead"
        tb_uci = None
    if tb_uci is not None:
        _LAST_KEY = key
        return tb_uci

    # critically low on time: don't search at all - even one depth-1 iteration could overrun
    # what this move has left. grab the best-looking move and return.
    if time_left_ms < PANIC_TIME_MS:
        best = panic_move(board)
        PLAYED[key] = best
        _LAST_KEY = key
        return move_uci(best)

    try:
        STATE.nodes = 0
        board.zobrist = np.uint64(key)
        start = time.monotonic()
        # hard deadline to abort at - the 50 ms is slack for the node batch past the last clock
        # check plus move-gen and the reply; soft cap past which no new depth starts
        endgame = popcount(board.occupancy[2]) <= ENDGAME_TIME_MEN
        hard_limit = HARD_LIMIT_ENDGAME if endgame else HARD_LIMIT
        soft_limit = SOFT_LIMIT_ENDGAME if endgame else SOFT_LIMIT
        deadline = start + time_left_ms / hard_limit / 1000 - 0.05
        soft_cap = time_left_ms / soft_limit / 1000
        best, _score = deepen(board, start, deadline, soft_cap, int(moves[0]))
        PLAYED[key] = int(best)
        _LAST_KEY = key

        # nothing bounds the table's size mid-game the way a real fixed array would - clear it
        # if it has filled enough slots to be a memory concern.
        if tt_used() > TT_MAX_ENTRIES:
            STATE.tt_depth[:] = -1
            STATE.tt_eval[:] = NO_EVAL

        return move_uci(int(best))
    
    except Exception as exc:  # numba failed to compile, or a bug in the port - hand off
        print(f"agent: numba search unavailable ({exc!r}); using the reference engine")
        GAME_HASHES.clear()  # the reference engine's replies are not folded in; drop the trail
        return reference.get_move(fen, time_left_ms)


def tt_used() -> int:
    return int(np.count_nonzero(STATE.tt_depth >= 0))


# tools/nodebench.py / tools/verify_search.py hook: fixed-depth search from a clean table,
# returning (uci, score, nodes). The same shape as reference.bench_search.
def bench_search(fen: str, depth: int) -> tuple[str, int, int]:
    STATE.nodes = 0
    STATE.aborted = 0
    STATE.deadline = time.monotonic() + 3600.0
    STATE.avoid = NO_MOVE
    STATE.game_n = 0
    STATE.tt_depth[:] = -1
    STATE.tt_eval[:] = NO_EVAL
    STATE.history[:] = 0
    STATE.killers[:] = NO_MOVE

    board = parse_fen(fen)
    board.zobrist = np.uint64(zobrist_hash(board))
    move, score = search_root(STATE, board, depth, -MATE_SCORE, MATE_SCORE, NO_MOVE)
    
    return move_uci(int(move)), int(score), int(STATE.nodes)


# Compile the search inside the platform's 90 s init budget - a shallow run still reaches every
# hot path (quiescence, null-move, LMR, futility) - so the first real move of a game does not
# pay a multi-second numba JIT stall on the clock. No timed opening search: rated games start
# from unpublished mid-opening positions that share almost none of that tree. reference.py has
# no equivalent - it is plain Python, so there is nothing to compile ahead of the clock.
def warm_up() -> None:
    board = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    board.zobrist = np.uint64(zobrist_hash(board))
    evaluate(board)  # compile the NNUE forward before the search leans on it
    search_root(STATE, board, 4, -MATE_SCORE, MATE_SCORE, NO_MOVE)
    # a position with captures on the board so see / attackers_to compile here, not on the clock
    capt = parse_fen("r1bqkbnr/ppp2ppp/2n5/1B1pp3/3PP3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 4")
    moves, count = legal_moves(capt)
    for j in range(count):
        see(capt, int(moves[j]))
        see_ge(capt, int(moves[j]), 0)
        move_ordering_score(capt, int(moves[j]))
    # compile the in-search tablebase probe (objmode -> chess.syzygy) off the clock
    tb_probe_score(parse_fen("8/8/8/4k3/8/3RK3/8/8 w - - 0 1"), 0)


warm_up()
