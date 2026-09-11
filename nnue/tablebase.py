"""Syzygy tablebase probe - at the root (best move) and inside the search (exact WDL score).

For a position with few enough men, python-chess's chess.syzygy gives the exact result and the
move that keeps it - perfect endgame play instead of a Lichess-trained net that never learned to
convert. get_move calls best_tb_move before the search; negamax / quiescence_search call
tb_score at every <= TB_MEN node so the whole search steers toward won conversions and away from
tablebase draws with ground truth instead of the blind net's saturated "winning-ish" number.

nnue/syzygy/ ships the full 3-4-man Syzygy set, WDL + DTZ, 4.3 MB of public-domain generated
data (identical whoever produces it; the rules allow endgame tablebases). No network, no search;
the tables are mmap'd read-only at import.

A curated 5-man WDL slice (KRPvKR, KRPvKP, KRRvKR, KPPvKP, KRPPvK - 28 MB, now parked in
nnue/syzygy5/) was built and measured on 2026-09-09: 80 games at 5 s + 0.1 s against the same
engine at TB_MEN = 4 scored -48 Elo [-114, +15], corroborated by three other runs. It taxes
every <= 5-man node with an objmode probe in exactly the phase it was meant to help, an exact 0
for a known draw outweighs our +-30 contempt so the search liquidates into certain draws, and
every tablebase win scores TB_WIN - ply, which gives no gradient to convert by. Do not ship it
again without a measurement that says otherwise. Note DTZ is what makes the root ranking exact,
and DTZ for 5 men does not fit the 50 MB cap.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.syzygy

TB_MEN = 4  # the shipped set covers every 3-4-man ending, WDL and DTZ both
TB_NONE = -999  # tb_score sentinel: position not covered / probe failed

_DIR = Path(__file__).resolve().parent / "syzygy"
try:
    _TB: chess.syzygy.Tablebase | None = chess.syzygy.open_tablebase(str(_DIR))
except OSError:
    _TB = None

_PROBE_ERRORS = (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError)

# Which material configs actually have a table, so the numba search can skip the objmode probe
# for a <= TB_MEN position we cannot answer (an uncovered ending was crawling at ~15 knps -
# one failed probe per node). Key: counts of (P,N,B,R,Q) per side, 3 bits each, low 15 bits
# white; canonicalised to the smaller of (key, colour-swapped key) since Syzygy is symmetric.
_PIECE_ORDER = "PNBRQ"


def _material_key(white_pieces: str, black_pieces: str) -> int:
    def side(spec: str) -> int:
        return sum(spec.count(p) << (3 * i) for i, p in enumerate(_PIECE_ORDER))
    w, b = side(white_pieces), side(black_pieces)
    key = w | (b << 15)
    swapped = b | (w << 15)
    return min(key, swapped)


def _covered_material_keys() -> list[int]:
    keys = {_material_key("", "")}  # bare kings: probe_wdl special-cases it to a draw
    for path in _DIR.glob("*.rtbw"):
        white, _, black = path.stem.partition("v")
        keys.add(_material_key(white.lstrip("K"), black.lstrip("K")))
    return sorted(keys)


COVERED_MATERIAL = _covered_material_keys()  # sorted; nnue.agent freezes it into the jitted probe
CORNERS = (chess.A1, chess.A8, chess.H1, chess.H8)  # the mate drivers in rank_move


# A bare chess.Board carrying just the material of the numba engine's piece bitboards `pieces`
# (uint64 [2, 6], colour then pawn..king) with `side` to move (0 = white). No castling / ep -
# irrelevant to a Syzygy probe. Built the low-level way python-chess uses internally so there is
# no FEN round-trip on the hot path.
def _board_from_pieces(pieces: object, side: int) -> chess.Board:
    board = chess.Board.empty()
    board.pawns = int(pieces[0][0]) | int(pieces[1][0])
    board.knights = int(pieces[0][1]) | int(pieces[1][1])
    board.bishops = int(pieces[0][2]) | int(pieces[1][2])
    board.rooks = int(pieces[0][3]) | int(pieces[1][3])
    board.queens = int(pieces[0][4]) | int(pieces[1][4])
    board.kings = int(pieces[0][5]) | int(pieces[1][5])
    white = black = 0
    for piece_type in range(6):
        white |= int(pieces[0][piece_type])
        black |= int(pieces[1][piece_type])
    board.occupied_co[chess.WHITE] = white
    board.occupied_co[chess.BLACK] = black
    board.occupied = white | black
    board.turn = side == 0
    board.castling_rights = 0
    board.ep_square = None
    return board


# Exact WDL for the position, side-to-move relative: +1 win, 0 draw, -1 loss, or TB_NONE when
# the tables cannot answer. Syzygy reports +-1 for a cursed win / blessed loss - mate is on the
# board but cannot be reached inside the fifty-move rule, so the referee claims the draw first
# and it is a draw in play. Only +-2 is a result. The numba search calls this through objmode at
# every <= TB_MEN node; `pieces` / `side` are the jitclass fields.
def tb_score(pieces: object, side: int) -> int:
    if _TB is None:
        return TB_NONE
    try:
        wdl = _TB.probe_wdl(_board_from_pieces(pieces, side))
    except _PROBE_ERRORS:
        return TB_NONE
    if wdl > 1:
        return 1
    if wdl < -1:
        return -1
    return 0


# Ordering key for the move that led to `child`, larger is better, from the moving side's view.
# The five-valued WDL leads: a real win outranks a cursed one, so a move that converts a cursed
# win into a real one (a pawn push resets the clock the curse is about) is preferred over one
# that keeps it. Then a zeroing move, so a win cannot be shuffled into a fifty-move claim; then
# distance, the fastest win and the slowest loss; then, when winning, the classic mate drivers -
# the defending king toward a corner, our king closer to it - which is what makes progress in an
# ending whose DTZ we did not ship.
def rank_move(tb: chess.syzygy.Tablebase, child: chess.Board, zeroing: bool) -> tuple[int, ...]:
    if child.is_checkmate():
        return (3, 1, 0, 0, 0)

    wdl = -tb.probe_wdl(child)  # child is opponent-to-move; negate to the mover's view
    try:
        dtz = -tb.probe_dtz(child)
    except _PROBE_ERRORS:
        # DTZ only ships for the 3-4-man set. probe_dtz answers a drawn 5-man child from the WDL
        # table alone but raises on a decisive one, so treating a raise as "unrankable" would
        # keep only the drawn moves and hand back a draw from a won position. Neutral here; the
        # drivers below break the tie instead.
        dtz = 0

    drive = (0, 0)
    defender = child.king(child.turn)
    attacker = child.king(not child.turn)
    if wdl > 0 and defender is not None and attacker is not None:
        corner = min(chess.square_distance(defender, square) for square in CORNERS)
        drive = (-corner, -chess.square_distance(defender, attacker))

    return (wdl, 1 if zeroing else 0, -abs(dtz) if wdl > 0 else abs(dtz), *drive)


# UCI of the move Syzygy says holds the best achievable result from `fen`, or None when the
# position has too many men or the tables cannot rank every move - a promotion out of the
# shipped 5-man material has no table, and picking the best of the moves we can rank would mean
# passing over the move that wins. The whole position goes to the search in that case; the
# in-search probe still holds the line to a won or drawn result.
def best_tb_move(fen: str) -> str | None:
    if _TB is None:
        return None
    board = chess.Board(fen)
    if chess.popcount(board.occupied) > TB_MEN:
        return None

    try:
        _TB.probe_wdl(board)  # confirm this position is covered before scanning children
    except _PROBE_ERRORS:
        return None

    best_move: chess.Move | None = None
    best_key: tuple[int, ...] | None = None
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            key: tuple[int, ...] | None = rank_move(_TB, board, zeroing)
        except _PROBE_ERRORS:
            key = None
        finally:
            board.pop()

        if key is None:
            return None
        if best_key is None or key > best_key:
            best_key, best_move = key, move

    return best_move.uci() if best_move is not None else None
