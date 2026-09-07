"""Root-node Syzygy tablebase probe.

For a position with few enough men, python-chess's chess.syzygy gives the exact result and the
move that keeps it - perfect endgame play instead of a Lichess-trained net that never learned to
convert. get_move calls best_tb_move before the search and returns its answer when there is one.

The standard 3-4-man Syzygy set (WDL + DTZ) ships in nnue/syzygy/ (~4.3 MB, public-domain
generated data - identical whoever produces it; the rules allow endgame tablebases). No network,
no search; the tables are mmap'd read-only at import.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.syzygy

TB_MEN = 4  # tables present cover up to this many pieces on the board

_DIR = Path(__file__).resolve().parent / "syzygy"
try:
    _TB: chess.syzygy.Tablebase | None = chess.syzygy.open_tablebase(str(_DIR))
except OSError:
    _TB = None


# UCI of the move Syzygy says holds the best achievable result from `fen`, or None when the
# position has too many men or the tables cannot answer it. Ranking, from our side's view:
# better WDL first (win > cursed win > draw > blessed loss > loss); then a move that resets the
# fifty-move clock (a capture or a pawn move), so a win cannot be shuffled away; then distance -
# the fastest win, the slowest loss.
def best_tb_move(fen: str) -> str | None:
    if _TB is None:
        return None
    board = chess.Board(fen)
    if chess.popcount(board.occupied) > TB_MEN:
        return None

    try:
        _TB.probe_wdl(board)  # confirm this position is covered before scanning children
    except (KeyError, chess.syzygy.MissingTableError, ValueError):
        return None

    best_move: chess.Move | None = None
    best_key: tuple[int, int, int] | None = None
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            wdl = -_TB.probe_wdl(board)      # child is opponent-to-move; negate to our view
            dtz = -_TB.probe_dtz(board)
        except (KeyError, chess.syzygy.MissingTableError, ValueError):
            board.pop()
            return None
        board.pop()
        key = (wdl, 1 if zeroing else 0, -dtz)
        if best_key is None or key > best_key:
            best_key, best_move = key, move

    return best_move.uci() if best_move is not None else None
