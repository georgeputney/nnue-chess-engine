"""Submission entrypoint - the platform imports this module and calls get_move()."""

# hand-rolled alpha-beta engine, built in numbered phases. the roadmap, the reference for each
# technique, and the measurement discipline live in docs/plan.md (not shipped in the zip)

import chess


PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


# material only, in centipawns, from `side`'s point of view. king omitted: both sides always
# have exactly one, so it cancels, and scoring it risks swamping the real terms near mate
def evaluate(board: chess.Board, side: chess.Color) -> int:
    return sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUE.items()
    )


# the one entry point. returns a legal UCI move ("e2e4", or "e7e8q" to promote) for the side
# to move in `fen`; `time_left_ms` is our clock before this move, unused until time management
# lands. phase 1: no search yet, just the move that leaves the best static material.
# the process is reused across our moves within a game (module state persists) but not across
# games. printing to stdout is safe; it is kept out of the protocol stream
def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    mover = board.turn

    best_score = -1_000_000
    best: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        score = evaluate(board, mover)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)

    return best[0].uci()
