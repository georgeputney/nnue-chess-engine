"""
Submission entry point. The platform imports this module once per game and calls
get_move(fen, time_left_ms) per move; module state survives between our moves but not into
the next game, and import gets a 60 s budget before the clock starts.

Negamax Alpha-Beta over a material evaluation. docs/plan.md is the phased roadmap.
"""

import chess

# centipawn values; no king, it cancels in any material difference.
# ref: https://www.chessprogramming.org/Point_Value
PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# larger than any real score: +MATE = we mate, -MATE = we are mated / no move yet.
MATE = 1_000_000

# get_move search depth. low because it is unordered and flags deeper; phase 4 is meant to
# make it a time budget.
DEPTH = 3

# nodes seen this search, for tools/nodebench.py; bench_search resets it.
NODES = 0


# material balance from `side`'s view, in centipawns, no lookahead.
# ref: https://www.chessprogramming.org/Evaluation
def evaluate(board: chess.Board, side: chess.Color) -> int:
    return sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUE.items()
    )


# value of `board` searched `depth` plies, from the side to move. the caller negates the
# result and swaps/negates the window (`-beta, -alpha`), since good for the mover is bad
# for the previous mover. no legal moves = mate (in check) or stalemate. mates are not
# distance-scored yet (phase 17).
# ref: https://www.chessprogramming.org/Negamax
# ref: https://www.chessprogramming.org/Alpha-Beta
def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    global NODES
    NODES += 1

    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0
    if depth == 0:
        return evaluate(board, board.turn)

    best = -MATE
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  # beta cutoff

    return best


# root ply: negamax's loop, but keeps the move. seeded so a move is always returned.
def search_root(board: chess.Board, depth: int) -> tuple[chess.Move, int]:
    best_move = next(iter(board.legal_moves))
    best_score = -MATE

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -MATE, MATE)
        board.pop()
        if score > best_score:  # strict, so ties keep the earlier move
            best_score = score
            best_move = move

    return best_move, best_score


# the platform entry point. returns UCI ("e2e4", "e7e8q" to promote); time_left_ms is
# unused until phase 4.
def get_move(fen: str, time_left_ms: int) -> str:
    move, _ = search_root(chess.Board(fen), DEPTH)
    
    return move.uci()


# tools/nodebench.py hook: same search to a set depth, returning (move, score, nodes).
def bench_search(fen: str, depth: int) -> tuple[str, int, int]:
    global NODES
    NODES = 0
    move, score = search_root(chess.Board(fen), depth)

    return move.uci(), score, NODES
