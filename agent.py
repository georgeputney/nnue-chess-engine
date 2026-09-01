"""
submission entry point. the platform imports this module once per game and calls
get_move(fen, time_left_ms) for each of our moves. module state survives between those
calls but not into the next game. import runs inside a 60 s budget before the clock
starts, so heavy setup belongs at module level, not inside get_move.

a plain negamax alpha-beta search over a hand-written evaluation, built in numbered
phases. docs/plan.md is the roadmap - what each phase adds, its wiki reference, and how
the change is measured. it is not shipped in the zip.

    evaluate      static score of one position, no lookahead
    negamax       the search: looks DEPTH plies ahead, scoring leaves with evaluate
    search_root   one negamax level that also keeps the best move
    get_move      the entry point: runs search_root, returns the move as UCI
    bench_search  the same search with a node count, for tools/nodebench.py
"""

import chess

# centipawn piece values (100 = one pawn). no king entry: both sides always have exactly
# one, so it cancels in any material difference. for now this is the whole evaluation;
# positional terms are planned for phase 18.
# ref: https://www.chessprogramming.org/Point_Value
PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# sentinel bigger than any real material score: +MATE = we force mate, -MATE = we are
# mated, and -MATE seeds "best score so far" so any real move beats it.
MATE = 1_000_000

# plies get_move searches. fixed for now; phase 4 is meant to replace it with a time
# budget. low on purpose: no move ordering or pruning yet, so each ply multiplies the work
# by ~35, and depth 3 already flags under the 5-10 s local clocks. phases 3 and 4 should
# make more depth affordable.
DEPTH = 3

# nodes visited by the current search. module-global so negamax can count without an
# accumulator arg. bench_search zeroes it per search; get_move ignores it.
NODES = 0


# static evaluation of `board` from `side`'s point of view, in centipawns, no search: pure
# material for now (+300 = a knight up). `side` is a parameter, not board.turn, because
# negamax scores each leaf from whoever is to move there. phase 18 is planned to add
# piece-square tables, mobility, pawn structure and a tapered midgame/endgame blend.
# ref: https://www.chessprogramming.org/Evaluation
def evaluate(board: chess.Board, side: chess.Color) -> int:
    return sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUE.items()
    )


# negamax: value of `board` with both sides playing best for `depth` more plies, scored
# from the side to move. minimax's two cases collapse to one - every node maximises, the
# caller negates (`-negamax(...)`) - since what is good for the mover is bad for the
# previous mover. three cases, and the order matters:
#   1. no legal moves: checkmate (-MATE) if in check, else stalemate (0). before the depth
#      check, or a mate on the last ply is scored as plain material and we walk into it.
#   2. depth 0: return the static evaluation.
#   3. otherwise: try every move, recurse a ply shallower, keep the best.
# for now mate in 1 and mate in 5 both score +/-MATE, so mates are found but not raced;
# phase 17 is where a distance term is planned.
# ref: https://www.chessprogramming.org/Negamax
def negamax(board: chess.Board, depth: int) -> int:
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
        best = max(best, -negamax(board, depth - 1))  # negate: child scores from their side
        board.pop()

    return best


# negamax's recursive case at the root, keeping the move that scored best, not just the
# score. best_move is seeded with the first legal move so a real move is always returned
# even when every line loses to -MATE.
def search_root(board: chess.Board, depth: int) -> tuple[chess.Move, int]:
    best_move = next(iter(board.legal_moves))
    best_score = -MATE

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1)
        board.pop()
        if score > best_score:  # strict: ties keep the earlier move, so runs are deterministic
            best_score = score
            best_move = move

    return best_move, best_score


# the platform's entry point, once per move. `fen` has our colour to move; `time_left_ms`
# is our clock, unused for now (phase 4 is meant to turn it into a time budget). returns
# UCI ("e2e4", or "e7e8q" to promote). stdout is safe to print to - the runner keeps it
# out of the move protocol.
def get_move(fen: str, time_left_ms: int) -> str:
    move, _ = search_root(chess.Board(fen), DEPTH)
    return move.uci()


# test hook for tools/nodebench.py, not used in a game: the same search to a given depth,
# node counter zeroed first, returning (move, score, nodes) so nodebench can confirm a
# refactor left the move and score unchanged while cutting nodes.
def bench_search(fen: str, depth: int) -> tuple[str, int, int]:
    global NODES
    NODES = 0
    move, score = search_root(chess.Board(fen), depth)
    return move.uci(), score, NODES
