"""Checks zobrist.zobrist_hash's basic properties: identical positions hash identically,
different positions (almost always) hash differently, and reaching the same position by two
different move orders (a transposition) produces the same hash - the property the whole point
of Zobrist hashing depends on.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import numpy as np  # noqa: E402

from engine.board import Board, from_chess_board  # noqa: E402
from engine.movegen import legal_moves, make_move  # noqa: E402
from engine.zobrist import zobrist_hash  # noqa: E402


def main() -> None:
    random.seed(9)
    failures = 0

    # identical positions
    p1 = from_chess_board(chess.Board())
    p2 = from_chess_board(chess.Board())
    if zobrist_hash(p1) != zobrist_hash(p2):
        failures += 1
        print("FAIL: two builds of the same position hashed differently")

    # collisions: walk random games via python-chess (so positions are realistic/varied),
    # hashing every position reached and checking no two different FENs share a hash
    seen: dict[int, str] = {}
    collisions = 0
    positions_checked = 0
    for _game in range(40):
        board = chess.Board()
        for _ply in range(60):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
            pos = from_chess_board(board)
            h = zobrist_hash(pos)
            fen_key = " ".join(board.fen().split()[:4])  # ignore halfmove/fullmove counters
            positions_checked += 1
            if h in seen and seen[h] != fen_key:
                collisions += 1
                print(f"FAIL: collision between {seen[h]!r} and {fen_key!r}")
            seen[h] = fen_key
    if collisions:
        failures += 1
    print(f"checked {positions_checked} random positions, {collisions} collisions")

    # transposition: two knight moves out and back reaches the exact starting position again
    def find_move(pos: Board, uci: str) -> int:
        moves, count = legal_moves(pos)
        from engine.move import move_uci

        for i in range(count):
            m = int(moves[i])
            if move_uci(m) == uci:
                return m
        raise ValueError(uci)

    start = from_chess_board(chess.Board())
    pos = start
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        pos = make_move(pos, find_move(pos, uci))  # type: ignore[assignment, type-var, call-arg]
    if zobrist_hash(pos) != zobrist_hash(start):
        failures += 1
        print("FAIL: knights out and back did not transpose to the same hash as the start")
    else:
        print("transposition check: ok")

    # incremental hash: make_move must keep Board.zobrist equal to a from-scratch
    # zobrist_hash after every move - across castling, promotions, en passant and rights loss.
    seeds = [
        chess.STARTING_FEN,
        # castling both sides, then promotions imminent, then an en-passant target
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    ]
    inc_checked = 0
    inc_bad = 0
    for seed in seeds:
        for _game in range(30):
            pos = from_chess_board(chess.Board(seed))
            pos.zobrist = np.uint64(zobrist_hash(pos))  # type: ignore[assignment]
            for _ply in range(60):
                moves, count = legal_moves(pos)
                if count == 0:
                    break
                mv = int(moves[random.randrange(count)])
                pos = make_move(pos, mv)  # type: ignore[assignment, type-var, call-arg]
                inc_checked += 1
                inc, scratch = int(pos.zobrist), zobrist_hash(pos)
                if inc != scratch:
                    inc_bad += 1
                    if inc_bad <= 5:
                        print(f"FAIL: incremental {inc} != scratch {scratch}")
    if inc_bad:
        failures += 1
    print(f"incremental hash: {inc_checked - inc_bad}/{inc_checked} match zobrist_hash")

    print("\nPASS" if not failures else "\nFAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
