#!/usr/bin/env python3
"""
A UCI front end for engine/agent.py, so the engine that played the Chessathon can play on
lichess through lichess-bot (https://github.com/lichess-bot-devs/lichess-bot), or in any UCI
GUI. See lichess/README.md for the bot setup.

Nothing under engine/ changes: this speaks UCI on one side and calls the same
get_move(fen, time_left_ms) the competition platform called on the other. The only real work
here is the clock - the platform ran a single time control and get_move budgets a move as a
fraction of whatever clock it is handed, so lichess's many time controls are converted into an
equivalent one (effective_clock_ms) rather than the budget being rewritten.

Drive it by hand to check it works:

    .venv/bin/python lichess/uci.py
    uci
    position startpos moves e2e4
    go wtime 60000 btime 60000 winc 0 binc 0
"""

import os
import sys
import time
from itertools import pairwise
from pathlib import Path

import chess

# fd 1 becomes stderr before the engine loads, exactly as harness/runner.py does it for the
# platform: a diagnostic from the agent (the numba fallback notice) or from a library can then
# never land in the middle of the UCI stream. The protocol gets the real stdout, privately.
protocol = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported down here on purpose, after the swap above: importing it runs agent.warm_up(), which
# compiles the search, and anything that might say so belongs on stderr rather than in the
# middle of a handshake. The path insert is what lets it resolve at all - lichess-bot starts
# the engine from its own clone, so this repo is not otherwise on sys.path.
from engine import agent  # noqa: E402

ENGINE_NAME = "NNUE Chess Engine"
ENGINE_AUTHOR = "George Putney"

# get_move spends at most time_left_ms / agent.HARD_LIMIT on a move and starts no new deepening
# iteration past time_left_ms / agent.SOFT_LIMIT. Those fractions were fitted to one time
# control (120 s + 0.5 s a move), so an increment is folded into the clock handed over rather
# than left for the fractions to rediscover: this many moves' worth of it, the length of a game
# from here in the middle game.
INCREMENT_MOVES = 25

# Held back from every budget, on top of the move_overhead lichess-bot has already subtracted.
DEFAULT_MOVE_OVERHEAD_MS = 300

# What to think for when a GUI asks for a move without saying anything about a clock.
DEFAULT_THINK_MS = 3000

# What a tablebase win is worth on an info line, in centipawns. A probe scores a win as
# TB_WIN_SCORE less the ply it was found at, so anything inside a search's depth of that
# ceiling is one - the same slack MATE_THRESHOLD leaves below MATE_SCORE.
TB_WIN_CP = 10_000

GO_INTS = frozenset({"wtime", "btime", "winc", "binc", "movetime", "depth"})

board = chess.Board()
move_overhead_ms = DEFAULT_MOVE_OVERHEAD_MS


# One line out on the protocol stream, flushed: a GUI reads us line by line and a buffered
# bestmove is a flag fall.
def send(line: str) -> None:
    protocol.write(line + "\n")
    protocol.flush()


# Everything get_move carries from one move to the next: the anti-repetition bookkeeping, the
# game's hash trail, and the search tables. The platform gets this reset for free because it
# starts a process per game, and so does lichess-bot; it matters for a GUI that keeps one
# process across several games.
def new_game() -> None:
    agent.SEEN.clear()
    agent.PLAYED.clear()
    agent.GAME_HASHES.clear()
    agent._LAST_KEY = 0
    agent.STATE.tt_depth[:] = -1
    agent.STATE.tt_eval[:] = agent.NO_EVAL
    agent.STATE.history[:] = 0
    agent.STATE.killers[:] = agent.NO_MOVE
    agent.STATE.game_n = 0


# The clock to hand get_move so its own fractions come out at the budget this move deserves.
# The increment is worth a few dozen moves of itself, but the hard deadline that follows from
# the total (effective / HARD_LIMIT) must still fit inside the time really on the clock - at
# 10 s left in a 3+2 game the increment is not yet in hand and cannot be spent.
def effective_clock_ms(clock_ms: int, increment_ms: int) -> int:
    usable = max(0, clock_ms - move_overhead_ms)
    return min(usable + increment_ms * INCREMENT_MOVES, usable * agent.HARD_LIMIT)


# What `go` asked for, as a clock get_move understands.
def go_clock_ms(params: dict[str, int]) -> int:
    if "movetime" in params:
        # A fixed think time. Handing over movetime * HARD_LIMIT puts the hard deadline exactly
        # on it: the engine usually returns well inside that, since it starts no new iteration
        # past a tenth of the clock, but it cannot overrun. lichess-bot asks for the first move
        # of every game this way (a flat 10 s, to stay under lichess's 30 s first-move limit).
        return max(0, params["movetime"] - move_overhead_ms) * agent.HARD_LIMIT

    side = "w" if board.turn == chess.WHITE else "b"
    if f"{side}time" not in params:
        return DEFAULT_THINK_MS * agent.HARD_LIMIT  # `go infinite`, or a bare `go`

    return effective_clock_ms(params[f"{side}time"], params.get(f"{side}inc", 0))


# UCI's two score forms. The search scores a mate as MATE_SCORE less the ply it lands on,
# and a tablebase result just under that - forced, but carrying no distance, so it has no mate
# score to report and goes out as a large flat centipawn one instead.
def score_field(score: int) -> str:
    if abs(score) >= agent.MATE_THRESHOLD:
        moves = (agent.MATE_SCORE - abs(score) + 1) // 2
        return f"mate {moves if score > 0 else -moves}"
    if abs(score) >= agent.TB_WIN_SCORE - 2 * agent.MAX_DEPTH:
        return f"cp {TB_WIN_CP if score > 0 else -TB_WIN_CP}"
    return f"cp {score}"


# Search the position set by the last `position` command and answer with a move.
def handle_go(params: dict[str, int]) -> None:
    start = time.monotonic()

    try:
        if "depth" in params:
            # Fixed depth, for testing the bridge and for fixed-depth matches in a GUI. This is
            # bench/'s entry point, not the game one: it searches from an empty table and skips
            # the tablebase probe and the repetition bookkeeping, so never put `depth` in
            # lichess-bot's go_commands - it would throw away what the game had learned so far.
            uci, score, nodes = agent.bench_search(board.fen(), params["depth"])
            info = f"info depth {params['depth']} score {score_field(int(score))}"
        else:
            # get_move zeroes the node count only on the path that searches - a move answered
            # straight from the tablebase leaves the last search's total standing - so zero it
            # here, and the info line is this move's own work either way.
            agent.STATE.nodes = 0
            uci = agent.get_move(board.fen(), go_clock_ms(params))
            nodes = int(agent.STATE.nodes)
            # No depth and no score on this line: get_move returns a move and nothing else, and
            # reaching past it for the root score would mean editing engine/, which stays what
            # the competition ran. lichess-bot copes - a move with no score simply sits out its
            # draw_or_resign logic, which config.yml turns off for that reason.
            info = "info"
    except Exception as exc:
        # A `go` that goes unanswered hangs the game until lichess aborts it, so whatever gets
        # past get_move's own fallback to the reference engine still has to leave with a move.
        print(f"uci: search failed ({exc!r}); playing the first legal move")
        legal = next(iter(board.legal_moves), None)
        send(f"bestmove {legal.uci() if legal else '0000'}")
        return

    elapsed = max(1, int((time.monotonic() - start) * 1000))
    send(f"{info} time {elapsed} nodes {nodes} nps {nodes * 1000 // elapsed}")
    send(f"bestmove {uci}")


# "position startpos|fen <fen> [moves <uci> ...]". The engine is handed a FEN and nothing else,
# so the move list is replayed here to get one - and the halfmove clock in it is what bounds
# the window get_move keeps for repetition detection.
def parse_position(tokens: list[str]) -> chess.Board:
    end = tokens.index("moves") if "moves" in tokens else len(tokens)
    startpos = not tokens or tokens[0] == "startpos"
    position = chess.Board() if startpos else chess.Board(" ".join(tokens[1:end]))
    for uci in tokens[end + 1:]:
        position.push_uci(uci)
    return position


# The integer arguments of `go`, in any order. Anything else it carries - searchmoves, ponder,
# movestogo, nodes - is dropped: none of it is supported, and config.yml keeps lichess-bot from
# asking for any of it.
def parse_go(tokens: list[str]) -> dict[str, int]:
    return {
        name: int(value)
        for name, value in pairwise(tokens)
        if name in GO_INTS and value.isdigit()
    }


# "setoption name <words> value <words>". Move Overhead is the only option offered: the
# transposition table is a fixed 2^22 slots and the search is single-threaded, so there is no
# honest Hash or Threads to advertise.
def set_option(tokens: list[str]) -> None:
    global move_overhead_ms
    if "value" not in tokens:
        return
    split = tokens.index("value")
    name, value = " ".join(tokens[1:split]), " ".join(tokens[split + 1:])
    if name == "Move Overhead" and value.isdigit():
        move_overhead_ms = int(value)


def main() -> None:
    global board

    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        command, rest = tokens[0], tokens[1:]

        if command == "uci":
            send(f"id name {ENGINE_NAME}")
            send(f"id author {ENGINE_AUTHOR}")
            send(f"option name Move Overhead type spin default {DEFAULT_MOVE_OVERHEAD_MS} "
                 f"min 0 max 5000")
            send("uciok")
        elif command == "isready":
            send("readyok")  # the search compiled at import, so this is always immediate
        elif command == "setoption":
            set_option(rest)
        elif command == "ucinewgame":
            new_game()
        elif command == "position":
            board = parse_position(rest)
        elif command == "go":
            handle_go(parse_go(rest))
        elif command == "quit":
            break
        # Anything else is ignored, as UCI requires. `stop` and `ponderhit` never arrive: the
        # search runs synchronously inside `go` and no ponder support is advertised, so a GUI
        # has nothing to interrupt.


if __name__ == "__main__":
    main()
