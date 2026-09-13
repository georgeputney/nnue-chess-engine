"""
The HTTP side of the play-it-in-a-browser demo: one warm engine process, one search at a time,
answering moves for web/index.html. Run it with

    make web

and open http://localhost:8000. See web/README.md for putting it on a real host.

The engine is the same engine - this calls get_move(fen, time_left_ms) exactly as the
competition platform and lichess/uci.py do, and nothing under engine/ changes for it. What
does need care, and neither of those needed:

  - The server owns the board. The page sends a move and gets a position back; it has no chess
    logic of its own and cannot be talked into an illegal one.
  - The engine's per-game state - the anti-repetition bookkeeping and the game's hash trail -
    lives in module globals, because the platform gave every game a process of its own. Here
    many games share one, so a Session owns that state and install() swaps it in around each
    search. The transposition table stays shared: it is a cache, and a position one visitor
    paid for is one the next gets free.
  - Nothing here is behind a login, so a visitor's move costs CPU on a machine with one core to
    give. Hence the lock, the clamped think time, and the per-address rate limit.
"""

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from engine import agent  # noqa: E402  (after the path insert; importing compiles the search)
from engine.board import parse_fen  # noqa: E402

# The page runs a real clock and sends the engine its own remaining time, which is converted
# into a budget exactly the way lichess/uci.py does it: get_move spends at most a quarter of
# the clock it is handed and starts no new deepening iteration past a fortieth, and an
# increment is folded in as this many moves' worth.
INCREMENT_MOVES = 25
MOVE_OVERHEAD_MS = 300

# The engine's clock is owned here, not sent by the page. That is what lets the budget be the
# real one: a page that could name Talos's remaining time could claim an hour and tie the core
# up, so the server keeps the clock and the page only displays it. No artificial ceiling sits
# above it - the time control is the CPU budget, and the whole game is bounded by the two
# minutes plus increments that Talos actually has.
# 2+1, near enough the control this engine was tuned for: the Chessathon ran 120 s plus half a
# second a move. At this clock Talos answers in about 5 s early and 3-4 s later on.
START_MS = 2 * 60 * 1000
INCREMENT_MS = 1000

# One core, one search: everything else waits, and past the wait is turned away rather than
# queued behind a crowd.
SEARCH_LOCK = threading.Lock()
LOCK_TIMEOUT_S = 30.0

# Per-address budget, as a token bucket: this many moves, refilled over this many seconds.
RATE_BURST = 20
RATE_REFILL_S = 60.0

# Sessions are cheap - a board and three small containers - but nothing makes a visitor say
# goodbye, so they expire, and there is a ceiling on how many live at once.
SESSION_IDLE_S = 3600.0
MAX_SESSIONS = 500

STATIC = {"/": "index.html", "/app.js": "app.js", "/sounds.js": "sounds.js",
          "/style.css": "style.css", "/favicon.svg": "favicon.svg",
          "/favicon.ico": "favicon.ico", "/apple-touch-icon.png": "apple-touch-icon.png",
          "/icon-192.png": "icon-192.png", "/icon-512.png": "icon-512.png",
          "/manifest.json": "manifest.json"}

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
                 ".png": "image/png", ".ico": "image/x-icon", ".json": "application/json"}


# The clock to hand get_move for this move, from the engine's own remaining time.
def think_clock_ms(clock_ms: float, increment_ms: int) -> int:
    usable = max(0, int(clock_ms) - MOVE_OVERHEAD_MS)
    return min(usable + increment_ms * INCREMENT_MOVES, usable * agent.HARD_LIMIT)


# One visitor's game: the board, and what get_move would otherwise keep in module globals.
@dataclass
class Session:
    board: chess.Board = field(default_factory=chess.Board)
    human_white: bool = True
    engine_ms: float = START_MS
    clock_trail: list[float] = field(default_factory=list)   # engine_ms before each of its moves
    seen: dict[int, int] = field(default_factory=dict)
    played: dict[int, int] = field(default_factory=dict)
    game_hashes: list[int] = field(default_factory=list)
    last_key: int = 0
    touched: float = field(default_factory=time.monotonic)

    # Back to move one. The search tables are shared and stay; only this game's state resets.
    def restart(self, human_white: bool) -> None:
        self.board = chess.Board()
        self.human_white = human_white
        self.engine_ms = START_MS
        self.clock_trail.clear()
        self.seen.clear()
        self.played.clear()
        self.game_hashes.clear()
        self.last_key = 0


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()
BUCKETS: dict[str, tuple[float, float]] = {}
BUCKETS_LOCK = threading.Lock()


# Point the engine's module globals at this session's state. get_move reads them by name at
# call time, so rebinding the module attributes is all it takes; the containers are mutated in
# place from there, and only last_key has to be copied back out afterwards.
def install(session: Session) -> None:
    agent.SEEN = session.seen
    agent.PLAYED = session.played
    agent.GAME_HASHES = session.game_hashes
    agent._LAST_KEY = session.last_key


# The session for this id, created if it is new, unknown, or long enough idle to have been
# swept. The id comes back alongside, so an expired visitor gets a fresh game, not an error.
def session_for(session_id: str) -> tuple[str, Session]:
    now = time.monotonic()
    with SESSIONS_LOCK:
        for stale in [k for k, v in SESSIONS.items() if now - v.touched > SESSION_IDLE_S]:
            del SESSIONS[stale]
        if session_id not in SESSIONS:
            if len(SESSIONS) >= MAX_SESSIONS:
                del SESSIONS[min(SESSIONS, key=lambda k: SESSIONS[k].touched)]
            session_id = uuid.uuid4().hex
            SESSIONS[session_id] = Session()
        SESSIONS[session_id].touched = now
        return session_id, SESSIONS[session_id]


# Token bucket, one per address: RATE_BURST moves, refilling over RATE_REFILL_S.
def within_rate_limit(address: str) -> bool:
    now = time.monotonic()
    with BUCKETS_LOCK:
        tokens, last = BUCKETS.get(address, (float(RATE_BURST), now))
        tokens = min(RATE_BURST, tokens + (now - last) * RATE_BURST / RATE_REFILL_S)
        if tokens < 1.0:
            BUCKETS[address] = (tokens, now)
            return False
        BUCKETS[address] = (tokens - 1.0, now)
        return True


# The engine's own static evaluation, in centipawns from White's point of view: the NNUE
# forward pass alone, before the search has a say. The network's opinion, which is the part of
# this engine worth putting on a screen.
def static_eval(board: chess.Board) -> int:
    position = parse_fen(board.fen())
    score = int(agent.evaluate(position))
    return score if position.side == 0 else -score


# How the game ended, structured rather than as a sentence: the page words it in terms of you
# and Talos, which the server has no business deciding.
def outcome_of(board: chess.Board) -> dict[str, Any]:
    if board.is_checkmate():
        return {"kind": "checkmate", "winner": "black" if board.turn == chess.WHITE else "white"}
    if board.is_stalemate():
        return {"kind": "stalemate", "winner": None}
    if board.is_insufficient_material():
        return {"kind": "insufficient", "winner": None}
    if board.can_claim_threefold_repetition():
        return {"kind": "repetition", "winner": None}
    if board.can_claim_fifty_moves():
        return {"kind": "fifty", "winner": None}
    return {"kind": None, "winner": None}


# The game so far in algebraic notation. Replayed from the start each time rather than kept as
# the moves are made: a few dozen pushes is nothing next to a search, and nothing can drift.
def movelist(board: chess.Board) -> list[str]:
    replay = chess.Board()
    notation = []
    for move in board.move_stack:
        notation.append(replay.san(move))
        replay.push(move)
    return notation


# Everything the page needs to draw itself: the position, what it may do next, and how the last
# search went. The page holds no chess logic, so this is the whole of its world.
def snapshot(session_id: str, session: Session, engine: dict[str, Any] | None) -> dict[str, Any]:
    board = session.board
    last = board.move_stack[-1].uci() if board.move_stack else None
    return {
        "moves": movelist(board),
        "session": session_id,
        "fen": board.fen(),
        "legal": sorted(move.uci() for move in board.legal_moves),
        "check": board.is_check(),
        "engineMs": max(0, round(session.engine_ms)),
        "outcome": outcome_of(board) if session.engine_ms > 0
                   else {"kind": "flag", "winner": "black" if session.human_white else "white"},
        "over": board.is_game_over(claim_draw=True) or session.engine_ms <= 0,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "humanWhite": session.human_white,
        "lastMove": last,
        "eval": static_eval(board),
        "engine": engine,
    }


# Search the session's position and play the move. Everything touching the engine happens under
# SEARCH_LOCK: there is one core, and one set of module globals, to go round.
def engine_move(session: Session) -> dict[str, Any]:
    if not SEARCH_LOCK.acquire(timeout=LOCK_TIMEOUT_S):
        raise TimeoutError("engine busy")
    try:
        install(session)
        session.clock_trail.append(session.engine_ms)
        agent.STATE.nodes = 0
        start = time.monotonic()
        uci = agent.get_move(session.board.fen(), think_clock_ms(session.engine_ms, INCREMENT_MS))
        elapsed = max(1, int((time.monotonic() - start) * 1000))
        nodes = int(agent.STATE.nodes)
        session.last_key = agent._LAST_KEY
        session.board.push_uci(uci)
        # it spent what it spent, then takes the increment like any other side
        session.engine_ms = session.engine_ms - elapsed + INCREMENT_MS
        return {"uci": uci, "nodes": nodes, "ms": elapsed, "nps": nodes * 1000 // elapsed}
    finally:
        SEARCH_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "nnue-demo"
    protocol_version = "HTTP/1.1"

    def reply(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # the page, its script and its stylesheet are a few kilobytes and change together on a
        # redeploy - a browser holding a stale one of the three is a broken demo, so make it ask
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def reply_json(self, code: int, payload: dict[str, Any]) -> None:
        self.reply(code, json.dumps(payload).encode(), "application/json")

    # do_GET / do_POST are the names BaseHTTPRequestHandler dispatches to.
    def do_GET(self) -> None:
        name = STATIC.get(self.path.split("?")[0])
        if name is None:
            self.reply_json(404, {"error": "not found"})
            return
        path = ROOT / name
        self.reply(200, path.read_bytes(), CONTENT_TYPES[path.suffix])

    def do_POST(self) -> None:
        address = self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0]
        if not within_rate_limit(address.strip()):
            self.reply_json(429, {"error": "too many moves - give it a moment"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length)) if length else {}
        except (ValueError, json.JSONDecodeError):
            self.reply_json(400, {"error": "bad request"})
            return

        session_id, session = session_for(str(request.get("session", "")))

        try:
            if self.path == "/api/new":
                self.handle_new(session_id, session, request)
            elif self.path == "/api/move":
                self.handle_move(session_id, session, request)
            elif self.path == "/api/engine":
                self.handle_engine(session_id, session)
            elif self.path == "/api/undo":
                self.handle_undo(session_id, session)
            else:
                self.reply_json(404, {"error": "not found"})
        except TimeoutError:
            self.reply_json(503, {"error": "the engine is busy - try again in a moment"})

    # A new game. The page asks for Talos's opening move separately if it took Black.
    def handle_new(self, session_id: str, session: Session, request: dict[str, Any]) -> None:
        session.restart(human_white=str(request.get("colour", "white")) != "black")
        self.reply_json(200, snapshot(session_id, session, None))

    # The visitor's move, and only that. Playing both moves in one request meant the page could
    # not show yours until Talos had answered, several seconds later - so the board, the clock
    # and the move list all waited on a search that had nothing to do with the move just made.
    def handle_move(self, session_id: str, session: Session, request: dict[str, Any]) -> None:
        uci = str(request.get("uci", ""))
        try:
            move = session.board.parse_uci(uci)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            self.reply_json(400, {"error": f"{uci} is not legal here"})
            return

        session.board.push(move)
        self.reply_json(200, snapshot(session_id, session, None))

    # Talos's reply, asked for once the page has painted yours.
    def handle_engine(self, session_id: str, session: Session) -> None:
        over = session.board.is_game_over() or session.engine_ms <= 0
        engine_turn = (session.board.turn == chess.WHITE) != session.human_white
        engine = engine_move(session) if engine_turn and not over else None
        self.reply_json(200, snapshot(session_id, session, engine))

    # Take back the pair of moves that ended the position on screen.
    def handle_undo(self, session_id: str, session: Session) -> None:
        for _ in range(2):
            if session.board.move_stack:
                session.board.pop()
        # taking the move back gives the time back with it
        if session.clock_trail:
            session.engine_ms = session.clock_trail.pop()
        # the engine's repetition bookkeeping counted positions that no longer happened
        session.seen.clear()
        session.played.clear()
        session.game_hashes.clear()
        session.last_key = 0
        self.reply_json(200, snapshot(session_id, session, None))

    # One line per request on stdout rather than stderr, so a log file reads in order.
    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} {format % args}")


def main() -> None:
    # argv wins, then PORT - which is how Spaces, Render and the rest hand one over. HOST is
    # every interface by default, which is what a container and a laptop both want; behind a
    # reverse proxy set it to 127.0.0.1 so the port is not reachable from outside directly.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "")
    print(f"talos demo on {host or '0.0.0.0'}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
