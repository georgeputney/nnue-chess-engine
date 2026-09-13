/*
 * Talos play page, wired to the Python engine behind /api.
 *
 * The server owns the board: it validates every move and sends back the legal moves for the
 * position it returns, so nothing here can be talked into an illegal one, and there is no
 * chess knowledge on this side at all.
 *
 * A move and Talos's reply are two requests, not one. Asking for both together meant your own
 * move could not appear - on the board, in the clock or in the notation - until a search that
 * had nothing to do with it had finished several seconds later.
 */

const START_MS = 2 * 60 * 1000;    // 2+1, matching web/server.py
const INCREMENT_MS = 1000;
const TICK_MS = 100;
const EVAL_RANGE_CP = 600;

// The solid Unicode set for both colours. Hue carries ownership, value carries chess colour,
// so the outlined white glyphs are not wanted - they read as translucent at this size.
// U+FE0E, the text presentation selector, after every glyph. Without it iOS renders the pawn
// as an emoji - a different font, a different weight, and nothing like the other five pieces.
const GLYPH = {
  k: "\u265A\uFE0E", q: "\u265B\uFE0E", r: "\u265C\uFE0E",
  b: "\u265D\uFE0E", n: "\u265E\uFE0E", p: "\u265F\uFE0E",
};
const FILES = "abcdefgh";

const el = (id) => document.getElementById(id);
const board = el("board");

let snap = null;        // the last position the server sent
let selected = null;
let thinking = false;
let humanWhite = true;
let ended = null;       // client-side endings: resignation and flag fall
let humanMs = START_MS;      // this side is the page's to keep
let engineMs = START_MS;     // this side belongs to the server, which owns Talos's budget
let clockRunning = false;   // the clock does not run before the first move
let lastTick = null;        // wall time of the previous tick, so drift does not accumulate
let searchStart = null;     // when the request went out, and what Talos's clock was then
let engineMsAtSearch = START_MS;
let clockHistory = [];      // one entry per move pair, so Undo can restore both clocks

// ---- server -------------------------------------------------------------------------

async function send(path, body) {
  el("error").textContent = "";
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session: localStorage.getItem("session") || "", ...body }),
    });
    const data = await response.json();
    if (!response.ok) {
      el("error").textContent = data.error || "The engine could not answer that.";
      return null;
    }
    localStorage.setItem("session", data.session);
    return data;
  } catch {
    el("error").textContent = "Lost contact with the engine.";
    return null;
  }
}

async function newGame(asWhite) {
  humanWhite = asWhite;
  humanMs = engineMs = START_MS;
  clockRunning = false;
  lastTick = null;
  searchStart = null;
  clockHistory = [];
  ended = null;
  selected = null;
  el("seg-white").classList.toggle("on", asWhite);
  el("seg-black").classList.toggle("on", !asWhite);

  const data = await send("/api/new", { colour: asWhite ? "white" : "black" });
  if (data) adopt(data);
  if (!asWhite) await askTalos();    // playing black means Talos opens
}

// Talos's clock comes back from the server, which is the only side that knows what the search
// actually cost. The page ticks it down between replies purely so it reads smoothly.
function adopt(data) {
  engineMs = data.engineMs;
  if (data.moves.length) clockRunning = true;    // the clock starts on the first move played
  if (data.engine) play(soundForSan(data.moves[data.moves.length - 1]));
  snap = data;
  selected = null;
  render();
}

// Ask Talos for his reply, having already painted yours.
async function askTalos() {
  if (!snap || snap.over) return;
  beginSearch();
  render();
  const data = await send("/api/engine", {});
  endSearch();
  if (data) adopt(data); else render();
}

// ---- clock --------------------------------------------------------------------------

function gameOver() {
  return Boolean(ended) || Boolean(snap && snap.over);
}

// Whose clock is running. While Talos is searching it is his, and `snap` cannot answer that:
// it still holds the position from before your move, because his reply has not arrived yet.
function engineOnClock() {
  return searchStart !== null;
}

// Talos's clock is derived from the wall time since his request went out, not accumulated a
// tick at a time. That is exactly what the server charges him, so the two cannot drift apart,
// and a tick that arrives late or not at all cannot leave the clock stuck.
function beginSearch() {
  thinking = true;
  searchStart = performance.now();
  engineMsAtSearch = engineMs;
}

function endSearch() {
  thinking = false;
  searchStart = null;
}

function tick() {
  // measure the real interval rather than trusting setInterval, which drifts - and drifts
  // worst while a request is in flight, which is exactly when the clock matters
  const now = performance.now();
  const elapsed = lastTick === null ? 0 : now - lastTick;
  lastTick = now;

  if (!clockRunning || gameOver() || !snap) return;

  if (searchStart !== null) {
    engineMs = Math.max(0, engineMsAtSearch - (now - searchStart));
  } else {
    humanMs -= elapsed;
  }

  if (humanMs <= 0) {
    humanMs = 0;
    ended = { kind: "flag", humanLost: true };
    clockRunning = false;
    render();
    return;
  }
  paintClocks();
}

function clockText(ms) {
  const total = Math.max(0, Math.ceil(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function paintClocks() {
  el("bot-clock").textContent = clockText(humanMs);
  el("top-clock").textContent = clockText(engineMs);

  const live = Boolean(clockRunning && !gameOver() && snap);
  const engineTurn = engineOnClock();
  el("bot-clock").classList.toggle("running", live && !engineTurn);
  el("top-clock").classList.toggle("running", live && engineTurn);

  const playing = Boolean(!gameOver() && snap);
  el("bot-dot").classList.toggle("live", playing && !engineTurn);
  el("top-dot").classList.toggle("live", playing && engineTurn);
}

// ---- board ---------------------------------------------------------------------------

// The board half of a FEN, as a map from square name to piece letter.
function positionOf(fen) {
  const squares = {};
  fen.split(" ")[0].split("/").forEach((row, index) => {
    const rank = 8 - index;
    let file = 0;
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") file += Number(ch);
      else squares[FILES[file++] + rank] = ch;
    }
  });
  return squares;
}

function targetsFrom(square) {
  if (!square || !snap) return [];
  return snap.legal.filter((uci) => uci.startsWith(square)).map((uci) => uci.slice(2, 4));
}

function humanToMove() {
  return snap && !gameOver() && !thinking && (snap.turn === "white") === humanWhite;
}

function render() {
  if (!snap) return;
  const pieces = positionOf(snap.fen);
  const last = snap.lastMove ? [snap.lastMove.slice(0, 2), snap.lastMove.slice(2, 4)] : [];
  const targets = targetsFrom(selected);
  const ranks = humanWhite ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = humanWhite ? FILES : [...FILES].reverse().join("");

  board.innerHTML = "";
  for (const rank of ranks) {
    for (const file of files) {
      const name = file + rank;
      const piece = pieces[name];
      const cell = document.createElement("div");
      const dark = (FILES.indexOf(file) + rank) % 2 === 1;   // a1 is dark
      cell.className = "cell " + (dark ? "dark" : "light");
      if (last.includes(name)) cell.classList.add("last");
      if (name === selected) cell.classList.add("selected");

      let inner = "";
      if (targets.includes(name)) {
        inner += piece ? '<span class="capture-ring"></span>'
                       : '<span class="target-dot"><span></span></span>';
      }
      if (rank === ranks[7]) inner += `<span class="coord-file">${file}</span>`;
      if (file === files[0]) inner += `<span class="coord-rank">${rank}</span>`;
      if (piece) {
        const white = piece === piece.toUpperCase();
        cell.classList.add((white === humanWhite ? "you-" : "ai-") + (white ? "w" : "b"));
        inner += GLYPH[piece.toLowerCase()];
      }
      cell.innerHTML = inner;
      cell.dataset.square = name;
      board.appendChild(cell);
    }
  }

  paintPanel();
  paintClocks();
}

// ---- panel ---------------------------------------------------------------------------

// How the game ended, or null while it is still running. This goes on the board.
function verdictText() {
  if (ended && ended.kind === "resign") return "You resigned. Talos wins.";
  if (ended && ended.kind === "flag") return "Flagged. Talos wins on time.";
  const outcome = snap ? snap.outcome : null;
  if (!outcome || !outcome.kind) return null;
  if (outcome.kind === "flag") return "Talos flagged. You win.";
  if (outcome.kind === "checkmate") {
    return (outcome.winner === "white") === humanWhite ? "Checkmate. You win."
                                                       : "Checkmate. Talos wins.";
  }
  if (outcome.kind === "stalemate") return "Stalemate, drawn.";
  if (outcome.kind === "insufficient") return "Drawn, neither side can mate.";
  if (outcome.kind === "repetition") return "Drawn by repetition.";
  return "Drawn by the fifty-move rule.";
}

// Whose move it is, which is all the status line has to say now.
function statusText() {
  if (verdictText()) return "Game over.";
  if (thinking) return "Talos is thinking…";
  if (!snap) return "Talos is waking up…";
  if (humanToMove()) return snap.check ? "Your move, you are in check." : "Your move.";
  return "Talos to move.";
}

// The bar reads from your side, not White's: this page is you against Talos.
function paintEval() {
  const yours = humanWhite ? snap.eval : -snap.eval;
  const clamped = Math.max(-EVAL_RANGE_CP, Math.min(EVAL_RANGE_CP, yours));
  el("evalbar-fill").style.height = `${50 + (clamped / EVAL_RANGE_CP) * 50}%`;
  el("evalbar").title = `Static evaluation ${pawns(yours)} from your side`;
}

function pawns(centipawns) {
  const sign = centipawns > 0 ? "+" : centipawns < 0 ? "-" : "";
  return sign + (Math.abs(centipawns) / 100).toFixed(2);
}

function paintPanel() {
  el("status").textContent = statusText();

  const verdict = verdictText();
  el("verdict").hidden = !verdict;
  if (verdict) el("verdict-line").textContent = verdict;

  paintEval();

  const rows = el("moves");
  rows.innerHTML = "";
  const moves = snap ? snap.moves : [];
  if (!moves.length) {
    rows.innerHTML = '<div class="moves-empty">Click a piece, then a square.</div>';
  } else {
    for (let i = 0; i < moves.length; i += 2) {
      const row = document.createElement("div");
      row.className = "move-row";
      row.innerHTML = `<span class="n">${i / 2 + 1}</span><span>${moves[i]}</span>` +
                      `<span class="b">${moves[i + 1] || ""}</span>`;
      rows.appendChild(row);
    }
    rows.scrollTop = rows.scrollHeight;
  }

  el("undo").disabled = thinking || !snap || snap.moves.length < 2;
  el("resign").disabled = gameOver() || !snap;
}

// ---- interaction ----------------------------------------------------------------------

board.addEventListener("click", async (event) => {
  const cell = event.target.closest(".cell");
  if (!cell || !humanToMove()) return;
  const name = cell.dataset.square;

  if (targetsFrom(selected).includes(name)) {
    // promotions arrive as four separate moves; take the queen unasked, as the design does
    const candidates = snap.legal.filter((uci) => uci.startsWith(selected + name));
    const uci = candidates.find((u) => u.length === 4) || candidates[0];

    clockHistory.push(humanMs);
    humanMs += INCREMENT_MS;
    clockRunning = true;

    // paint your move first, then ask
    // your own move sounds on the click, not on the round trip
    const before = positionOf(snap.fen);
    const moved = before[selected];
    const castled = moved && moved.toLowerCase() === "k"
      && Math.abs(FILES.indexOf(selected[0]) - FILES.indexOf(name[0])) === 2;
    play(castled ? "castle" : before[name] ? "capture" : "move");

    selected = null;
    const data = await send("/api/move", { uci });
    if (data) adopt(data);       // board, clock and notation all move together, now
    await askTalos();
    return;
  }

  const piece = positionOf(snap.fen)[name];
  const mine = piece && (piece === piece.toUpperCase()) === humanWhite;
  selected = mine && name !== selected ? name : null;
  render();
});

el("undo").onclick = async () => {
  if (thinking) return;
  const data = await send("/api/undo", {});
  const restored = clockHistory.pop();
  if (restored !== undefined) humanMs = restored;
  ended = null;
  if (data) adopt(data);
};

el("mute").onclick = () => paintSound(toggleSound());

el("resign").onclick = () => {
  if (!snap || gameOver()) return;
  ended = { kind: "resign" };
  clockRunning = false;
  render();
};

el("verdict-new").onclick = () => newGame(humanWhite);
el("seg-white").onclick = () => newGame(true);
el("seg-black").onclick = () => newGame(false);
el("new-game").onclick = () => newGame(humanWhite);

// the icon shows the current state; the label says what clicking it will do
function paintSound(isMuted) {
  const button = el("mute");
  button.classList.toggle("muted", isMuted);
  button.setAttribute("aria-label", isMuted ? "Unmute" : "Mute");
  button.title = isMuted ? "Unmute" : "Mute";
}

paintSound(soundMuted());

setInterval(tick, TICK_MS);
newGame(true);
