# Building our own agent.py

Goal: a hand-rolled alpha-beta chess engine in `agent.py`. The feature list below came out of
day 1 spent surveying established chess-programming techniques; from here we build each one
ourselves and measure it, so every heuristic and every tuned number traces back to a
measurement we ran rather than a value copied from somewhere.

Entry point contract: `get_move(fen: str, time_left_ms: int) -> str` returning UCI. Our colour
is `chess.Board(fen).turn`. Process starts once per game; module state survives between our
moves in that game only. Heavy setup goes at import time (60 s budget).

---

## Measurement (build/keep these working the whole way)

Nothing here is optional. A feature we cannot measure is a feature we cannot defend. Two of
these tools do not exist yet; we build each one in the phase that first needs it.

- **`make gate`** — ruff + mypy strict + two 5 s games that must finish. This is the
  submission-validity bar. Run after every change. (Exists: stock.)
- **`make arena`** — 20 fast games vs one baseline from the start position, quick smoke signal
  only; too few games, and too correlated, to trust a small delta. (Exists: stock.)
- **`tools/bench.py`** — the real A/B, to build in Phase 1. Fixed near-level opening FENs, both
  colours, prints score + Elo + 95% CI. Accept or reject any *heuristic* change on this:
  `baselines/greedy`, then `baselines/minimax`, then an older copy of our own engine. Lives in
  `tools/`, not `harness/`, so mypy strict leaves it alone and we honour "don't edit
  `harness/`".
- **`tools/nodebench.py`** — the exact-refactor check, to build in Phase 3. Calls
  `agent.bench_search` (added in Phase 2), searches a fixed FEN list to a fixed depth, prints
  nodes + bestmove + score per position. For the changes that must not alter the result
  (alpha-beta, TT, PVS, aspiration): bestmove and score identical, node count down.

Rule of thumb: exact-equivalent change -> nodebench must match. Heuristic change -> bench.py
Elo must be >= 0 (inside CI) or we revert it and write down why.

Per feature: change behind a named constant, nodebench, then bench.py vs a baseline, record
the Elo line in the commit message.

---

## Phase 0 - skeleton

- [ ] `agent.py`: the stock starter already is the skeleton (parses FEN, returns a legal move).
      Pure engine, no `torch`.
- [ ] Never create `chess.py` / `types.py` / `random.py` next to `agent.py` (sys.path shadow).
- [ ] `make gate` green; both smoke games finish cleanly.

## Phase 1 - static eval v0 (material)

Ref: https://www.chessprogramming.org/Evaluation

- [ ] `evaluate(board, side) -> int`, centipawns, from `side`'s point of view.
- [ ] Values 100 / 320 / 330 / 500 / 900 over `board.pieces(pt, colour)`.
- [ ] Wire a 1-ply pick into `get_move` so `evaluate` is exercised.
- [ ] Verify: symmetric position -> 0; side up a knight -> ~ +320.
- [ ] Build `tools/bench.py`: openings Elo A/B (fixed near-level FENs, both colours, score +
      Elo + 95% CI). In `tools/`, not `harness/`. Sanity-check: roughly even vs `baselines/greedy`.

## Phase 2 - negamax, fixed depth

Ref: https://www.chessprogramming.org/Negamax

- [ ] `negamax(board, depth) -> int`; depth 0 -> `evaluate`; else max over `-negamax(child)`.
- [ ] Checkmate / stalemate handled (no legal moves).
- [ ] Add `bench_search(fen, depth) -> (uci, score, nodes)` for `tools/nodebench.py` to call.
      Thin wrapper over the search; keep its signature stable as internals change.
- [ ] Beats `baselines/random` at depth 3.

## Phase 3 - alpha-beta (exact refactor)

Ref: https://www.chessprogramming.org/Alpha-Beta

- [ ] Add `alpha, beta`; `alpha = max(alpha, score)`; cut on `alpha >= beta`.
- [ ] Build `tools/nodebench.py`: fixed FEN list (opening / middlegame / tactical / endgame),
      fixed depth, prints `nodes  bestmove  score` per FEN via `agent.bench_search`, plus a
      `--baseline` / `--check` snapshot compare.
- [ ] nodebench: identical bestmove + score at each depth, fewer nodes. If move changes, bounds
      are wrong.

## Phase 4 - iterative deepening + time management

Refs: https://www.chessprogramming.org/Iterative_Deepening
      https://www.chessprogramming.org/Time_Management

- [ ] Root loop `depth = 1, 2, 3, ...`, keep best move of last *completed* depth.
- [ ] Hard cap: abort once elapsed > `time_left_ms / 4`.
- [ ] Soft cap: do not *start* a new depth past `time_left_ms / 40`.
- [ ] Abort = exception raised off a node counter (`nodes % 2048 == 0 and time_up()`), caught at
      root, board rewound as the stack unwinds. Tune 2048 so the clock check isn't a cost.
- [ ] bench.py vs greedy: no flag/loss terminations, score up.

## Phase 5 - move ordering: MVV-LVA

Ref: https://www.chessprogramming.org/MVV-LVA

- [ ] Captures first, key `value(victim) - value(attacker)` (`board.piece_at(to)`, `move.promotion`).
- [ ] Quiets after, unordered for now.
- [ ] nodebench: depth-N result unchanged, node count down hard (3-10x).

## Phase 6 - quiescence search

Refs: https://www.chessprogramming.org/Quiescence_Search
      https://www.chessprogramming.org/Delta_Pruning

- [ ] At `depth <= 0`: stand-pat = `evaluate`; `if stand_pat >= beta: return beta`;
      `alpha = max(alpha, stand_pat)`; then search **captures only**
      (`board.generate_legal_captures()`), MVV-LVA, recurse. Ply cap on the tail.
- [ ] Delta pruning: skip a capture if `stand_pat + value(victim) + margin < alpha`.
      Start with our own round guess: one flat margin, ~200 cp, for every victim. Split it
      per-victim later only if the tactical suite says it helps. Tune with bench.py.
- [ ] Verify: hanging-piece-behind-a-capture position no longer misevaluated.

## Phase 7 - transposition table

Ref: https://www.chessprogramming.org/Transposition_Table

- [ ] Key `chess.polyglot.zobrist_hash(board)` (or `board._transposition_key()`).
- [ ] `dict[int, (key, depth, score, flag, move)]`, fixed cap, always-replace.
      flag in {UPPER, EXACT, LOWER}.
- [ ] Probe: if stored `depth >= remaining`, cut when bound allows (EXACT always; LOWER if
      `score >= beta`; UPPER if `score <= alpha`). Always reuse stored move for ordering.
- [ ] Store: cutoff -> LOWER; never raised alpha -> UPPER; else EXACT.
- [ ] Mate scores stored relative to node (`+ply` in, `-ply` out); clamp stored score to
      +/- 20000.
- [ ] Feed stored score into the static-eval slot.
- [ ] nodebench: bestmove + score unchanged, nodes down; re-search of same FEN near-instant.

## Phase 8 - principal variation search (exact refactor)

Ref: https://www.chessprogramming.org/Principal_Variation_Search

- [ ] First move full window `(-beta, -alpha)`; later moves null window `(-alpha-1, -alpha)`;
      re-search full window if it returns `> alpha` and `< beta`.
- [ ] Re-search loop drops the LMR reduction first, then widens the window (order matters: an
      un-reduced re-search is cheaper than a full-width one).
- [ ] nodebench: matches alpha-beta at depth N, nodes down on well-ordered positions.

## Phase 9 - history heuristic + killers

Refs: https://www.chessprogramming.org/History_Heuristic
      https://www.chessprogramming.org/Killer_Heuristic

- [ ] `history[piece_type][to_square]` ints. Quiet causing beta cutoff: `bonus = depth*depth`,
      gravity update `h += bonus - h*abs(bonus)//MAX` (saturates).
- [ ] Order quiets by history desc. Killers: 1-2 quiet cutoff-moves per ply, tried after captures.
- [ ] bench.py vs minimax: expect real Elo.

## Phase 10 - null-move pruning

Ref: https://www.chessprogramming.org/Null_Move_Pruning

- [ ] Non-PV, not in check, enough material: `board.push(chess.Move.null())`, search
      `depth - 1 - R` null-window around beta, pop. `score >= beta` -> return beta.
- [ ] `R = 2 + depth // 6` (tune). Skip in K+P endings (zugzwang).
- [ ] bench.py Elo >= 0; spot-check a known zugzwang position.

## Phase 11 - check extension

Ref: https://www.chessprogramming.org/Check_Extensions

- [ ] `if board.gives_check(move): search child at depth - 1 + 1`. Cap extensions per line.
- [ ] bench.py Elo >= 0.

## Phase 12 - reverse futility pruning

Ref: https://www.chessprogramming.org/Reverse_Futility_Pruning

- [ ] Shallow non-PV, not in check: `if static_eval - margin*depth >= beta: return static_eval`.
- [ ] Margin ~70-120 cp/ply. Tune by bench.py.

## Phase 13 - late move reductions

Ref: https://www.chessprogramming.org/Late_Move_Reductions

- [ ] Quiet moves past the first few: search at `depth - r`; re-search full depth if it beats
      alpha.
- [ ] `r` grows with move index and depth, shrinks with the move's history score.
      Base `r = 0.75 + ln(depth)*ln(idx)/2.25`, integerised (a widely used log-formula shape).
- [ ] Biggest node saver, most finicky. Sweep the constants in bench.py.

## Phase 14 - late move pruning + move-loop futility

Ref: https://www.chessprogramming.org/Futility_Pruning

- [ ] LMP: shallow non-PV, after `LMP[depth]` quiets stop trying quiets. Start from our own
      cap that grows roughly with depth (e.g. `3 + depth*depth`: 4, 7, 12, 19 for depths
      1-4), then tune with bench.py.
- [ ] Futility: before a quiet at shallow depth, `if static_eval + margin*depth <= alpha: skip`.
- [ ] bench.py only; expect a small gain; justify with the score line.

## Phase 15 - internal iterative reduction

Ref: https://www.chessprogramming.org/Internal_Iterative_Reductions

- [ ] No TT move at this node and depth high -> `depth -= 1` before searching.
- [ ] bench.py Elo >= 0.

## Phase 16 - aspiration windows

Ref: https://www.chessprogramming.org/Aspiration_Windows

- [ ] From depth ~4: `alpha, beta = prev - 20, prev + 20`. Fail-low -> widen alpha (or -INF);
      fail-high -> widen beta. Make the re-search actually use the widened window (easy bug:
      re-searching with the same window the search just failed).
- [ ] Tune the +/- 20 and the widening schedule. nodebench: same result, nodes down.

## Phase 17 - mate-distance scoring

Ref: https://www.chessprogramming.org/Score#Mate_Scores

- [ ] Checkmate returns `MATE - ply` so faster mates score higher. Mate-distance pruning:
      clamp alpha/beta to best/worst mate still reachable.
- [ ] `MATE = 30000`, `MATE_BOUND = MATE - 1000`.

## Phase 18 - evaluation build-out (each term its own bench.py test)

Refs: https://www.chessprogramming.org/Piece-Square_Tables
      https://www.chessprogramming.org/Tapered_Eval
      https://www.chessprogramming.org/Mobility

- [ ] Piece-square tables: 6x64, seed from a published set, `chess.square_mirror` for black.
- [ ] Tapered eval: MG + EG sets, `phase = sum(PHASE[pt])` clamped 24,
      `(mg*phase + eg*(24-phase)) // 24`.
- [ ] Tempo bonus for side to move (start ~10 cp, roughly equal in MG and EG).
- [ ] Mobility: slider `board.attacks(sq)` count, weighted per type; king scored as a queen
      (more mobility = more exposed = worse) as the king-safety term.
- [ ] Pawn structure: friendly pawns ahead on the same file = doubled-pawn penalty (one term,
      two jobs).
- [ ] "Far pawn" virtual piece: pawn on the opposite board half from its own king scored as its
      own piece type (`FAR_PAWN = 0`) with its own table rows. Add last.

## Phase 19 - Texel tuning

Ref: https://www.chessprogramming.org/Texel%27s_Tuning_Method

Offline only. The rules allow training on engine-annotated data; the ban is on what ships and
runs inside the zip. Plan: label a large set of quiet positions with a strong engine (e.g.
Stockfish) and tune our eval weights against those labels. Target a substantial dataset -
several hundred thousand positions or more - so the weights are well-constrained.

- [ ] Build the extraction + labelling pipeline: sample quiet positions (no side in check, no
      winning capture on the stand-pat) from games, label each with an engine (cp score, or a
      win/draw/loss expectation).
- [ ] Loss = MSE of `sigmoid(our_eval / K)` vs the label; fit the scaling `K` once by
      minimising over it.
- [ ] Optimise the weight vector (PST entries, term weights, MG/EG halves) by coordinate
      descent: nudge each param +/- 1, keep the change if loss drops, repeat to convergence.
- [ ] Re-run bench.py after: keep the version that wins games, not the one with the lowest
      tuning loss - they diverge.
- [ ] Tuner and dataset live outside `agent.py`; only the resulting numbers ship.

---

## Platform gotchas

- Read-only FS except 256 MB `/tmp`; `HOME` and caches already point there.
- No network, one core (`torch.set_num_threads(1)` only if torch is imported), 2 GB, no GPU.
- Zip has `agent.py` at the root, everything unzipped < 50 MB.
- Illegal / malformed / crash / OOM / flag loses the game; reply > 4 KB counts as illegal;
  300 plies -> material adjudication.
- Pure python-chess is ~10-50k nps; that caps depth. If too slow, the fix is a numba bitboard
  core (see `baselines/numba`) - a real rewrite, decide before Phase 13.
