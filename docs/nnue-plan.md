# NNUE engine - roadmap

Branch `nnue`. On this branch `agent.py` keeps the linear tapered eval; `nnue/agent.py` is the
NNUE submission entry point (its own numba board / movegen copy with an accumulator hook). This
file records where the strength gap is and the four things we are building to close it.

## Where we stand (2026-09-06)

Local arena through `harness/`: our `agent.py` (numba engine, linear eval) vs two other public
forks of the starter, 6 games each, 8 s + 0.1 s, colours alternating.

| Opponent | Their approach | Score |
|---|---|---|
| `TobyCoad/aichessathon-starter` | numba engine + trained NNUE (768 -> 512x2 -> 32 -> 1, 16 king zones, 8 output buckets), Lichess + Stockfish-binpack data, Syzygy + Polyglot book | **0.0%** (+0 =0 -6, checkmated every game) |
| `frozenexplorer/aichessathon-starter` | numba engine + hand-tuned classical eval, no net shipped | **100.0%** (+6 =0 -0) |

Read: our search skeleton is sound - it sweeps the classical-eval fork on both colours. The gap
to the NNUE fork is evaluation first, a few search accuracy terms second, time management
third. 8 s is not 120 s and TobyCoad has had a week of autonomous iteration behind it, so the
0-6 is a real gap but partly a maturity gap.

Reference clones for longer runs live outside the repo (scratchpad `toby/`, `frozen/`).

## What TobyCoad's repo shows (their measured numbers)

- **King-zone bucketed input: +31 Elo** (8 -> 16 zones; 16 -> 32 was worse - data-per-zone is
  the binding constraint). Horizontal mirroring of the buckets is their next step: free,
  doubles data per weight.
- **Output buckets by piece count** (they use 8).
- **Every loss reached <= 16-piece endgames** - static error 475 cp there vs 70 cp in the
  middlegame. Volume is not the fix (25% of their corpus is <= 16 pieces); label quality and
  distribution are. They moved to Stockfish-labelled positions (~depth 21) mixed with Lichess,
  and ship Syzygy 3-4-man tablebases as hard insurance.
- **Width is a trap**: 512 -> 768 cost 12-18% node rate for ~0 Elo. Keep the net callable
  thousands of times per move; bucket it, do not widen it.
- **Validation loss stops predicting Elo** past a point (their 1024-wide net: 15% better loss,
  ~0 Elo). Judge nets by games; keep a fixed endgame suite as a veto instrument only.
- **Their Elo model of their own engine**: EBF ~ 2.1, one speed doubling ~ +65 Elo @ 8 s /
  +32 @ 120 s; exact speed-ups realise 100% of that, pruning ~27%, ordering ~55%. "Saves 8% of
  nodes = +2 Elo, do not spend a day on it."
- **Their process**: every change is a switch OFF by default, tree always = current best,
  compiled kernel held bit-identical to a Python reference; bench nodes at fixed depth as the
  cheap objective; no SPSA on this timeline; post-mortem every rated game and bucket the loss
  cause (search / horizon / eval / time); measure `import` wall time on every change (numba
  compile is charged to the 90 s init budget).

## The four we are building

Ordered by leverage. Each lands behind a named constant, OFF in the tree, with the measurement
that can see it.

### 1. Bucket and endgame-train the net

Ref: https://www.chessprogramming.org/NNUE

`nnue/arch.py` today: 768 -> 256 x2 -> 32 -> 32 -> 1, dual perspective, **no king buckets, no
output buckets**. `nnue/net.npz` is ~240 KB.

- [ ] King-zone buckets on the feature transformer: pick the perspective side's first-layer
      weight slab by that side's king zone. Start with 16 zones (their +31 Elo point); build
      the zone map so a later mirror to 8x2 is a one-line change.
- [ ] Output buckets by piece count (start 8), selected at `evaluate` time.
- [ ] Keep FT_OUT at 256. Do not widen.
- [ ] Endgame data: label a <= 16-piece slice with the Stockfish pipeline (`tools/label.py`)
      and mix it into `data/` at a share we tune, not weight-dumped in. Track slope on a
      held-out Lichess set (`sum(pred*target) / sum(target**2)` ~ 1.0) so a rescaled head is
      caught.
- [ ] Ship Syzygy 3-4-5-man tablebases and a Polyglot opening book with `nnue/agent.py` -
      probe TB at the root and in search, book for the first plies. Cheap insurance against
      the endgame band and against early clock burn.
- Measure: `tools/ab_nnue.py` (net vs linear, both colours) for the net changes;
  `tools/bench.py` Elo vs an older copy of `nnue/agent.py`; `tools/endgame_bench.py` as a veto
  (a regression there rejects the change even if the A/B likes it). Judge by games, not val
  loss.

### 2. Search accuracy terms (one bundle)

Refs: https://www.chessprogramming.org/History_Heuristic#Continuation_History
      https://www.chessprogramming.org/Improving

Landing together because each is individually below A/B resolution; together they unlock harder
LMR.

- [ ] **Continuation history**: a 1-ply (piece, to) -> (piece, to) table, updated on a quiet
      cutoff with the same gravity as the main history. Used in three places - quiet ordering,
      the LMR reduction (`r -= (history + conthist) // K`, clamped +-2, replacing the coarse
      +-1-at-a-threshold step), and quiet-move futility / history pruning. We have no
      counter-move table today, so this is new.
- [ ] **Improving flag**: `static_eval[ply] > static_eval[ply-2]`. Needs a per-ply static-eval
      stack. Gates the RFP margin (`RFP_MARGIN * (depth - improving)`), the futility margins,
      the LMP count, and adds `+1` to the LMR reduction when not improving. Default improving =
      True at ply 0-1 (no grandparent); skip the write after a null move.
- [ ] **Cut-node flag**: one bool passed down. The child of a null-window search is a cut node
      iff its parent was not; the null-move child is always a cut node; the first child of a PV
      node is PV. Use: `+1` (later `+2` at high depth) to the LMR reduction at cut nodes.
- [ ] Cheap correctness in the same bundle: clear killers between real moves (ours persist for
      the whole game); check the quiet list handed to the history malus is not stale when a
      capture precedes the cutoff.
- Not in this bundle: our null-move reduction already carries an eval-margin term
  (`(depth * W - margin) // S - 1`), which is the shape TobyCoad's `2 + d/6` is missing. Leave
  it; re-tune the constants later only if a sweep is cheap.
- Measure: `tools/nodebench.py` at depth 8 and 10 first - target <= 0.90x nodes for the
  bundle; if it is not clearly below 0.95x the ordering wiring is doing nothing, debug before
  spending games. Then `tools/bench.py` Elo vs the pre-bundle copy. Keep the flags-off path
  bit-identical to `reference.py` (`tools/nodebench.py` / `tools/verify_*.py`).

### 3. Allocation-free SEE

Ref: https://www.chessprogramming.org/Static_Exchange_Evaluation

`agent.see()` (and the `nnue/` copy) allocates `np.empty(34)` on every call - in quiescence
that is the hottest loop in the engine. Both call sites only compare the result against a
threshold.

- [ ] Add `see_ge(board, move, threshold) -> bool`: a scalar running balance, early exit as
      soon as the sign is settled, no array, no unwind pass. Standard swap-loop formulation.
- [ ] Golden-test `see_ge(m, t) == (see(m) >= t)` over a few thousand random capture positions
      before wiring it in; keep `see()` for anywhere the value itself is wanted.
- [ ] Route the quiescence capture filter and the depth <= 5 main-search SEE prune through
      `see_ge`.
- Measure: `tools/nodebench.py` - node counts **byte-identical** (this is exact), knps up.
  Fold into the next bundle without its own A/B; an exact speed-up is invisible to the game
  harness. TobyCoad's estimate for the same change: +4-6% knps.

### 4. Time-budget shape

Ref: https://www.chessprogramming.org/Time_Management

TobyCoad's post-mortems put time as a top-2 loss cause. Three separable defects, likely ours
too - audit `agent.py`'s budget against them:

- [ ] **Increment under-credited.** 0.5 s / move over a 60-move game is ~30 s, a quarter of the
      clock. Credit `~0.8 * increment` in the soft budget, not a small constant.
- [ ] **Reserve banked and never spent.** A fixed-% reserve that collapses the budget to
      `remaining / N` once the clock enters it parks there for the rest of the game. Release
      most of it; keep ~1 s / move of headroom at a 1.5x charge.
- [ ] **Budget never reacts to the position.** Scale the soft budget by a factor in roughly
      [0.6, 1.6] from (a) best-move stability across the last 3 iterations and (b) effort - the
      fraction of this move's nodes spent under the best root move (a per-root-move node
      subtraction in the existing root loop).
- Measure: **not** the fast harness - below the low-clock floor this is byte-identical play.
  Build an offline schedule model (replay the clock arithmetic over a 130-ply game at
  120 s + 0.5 s, print seconds / move at moves 20 / 40 / ... / 120 per variant), then a
  clock-replay test at 120 s + 0.5 s with a 1.5x charge and a lowest-clock floor >= 5 s, then a
  set of full 120 s games. Accept on the schedule model + the floor + no flags.

## Measurement tools

- `tools/nodebench.py` - fixed FENs, fixed depth, `nodes bestmove score` per position; the
  exact-refactor gate (item 3, and the flags-off path of item 2).
- `tools/bench.py` - near-level openings, both colours, score + Elo + 95% CI; the heuristic
  gate. Always vs an older copy of the same engine.
- `tools/ab_nnue.py` - in-process net vs linear, a quick read on whether the net earns its
  slower nodes.
- `tools/endgame_bench.py` - <= 16-piece positions labelled by an engine, mean cp loss at a
  fixed movetime; a **veto** on eval changes, not a ranking signal.
- `import` wall time - `time uv run python -c "import nnue.agent"` on every bundle; every njit
  branch is numba compile time against the 90 s init budget.

---

Carried from `docs/plan.md`: every tuned number traces to a measurement we ran; change behind a
named constant; exact-equivalent change -> nodebench matches; heuristic change -> bench Elo >= 0
inside CI, or revert it and write down why.
