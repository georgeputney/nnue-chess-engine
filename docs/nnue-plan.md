# NNUE engine - roadmap

Branch `nnue`. On this branch `agent.py` keeps the linear tapered eval; `nnue/agent.py` is the
NNUE submission entry point (its own numba board / movegen copy with an accumulator hook). This
file records where the strength gap is and what we are building to close it.

## Where we stand (updated 2026-09-07)

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

## Since 2026-09-06

- **Output buckets (8) shipped**, net widened 128 -> 256, retrained on 45M Lichess positions
  (`data/lichess_big`, `ingest_lichess --shard-size` streams the full dump). arch is now
  768 -> 256 x2 -> 32 -> 8 heads -> 1.
- **TT static-eval cache** (`EVAL_CACHE`, `e95e86f`): negamax RFP/null-move gate and quiescence
  stand-pat reuse the TT slot's stored NNUE eval on a key match. Exact (nodebench byte-identical
  with the flag flipped), ~3-5% at fixed depth, more with iterative deepening reuse.
- **`tools/bundle_nnue.py`**: flattens `nnue/` into a submittable `candidates/nnue/` dir
  (`from nnue.x` -> `from x`, root modules + `net.npz` + `syzygy/` copied in). Needed because
  `harness/runner.py` puts only the agent dir on `sys.path`.
- Deep read of `TobyCoad/aichessathon-starter`'s current build (their flag comments are a
  measured notebook). The items below fold in what they *promoted* and skip what they rejected.

## What TobyCoad promoted vs rejected (their flag comments, latest build)

**Promoted** (measured wins, worth cherry-picking - see items 2 and 6):
`TT_BUCKETS` (2-slot: deep + always-replace), `SEE_MAIN` (skip captures losing > 20*d^2 at
d<=5), `SINGULAR` (d>=7, hash move, half-depth verify), `NMP_V2` (dynamic R = 3 + d//4 +
min((eval-beta)//200, 3)) + `NMP_V2B` (verify at d>=10), `HISTORY2` + `CAPTURE_ORDER` (counter
-move + capture history, SEE-losing captures below quiets), `LMR_AGGRESSIVE` (log(d)*log(m)/1.8
+ 0.5), `ASP_WIDE` (geometric 1.5x widening) with `ASPIRATION_WINDOW 15`, `PRUNE_V2`, `QS_TT`,
`QS_CAP 14`, `LAZY_ACC` (profiled 15.4% of search time), `REPETITION_TWOFOLD` (they *lost a won
game with mate on the board* without it).

**Rejected** - do not spend time here: correction history (`CORRECTION`, first version
-137 +/- 65 Elo, entries saturate in seconds), continuation-history `CONT_HIST`, `IMPROVING`,
`CUTNODE`, `RFP_PHASE` margin scaling, `ENDGAME_SHRINK` blend-to-material ("the cure is a
retrain"), the Polyglot book (~-20 cp per firing on curated starts, one Greek-gift loss),
pondering (our contract suspends the process between our moves).

## 2026-09-09: the endgame program, measured

Six A/B matches, each 80 games over 40 openings at 5 s + 0.1 s through `tools/bench.py`,
colours alternating. Every attempt to train or tabulate the endgame lost Elo. The only things
that gained were the cheap correctness fixes.

| change | isolates | result |
|---|---|---|
| net_215m vs net_768 | the endgame-mix retrain | **+30** [-34, +97] for the *unmixed* net |
| blend vs net_768, fixed code both sides | the cp-blended loss | **-84** [-159, -16] |
| fixed code vs old code, net_768 both sides | fifty-move + TB bundle incl. 5-man | **-26** [-84, +31] |
| `nnue_fix50` vs `o8-300m` | fifty-move + fixed ranking, 3-4-man | **+22** [-41, +86] |
| `nnue_fix215` vs `o8-300m` | the same plus the 5-man slice | **+4** [-59, +68] |
| `nnue_fix215` vs `nnue_fix50` | **the 5-man slice alone** | **-48** [-114, +15] |

**Shipped** (`nnue/` as it stands, 4.9 MB unzipped): the fifty-move rule scored as the draw the
referee claims, in negamax and quiescence, checkmate resolved first; a rewritten Syzygy root
ranking; `tb_score` counting only +-2 so a cursed win / blessed loss is the draw it is in play;
the in-search WDL probe capped at `TB_MEN = 4`.

**Rejected and parked.** The cp-blended net is `nnue/net_blend.npz`; `nnue/net.npz` is the
committed 215M net again. The 5-man WDL slice (KRPvKR, KRPvKP, KRRvKR, KPPvKP, KRPPvK, 28 MB)
is in `nnue/syzygy5/`, out of the bundler's reach. Do not ship either again without new
evidence. Why the slice loses, in order of confidence: the objmode probe fires at every
<= 5-man node and covered 5-man material is common, so it taxes exactly the phase it was meant
to help; an exact 0 for a known draw outweighs our +-30 contempt, so the search liquidates into
certain draws; and every tablebase win scores `TB_WIN - ply`, which gives no gradient to
convert by. DTZ is what makes the root ranking exact, and 5-man DTZ does not fit the 50 MB cap.

### The root-ranking bug this started from

`best_tb_move` was returning a **drawing move from a won position**, and `get_move` returns its
answer before the search runs, so nothing downstream could rescue it. python-chess's `probe_dtz`
short-circuits: a drawn child returns 0 without opening the DTZ file, a decisive child has to
read it and raises `MissingTableError`. We ship DTZ only for 3-4 men, so among 5-man children
only the drawn ones survived ranking and every winning move was skipped. Measured on won
positions: 47 of 103 KRPvKR, 45/150 KRPvKP, 35/116 KPPvKP, 66/149 KRRvKR; the 4-man control 0.
After the fix, 0 across all four.

The fix ranks on the five-valued WDL (a real win outranks a cursed one), treats a missing DTZ as
a neutral 0 rather than a skip, breaks ties with zeroing then the corner / king-distance mate
drivers, and returns None for the whole position when a *WDL* probe fails - a promotion out of
the shipped material - rather than picking the best of a partial set. It declines 33/200 KRPvKR,
36/200 KRPvKP, 56/200 KPPvKP positions that way; the search held the win in 45 of 45 sampled
declined-and-won positions.

### Two measurement rules that cost a day

- **`tools/eg_suite.py --net` was a no-op.** It set `NNUE_NET`, and `nnue/net.py` only ever
  loaded its own default path, so every net-vs-net comparison scored the shipped net against
  itself. Fixed; `DEFAULT_PATH` honours the variable. Pair it with `NUMBA_CACHE_DIR`.
- **Endgame static error is anti-correlated with Elo** across our three nets. Medians against
  Stockfish depth 18 on the cached 500-position suite, by piece band:

  | net | 2-4 | 5-6 | 7-9 | games |
  |---|---|---|---|---|
  | net_215m (committed, no mix) | 150 | 46 | **308** | best of the three |
  | net_768 (endgame mix) | 145 | 48 | 224 | -30 |
  | blend (mix + cp loss) | **84** | 60 | 238 | -84 |

  The net with the worst endgame static error wins games. `eg_suite` stays a veto instrument;
  never rank a net by it. This is toby's "validation loss stops predicting Elo" in our numbers.

### What the fifty-move rule actually does

It bites only within about three plies of the claim. On 40 won 7-10-man positions at depth 8, the
engine plays a clock-resetting move 30/40 at halfmove clock 0, 31/40 at 92, 34/40 at 97, 39/40 at
99; the pre-fix engine is identical at 0 and 92, being blind. Reverse futility and null-move cut
off above the leaves where the clock expires, so the hundredth halfmove never enters the tree
until it is nearly on top of us. It is a correctness floor - never claim a win the referee will
call a draw - not a steering term. Steering from a distance needs toby's ADJ_HORIZON shape: draw-
bound at the node level when `halfmove_clock + horizon >= 100`. Unbuilt, and it needs its own A/B.

### Rated games say the same thing

`games/` holds 31 rated games (120 s + 0.5 s, engine named "Talos", "Hephaestus" in round 58):
+11 =9 -11, 50.0%. 26 of 31 reached <= 12 men, **all 11 losses did**, and 8 of the 9 draws ended
in an endgame (2 fifty-move, 4 threefold, 3 insufficient material). Same shape as toby's finding
that every loss of theirs reached a <= 16-piece endgame. The post-mortem - bucketing each loss as
search / horizon / eval / time / rule, with the `[%clk]` tags giving time per move - has not been
done yet and is the next thing worth doing.

## What we are building

Ordered by leverage. Each lands behind a named constant, OFF in the tree, with the measurement
that can see it.

### 1. Bucket and endgame-train the net

Ref: https://www.chessprogramming.org/NNUE

`nnue/arch.py` today: 768 -> 256 x2 -> 32 -> **8 output heads** -> 1, dual perspective, **no
king buckets**. Trained on 45M Lichess positions. Remaining: king buckets, endgame data, TB.

- [x] **Output buckets by piece count** (8), selected at `evaluate` time. `nnue/arch.py`
      `OUTPUT_BUCKETS` + `output_bucket()`; the final 32 -> 1 layer is `[8, 32]`, the runtime
      picks one head from `popcount(occupancy)`. Trainer (`nnue/model.py` `nn.Linear(32, 8)` +
      per-sample gather), `tools/train_nn.py`, `tools/export_nn.py`, `nnue/net.py` oracle all
      carry the bucket. Selection only - eval cost is one head. verify_nnue: engine == numpy
      twin within 2 cp, accumulator exact.
- [~] King-bucketed HalfKA feature transformer - built, trained to `nnue/model_kb.pt`, **not
      exported or shipped**. It needs a games A/B, not a static-error comparison (see the
      2026-09-09 section). KB=16, `king_bucket(sq) = 4*(rank//2) +
      file//2`). BUILT and verified on a smoke net; from-scratch retrain running (2026-09-08,
      `nnue/model_kb.pt`, mix-frac 0.15). Chose a flat single transformer over the ModuleList
      sketch below:
  - `nnue/arch.py`: `KING_BUCKETS=16`, `PLANE=768`, `FEATURES = KB*PLANE = 12288`;
    `feature_index(perspective, own_king_sq, colour, type, sq)` folds the bucket into the flat
    index. `king_bucket()`.
  - net.npz: `ft_weight_t [FEATURES, ft_out]` int16 (~6.3 MB), one shared scale - no shape
    change to the export/quant code beyond the width.
  - `nnue/model.py`: transformer is one `nn.EmbeddingBag(FEATURES+1, ft_out, mode="sum",
    padding_idx=FEATURES)` + a separate `ft_bias`; forward takes the two perspectives'
    `[B, 32]` active-feature-index lists (padded with FEATURES). Sparse, no dense 12288 plane.
  - `nnue/net.py`: `feature_lists(packed, stm)` builds the `[N, 32]` index lists (the numpy
    oracle); `accumulators()` gathers rows of a `[FEATURES+1, ft_out]` padded weight.
  - `tools/train_nn.py`: `feature_index_batch()` - on-device torch twin of `feature_lists`,
    argsort trick to left-pack the set bits, no ragged ops.
  - `nnue/accumulator.py`: `feature_index` takes `own_king_square`; `fill_accumulator_side`
    rebuilds one half; `update_feature(acc, wk, bk, ...)` both halves; `update_feature_side`
    one half.
  - `nnue/movegen.py make_move`: accumulator deltas moved to one block after the bitboards
    settle. Non-king move -> `update_feature` both halves (buckets unchanged). King move ->
    `update_feature_side` the OTHER half for every changed piece (its bucket is unchanged),
    then `fill_accumulator_side(new.pieces, new.acc, colour)` to rebuild the mover's half.
    No `acc_bucket` field needed - the branch is just `piece == KING`.
  - Verified: `verify_nnue` oracle max 0.5 cp / accumulator 0 of 6013 disagree; quant vs torch
    rmse 0.00 cp; perft / verify_movegen / verify_zobrist all pass; cold import 10 s (< 90 s
    budget - numba can't disk-cache the 6 MB weight global, costs ~1 s/start).
- [x] FT_OUT back to 256 (arch default; the shipped net had been trained at 128). Not widening
      past that yet.
- [x] Endgame data - **built and rejected**. `tools/label_eg.py` labels a <= 16-piece slice
      (decisive positions kept, mates mapped to +- mate_cp) and `tools/train_nn.py --mix /
      --mix-frac` oversamples it. The mixed net measured **-30** and a further cp-magnitude loss
      term (`--cp-weight`, to stop the net saturating past "winning") measured **-84**. Both
      improved endgame static error and both lost games; see the 2026-09-09 section. Do not run
      a third endgame retrain without a reason that is not static error.
- [x] Syzygy - **3-4-man shipped, 5-man rejected**. `nnue/tablebase.py` ranks the root move and
      `tb_probe_score` gives negamax / quiescence an exact WDL at every <= 4-man node. A 5-man
      WDL slice measured -48 and is parked in `nnue/syzygy5/`. The Polyglot book is untried and
      toby measured theirs at ~-20 cp per firing on curated starts, so it stays unbuilt.
- Measure: `tools/ab_nnue.py` (net vs linear, both colours) for the net changes;
  `tools/bench.py` Elo vs an older copy of `nnue/agent.py`; `tools/endgame_bench.py` as a veto
  (a regression there rejects the change even if the A/B likes it). Judge by games, not val
  loss.

### 2. Search accuracy terms (one bundle)

Refs: https://www.chessprogramming.org/History_Heuristic#Continuation_History
      https://www.chessprogramming.org/Singular_Extensions
      https://www.chessprogramming.org/Static_Exchange_Evaluation

Each is individually below A/B resolution; landing together, they unlock harder LMR. Split into
**2a (toby-promoted, low risk)** and **2b (needs its own A/B)**. All behind named constants,
OFF in the tree, flags-off path held bit-identical by `tools/nodebench.py`.

**2a - promoted elsewhere, take as given, one A/B for the bundle:**

- [ ] **TT 2-slot buckets** (`TT_BUCKETS`). Even slot keeps the deeper entry, odd always takes
      the store; a probe checks both. We are single-slot always-replace - a deep entry dies to
      key traffic. ~exact, cheap.
- [ ] **SEE prune in the main search** (`SEE_MAIN`). Skip a capture losing more than
      `20 * depth^2` on `see_ge`, at depth <= 5, never the first move. We already have the
      allocation-free `see_ge`; this is one call site. (Quiescence SEE is item 3, done.)
- [ ] **Dynamic null-move R** (`NMP_V2`): `R = 3 + depth//4 + min((static - beta)//200, 3)`,
      only when `static >= beta`, skipped when the TT holds an upper bound below beta. Our R
      already carries an eval-margin term - re-tune to this shape, do not rebuild.
- [ ] **Null verification** (`NMP_V2B`): on a null cutoff at depth >= 10, re-search the node
      reduced before trusting it; disable null below `ply + 3*null_depth//4` in that subtree
      (Stockfish `nmpMinPly`). Protects the deep nodes a wrong null cutoff poisons.

**2b - build and A/B here:**

- [ ] **Continuation history**: a 1-ply `(piece, to) -> (piece, to)` table, gravity update on a
      quiet cutoff. Used in quiet ordering, the LMR term (`r -= (history + conthist) // K`,
      clamped +-2, replacing the coarse step), and quiet futility / history pruning. No
      counter-move table today, so this is new. **toby rejected plain conthist** - only build
      it if it is the LMR-history feeder, not standalone.
- [ ] **Singular extensions** (`SINGULAR`): at depth >= 7 with an exact-or-lower TT bound at
      depth >= depth-3, re-search without the hash move at half depth, window ~2 pawns/ply
      below the stored score; if nothing reaches it, extend the hash move a ply. Cap at 6
      check+singular extensions per line.
- [ ] **Capture history + capture ordering** (`CAPTURE_ORDER`): SEE-losing non-promo captures
      drop below every quiet; winning/equal keep MVV-LVA; a capture-history tiebreak
      (gravity bonus on a capture cutoff).
- [ ] **Improving flag**: `static_eval[ply] > static_eval[ply-2]` off a per-ply static-eval
      stack. Gates RFP margin (`* (depth - improving)`), futility margins, LMP count, `+1` LMR
      when not improving. Default True at ply 0-1; skip the write after a null move. **toby has
      this OFF** - low confidence, cheap to try, drop fast if flat.

- Measure: `tools/nodebench.py` at depth 8 and 10 first - target <= 0.90x nodes for the
  bundle; below 0.95x or the wiring is doing nothing, debug before spending games. Then
  `tools/bench.py` Elo vs the pre-bundle copy.

### 3. Allocation-free SEE  [done - 2026-09-06]

Ref: https://www.chessprogramming.org/Static_Exchange_Evaluation

`agent.see()` (and the `nnue/` copy) allocated `np.empty(34)` on every call - in quiescence
that is the hottest loop in the engine. The one call site left, the quiescence capture filter,
only compared the result against zero.

- [x] `see_ge(board, move, threshold) -> bool` in `agent.py`, `reference.py`, `nnue/agent.py`:
      a scalar running balance (Stockfish form), no array, early exit. The attacker set is
      recomputed each step as `see()` does, so no per-piece x-ray bookkeeping.
- [x] `tools/verify_see.py`: the jitted and plain ports agree at every threshold, and at
      threshold 0 - the only one the engine uses - `see_ge` equals `see() >= 0` over the whole
      capture suite (20602 captures, 0 mismatches). Note: the Stockfish form is deliberately
      *not* `see() >= t` at arbitrary `t` (the defender is assumed to keep recapturing), so the
      test only holds it to `t == 0`. `see()` is kept for `move_ordering_score`, which wants
      the value.
- [x] Quiescence capture filter routed through `not see_ge(board, move, 0)`. We have no
      depth <= 5 main-search SEE prune (that is a TobyCoad feature, not ours) - nothing else to
      route.
- [x] `tools/nodebench.py --depth 6`: node counts and every bestmove / score byte-identical
      (185434 nodes, exact refactor confirmed). Import unchanged at ~19 s. knps delta too noisy
      to quote off the micro-bench; folded in on the exactness evidence, no A/B.

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

### 5. Lazy accumulator  [NPS, ~+25 Elo at toby's 120 Elo / node-doubling]

Ref: https://www.chessprogramming.org/NNUE (Lazy Updates)

`make_move` updates `board.acc` (`update_feature` x2-3) and `copy_board` copies the ~1 KB
accumulator on **every** child. A child cut by the TT, a repetition, or the null move before
any `evaluate` never needs it. toby profiled this at 15.4% of search time; `LAZY_ACC` is one of
their promoted flags.

- Spiked the clean form (a `parent` ref on the `Board` jitclass so `evaluate` walks back to a
  clean accumulator): **does not compile in numba 0.67** - `NotImplementedError: No definition
  for lowering DeferredType.acc = array(int32, 1d)` (can't write an array field through a
  recursive/optional jitclass ref).
- [x] **Pragmatic form** (`LAZY_ACC`): `make_move` skips `update_feature`, `copy_board` skips
  the acc copy (fresh `np.empty`), the child is marked `acc_dirty`; `evaluate` calls
  `fill_accumulator` once if dirty then clears the flag. Built it behind a compile-time
  `LAZY_ACC` flag across `nnue/{accumulator,board,movegen,agent}.py`.
  **Rejected - it roughly halves NPS.** `nodebench` depth 8: identical nodes / bestmove / score
  (exact, as expected), but wall time 1.29 s -> 3.49 s (least-contended run of 5), and the
  contended cluster went ~208 knps -> ~110 knps. The plan's per-node cost estimate above was
  wrong: this search takes a static eval at nearly every node (qsearch standing-pat + negamax
  RFP/null-move), so "rebuild once on eval" means almost every node pays a full
  `fill_accumulator` (~16 k int16 accumulate ops mid-game) in place of ~3 `update_feature`
  calls (~1.5 k ops) + a 2 KB `acc.copy()`. Pruned-before-eval children are far too small a
  slice to pay for that. Reverted; the incremental accumulator stays.
- **True lazy** (ply-indexed accumulator stack in `SearchState` + delta records, `evaluate(state,
  ply)` threaded through negamax / quiescence) is the only form that could win here, and only if
  it also skips the per-node rebuild - i.e. it needs a real make/unmake stack, not copy-make.
  Out of scope unless the search is restructured. toby's 15.4% figure must be against a
  make/unmake engine where the copy cost dominates; it does not carry to this one.

### 5b. NPS profile + spent leads  (2026-09-08)

Microbench per-node split (least-contended run, `scratchpad/microbench.py`), 26-piece position:
`evaluate_accumulator` ~1270 ns (~45%), `make_move` incl `copy_board` ~690 ns, `argsort` move
order ~580 ns, `legal_moves` ~140 ns, `is_check` ~5 ns.

- **Skip-zero L1 matmul - REJECTED.** ~68% of feature-transformer activations clip to exactly 0,
  so L1 (32x512, the dominant loop) should skip them. Tried both forms, bit-exact, same-process
  A/B: gather (`L1_WEIGHT[j, live_index[t]]`) **~2x slower**, column-major (`L1WT[i] * a` into a
  32-wide `hidden1`) **~50% slower**. The dense contiguous reduction vectorises to a tight NEON
  FMA chain; L1 is only 16 k MACs, already cheap, and the branch + bookkeeping + non-contiguous
  access of any sparse form costs more than the ~68% arithmetic it saves. Stockfish's sparse L1
  wins because their L1 is far bigger and hand-written in AVX-512 with `vpdpbusd` + compress.
  numba on NEON has neither. Same shape of result as the int8 tail.
- **Still open, smaller:** lazy ("pick next best") move ordering to drop the per-node
  `np.argsort` alloc + full sort (cut usually comes in the first 1-3 moves); a per-ply `Board`
  scratch pool to drop the `copy_board` jitclass allocation. Both are single-digit-to-low-teens
  percent at best. The real lever is make/unmake (see 5's note); everything short of it is
  picked over.

### 6. Platform-correctness audit  [stops thrown games, not average Elo]

toby lost rated games to three of these. Audit `nnue/agent.py` against each; a fix here is
worth more than a Tier-2 heuristic because it converts losses, not draws.

- [ ] **Repetition seeded from game history** (`REPETITION_TWOFOLD`). The referee runs
  `board.outcome(claim_draw=True)` and python-chess lets the mover claim as soon as ONE move
  would make a third occurrence - counting the *game*, not just the search path. Our
  `repetitions()` starts from 0 at the root; it does not know a position is already at 2 from
  earlier real moves. Seed the per-ply hash trail (or a small game-history multiset) into
  `get_move`, and while winning score any position seen once before as a draw. toby: *lost a
  won game with mate on the board* (their round 11).
- [ ] **Null-after-null**. Two nulls restore the zobrist key; our path-based `repetitions()`
  can then see a false repeat and score the grandchild a draw. `tools/nodebench.py` check that
  NMP still cuts at depth >= 6; if not, forbid a null move directly after a null move.
- [ ] **BLAS thread pinning in the bundle**. Set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
  `MKL_NUM_THREADS` / `NUMEXPR_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS = 1` **before**
  `import numpy` in the shipped `agent.py`. Our runtime eval is pure njit (no BLAS), so this is
  insurance for the parallel-game referee, not a known bug - add it to `tools/bundle_nnue.py`'s
  output or a header the bundler prepends.
- [ ] **Import budget under load**. `time` the real `candidates/nnue` import; the platform box
  is ~2.1x a dev Mac and gives 90 s. Locally ~8-19 s, so headroom is fine, but confirm on the
  first upload's validation log and watch it on every njit branch added.
- (600-ply -> draw: `harness/referee.py` already does this. toby's fork shipped a stale
  300 + material harness; ours is correct.)

## Measurement tools

- `tools/nodebench.py` - fixed FENs, fixed depth, `nodes bestmove score` per position; the
  exact-refactor gate (item 3, and the flags-off path of item 2).
- `tools/bench.py` - near-level openings, both colours, score + Elo + 95% CI; the heuristic
  gate. Always vs an older copy of the same engine.
- `tools/ab_nnue.py` - in-process net vs linear, a quick read on whether the net earns its
  slower nodes.
- `tools/endgame_bench.py` - <= 16-piece positions labelled by an engine, mean cp loss at a
  fixed movetime; a **veto** on eval changes, not a ranking signal.
- `tools/eg_suite.py` - static error and move cp-loss per piece band, against Stockfish labels,
  with a cached suite in `tools/suites/` for reproducible runs. `--net` works as of 2026-09-09
  (it was silently scoring the shipped net before). **Veto only** - its static error is
  anti-correlated with Elo across the nets we have measured.
- A fixed-depth probe over generated positions, when the question is whether a rule changes what
  the engine plays. It is deterministic and immune to machine load, and it resolves in minutes
  what an 80-game bench cannot resolve at all: that bench carries a ~+-60 Elo CI, so it cannot
  see a 20 Elo change, and it has almost no power over something that fires in a few games.
- `import` wall time - `time uv run python -c "import nnue.agent"` on every bundle; every njit
  branch is numba compile time against the 90 s init budget.

**Retrain gotcha:** `nnue/net.npz` is a module global that `@njit(cache=True)` functions in
`nnue/accumulator.py` freeze into their on-disk cache. After any retrain / re-export, numba
keeps running the *old* weights until the cache is cleared:
`find . -name '*.nbc' -o -name '*.nbi' | xargs rm` (and drop `__pycache__/`). verify_nnue's
oracle diff blowing up to hundreds of cp with the accumulator check still passing is the tell.

---

Carried from `docs/plan.md`: every tuned number traces to a measurement we ran; change behind a
named constant; exact-equivalent change -> nodebench matches; heuristic change -> bench Elo >= 0
inside CI, or revert it and write down why.

## Post-mortem of the 30 rated games in games/ (2026-09-09)

Stockfish 18 depth 20 on every position (rounds 58, 69, 70 at 0.8 s/position after depth 20
stalled), clocks from the [%clk] tags, then each decisive position replayed through the real
`deepen` loop at the game's budget, x4 and x16, with the current tree and with the candidate
builds. Scripts and JSON: scratchpad `pm/` (annotate.py, analyse.py, probe.py, probe2.py).

**Which build played.** Only `candidates/o8-tb` (45M net, root-only Syzygy, no in-search probe,
no fifty-move rule, no game-trail repetition; built 09-07 15:57 BST) reproduces the R64, R67 and
R88 decisive moves; the current tree does not. Confirm against the upload log before treating
these games as evidence about the shipped build.

**Buckets, 11 losses + 3 thrown draws = 14 decisive events:** eval 10, horizon 2, rule 1,
search 1, time 0 (primary). 6 of the 11 losses were decided at 19-32 men (R60, R78, R80, R81,
R82, R85: our acpl above the opponent's in every one, same moves at every budget); "all losses
reached <= 12 men" is a lost engine playing on to mate. 6 of the 9 draws were never winning
(R62, R63, R66, R75, R76, R84) - held, not given away.

| R | event | men | spend / left | cause | bucket |
|---|---|---|---|---|---|
| 64 L | 41...Kb6 | 9 | 0.9 s / 25 s | drawn KR+2P v KR+3P scored -120..-250, all moves alike | eval |
| 69 L | 57...Rc3, 62...Kd1 | 12, 9 | 0.95, 0.54 s / 19 s | +541 -> 0, then 0 -> -288; scores flat +160..+290 for moves worth +738 vs 0 | eval |
| 70 L | 40-53 | 16 -> 10 | 0.7-1.2 s / 25-33 s | 20-60 cp lost per move in R+B v R+N; x4 time fixes 2 of 3 probes | eval (time secondary) |
| 73 L | 104...Bb3, 105...Bf7 | 8 | 0.4-0.5 s / 14 s | B+2P v R+2P fortress held 60 moves; SF -55, ours -440 for every move | eval |
| 88 L | 83...Qxf4 | 7 -> 4 | 0.66 s / 18 s | into lost KPvKP; o8-tb scores it -15, every build with the in-search TB rejects it | horizon (fixed by shipped probe) |
| 58 D | threefold at +2500..+4100 | 10 | 1 s / 30 s | mate in 25 needs Kh3xh4; ours +450..+690 flat, bishop/king shuffle; no game trail | rule (root cause eval) |
| 67 D | 53...Bg6, 56...fxg3 | 11 | 4.3, 1.2 s / 25 s | +481 and +516 -> 0; eval +8..+69 for +1178; x16 time picks the blunder | eval |
| 74 D | 32.Rh3 | 21 | 2.8 s | 16-ply perpetual; ours +204 even at x16 | horizon |
| 81 L | 42.Rg2, 44.Qe3 | 19 | 0.7-0.8 s / 30 s | Rg2 only fixed at x16 | search |

**Leads.** Draw-score shaping: killed - no decision in 30 games turned on the +-30, and no
lost fortress ever had a repetition on offer (0 repeated positions on our move in R73).
Fifty-move steering: killed - nothing was decided by the clock (R73 hit hm 95, the opponent
reset). Time: the multipliers named in the lead do not exist; the budget is a flat 1/40 of the
clock, so median spend falls 2.85 s (>= 20 men) -> 0.47 s (<= 8 men), 47% of our <= 12-men moves
took < 1 s with > 15 s left, and games end with 15-40 s unused. Never the primary cause; a
secondary lever with a free resource (x4 helped in 5 probes, hurt in 2, no effect in R64/R73/R88).
7-9 men eval: confirmed as the dominant cause - every endgame event is 8-12 men with the eval
250-1100 cp off and flat across moves. KB=16 net: no evidence either way.

## Endgame net + time budget: the overnight matrix (2026-09-10)

Built after the post-mortem. `nnue/accumulator.py` now loads a second net for positions with
`EG_MEN` = 12 men or fewer (`nnue/net_eg.npz`, optional - absent, the main net serves and the
build is bit-identical to a single-net one: 4000 static evals and 11 fixed-depth searches
identical, `tools/verify_nnue.py` oracle 0.5 cp, 8078 incremental moves exact). `make_move`
refills the accumulator when a capture crosses into the band. `tools/filter_shards.py` cuts
the <= 12-men slice of the 215M corpus (29.9M positions, 13.9%) into `data/endgame12`.
`tools/bundle_nnue.py --net-eg` ships it. Gotcha: numba freezes the weight arrays into its
cache, so swap a net and delete `nnue/__pycache__/*.nb?` (or use a fresh NUMBA_CACHE_DIR).

Nets, all 768 -> 256x2 -> 32 -> 8 heads, trained on `data/endgame12` (8 val shards held out):

| net | start | data | best val mse |
|---|---|---|---|
| eg12 | warm from model_215m | endgame12 | 0.00822 (main net ~0.0086 on the same val) |
| eg12s | scratch | endgame12 | 0.00818 |
| eg12m | warm from eg12 | endgame12 + 30% batches from data/endgame_mix (2.7M SF depth 12/14, <= 12 men) | 0.00859 |
| eg12sm | scratch | same mix | 0.00860 |

Benches vs `candidates/eg_base` (new code, no endgame net), 8 workers, endgame = 50 games of
tools/endgame_bench.py, openings = 80 games of tools/bench.py:

| candidate | control | endgame | openings |
|---|---|---|---|
| eg12 (net only) | 5 s + 0.1 | 48.0%, -14 [-80, +51] | 50.6%, +4 [-68, +77] |
| tm (time only: soft cap 1/20 at <= 12 men) | 30 s + 0.2 | 46.0%, -28 [-80, +23] | 58.1%, +57 [-6, +124] |
| eg12m **+ tm** | 5 s + 0.1 | 58.0%, +56 [-4, +120] | 54.4%, +30 [-38, +102] |
| eg12s **+ tm** | 5 s + 0.1 | 50.0%, 0 [-52, +52] | 59.4%, +66 [-1, +138], 1 flag (machine under 3 trainings + 8 workers) |
| eg12sm **+ tm** | 5 s + 0.1 | 57.0%, +49 [-6, +107] | 53.1%, +22 [-45, +90] |

**Confound:** the time change was edited into the tree before chain 2 bundled eg12m/eg12s/
eg12sm, so those three rows measure net + time against base. Read: the plain endgame nets do
nothing on the endgame suite (eg12 -14, eg12s 0); both Stockfish-mixed nets are +50ish there
with the lower CI at about -5, twice independently. Clean runs (eg12m vs tm, differing only in
net_eg.npz; tm vs eg_base replicated at 30 s + 0.2) are in the 09-10 daytime log below.

### The clean runs (2026-09-10 midday)

`candidates/eg12m` vs `candidates/tm` - identical bundles apart from `net_eg.npz`, so this is
the endgame net alone, with the time change present on both sides:

| suite | score | Elo |
|---|---|---|
| endgame (50) | +9 =40 -1, 58.0% | **+56 [+15, +99]** |
| openings (80) | +28 =24 -28, 50.0% | -0 [-65, +65] |

The lower bound clears zero and the openings are a wash - the Stockfish-mixed endgame net is a
real gain, and it is free where it does not fire. Nine wins to one loss with forty draws: it
converts, it does not out-tactic.

**Why it works - and it is not accuracy.** On the nine decisive positions from the post-mortem
(scratchpad `evaldiag.py`), asking whether the static eval ranks Stockfish's move above the one
we played, every net scores 5-6 of 9 - the mixed nets are no better than the shipped one. What
changed is the conversion gradient (`gradient.py`):

| position | main | eg12 plain | eg12m MIX | eg12sm MIX |
|---|---|---|---|---|
| K+R vs K | +72 | +201 | +419 | +463 |
| K+Q vs K | +585 | +790 | +952 | +920 |
| K+2R vs K | +384 | +474 | +804 | +748 |
| K+P vs K (won) | +10 | +17 | +283 | +84 |
| K+R vs K+R (drawn) | +3 | +6 | +2 | +14 |
| K+B vs K (dead) | +5 | +10 | +24 | +11 |

The shipped net rates a whole rook up at +72 - it never learned a won ending is won, exactly
the "+57 cp for KRvK" this file recorded on 09-07 - so the search had no reason to steer into
those endings. The Stockfish mix keeps decisive scores unclamped and lifts them to +419/+463
while leaving dead positions at zero. Move-ranking accuracy is unchanged; the gradient is what
moved. That also re-confirms the standing rule: static error would have ranked these nets wrong.

Next: `data/endgame_sf` (the 2.7M Stockfish-labelled positions alone, staged as shard_*.npz),
100% of batches, warm from eg12m (`eg12p`) and from scratch (`eg12ps`), benched against eg12m.
