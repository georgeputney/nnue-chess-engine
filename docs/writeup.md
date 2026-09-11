# Writeup

The retrospective. `docs/plan.md` and `docs/nnue-plan.md` are the build log this cites; this is
the narrative version - what was built, in what order, why, and what got measured and thrown
away along the way.

## The constraints, and why they shaped everything

AI Chessathon's sandbox: one core of an AMD EPYC 9V74, 2 GB, no network, no GPU, 120 s + 0.5 s
per move, a 90 s import budget, 50 MB unzipped. Two of those numbers decided the whole
architecture. `python-chess` runs at roughly 10-50k nodes/second - fine for a first engine, a
hard ceiling on how deep alpha-beta search alone can reach in a real game clock. And a 2 GB, no
GPU box rules out anything that wants to run a real neural net at inference time the way a
training-time setup would - the net has to be small enough, and cheap enough per call, to be
searched thousands of times a move on a CPU with no batching.

## The arc

**Material -> negamax -> alpha-beta -> a transposition table + PVS** ([`stages/01`](../stages/01-material-1ply)
through [`stages/03`](../stages/03-tt-pvs-history)) is the standard skeleton, each phase
measured against the one before it in `docs/plan.md`. **A tapered PeSTO piece-square eval**
([`stages/04`](../stages/04-pesto-tapered)) is the first evaluation that gives the search
something to actually distinguish - +57 Elo on its own, +338 against the starter kit's
minimax. **Texel tuning** ([`stages/05`](../stages/05-texel-tuned)) stops hand-picking those
weights: label positions offline, fit by coordinate descent against the labels, ship only the
numbers, keep the version that wins games over `bench/openings_bench.py` rather than the one
with the lowest tuning loss - those two diverge, and this project treats that divergence as the
whole reason to bench after every retune, not just log it once. **The full classical pruning
suite** ([`stages/06`](../stages/06-full-pruning)) - null-move, LMR, LMP, aspiration windows,
mate-distance pruning, a generalised pawn-structure term - is standard chess-programming
technique, each landed behind its own measurement.

**The numba bitboard rewrite** ([`stages/07`](../stages/07-numba-classical)) is where the
`python-chess` ceiling stopped being theoretical. A numba-jitted bitboard layer mirroring the
`python-chess` API, with the search and evaluation `@njit` too, so a node never leaves compiled
code. Measured against two other public forks of the same starter kit at 8 s + 0.1 s: **100%
(6-0)** against a numba engine with a hand-tuned classical eval and no trained net, **0% (0-6,
checkmated every game)** against a numba engine shipping a trained NNUE. The search skeleton
was sound - it swept a comparable classical engine on both colours - but the gap to a trained
net was not something more pruning was going to close.

## NNUE

The shipped evaluation ([`stages/08`](../stages/08-single-nnue), `engine/`) is a dual-perspective
network: a 768-feature transformer (`768 -> 256` per perspective, shared weights), an
own/other-perspective accumulator carried incrementally on the board (`make_move` updates a
few columns instead of a full forward pass), and a small tail (`512 -> 32 -> 32 -> 8 output
heads`) selected by piece count at evaluation time. Trained on 215M Lichess positions
(`tools/ingest_lichess.py`, streamed in parallel shards; `tools/train_nn.py`), quantised to
int16/int32 for the runtime (`tools/export_nn.py`), and checked bit-for-bit against a plain
numpy oracle (`bench/verify_nnue.py`) - the same discipline as the classical build, just with a
trained function standing in for a hand-tuned one.

**The retrain-cache gotcha**, because it cost real debugging time twice: `numba`'s
`@njit(cache=True)` functions in `engine/accumulator.py` freeze the net's weight arrays into
their on-disk cache. Swap `net.npz` without deleting `engine/__pycache__/*.nb?` and the engine
keeps running the *old* weights - `bench/verify_nnue.py`'s oracle check blowing up to hundreds
of centipawns while the accumulator check still passes is the tell.

### The endgame gap, and what actually closed it

A post-mortem of 30 rated games (rounds 58-88) replayed through the real search: of 14
decisive events, 10 were the evaluation, not the search or the clock. The all-position net
rated a won rook-up ending at +72 cp - accurate enough to rank moves, useless as a gradient to
steer a search toward converting it.

The fix that shipped was a **second net**, trained on a slice of positions with <= 16 men
mixing in 30% Stockfish-labelled data, selected by piece count at evaluation time
(`engine/net_eg.npz`, `engine/accumulator.py`'s `EG_MEN`), plus a loosened time budget below
that threshold. Together: **+89 Elo [+27, +157]** over the previous shipped build, 80 games at
30 s + 0.2 s. Net alone: +56 Elo on an endgame suite, ~0 from openings - free where it doesn't
fire. **The mechanism was not accuracy** - asked whether the static eval ranks the engine's own
best move above Stockfish's on nine decisive endgame positions, every net scored 5-6 of 9, the
mixed net no better. What moved was the *gradient*: K+R vs K went from +72 cp to +419 cp while
dead positions stayed at zero, giving the search a reason to walk into the ending instead of
shuffling around it.

### What was tried and rejected

Every one of these was built behind a flag, measured, and reverted or left unshipped - the
list is as much a part of the result as what shipped:

- **The lazy accumulator** (defer `make_move`'s incremental update, rebuild once on first
  `evaluate`) - bit-exact, but this search reads a static eval at nearly every node
  (quiescence stand-pat, RFP, null-move), so "rebuild once" made almost every node pay a full
  accumulator rebuild instead of a few incremental columns. Roughly **halved NPS**. Only a true
  make/unmake rewrite (not this copy-make engine's per-node board copy) could make a lazy
  accumulator win, and that's a different, much larger project.
- **A cp-magnitude-blended loss** for training: -84 Elo against the unmixed net, dead on
  arrival.
- **A 5-man Syzygy slice** (+28 MB of the 50 MB budget): -48 Elo head-to-head against the
  shipped 3-4-man-only probe, corroborated by two other runs. An exact 0 for a known draw
  outweighs the engine's own +/-30 contempt, so the search liquidated into certain draws, and
  every tablebase win scores `TB_WIN - ply` - no gradient to convert by, the same lesson as the
  eval gap above.
- **King-bucketed HalfKA features** (`nnue-king-buckets` branch) - built, verified exact, and
  retrained, motivated by the idea that a flat 768-feature net can't represent "this square is
  safe *because of where the king is*". Parked on its own branch, not merged into what shipped.
- **Wider nets** (768->1024): a real drop in nodes/second for effectively no Elo - width is a
  tax this engine pays per node, thousands of times a move; bucket the output, don't widen the
  input.
- A **Polyglot opening book**: measured negative against curated near-level starting positions.
- **Correction history** and a standalone **continuation history** (without also feeding LMR):
  both measured negative or flat when tried in isolation.

`bench/eg_suite.py`'s static-error metric is explicitly **a veto, never a ranking signal** -
across three endgame nets, the one with the *worst* static error was the one that won games; a
net with much better static error lost 84 Elo. Every eval change here was judged by games.

## The competition

`docs/games.csv` is the full 122-round record - a starting rating of 1498, decisive games
through checkmate more often than any other termination, and a final rank of **#48 of 465
entries (top 10%)** at **2314 Elo**. `docs/games/` has the individual game PGNs.

## Lessons

- **Measure the exact thing, not a proxy for it.** Static evaluation error and tuning loss both
  looked like reasonable proxies for playing strength and both pointed the wrong way at least
  once. Nothing shipped here without a same-shape A/B against real games.
- **An architecture change earns its keep by removing a ceiling, not by being fashionable.**
  The numba rewrite happened because `python-chess`'s node rate was the actual bottleneck,
  confirmed by depth-limited comparisons before a line of the bitboard layer was written.
- **A rejected idea is not wasted effort if it's written down.** Several of the rejected items
  above look, on paper, like they should have worked. They didn't, on this engine, with this
  training data, at this scale - and the next person (including a future version of the same
  project) gets to skip re-discovering that the slow way.
