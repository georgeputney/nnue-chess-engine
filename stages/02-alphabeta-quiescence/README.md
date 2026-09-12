# 02 - negamax, alpha-beta, iterative deepening, MVV-LVA, quiescence

Commit `717df8f` (`git show 717df8f`) - [docs/plan.md](../../docs/plan.md) Phases 2-6.

**What it adds.** Negamax replaces the 1-ply pick; alpha-beta prunes it (an exact refactor -
same result, fewer nodes); iterative deepening means there's always a move ready when the
clock runs out; captures are tried first, ordered by MVV-LVA; and at the horizon,
`quiescence_search` keeps searching captures (every reply if in check) until the position is
quiet, with delta pruning to skip captures that can't reach `alpha` even generously.

**How it works.** `evaluate` is unchanged from stage 01 - the gain here is entirely search
depth and not reading a mid-exchange position as final.

**Measured** (`docs/plan.md`, Phase 6, vs the starter's minimax baseline, 80 games): +15 =62
-3, **+53 Elo [+18, +88]**, up from +26 (crossing zero) at Phase 4. Losses dropped 11 -> 3; the
draws left are the material-only eval having no gradient in quiet positions - Phase 18's job.
Not an exact refactor end to end: depth-4 node count rises ~0.6M -> 1.45M because quiescence
searches past the fixed horizon.

**Run it:**

```
uv run python -m harness.play --white stages/02-alphabeta-quiescence --black stages/03-tt-pvs-history
```
