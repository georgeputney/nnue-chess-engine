# 03 - transposition table, PVS, killers + history

Commit `a7ecf05` (`git show a7ecf05`) - [docs/plan.md](../../docs/plan.md) Phases 7-9.

**What it adds.** A transposition table keyed on the Zobrist hash, replacing shallower entries
and reusing the stored move for ordering; principal variation search (full window on the first
move, a null-window scout on the rest, re-search only if it beats `alpha`); and quiet-move
ordering - `KILLERS` (the last two quiet cutoff moves at each remaining depth) and `HISTORY`
(a `[piece_type][to_square]` table with a gravity update so scores saturate rather than run
away). Both persist across a game.

**How it works.** The sort key is now TT move, then MVV-LVA captures, then killers, then
history - everything before this stage searched quiets in generation order.

**Measured** (`docs/plan.md`, Phase 9): `nodebench --check` vs Phase 8 - every score
identical, node count down ~4% at depth 4 (the transposition table and PVS are exact
refactors). Bench vs the starter's minimax baseline, 80 games: +14 =62 -4, **+44 Elo [+8,
+80]** - in the same band as stages 02-03 because the material-only eval and the draw rate
mask what the search alone is worth here.

**Run it:**

```
uv run python -m harness.play --white stages/03-tt-pvs-history --black stages/04-pesto-tapered
```
