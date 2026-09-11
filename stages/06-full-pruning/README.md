# 06 - the full pruning suite; end of the pure-python-chess engine

Commit `f074999` (`git show f074999`) - [docs/plan.md](../../docs/plan.md) Phases 10-18.

**What it adds.** Everything between the Texel tuner and the numba rewrite, folded into one
snapshot: null-move pruning, check extensions, reverse futility pruning, late move reductions,
late move pruning + move-loop futility, internal iterative reduction, aspiration windows,
mate-distance scoring, and - the last two evaluation terms - a "far pawn" virtual piece type
(a pawn on the far side of the board from its own king scores on its own row: pawn storms and
weak shelter play differently) and a generalised "friendly pawns ahead" term replacing the
flat doubled-pawn penalty. This is the last stage running on plain `python-chess` - everything
past here is the same engine ported onto the numba bitboard layer for speed.

**How it works.** Each of these landed individually behind its own measurement in
`docs/plan.md`; what's frozen here is the state after all of them, seeded so each addition
reproduces the prior behaviour until tuned separately (the far-pawn table starts as a literal
copy of `PAWN`'s rows).

**Measured**: `tools/tune.py --selfcheck` passes; `tools/nodebench.py --depth 6` is
bestmove/score-identical across the fixed suite bar one position off by 7 nodes (a
tripled-pawn case where the generalised per-piece sum correctly differs from the old
count-1 shortcut) - the structural changes here were verified exact, not just benched.

**Run it:**

```
uv run python -m harness.play --white stages/06-full-pruning --black stages/07-numba-classical
```
