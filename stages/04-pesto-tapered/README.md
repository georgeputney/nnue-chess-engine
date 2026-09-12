# 04 - PeSTO tapered piece-square eval

Commit `2b9dfe6` (`git show 2b9dfe6`) - [docs/plan.md](../../docs/plan.md) Phase 18 (pulled
forward - see `docs/plan.md`'s note on evaluation build-out).

**What it adds.** The first positional evaluation. PeSTO's midgame/endgame piece-square
tables replace the flat material count, material folded into the same tables, the king
tapered like every other piece instead of special-cased; `evaluate` carries a midgame and an
endgame accumulator and blends them by game phase. Plus four hand-picked terms: a 10 cp tempo
bonus, slider mobility (bishop/rook/queen, weighted attacked-square count), the king scored as
a phantom queen at midgame (open lines around it = danger, fading out as it should get active
in the endgame), and a 12 cp doubled-pawn penalty.

**How it works.** Search is unchanged from stage 03 - every gain here is `evaluate` alone
giving the search something to actually distinguish.

**Measured**: A/B vs the pre-PeSTO snapshot, 80 games: **+57 Elo [+12, +104]** (21-51-8),
roughly +25 over the PeSTO tables alone. Vs the starter's minimax baseline: **+338 Elo [+272,
+431]**. These weights are hand-picked starting points - splitting them midgame/endgame
properly and fitting them is stage 05's job.

**Run it:**

```
uv run python -m harness.play --white stages/04-pesto-tapered --black stages/05-texel-tuned
```
