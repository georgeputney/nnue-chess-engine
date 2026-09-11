# 01 - material eval + a 1-ply pick

Commit `5162cc6` (`git show 5162cc6`) - [docs/plan.md](../../docs/plan.md) Phase 1.

**What it adds.** The floor of the ladder: `evaluate(board, side)` sums piece values (100 /
320 / 330 / 500 / 900) from the side to move's point of view, and `get_move` plays whichever
legal move maximises it one ply deep. No search beyond that ply, no positional knowledge at
all - a symmetric position scores 0, a knight up scores ~+320.

**How it works.** `python-chess` supplies the board and legal-move generation; this file is
just the loop over `board.legal_moves` and the material count.

**Why it's here.** Every later stage in this ladder is measured against the stage before it -
this is the one nothing is measured against, the floor everything else builds up from.

**Run it:**

```
uv run python -m harness.play --white stages/01-material-1ply --black stages/02-alphabeta-quiescence
```
