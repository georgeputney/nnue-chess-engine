# 05 - Texel-tuned evaluation weights

Commit `4f36faf` (`git show 4f36faf`) - [docs/plan.md](../../docs/plan.md) Phase 19.

**What it adds.** `tools/tune.py`: every weight in `tables.py` (material, piece-square tables,
midgame/endgame halves, the other eval terms) stops being hand-picked and starts being fitted.
Positions are labelled offline by an engine; the loss is the MSE of `sigmoid(our_eval / K)`
against the label, `K` fit once by minimising over it; the weight vector is optimised by
coordinate descent - nudge each parameter +/-1, keep the change if the loss drops, repeat to
convergence.

**How it works.** `agent.py`'s search and `evaluate` shape are unchanged from stage 04;
`tables.py` is now a generated artefact of `tools/tune.py`, not typed by hand. The tuner and
its dataset live outside `agent.py` entirely - only the resulting numbers ship, which is why
`agent.py` here is byte-for-byte the same shape as stage 04's, just reading different numbers.

**The discipline, not just this stage:** the version that ships is the one that wins games
against `tools/bench.py`, not the one with the lowest tuning loss offline - the two diverge,
and re-running the bench after every retune is what catches it.

**Run it:**

```
uv run python -m harness.play --white stages/05-texel-tuned --black stages/06-full-pruning
```
