# 08 - single-net NNUE

Commit `711821a` (`git show 711821a`, "Replace the tapered eval with a trained NNUE") -
[docs/nnue-plan.md](../../docs/nnue-plan.md).

**What it adds over stage 07.** The linear tapered evaluation is replaced by a trained NNUE: a
dual-perspective feature transformer (768 -> 256x2), a small tail over both perspectives with
output buckets by piece count, an incremental int accumulator carried on the board so a leaf
eval is a few small matrix multiplies instead of a board scan, and a Syzygy 3-4-man tablebase
probe. This snapshot ships the single all-position net (`net.npz`, trained on 215M Lichess
positions) - the one further refinement past this point, an endgame-specialised second net for
positions with few pieces left, is what actually shipped in the competition and lives in
[`engine/`](../../engine) (`net_eg.npz`) and [docs/nnue-plan.md](../../docs/nnue-plan.md), not
as its own rung here: it's a specialised head on the same architecture, not a new concept.

**Why the single net matters on its own.** The all-position net rates a won rook-up ending at
+72 cp - accurate enough for move ranking, but flat where a search needs a gradient to steer
into a winning line. That's exactly the shape of gap the endgame net exists to close; this
stage is what NNUE alone bought before that fix.

**How it's built.** A flattened build of `engine/` via `tools/bundle_engine.py --net-eg
/nonexistent` (the single-net form). If `engine/` has moved on, rebuild with the same command.

**Measured.** Beat a comparable classical-eval fork 100% and lost 0-6 to a stronger NNUE fork
at 8 s + 0.1 s early in development (see stage 07's README) - the gap that motivated this
stage. The full, endgame-net-equipped build (`engine/`) is what played the competition:
**2314 Elo**, 49W-42L-23D-8void over 122 rated games ([docs/games.csv](../../docs/games.csv),
individual game PGNs in [docs/games/](../../docs/games)).

**Run it:**

```
uv run python -m harness.play --white stages/08-single-nnue --black stages/07-numba-classical
```
