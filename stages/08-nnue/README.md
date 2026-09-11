# 08 — NNUE

The engine this repo is actually about. Unlike stages 01-07, this isn't vendored history —
it's a flattened build of [`engine/`](../../engine), the live, canonical source, produced by
`tools/bundle_engine.py` so it can run through the same harness as every other stage. If
`engine/` has moved on since this snapshot, re-run:

```
uv run python tools/bundle_engine.py --out stages/08-nnue
```

**What it adds over stage 07.** The linear tapered evaluation is replaced by a trained NNUE: a
dual-perspective feature transformer (768 → 256×2), a small tail over both perspectives with
output buckets by piece count, an incremental int accumulator carried on the board so a leaf
eval is a few small matrix multiplies instead of a board scan, and a Syzygy 3-4-man tablebase
probe. A second, endgame-specialised net (`net_eg.npz`, trained on a Stockfish-mixed slice of
positions with ≤ 16 men) takes over below that piece count — the shipped all-position net rated
a won rook-up ending at +72 cp, not a gradient a search can steer on; the endgame net fixes the
gradient, not the move-ranking accuracy. Full story, including what was tried and rejected
(wider nets, a 5-man tablebase slice, a Polyglot book, correction history) in
[docs/nnue-plan.md](../../docs/nnue-plan.md) and [docs/writeup.md](../../docs/writeup.md).

**Measured.** Beat a comparable classical-eval fork 100% and lost 0-6 to a stronger NNUE fork
at 8 s + 0.1 s early in development (see stage 07's README) — the gap that motivated this
stage. In the actual competition: **2314 Elo**, 49W-42L-23D-8void over 122 rated games
([docs/games.csv](../../docs/games.csv), individual game PGNs in
[docs/games/](../../docs/games)).

**Run it:**

```
uv run python -m harness.play --white stages/08-nnue --black stages/07-numba-classical
```
