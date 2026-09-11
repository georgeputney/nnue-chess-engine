# 07 — the numba bitboard rewrite (final classical form)

Port began at commit `dca8816` (`git show dca8816`); this snapshot is the classical engine's
final state, right before its evaluation was replaced by the trained net in stage 08.

**What it adds.** `python-chess` is ~10-50k nps — a ceiling `docs/plan.md` flagged from the
start as the reason to eventually rewrite the core. `board.py` / `movegen.py` / `attacks.py` /
`move.py` / `zobrist.py` / `bitboard.py` here are a numba-jitted bitboard layer mirroring the
`python-chess` API (magic bitboards for sliders, a de Bruijn lookup for bitscan, SWAR
popcount), and `agent.py`'s search and evaluation are `@njit` too, so a node never leaves
compiled code. `reference.py` is the plain `python-chess` engine this was ported from — same
function names, kept as the golden twin the `verify_*` tools check against and the runtime
fallback if numba fails to compile on the platform. On top of the port: static exchange
evaluation for move ordering and quiescence, a contempt factor, and passed-pawn / king-activity
eval terms.

**How it works.** Every `@njit` function is warmed once at import so compilation lands in the
90 s init budget, not the match clock; a node still walks the identical alpha-beta / PVS /
pruning shape as stage 06, just against bitboards instead of `python-chess` objects.

**Measured** ([docs/nnue-plan.md](../../docs/nnue-plan.md)): 6 games each, 8 s + 0.1 s, both
colours, against two other public forks of the same starter kit. **100% (+6 =0 -0)** against a
numba engine with a hand-tuned classical eval and no trained net — this stage's search
skeleton sweeps a comparable classical engine. **0% (+0 =0 -6, checkmated every game)** against
a numba engine shipping a trained NNUE. That gap — not search, not speed, the evaluation — is
what motivated stage 08.

**Run it:**

```
uv run python -m harness.play --white stages/07-numba-classical --black stages/08-nnue
```
