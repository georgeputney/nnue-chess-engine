# The build, one measured stage at a time

Every rung here is a runnable snapshot with its own `README.md` — what it adds, how it works,
and the measurement that justified it at the time. `stages/01-material-1ply` is the floor
(near-random, legal moves only); `stages/08-nnue` is a flattened build of
[`engine/`](../engine), the engine that actually played the competition. Nothing skipped a
measurement: an exact refactor (alpha-beta, the transposition table, PVS, aspiration windows)
had to match `bench/nodebench.py`'s node counts and best moves bit for bit; a heuristic change
(an eval term, a pruning margin, a new net) had to clear zero on `bench/openings_bench.py`'s 95%
confidence interval against the stage before it, or it didn't ship. `docs/plan.md` and
`docs/nnue-plan.md` are the full build log this table summarises.

| Stage | Adds | Measured |
|---|---|---|
| [01-material-1ply](01-material-1ply) | material eval, 1-ply pick | the floor — nothing to compare against yet |
| [02-alphabeta-quiescence](02-alphabeta-quiescence) | negamax, alpha-beta, iterative deepening, MVV-LVA, quiescence | +53 Elo [+18, +88] vs the starter's minimax |
| [03-tt-pvs-history](03-tt-pvs-history) | transposition table, PVS, killers + history | +44 Elo [+8, +80] vs the starter's minimax; TT/PVS exact vs stage 02 |
| [04-pesto-tapered](04-pesto-tapered) | tapered PeSTO piece-square eval, mobility, king safety | +57 Elo [+12, +104] vs the pre-PeSTO snapshot |
| [05-texel-tuned](05-texel-tuned) | offline-fitted eval weights (`tools/tune.py`) | ships only if it beats the prior version's bench score, not the lowest tuning loss |
| [06-full-pruning](06-full-pruning) | null-move, LMR, LMP, aspiration, mate-distance, far-pawn eval | each landed on its own bench line in `docs/plan.md`; structural changes nodebench-exact |
| [07-numba-classical](07-numba-classical) | ported onto a numba bitboard layer; SEE, contempt, passed pawns | 100% vs a comparable classical fork; 0-6 vs a fork with a trained net |
| [08-nnue](08-nnue) | NNUE evaluation (single net), Syzygy probe | the gap that closed 0-6 vs the fork above |

`engine/` itself carries one further, unshipped-as-a-stage refinement past 08: an
endgame-specialised second net (`net_eg.npz`) for positions with few pieces left — a
specialised head on the same architecture, not a new concept, so it doesn't earn its own rung.
That's the build that actually played the competition: **2314 Elo**, 49W-42L-23D-8void over
122 rated games.

## Running the ladder

Every stage is a self-contained agent directory — `harness/play.py` / `harness/arena.py` take
any two of them:

```
uv run python -m harness.play --white stages/04-pesto-tapered --black stages/06-full-pruning
uv run python -m harness.arena --opponent stages/01-material-1ply --games 20
```

Stages 01-07 are frozen extracts of the commits named in each `README.md`
(`git show <sha>:agent.py`) — they will never change. Stage 08 is a single-net build artefact
of `engine/` (`tools/bundle_engine.py --net-eg /nonexistent`); re-run that after any engine
change to refresh it.
