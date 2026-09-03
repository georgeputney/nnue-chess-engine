# Provenance

An AI Chessathon entry. This repo holds my work plus the competition's unmodified starter
kit; this file records which is which.

## My work

| Path | What it is |
|---|---|
| `agent.py` | The engine. `get_move`'s signature is the platform contract; everything below it is mine, built in the numbered phases in `docs/plan.md`. |
| `docs/plan.md` | The build roadmap and the measurement discipline behind each change. |
| `tools/bench.py` | Openings-based A/B match runner (score, Elo, 95% band). |
| `tools/nodebench.py` | Fixed-depth node-count bench for verifying search refactors are exact. |
| `tools/tune.py` | Offline Texel tuner: fits the evaluation weights to game results and writes `tables.py`. |
| `README.md` | Rewritten (the starter shipped its own). |

## Vendored: the AI Chessathon starter

By Advit Arora, used under the MIT License (see `LICENSE`). Source:
<https://github.com/advitrocks9/aichessathon-starter>

- `harness/` — local runner, referee, clock, packaging, and match drivers that mirror the
  platform protocol. **Unmodified.** `AGENTS.md` forbids editing it: changing it makes local
  results meaningless.
- `baselines/` — `random`, `greedy`, `minimax`, `numba`; opponents to measure against.
- `Makefile`, `pyproject.toml`, `uv.lock` — build targets and the pinned dependency set.
- `AGENTS.md` (and its `CLAUDE.md` alias) — the competition contract and repo rules.
- `.github/workflows/ci.yml`, `LICENSE`, `.gitignore` — CI, licence, and ignores
  (`.gitignore` has one added line for a local scratch directory).

## Verifying the split

`git log -- harness/ baselines/ Makefile pyproject.toml` shows only the initial commit — the
starter has not been touched since. `git log -- agent.py` is the full phase history of the
engine.

## What ships

`make zip` builds the submission from `agent.py` alone (plus model weights, once there are
any). Nothing under `harness/`, `baselines/`, or `tools/` goes in it.
