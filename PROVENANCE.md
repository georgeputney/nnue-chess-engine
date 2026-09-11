# Provenance

An AI Chessathon entry, restructured after the competition ended into a portfolio repo. This
file records what's mine and what's the competition's unmodified starter kit.

## My work

| Path | What it is |
|---|---|
| `engine/` | The engine that played the competition (2314 Elo, `docs/games.csv`). `agent.py`'s `get_move` signature is the platform contract; everything else here is mine. |
| `stages/01` through `stages/07` | Frozen extracts of the classical engine's own commit history — see each stage's `README.md` for the exact commit. |
| `stages/08-nnue` | A single-net build of `engine/`, not vendored history. |
| `docs/plan.md`, `docs/nnue-plan.md` | The build roadmap and the measurement behind every change, classical then NNUE. |
| `docs/writeup.md` | The retrospective. |
| `docs/games.csv`, `docs/games/*.pgn` | The competition's actual rated-game record, downloaded from the platform. |
| `bench/` | Every measurement tool: `openings_bench.py` (Elo A/B), `nodebench.py` (exact-refactor check), `endgame_bench.py`, `eg_suite.py`, `perft.py`, and every `verify_*.py` twin check. |
| `tools/` | The offline pipeline: `train_nn.py`, `export_nn.py`, `ingest_lichess.py`, `label.py`, `label_eg.py`, `filter_shards.py`, `tune.py`, `gen_magics.py`, `bundle_engine.py`, `model.py`. |
| `README.md` | Rewritten (the starter shipped its own; see below). |

`engine/net.npz` and `engine/net_eg.npz` are nets I trained (`tools/train_nn.py` /
`tools/export_nn.py`) on Lichess and Stockfish-labelled positions — training on
engine-annotated data is explicitly allowed; what's banned is shipping someone else's engine
or net, and neither net is either.

## Vendored: the AI Chessathon starter

By Advit Arora, used under the MIT License (see `LICENSE`). Source:
<https://github.com/advitrocks9/aichessathon-starter>

- `harness/` — local runner, referee, clock, and match drivers that mirror the platform
  protocol. **Unmodified.** The competition rules forbade editing it: changing it makes local
  results meaningless, and that's still true for reproducing the numbers in `docs/`.
- `LICENSE`, `.gitignore` (most of it) — licence and ignores.

`baselines/` (the starter's `random` / `greedy` / `minimax` / `numba` opponents) and the
starter's own `Makefile` / `pyproject.toml` are gone — superseded by `stages/`, which tells the
same "something to measure against" story as the engine's own history instead of a fixed set
of throwaway opponents.

## Verifying the split

`git log -- harness/` shows only the initial commit — the one vendored piece has not been
touched since. `git log -- engine/agent.py stages/` (and, before this restructure, `agent.py`
and `nnue/`) is the full phase history of the engine, classical through NNUE.

## What ships

`make zip` (`tools/bundle_engine.py --zip`) builds the submission from a flattened `engine/`:
`agent.py` at the zip root plus the shared bitboard-layer modules, both nets, and the Syzygy
tables. Nothing under `harness/`, `bench/`, `tools/`, or `stages/` goes in it.
