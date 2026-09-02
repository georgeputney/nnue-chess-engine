# alpha-beta chess engine

My entry for [AI Chessathon](https://aichessathon.com). The whole submission is `agent.py`. It
exposes one function and plays one side of a game under a 120 s + 0.5 s per move clock, on a
single core, with no network and no GPU.

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

## Status

`agent.py` is currently the starter's legal random-mover. The engine is being built on top of
it, one measured change at a time.

## Approach

The plan, in the order it earns its strength:

1. Negamax with alpha-beta, iterative deepening so there is always a move to return when the
   clock runs out.
2. Move ordering: captures first, ordered by MVV-LVA, plus a transposition table kept across
   moves.
3. Quiescence search at the leaves so the evaluation is never read mid-exchange.
4. Evaluation: material and piece-square tables, tapered between a midgame and an endgame set,
   plus mobility and pawn-structure terms. Every weight Texel-tuned against engine-labelled
   positions offline; only the resulting numbers ship.
5. Time management driven by the clock that gets handed in, not a constant, with a margin so a
   flag is never the reason a game is lost.

`docs/plan.md` has the full phase list and the measurement behind each change.

## Layout

```
agent.py       the submission
docs/plan.md   the build plan and how each change is measured
baselines/     random, greedy, minimax, numba, each a directory with an agent.py, to measure against
harness/       local runner, referee, and packaging that mirror the platform's protocol and clock
```

## Running it

```
make setup
make play      # one full-length game against a baseline, real time control
make arena     # 20 fast games against a baseline, prints a score
make zip       # build submission.zip with agent.py at the root
make gate      # ruff, mypy, and two games that have to finish cleanly
```

Anything the agent prints to stdout or stderr shows up under the result. The platform discards
it in rated games and shows it in the validation log.

## Credits

Most of this repo, the `harness/`, the `baselines/`, the packaging, and the `agent.py` scaffold,
is the [AI Chessathon starter](https://github.com/advitrocks9/aichessathon-starter) by Advit
Arora, used under the MIT License (see `LICENSE`). My work is the engine in `agent.py`.
`PROVENANCE.md` has the file-by-file split.

## The competition

The agent contract and the rules live at [aichessathon.com/docs](https://aichessathon.com/docs).
They are canonical and they change, so read them before uploading.
