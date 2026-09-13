---
title: Talos
emoji: ♞
colorFrom: gray
colorTo: red
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Play Talos, a chess engine built from scratch that placed 48th of 465
---

# Talos

Play a chess engine written from scratch: alpha-beta search over numba-JIT magic bitboards,
evaluated by a neural network trained on 215M positions. It finished **#48 of 465** at the
Optiver-sponsored AI Chessathon on a 2314 rating.

No third-party engine or network went into it. Every heuristic and every tuned weight traces
to a measurement in the source repo rather than a value copied from somewhere.

- Source, build log, and the full competition record: see the repository link on the page.
- The same engine plays rated games on lichess, where its rating is public.

This Space runs the real thing - the same `get_move(fen, time_left_ms)` the competition
platform called - one warm process answering a move at a time, on its own 2+1 clock. It sleeps when nobody is
playing, so the first move after a quiet spell takes a minute while it wakes and recompiles.
