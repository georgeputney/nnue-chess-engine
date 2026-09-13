# The browser demo

A page that lets anyone play the engine with no account and nothing to install:
[`server.py`](server.py) keeps one warm engine process and answers moves for
[`index.html`](index.html). This is the real engine - the same `get_move(fen, time_left_ms)`
the competition platform called, NNUE and all, at full speed. Nothing under `engine/` changes
for it.

```
make web            # http://localhost:8000, PORT=8080 to move it
```

Give it ~10 s on first start: that is `warm_up()` compiling the numba search, plus a
fixed-depth search to measure this machine's nodes/sec. It is ready when it logs its port,
and the page then opens on that measurement rather than on a placeholder.

**Before deploying, fill in the two URLs at the top of [`app.js`](app.js)** - `REPO_URL` and
`LICHESS_URL`. They are what the demo is advertising, and they ship pointing at nothing.

## How it holds together

The page has no chess logic. It sends a move and gets a position back - fen, the legal moves
from it, the game so far in algebraic, the network's static evaluation, and what the last
search cost. So there is nothing in the client to talk into an illegal move, and no chess
library to load.

Three things the competition never had to deal with, because the platform gave every game a
process of its own:

- **Per-game state.** The engine's anti-repetition bookkeeping and hash trail live in module
  globals. Here many games share one process, so a `Session` owns that state and `install()`
  swaps it in around each search. The transposition table stays shared on purpose - it is a
  cache, and a position one visitor paid for is one the next gets free.
- **One core.** Every search takes `SEARCH_LOCK`; a visitor who waits more than
  `LOCK_TIMEOUT_S` gets a 503 rather than a queue.
- **No login.** A per-address token bucket (`RATE_BURST` moves, refilling over
  `RATE_REFILL_S`) and a session ceiling with idle expiry.

## The clock

The game is 2+1, near enough the control the engine was tuned for - the Chessathon ran 120 s
plus half a second a move. Talos is not handicapped here: it spends its clock exactly as it
does on lichess, roughly a twentieth of what is left per move, and `think_clock_ms` converts
the remaining time into the budget `get_move` wants the same way [`lichess/uci.py`](../lichess/uci.py)
does. Measured across three positions, per control:

| control | first move | middlegame |
|---|---|---|
| 5+2 | 9.8 s | 7.0 s |
| **2+1** | **4.9 s** | **3.6 s** |
| 1+1 | 2.3 s | 2.5 s |

**Talos's clock is owned by the server, not sent by the page.** That is what makes running it
uncapped safe: a page that could name the engine's remaining time could claim an hour of it and
tie up the core. The time control is therefore also the CPU budget - a whole game costs at most
the two minutes plus increments that Talos actually has.

## What it needs from a host

**One core and ~600 MB.** Measured: 483 MB resident steady, 560 MB peak, of which 192 MB is the
transposition table (4.2M slots x 6 int64 arrays). Not *more* cores - the search is
single-threaded and every extra one sits idle. Core speed is the whole game: the times above
are a 2.6 GHz-class core.

That rules out the usual free web tiers, which give 512 MB and a fraction of a vCPU - it would
OOM on startup, and at a tenth of a core the "normal" level's 350 ms becomes 3.5 s.

## Hugging Face Spaces

Free hardware there is 2 vCPU and 16 GB, which is more headroom than a small VPS, and Spaces is
built for exactly this sort of demo.

**Before you assemble it**, fill in `REPO_URL` and `LICHESS_URL` at the top of
[`app.js`](app.js) - they ship pointing at nothing.

1. Create the Space at <https://huggingface.co/new-space>: **SDK: Docker**, hardware **CPU
   basic (free)**, public. Do not pick a template.
2. `make space` - assembles `candidates/space`, a 5.5 MB tree with `Dockerfile` and `README.md`
   at its root, which is where Hugging Face looks for them. [`space/README.md`](space/README.md)
   is the Space card; its `app_port: 8000` matches the Dockerfile's `PORT`.
3. Push it, with a **write** token from <https://huggingface.co/settings/tokens> as the
   password when git asks (the username is your HF username):

```
cd candidates/space
git init -b main
git remote add origin https://huggingface.co/spaces/<you>/<space-name>
git add -A && git commit -m "NNUE chess engine demo"
git push -u origin main
```

Later deploys are `make space` again, then commit and push from the same directory - the target
leaves `candidates/space/.git` alone.

The build takes a few minutes: numba and numpy install from wheels, python-chess builds from an
sdist (pure Python, no compiler needed), and then the image runs the warm-up so everything numba
*can* cache to disk is cached at build rather than on the first visitor. Watch it in the Space's
**Logs** tab. It is up when the log shows the port.

Free Spaces sleep after a long idle stretch, so the first visitor after a quiet spell waits for
the container to wake and for the ~7 s of compiling that cannot be cached.

**Embedding it in your own site**, so the visitor sees your domain rather than huggingface.co:

```html
<iframe src="https://<you>-<space-name>.hf.space"
        style="width:100%;height:780px;border:0" title="NNUE chess engine"></iframe>
```

## On a server of your own

```
docker build --platform linux/amd64 -f web/Dockerfile -t nnue-demo .
docker run -d --name nnue-demo -p 8000:8000 --restart unless-stopped nnue-demo
```

Build for the architecture you will run on - numba's disk cache is native code, so one warmed
on an arm64 laptop does nothing for an x86 container.

Put TLS in front of it rather than in it. With Caddy that is the whole config:

```
chess.example.com {
    reverse_proxy localhost:8000
}
```

`server.py` reads `X-Forwarded-For` for the rate limit, so the proxy's address does not become
everybody's address. Sharing a box with a website? Run it niced (`Nice=10` in a systemd unit,
`--cpu-shares 512` in Docker) so the web server always wins the tie.

## The knobs worth watching

All at the top of [`server.py`](server.py). If the box starts feeling the load, shorten the
time control - `START_MS` and `INCREMENT_MS` - before anything else; it is the single biggest
lever on CPU per visitor, and it costs strength honestly rather than by capping the search.
Then `RATE_BURST`. If you see 503s in the log, more than one person is playing at once and a
search is holding the lock; the answer is a faster core, not a bigger one.
