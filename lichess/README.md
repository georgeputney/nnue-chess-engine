# Playing on lichess

The engine in [`engine/`](../engine) was built for the Chessathon platform, which imports
`agent.py` and calls `get_move(fen, time_left_ms)` over a JSON protocol of its own. Lichess
bots speak UCI instead, so [`uci.py`](uci.py) is the bridge: the same shim
[`harness/runner.py`](../harness/runner.py) is for the competition, pointed at a different
protocol. Nothing under `engine/` changes - what plays here is what played there.

[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) does the rest: it holds the
lichess connection, accepts challenges, and runs this engine as an ordinary UCI child process.
It stays in its own clone; only [`config.yml`](config.yml) here points back at this repo.

## Setup

**1. A bot account.** Bots need their own lichess account with no rated games ever played on
it. Register one, then create a personal access token with the **`bot:play`** scope at
<https://lichess.org/account/oauth/token/create>.

**2. This repo.**

```
make setup          # uv sync - torch, numpy, numba, python-chess
make lichess-check  # a scripted UCI session through the bridge; ends in `bestmove g1f3`
```

**3. lichess-bot**, in a clone of its own, with its own virtualenv - it has dependencies this
repo does not, and this repo pins versions it should not have to share:

```
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
python3 -m venv venv && source venv/bin/activate
python3 -m pip install -r requirements.txt
```

**4. The config.** `config.yml` here is a cut-down lichess-bot config set up for this engine,
with `/path/to/neural-chess-engine` where the absolute paths go. Fill them in:

```
make lichess-config   # writes candidates/config.yml with this checkout's paths
```

**5. Run it.** The token goes in the environment rather than in the file - `LICHESS_BOT_TOKEN`
overrides whatever `token:` says:

```
export LICHESS_BOT_TOKEN=lip_xxxxxxxxxxxx
cd /path/to/lichess-bot
python3 lichess-bot.py -u --config /path/to/neural-chess-engine/candidates/config.yml
```

`-u` upgrades the account to a BOT account, which is permanent and only needed once; drop it
afterwards. lichess-bot starts the engine once at boot to check the config, so a clean start
proves the whole chain before any game begins. Challenge the bot from another account, or set
`matchmaking.allow_matchmaking: true` to have it go looking for opponents.

## What the bridge supports

| | |
|---|---|
| `position startpos\|fen ... [moves ...]` | Yes. The move list is replayed with python-chess to get the FEN the engine wants. |
| `go wtime/btime/winc/binc` | Yes - the normal case. See the clock note below. |
| `go movetime` | Yes, as a hard cap. lichess-bot asks for the first move of every game this way. |
| `go depth` | Yes, but it is `bench_search`: an empty table, no tablebase probe, no repetition bookkeeping. For fixed-depth testing in a GUI, never in `go_commands`. |
| `go infinite`, bare `go` | Treated as a 3 s think, and answered rather than waiting for `stop`. |
| `setoption` | `Move Overhead` only. The transposition table is a fixed 2^22 slots and the search is single-threaded, so there is no honest `Hash` or `Threads` to advertise - and an option in `uci_options` that the engine never declared makes python-chess refuse to start it. |
| `stop`, `ponderhit` | No. The search runs synchronously inside `go`, so `ponder: false` in the config is required, not a preference. |
| `info score` | Only on the `go depth` path. `get_move` returns a move and nothing else, and reaching past it for the root score would mean editing `engine/`. lichess-bot handles a scoreless move fine - its draw/resign logic simply never fires, which is why the config turns it off. |
| `searchmoves`, `multipv`, variants | No. The config keeps lichess-bot from asking: no `online_moves`, no `move_quality: suggest`, standard chess only. |

## The clock

`get_move` budgets a move as a fraction of the clock it is handed - at most a quarter of it on
one move, no new deepening iteration past a fortieth (a twentieth in the endgame). Those
fractions were fitted against one time control, 120 s + 0.5 s a move, because that is all the
competition ran.

Lichess runs many, so `effective_clock_ms` converts rather than re-deriving the budget: the
clock handed over is the real remaining time plus 25 moves' worth of the increment, capped so
the quarter-of-the-clock hard deadline still fits inside the time actually on the clock. At
5+3 that is about 9 s a move early on, tightening by itself as the clock drains; with no
increment it is exactly the competition behaviour.

`go movetime n` hands over `n * 4`, which puts the hard deadline on `n` exactly. The engine
normally returns in a fraction of that, since it starts no new iteration past a tenth of the
clock - so the 10 s lichess-bot allows for the first move of a game costs about 1 s of it.
That is the right way round: the first move out of the opening is not where the thinking pays.

## Startup cost

Importing the engine compiles the numba search in `warm_up()`, and the parts that recurse or
drop into objmode cannot be cached to disk. Measured: **about 7 s on a laptop, 28 s on the
Ampere A1 this bot is hosted on**, where the cores are roughly four times slower. The
competition allowed 90 s for imports before the clock started, so it was free there.
lichess-bot is not so generous - it starts a fresh engine per game, and arms its own 30 s
abort timer *before* doing so, so on the slower box every game was aborted before the first
move. Even had it not been, 28 s would have come off a 120 s clock.

[`warm.py`](warm.py) is the answer to that. It imports the engine once at boot and then waits
on a unix socket; [`uci_client.py`](uci_client.py) - what `config.yml` points lichess-bot at -
hands over its own stdin, stdout and stderr, and the daemon forks a child that already holds
the compiled search. Starting a game drops from 28 s to **0.2 s**, measured on the box.

`fork()` is what makes that safe rather than clever. The parent never searches, so every child
begins from the same untouched module state a fresh process would have had, and two games at
once cannot see each other's transposition table; copy-on-write means they share the ~670 MB
of weights and compiled code rather than each paying for it.

With no daemon running, `uci_client.py` runs the engine in its own process exactly as before,
so a laptop, a UCI GUI and `make lichess-check` need nothing set up, and a dead daemon costs a
slow first game rather than no bot.

It also removes what used to be an argument for a floor on the time control. `config.yml`
plays 2+1 and nothing else now - near the control the engine was tuned for, and a control the
startup cost would once have ruled out entirely.

## In a GUI instead

The bridge is a plain UCI engine, so anything that speaks UCI can run it - cutechess-cli, Arena,
Banksia. Point them at `.venv/bin/python` with `lichess/uci.py` as the argument, or run
`lichess/uci.py` directly if `.venv` is the active environment.
