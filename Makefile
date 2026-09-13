SHELL := /bin/bash

.PHONY: setup bundle play arena zip gate bench verify lichess-check lichess-config web space

setup:
	uv sync

# engine/ is a real package; harness/ (vendored, unmodified) needs a flat agent directory with
# agent.py at its root - this is the one place those two facts meet. candidates/ is gitignored.
bundle:
	uv run python tools/bundle_engine.py --out candidates/engine

play: bundle
	uv run python -m harness.play --white candidates/engine --black stages/07-numba-classical $(if $(FEN),--fen "$(FEN)")

arena: bundle
	uv run python -m harness.arena --agent candidates/engine --opponent stages/07-numba-classical --games 20

zip:
	uv run python tools/bundle_engine.py --out candidates/engine --zip

gate:
	uv run ruff check .
	uv run mypy
	$(MAKE) bundle
	uv run python -m harness.arena --agent candidates/engine --opponent stages/01-material-1ply --games 2 --base-ms 5000

# the Elo ladder: engine vs. the classical build it replaced, both directions of the harness
bench: bundle
	uv run python bench/openings_bench.py --agent candidates/engine --opponent stages/07-numba-classical

# the exact-refactor / correctness checks (nodebench + the numba-vs-reference twins)
verify:
	uv run python bench/perft.py
	uv run python bench/verify_attacks.py
	uv run python bench/verify_move.py
	uv run python bench/verify_movegen.py
	uv run python bench/verify_zobrist.py
	uv run python bench/verify_see.py
	uv run python bench/verify_nnue.py

# the UCI bridge in lichess/: the same handshake lichess-bot opens a game with, run by hand
lichess-check:
	printf 'uci\nisready\nucinewgame\nposition startpos moves e2e4 e7e5\ngo movetime 3000\nquit\n' | uv run python lichess/uci.py

# lichess-bot runs from its own clone, so its config needs this repo's paths spelled out in
# full - fill them in rather than typing them. candidates/ is gitignored.
lichess-config:
	@mkdir -p candidates
	@sed 's|/path/to/neural-chess-engine|$(CURDIR)|g' lichess/config.yml > candidates/config.yml
	@echo "wrote candidates/config.yml - pass it to lichess-bot.py with --config"

# the browser demo: one warm engine process behind a small HTTP server. PORT=8080 to move it.
web:
	uv run python web/server.py $(PORT)

# assemble the Hugging Face Space: Hugging Face looks for Dockerfile and README.md at the repo
# root, so this builds a minimal tree with them there. Idempotent, and it leaves any .git in
# candidates/space alone - the second deploy is just commit and push.
space:
	@mkdir -p candidates/space
	@rm -rf candidates/space/engine candidates/space/web
	@rsync -a --exclude '__pycache__' engine web candidates/space/
	@cp web/Dockerfile candidates/space/Dockerfile
	@cp web/space/README.md candidates/space/README.md
	@echo "assembled candidates/space ($$(du -sh candidates/space | cut -f1)) - push steps in web/README.md"
