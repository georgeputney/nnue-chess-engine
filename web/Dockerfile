# The demo server. Build from the repo root:
#
#     docker build -f web/Dockerfile -t nnue-demo .
#     docker run -d --name nnue-demo -p 8000:8000 --restart unless-stopped nnue-demo
#
# Build for the architecture you will run on (--platform linux/amd64 for a normal VPS): numba's
# on-disk cache is native code, so a cache warmed on arm64 is no use to an x86 container.
FROM python:3.12-slim

# The engine itself needs numpy, numba and python-chess and nothing else - torch is in this
# repo's pyproject for the training pipeline in tools/, which does not ship here. Same pins.
RUN pip install --no-cache-dir numpy==2.5.2 numba==0.67.0 chess==1.11.2

WORKDIR /app
COPY engine/ engine/
COPY web/ web/

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app NUMBA_CACHE_DIR=/app/.numba-cache PORT=8000

# Compile what numba can cache to disk at build time rather than on the first visitor. The
# recursive and objmode parts (negamax, quiescence, the tablebase probe) cannot be cached, so
# the container still spends ~7 s compiling at startup - it is ready when it logs its port.
# a+rwX, not a+rX: Hugging Face Spaces runs the container as a non-root user, and numba
# still wants to write cache entries for anything the warm-up did not reach.
RUN python -c "import engine.agent" && chmod -R a+rwX /app/.numba-cache

EXPOSE 8000
CMD ["python", "web/server.py"]
