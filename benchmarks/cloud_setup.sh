#!/usr/bin/env bash
# Set up a fresh Linux GPU box (tested on Ubuntu 24.04, CUDA 13, Blackwell 48 GB) for the benchmarks:
# Python env, CUDA llama-server, Q4 GGUF, .env, frozen corpus and the model server.
# Every step is idempotent, so rerunning after an interruption only does what is missing.
#
#   bash benchmarks/cloud_setup.sh                  # everything
#   bash benchmarks/cloud_setup.sh --parallel 4     # llama-server slots (>= --workers)
#   SETUP_STEPS="env corpus" bash benchmarks/cloud_setup.sh
#
# Steps (SETUP_STEPS, default all): venv build model env corpus server check
# Overrides: REPO_ROOT LLAMA_DIR LLAMA_TAG HF_HOST MODEL_PATH CUDA_ARCH TMUX_SESSION
#   HF_HOST=hf-mirror.com when huggingface.co is not reachable from the box.
#
# The repo is private, so upload it with `python -m benchmarks.pack_for_cloud` (git bundle) first.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_TAG="${LLAMA_TAG:-b11586}"
MODEL_FILE="Qwen3.8-27B-Q4_K_M.gguf"
MODEL_BYTES=17442399968
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/$MODEL_FILE}"
MODEL_URL="https://${HF_HOST:-huggingface.co}/bartowski/Qwen3.8-27B-GGUF/resolve/main/$MODEL_FILE"
TMUX_SESSION="${TMUX_SESSION:-server}"
PARALLEL="${LLAMA_PARALLEL:-1}"
STEPS="${SETUP_STEPS:-venv build model env corpus server check}"

while [ $# -gt 0 ]; do
  case "$1" in
    --parallel) PARALLEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '\n== %s\n' "$*"; }
want() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
BIN="$LLAMA_DIR/build/bin/llama-server"
health() { curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null | grep -q '"ok"'; }

# Set KEY=VALUE in .env, replacing an existing line or appending one.
set_env() {
  local key="$1" value="$2" file="$REPO_ROOT/.env"
  if grep -q "^${key}=" "$file"; then sed -i "s#^${key}=.*#${key}=${value}#" "$file"; else echo "${key}=${value}" >> "$file"; fi
}

step_venv() {
  log "Python environment"
  if [ -x "$PY" ] && "$PY" -c "import mcp, httpx, dotenv" 2>/dev/null; then echo "already installed"; return; fi
  if command -v uv >/dev/null; then
    uv venv --python 3.12 "$REPO_ROOT/.venv" 2>/dev/null || uv venv "$REPO_ROOT/.venv"
    uv pip install --python "$PY" -r "$REPO_ROOT/requirements.txt"
  else
    python3 -m venv "$REPO_ROOT/.venv" && "$PY" -m pip install -r "$REPO_ROOT/requirements.txt"
  fi
}

step_build() {
  log "llama.cpp $LLAMA_TAG (CUDA)"
  if [ -x "$BIN" ] && "$BIN" --version >/dev/null 2>&1; then echo "already built: $BIN"; return; fi
  command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
  command -v nvcc >/dev/null || { echo "nvcc not found: install the CUDA toolkit first" >&2; exit 1; }
  local arch="${CUDA_ARCH:-}"
  if [ -z "$arch" ]; then   # compute capability 12.0 -> 120, 8.9 -> 89
    arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
  fi
  echo "CUDA architecture: $arch"
  [ -d "$LLAMA_DIR/.git" ] || git clone --branch "$LLAMA_TAG" --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_CUDA_ARCHITECTURES="$arch" -DTHREADS_PREFER_PTHREAD_FLAG=ON
  cmake --build "$LLAMA_DIR/build" --config Release --target llama-server -j "$(nproc)"
}

step_model() {
  log "Model $MODEL_FILE"
  mkdir -p "$(dirname "$MODEL_PATH")"
  local size=0
  [ -f "$MODEL_PATH" ] && size="$(stat -c %s "$MODEL_PATH")"
  if [ "$size" -eq "$MODEL_BYTES" ]; then echo "already complete ($size bytes)"; return; fi
  [ "$size" -gt "$MODEL_BYTES" ] && { echo "$MODEL_PATH is larger than expected; remove it and rerun" >&2; exit 1; }
  echo "downloading from $MODEL_URL (have $size of $MODEL_BYTES bytes)"
  curl --location --fail --retry 5 --retry-all-errors --continue-at - --output "$MODEL_PATH" "$MODEL_URL"
  [ "$(stat -c %s "$MODEL_PATH")" -eq "$MODEL_BYTES" ] || { echo "size mismatch after download" >&2; exit 1; }
}

step_env() {
  log ".env"
  [ -f "$REPO_ROOT/.env" ] || cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  set_env GGUF_MODEL_PATH "$MODEL_PATH"
  set_env LLAMA_SERVER_BIN "$BIN"
  set_env LLM_TEMPERATURE 0          # benchmarks need reproducible decoding
  set_env LLM_ENABLE_THINKING 0
  set_env LLAMA_PARALLEL "$PARALLEL"
  grep -E '^(GGUF_MODEL_PATH|LLAMA_SERVER_BIN|LLM_TEMPERATURE|LLM_ENABLE_THINKING|LLM_CONTEXT_TOKENS|LLAMA_PARALLEL)=' "$REPO_ROOT/.env"
}

step_corpus() {
  log "Frozen corpus"
  (cd "$REPO_ROOT" && "$PY" -m benchmarks.prepare_corpus unpack)
}

step_server() {
  log "Model server (llama-server, $PARALLEL slot(s))"
  local running_slots=""
  if health; then
    running_slots="$(curl -s -m 3 http://127.0.0.1:8000/props | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("total_slots",""))' 2>/dev/null || true)"
    if [ "$running_slots" = "$PARALLEL" ]; then echo "already running with $PARALLEL slot(s)"; return; fi
    echo "running with ${running_slots:-?} slot(s), restarting with $PARALLEL"
    if command -v tmux >/dev/null; then tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true; fi
    pkill -f llama-server || true
    sleep 3
  fi
  local cmd="cd $REPO_ROOT && LLAMA_PARALLEL=$PARALLEL $PY deploy/serve_gguf.py > $REPO_ROOT/server.log 2>&1"
  if command -v tmux >/dev/null; then tmux new-session -d -s "$TMUX_SESSION" "$cmd"; else nohup bash -c "$cmd" >/dev/null 2>&1 & fi
  for _ in $(seq 1 120); do health && break; sleep 3; done
  health || { echo "server did not become healthy; see $REPO_ROOT/server.log" >&2; exit 1; }
  echo "healthy"
}

step_check() {
  log "Preflight"
  (cd "$REPO_ROOT" && "$PY" - <<'PYEOF'
from dotenv import load_dotenv
load_dotenv()
from benchmarks import preflight
report = preflight.check(full=True)
print(preflight.render(report))
raise SystemExit(0 if report.ok else 1)
PYEOF
  )
}

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
for step in venv build model env corpus server check; do want "$step" && "step_$step"; done
log "done. Next: python -m benchmarks.run_all --label baseline-q4"
