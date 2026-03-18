#!/usr/bin/env bash
# =============================================================================
# InfraBrain RunPod Setup Script
# =============================================================================
#
# Usage (on the RunPod pod):
#   bash /workspace/scripts/runpod-setup.sh
#
# Or as Start Command (copy the script to your volume first):
#   bash /workspace/scripts/runpod-setup.sh
#
# Tested on:
#   GPU:    RTX PRO 6000 (98GB VRAM)
#   Driver: NVIDIA 580.126.20, CUDA 13.0
#   Image:  runpod/pytorch:1.0.2-cu128-torch280-ubuntu2404
#   Volume: /workspace (300GB, Network Volume "infrabrain-model-vault")
#
# What it does:
#   1. Checks GPU, CUDA, Python, disk space
#   2. Installs Ollama with GPU detection fix for RunPod containers
#   3. Starts Ollama (port 11434) using cached models from volume
#   4. Pulls any missing Ollama models (skips cached)
#   5. Installs vLLM (pip cached on volume for fast restarts)
#   6. Starts vLLM with Qwen2.5-7B for triage/routing (port 8000)
#   7. Verifies both backends respond + quick inference test
#   8. Prints connection info for local Mac config
#
# Required RunPod settings:
#   HTTP ports: 8000, 11434
#   Volume:     mounted at /workspace
#   GPU:        any NVIDIA with >= 24GB VRAM
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[SETUP]${NC} $1"; }
ok()   { echo -e "${GREEN}[  OK ]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN ]${NC} $1"; }
fail() { echo -e "${RED}[FAIL ]${NC} $1"; exit 1; }
step() { echo -e "\n${BOLD}━━━ Step $1 ━━━${NC}"; }

# =============================================================================
# Config — edit these to change models/ports
# =============================================================================
OLLAMA_MODEL_DIR="/workspace/ollama-models"
VLLM_CACHE_DIR="/workspace/hf-cache"
VLLM_PIP_CACHE="/workspace/pip-cache"

# vLLM serves the small fast model for triage/routing
VLLM_MODEL="Qwen/Qwen2.5-7B"
VLLM_PORT=8000
VLLM_GPU_UTIL=0.3           # 30% GPU for vLLM (~29GB on 98GB card)
VLLM_MAX_MODEL_LEN=4096

# Ollama serves the heavy models (uses remaining GPU + CPU offload)
OLLAMA_MODELS_TO_PULL=("qwen3.5:9b" "qwen3.5:122b-a10b" "bge-m3")

# Minimum requirements
MIN_VRAM_MB=20000            # 20GB minimum
MIN_DISK_GB=50               # 50GB free on /workspace

# =============================================================================
# Step 0: Environment check
# =============================================================================
step "0/7: Environment check"

# GPU
log "Checking GPU..."
if ! nvidia-smi &>/dev/null; then
  fail "nvidia-smi not found. Is this a GPU pod?"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
ok "GPU: ${GPU_NAME} (${GPU_VRAM} MiB VRAM)"
ok "Driver: ${GPU_DRIVER} | CUDA: ${CUDA_VER}"

if [ "${GPU_VRAM}" -lt "${MIN_VRAM_MB}" ]; then
  fail "Need at least ${MIN_VRAM_MB} MiB VRAM, got ${GPU_VRAM} MiB"
fi

# Python
log "Checking Python..."
PYTHON_VER=$(python3 --version 2>/dev/null || echo "NOT FOUND")
PIP_VER=$(pip --version 2>/dev/null | awk '{print $2}' || echo "NOT FOUND")
ok "Python: ${PYTHON_VER} | pip: ${PIP_VER}"

# Disk space
log "Checking disk space..."
DISK_FREE_GB=$(df /workspace --output=avail -BG 2>/dev/null | tail -1 | tr -d ' G' || echo "0")
ok "Disk free: ${DISK_FREE_GB} GB on /workspace"
if [ "${DISK_FREE_GB}" -lt "${MIN_DISK_GB}" ]; then
  warn "Low disk space (${DISK_FREE_GB}GB < ${MIN_DISK_GB}GB). Models may not fit."
fi

# Volume check
if [ -d "/workspace" ]; then
  ok "Volume mounted at /workspace"
  if [ -d "${OLLAMA_MODEL_DIR}" ]; then
    CACHED_SIZE=$(du -sh "${OLLAMA_MODEL_DIR}" 2>/dev/null | awk '{print $1}' || echo "0")
    ok "Cached Ollama models: ${CACHED_SIZE} at ${OLLAMA_MODEL_DIR}"
  else
    warn "No cached Ollama models found — will download on first pull"
  fi
else
  fail "/workspace not mounted. Attach a volume in RunPod settings."
fi

# CUDA libraries
log "Checking CUDA libraries..."
if [ -f "/usr/local/cuda/lib64/libcudart.so" ]; then
  ok "CUDA runtime libs found at /usr/local/cuda/lib64"
elif [ -f "/usr/lib/x86_64-linux-gnu/libcudart.so" ]; then
  ok "CUDA runtime libs found at /usr/lib/x86_64-linux-gnu"
else
  warn "CUDA runtime libs not in expected paths — Ollama may fall back to CPU"
fi

echo ""
log "Environment OK. Starting setup..."

# =============================================================================
# Step 1: Install Ollama
# =============================================================================
step "1/7: Install Ollama"

if command -v ollama &>/dev/null; then
  OLLAMA_VER=$(ollama --version 2>/dev/null || echo 'unknown')
  ok "Ollama already installed: ${OLLAMA_VER}"
else
  log "Downloading and installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  ok "Ollama installed"
fi

# =============================================================================
# Step 2: Start Ollama with GPU support
# =============================================================================
step "2/7: Start Ollama (port 11434)"

# Kill any existing Ollama process
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# Create model directory on volume if needed
mkdir -p "$OLLAMA_MODEL_DIR"

# Clear old log
> /workspace/ollama.log

# Set environment for GPU detection on RunPod containers
# OLLAMA_HOST=0.0.0.0    — listen on all interfaces (RunPod proxy needs this)
# OLLAMA_MODELS          — persistent model storage on volume
# LD_LIBRARY_PATH        — ensure CUDA libs are found in container
# CUDA_VISIBLE_DEVICES   — explicit GPU selection
# NVIDIA_VISIBLE_DEVICES — Docker-level GPU visibility
export OLLAMA_HOST="0.0.0.0"
export OLLAMA_MODELS="$OLLAMA_MODEL_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
export CUDA_VISIBLE_DEVICES="0"
export NVIDIA_VISIBLE_DEVICES="all"

log "Starting ollama serve..."
ollama serve &>> /workspace/ollama.log &
OLLAMA_PID=$!

# Wait for Ollama to be ready
log "Waiting for Ollama API..."
OLLAMA_READY=false
for i in $(seq 1 30); do
  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    OLLAMA_READY=true
    break
  fi
  sleep 1
done

if [ "$OLLAMA_READY" = true ]; then
  ok "Ollama running (PID: $OLLAMA_PID)"
else
  warn "Ollama slow to start. Log tail:"
  tail -20 /workspace/ollama.log
  fail "Ollama failed to start after 30 seconds"
fi

# Check GPU detection
OLLAMA_GPU_LOG=$(grep -i "gpu\|cuda\|vram" /workspace/ollama.log 2>/dev/null | head -5 || echo "")
if echo "$OLLAMA_GPU_LOG" | grep -qi "vram"; then
  ok "Ollama detected GPU"
else
  warn "Ollama may not see the GPU. Log says:"
  echo "$OLLAMA_GPU_LOG"
  warn "Continuing anyway — inference may use CPU (slower but works)"
fi

# =============================================================================
# Step 3: Pull Ollama models
# =============================================================================
step "3/7: Ollama models"

log "Checking cached models..."
EXISTING_MODELS=$(ollama list 2>/dev/null || echo "")

for model in "${OLLAMA_MODELS_TO_PULL[@]}"; do
  if echo "$EXISTING_MODELS" | grep -q "${model}"; then
    ok "${model} — cached on volume"
  else
    log "Pulling ${model} (first time may take several minutes)..."
    if ollama pull "$model"; then
      ok "${model} — pulled successfully"
    else
      warn "Failed to pull ${model} — skipping (can retry later with: ollama pull ${model})"
    fi
  fi
done

echo ""
log "Available Ollama models:"
ollama list 2>/dev/null || warn "Could not list models"

# =============================================================================
# Step 4: Install vLLM
# =============================================================================
step "4/7: Install vLLM"

mkdir -p "$VLLM_PIP_CACHE" "$VLLM_CACHE_DIR"

if python -c "import vllm" &>/dev/null; then
  VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
  ok "vLLM already installed: v${VLLM_VER}"
else
  log "Installing vLLM via pip (caching at ${VLLM_PIP_CACHE})..."
  log "This takes 2-5 minutes on first install..."
  if pip install vllm --cache-dir "$VLLM_PIP_CACHE" --quiet 2>&1; then
    VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    ok "vLLM installed: v${VLLM_VER}"
  else
    fail "vLLM installation failed. Check pip output above."
  fi
fi

# =============================================================================
# Step 5: Start vLLM
# =============================================================================
step "5/7: Start vLLM (port ${VLLM_PORT})"

# Kill any existing vLLM process
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

# Clear old log
> /workspace/vllm.log

export HF_HOME="$VLLM_CACHE_DIR"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

log "Starting vLLM with ${VLLM_MODEL}..."
log "  GPU util: ${VLLM_GPU_UTIL} (${VLLM_GPU_UTIL}% of VRAM)"
log "  Max context: ${VLLM_MAX_MODEL_LEN} tokens"
log "  Model cache: ${VLLM_CACHE_DIR}"

python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --port "$VLLM_PORT" \
  --host 0.0.0.0 \
  --download-dir "$VLLM_CACHE_DIR" \
  &>> /workspace/vllm.log &
VLLM_PID=$!

log "Waiting for vLLM to load model (1-3 minutes on first run, ~30s cached)..."
VLLM_READY=false
for i in $(seq 1 180); do
  if curl -sf http://localhost:${VLLM_PORT}/v1/models &>/dev/null; then
    VLLM_READY=true
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    warn "vLLM process died. Last log lines:"
    tail -20 /workspace/vllm.log
    fail "vLLM failed to start. Check /workspace/vllm.log"
  fi
  # Progress indicator every 15 seconds
  if [ $((i % 15)) -eq 0 ]; then
    log "  Still loading... (${i}s elapsed)"
  fi
  sleep 1
done

if [ "$VLLM_READY" = true ]; then
  ok "vLLM running (PID: $VLLM_PID)"
else
  warn "vLLM still loading after 3 minutes. Last log lines:"
  tail -10 /workspace/vllm.log
  fail "vLLM timeout. Check /workspace/vllm.log"
fi

# =============================================================================
# Step 6: Verify both backends
# =============================================================================
step "6/7: Verification"

echo ""
log "Checking Ollama (port 11434)..."
OLLAMA_MODELS_JSON=$(curl -sf http://localhost:11434/v1/models 2>/dev/null || echo "FAILED")
if [ "$OLLAMA_MODELS_JSON" != "FAILED" ]; then
  OLLAMA_MODEL_COUNT=$(echo "$OLLAMA_MODELS_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo "?")
  ok "Ollama responding — ${OLLAMA_MODEL_COUNT} models available"
else
  warn "Ollama not responding on /v1/models"
fi

log "Checking vLLM (port ${VLLM_PORT})..."
VLLM_MODELS_JSON=$(curl -sf http://localhost:${VLLM_PORT}/v1/models 2>/dev/null || echo "FAILED")
if [ "$VLLM_MODELS_JSON" != "FAILED" ]; then
  VLLM_MODEL_NAME=$(echo "$VLLM_MODELS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',[]); print(d[0]['id'] if d else '?')" 2>/dev/null || echo "?")
  ok "vLLM responding — serving: ${VLLM_MODEL_NAME}"
else
  warn "vLLM not responding on port ${VLLM_PORT}"
fi

# Quick inference tests
echo ""
log "Inference test — vLLM (${VLLM_MODEL})..."
VLLM_TEST=$(curl -sf --max-time 30 http://localhost:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$VLLM_MODEL"'",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }' 2>/dev/null || echo "FAILED")

if [ "$VLLM_TEST" != "FAILED" ]; then
  VLLM_REPLY=$(echo "$VLLM_TEST" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "parse error")
  ok "vLLM: \"${VLLM_REPLY}\""
else
  warn "vLLM inference test failed (may need more warmup time)"
fi

log "Inference test — Ollama (qwen3.5:9b)..."
OLLAMA_TEST=$(curl -sf --max-time 60 http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }' 2>/dev/null || echo "FAILED")

if [ "$OLLAMA_TEST" != "FAILED" ]; then
  OLLAMA_REPLY=$(echo "$OLLAMA_TEST" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "parse error")
  ok "Ollama: \"${OLLAMA_REPLY}\""
else
  warn "Ollama inference test failed (model may still be loading into GPU)"
fi

# =============================================================================
# Step 7: Connection info
# =============================================================================
step "7/7: Connection info"

# Detect RunPod pod ID for proxy URL hints
POD_ID="${RUNPOD_POD_ID:-$(hostname)}"

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${GREEN}  SETUP COMPLETE${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo "  Processes:"
echo "    Ollama:  PID $OLLAMA_PID  (port 11434)"
echo "    vLLM:    PID $VLLM_PID  (port $VLLM_PORT)"
echo ""
echo "  Logs:"
echo "    tail -f /workspace/ollama.log"
echo "    tail -f /workspace/vllm.log"
echo ""
echo "  RunPod Proxy URLs (for InfraBrain config):"
echo "    Ollama: https://${POD_ID}-11434.proxy.runpod.net/v1"
echo "    vLLM:   https://${POD_ID}-${VLLM_PORT}.proxy.runpod.net/v1"
echo ""
echo "  SSH Tunnel (alternative — faster, more reliable):"
echo "    ssh -L 11434:localhost:11434 -L ${VLLM_PORT}:localhost:${VLLM_PORT} root@<runpod-ip> -p <port> -i ~/.ssh/id_ed25519"
echo "    Then use http://localhost:11434/v1 and http://localhost:${VLLM_PORT}/v1"
echo ""
echo "  InfraBrain config.json:"
echo "    {"
echo "      \"defaultBaseUrl\": \"https://${POD_ID}-11434.proxy.runpod.net/v1\","
echo "      \"modelMap\": {"
echo "        \"triage\": { \"model\": \"${VLLM_MODEL}\", \"baseUrl\": \"https://${POD_ID}-${VLLM_PORT}.proxy.runpod.net/v1\" },"
echo "        \"default\": \"qwen3.5:122b-a10b\","
echo "        \"strategic\": \"qwen3.5:122b-a10b\","
echo "        \"forensic\": \"qwen3.5:122b-a10b\","
echo "        \"worker\": { \"model\": \"${VLLM_MODEL}\", \"baseUrl\": \"https://${POD_ID}-${VLLM_PORT}.proxy.runpod.net/v1\" },"
echo "        \"vision\": \"qwen3.5:122b-a10b\","
echo "        \"embedding\": \"bge-m3\""
echo "      }"
echo "    }"
echo ""
echo "  Quick test from your Mac:"
echo "    curl https://${POD_ID}-11434.proxy.runpod.net/v1/models"
echo "    curl https://${POD_ID}-${VLLM_PORT}.proxy.runpod.net/v1/models"
echo ""
echo -e "${BOLD}============================================================${NC}"
