#!/usr/bin/env bash
#
# Build the Ornith-1.5-35B-A3B (MoE + distilled MTP head) .ninfer artifact.
#
# Same shape as build.sh, but for NInfer's qwen3_6_35b_a3b target:
# downloads the Ornith merged checkpoint (an exact tensor-contract match for
# Qwen3.6-35B-A3B — 1045 tensors, identical names/shapes/dtypes, MTP head in
# model-mtp.safetensors), the DFlash companion draft model, grafts the official
# Qwen3.6-35B-A3B frontend resources (the converter sha-pins them), and runs
# NInfer's converter to emit a single-file .ninfer artifact for an RTX 5090.
#
# Everything runs in pinned Docker containers, so the only host requirement is
# Docker with the NVIDIA container runtime. See README.md for the walk-through.
#
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────────────

# Working directory. Holds ./ckpt (base weights, ~72 GB), ./dflash (~1.7 GB)
# and ./out (artifact, ~21.2 GB).
W="${W:-$(pwd)/work-ornith}"

# NInfer engine/converter revision. This is the converter that defines the
# groupwise-int recipe (RECIPE_ID qwen3_6_35b_a3b-v2) and the frontend sha pins.
NINFER_REPO="${NINFER_REPO:-https://github.com/Neroued/ninfer}"
NINFER_COMMIT="${NINFER_COMMIT:-b2b96bae4dd88f95b9ea8126d68fae3b88caa374}" # 2026-08-18

# Base weights: Ornith-1.5-35B-A3B with the MTP head distilled from
# Qwen3.6-35B-A3B, merged into the Qwen3.6-35B-A3B checkpoint layout.
BASE_REPO="${BASE_REPO:-shisa-ai/Ornith-1.5-35B-A3B-MTP}"

# DFlash companion draft model (required by the converter; the artifact embeds
# it and ninfer-serve can use it for text-only speculation).
DFLASH_REPO="${DFLASH_REPO:-z-lab/Qwen3.6-35B-A3B-DFlash}"

# Official frontend source: the tokenizer / chat template / preprocessor
# resources the converter verifies against its built-in sha pins.
FRONTEND_REPO="${FRONTEND_REPO:-Qwen/Qwen3.6-35B-A3B}"

# Output artifact name.
OUT_FILE="${OUT_FILE:-ornith_1_5_35b_a3b.ninfer}"

# Container images (must match the runtime you serve on: CUDA 13.1, Ubuntu 24.04).
CONVERT_IMAGE="${CONVERT_IMAGE:-nvidia/cuda:13.1.2-runtime-ubuntu24.04}"

# GPU for the quantize step. "all" uses every visible GPU; pin a single card
# with a UUID (nvidia-smi -L) or index, e.g. GPU_SELECT="device=GPU-....".
# The quantize does NOT have to run on the 5090 you ultimately serve on.
# Set DEVICE=cpu to skip the GPU entirely (slower, but cannot OOM and needs no
# NVIDIA runtime).
GPU_SELECT="${GPU_SELECT:-all}"
DEVICE="${DEVICE:-cuda}"

# Optional: pin the heavy steps to a CPU set, e.g. CPUSET="0-7".
CPUSET="${CPUSET:-}"

# Optional: a HuggingFace token for the weight downloads. Not required — all
# inputs are public — but authenticated requests get higher rate limits, which
# matters for the ~72 GB base pull.
export HF_TOKEN="${HF_TOKEN:-}"

# ── Derived ──────────────────────────────────────────────────────────────────

CPUSET_FLAG=(); [ -n "$CPUSET" ] && CPUSET_FLAG=(--cpuset-cpus "$CPUSET")
GPU_FLAG=();    [ "$DEVICE" = "cuda" ] && GPU_FLAG=(--gpus "$GPU_SELECT")

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$W"/ckpt "$W"/dflash "$W"/out
cd "$W"

# ── [1/5] NInfer converter source ────────────────────────────────────────────
echo "=== [1/5] NInfer converter @ ${NINFER_COMMIT:0:12}"
if [ ! -f ninfer/tools/convert/qwen3_6_35b_a3b/convert.py ]; then
    mkdir -p ninfer
    curl -fsSL "${NINFER_REPO}/archive/${NINFER_COMMIT}.tar.gz" \
        | tar xz -C ninfer --strip-components=1
fi

# ── [2/5] Base weights (BF16, ~72 GB, resumable) ─────────────────────────────
echo "=== [2/5] base weights: ${BASE_REPO}"
docker run --rm "${CPUSET_FLAG[@]}" -e HF_TOKEN -v "$W/ckpt:/ckpt" python:3.12-slim bash -ec '
    pip -q install "huggingface_hub[hf_transfer]" >/dev/null
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 - << PY
from huggingface_hub import snapshot_download
snapshot_download("'"$BASE_REPO"'", local_dir="/ckpt",
                  allow_patterns=["*.safetensors","*.json","*.jinja","*.txt"])
print("base weights complete")
PY'

# ── [3/5] DFlash companion weights (~1.7 GB) ─────────────────────────────────
echo "=== [3/5] dflash weights: ${DFLASH_REPO}"
docker run --rm "${CPUSET_FLAG[@]}" -e HF_TOKEN -v "$W/dflash:/dflash" python:3.12-slim bash -ec '
    pip -q install "huggingface_hub[hf_transfer]" >/dev/null
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 - << PY
from huggingface_hub import snapshot_download
snapshot_download("'"$DFLASH_REPO"'", local_dir="/dflash",
                  allow_patterns=["config.json","model.safetensors"])
print("dflash weights complete")
PY'

# ── [4/5] Graft the official Qwen frontend (converter sha-pins these six) ─────
# Ornith already ships five of the six byte-identically; only its
# chat_template.jinja differs (it keeps prior-turn <think> blocks). The
# converter refuses anything whose sha does not match its built-in pins, so we
# overwrite all six with the official ones, then verify.
echo "=== [4/5] graft official frontend from ${FRONTEND_REPO}"
# Runs in a container because the download step leaves ckpt/ root-owned under
# rootful Docker, so a host-side curl cannot overwrite the six files.
docker run --rm -i "${CPUSET_FLAG[@]}" -v "$W/ckpt:/ckpt" python:3.12-slim python3 - << PY
import urllib.request
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "preprocessor_config.json",
          "video_preprocessor_config.json"]:
    urllib.request.urlretrieve(
        "https://huggingface.co/${FRONTEND_REPO}/resolve/main/" + f, "/ckpt/" + f)
    print("grafted", f)
PY
python3 "$REPO_DIR/verify_frontend.py" ckpt "$REPO_DIR/frontend-qwen3_6_35b_a3b.sha256" \
    || { echo "frontend pin check failed — see README (pins may have changed with NINFER_COMMIT)"; exit 1; }

# ── [5/5] Convert → groupwise-int .ninfer artifact ───────────────────────────
# NOTE: the converter shells out to `git rev-parse HEAD` to stamp the converter
# revision into its report. The stock CUDA image has no git, which makes the
# very last step crash *after* the artifact is fully written. We install git so
# the run completes cleanly (in a tarball checkout there is no .git, so the
# stamp is simply empty — that is fine and expected).
echo "=== [5/5] convert on --device ${DEVICE}"
docker run --rm "${CPUSET_FLAG[@]}" "${GPU_FLAG[@]}" \
    -v "$W:/w" -w /w/ninfer "$CONVERT_IMAGE" bash -ec '
    export DEBIAN_FRONTEND=noninteractive PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    apt-get update -qq
    apt-get install -qq --yes --no-install-recommends python3 python3-pip python3-venv git >/dev/null
    python3 -m venv /tmp/venv && . /tmp/venv/bin/activate
    pip -q install torch numpy safetensors >/dev/null
    python3 -m tools.convert.qwen3_6_35b_a3b.convert \
        --model /w/ckpt --dflash-model /w/dflash \
        --out "/w/out/'"$OUT_FILE"'" --device "'"$DEVICE"'"'

echo "=== DONE"
ls -la out/
sha256sum "out/${OUT_FILE}"
[ -f "out/${OUT_FILE}.conversion.json" ] && echo "report: out/${OUT_FILE}.conversion.json"
