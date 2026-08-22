#!/usr/bin/env bash
#
# Build the abliterated Qwen3.8-27B groupwise-int .ninfer artifact.
#
# End-to-end, reproducible: downloads the public abliterated base weights,
# grafts the official Qwen frontend resources (the converter sha-pins them),
# and runs NInfer's own converter to emit a single-file .ninfer artifact that
# ninfer-serve loads on an RTX 5090.
#
# Everything runs in pinned Docker containers, so the only host requirement is
# Docker with the NVIDIA container runtime. See README.md for the full walk-through.
#
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────────────

# Working directory. Holds ./ckpt (base weights, ~55 GB) and ./out (artifact).
W="${W:-$(pwd)/work}"

# NInfer engine/converter revision. This is the converter that defines the
# groupwise-int recipe (RECIPE_ID qwen3_8_27b-v1) and the frontend sha pins.
NINFER_REPO="${NINFER_REPO:-https://github.com/Neroued/ninfer}"
NINFER_COMMIT="${NINFER_COMMIT:-b2b96bae4dd88f95b9ea8126d68fae3b88caa374}" # 2026-08-18

# Base weights: the public abliterated ("uncensored") BF16 Qwen3.8-27B.
BASE_REPO="${BASE_REPO:-JonathanColetti/Qwen3.8-27B-Uncensored}"

# Official frontend source: the tokenizer / chat template / preprocessor
# resources the converter verifies against its built-in sha pins.
FRONTEND_REPO="${FRONTEND_REPO:-Qwen/Qwen3.8-27B}"

# Output artifact name.
OUT_FILE="${OUT_FILE:-qwen3_8_27b_uncensored.ninfer}"

# Container images (must match the runtime you serve on: CUDA 13.1, Ubuntu 24.04).
CONVERT_IMAGE="${CONVERT_IMAGE:-nvidia/cuda:13.1.2-runtime-ubuntu24.04}"

# GPU for the quantize step. "all" uses every visible GPU; pin a single card
# with a UUID (nvidia-smi -L) or index, e.g. GPU_SELECT="device=GPU-....".
# The quantize needs only ~10-11 GB of VRAM, so a small card is fine — it does
# NOT have to be the 5090 you ultimately serve on. Set DEVICE=cpu to skip the
# GPU entirely (slower, but cannot OOM and needs no NVIDIA runtime).
GPU_SELECT="${GPU_SELECT:-all}"
DEVICE="${DEVICE:-cuda}"

# Optional: pin the heavy steps to a CPU set, e.g. CPUSET="0-7".
CPUSET="${CPUSET:-}"

# ── Derived ──────────────────────────────────────────────────────────────────

CPUSET_FLAG=(); [ -n "$CPUSET" ] && CPUSET_FLAG=(--cpuset-cpus "$CPUSET")
GPU_FLAG=();    [ "$DEVICE" = "cuda" ] && GPU_FLAG=(--gpus "$GPU_SELECT")

mkdir -p "$W"/ckpt "$W"/out
cd "$W"

# ── [1/4] NInfer converter source ────────────────────────────────────────────
echo "=== [1/4] NInfer converter @ ${NINFER_COMMIT:0:12}"
if [ ! -f ninfer/tools/convert/qwen3_8_27b/convert.py ]; then
    mkdir -p ninfer
    curl -fsSL "${NINFER_REPO}/archive/${NINFER_COMMIT}.tar.gz" \
        | tar xz -C ninfer --strip-components=1
fi

# ── [2/4] Base weights (abliterated BF16, ~55 GB, resumable) ─────────────────
echo "=== [2/4] base weights: ${BASE_REPO}"
docker run --rm "${CPUSET_FLAG[@]}" -v "$W/ckpt:/ckpt" python:3.12-slim bash -ec '
    pip -q install "huggingface_hub[hf_transfer]" >/dev/null
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 - << PY
from huggingface_hub import snapshot_download
snapshot_download("'"$BASE_REPO"'", local_dir="/ckpt",
                  allow_patterns=["*.safetensors","*.json","*.jinja","*.txt"])
print("base weights complete")
PY'

# ── [3/4] Graft the official Qwen frontend (converter sha-pins these six) ─────
# The abliterated repo ships trivially-different tokenizer/config files; the
# converter refuses anything whose sha does not match its built-in pins, so we
# overwrite the six frontend resources with the official ones, then verify.
echo "=== [3/4] graft official frontend from ${FRONTEND_REPO}"
for f in tokenizer.json tokenizer_config.json chat_template.jinja \
         generation_config.json preprocessor_config.json video_preprocessor_config.json; do
    curl -fsSL -o "ckpt/$f" "https://huggingface.co/${FRONTEND_REPO}/resolve/main/$f"
done
python3 "$(dirname "$0")/verify_frontend.py" ckpt \
    || { echo "frontend pin check failed — see README (pins may have changed with NINFER_COMMIT)"; exit 1; }

# ── [4/4] Convert → groupwise-int .ninfer artifact ───────────────────────────
# NOTE: the converter shells out to `git rev-parse HEAD` to stamp the converter
# revision into its report. The stock CUDA image has no git, which makes the
# very last step crash *after* the artifact is fully written. We install git so
# the run completes cleanly (in a tarball checkout there is no .git, so the
# stamp is simply empty — that is fine and expected).
echo "=== [4/4] convert on --device ${DEVICE}"
docker run --rm "${CPUSET_FLAG[@]}" "${GPU_FLAG[@]}" \
    -v "$W:/w" -w /w/ninfer "$CONVERT_IMAGE" bash -ec '
    export DEBIAN_FRONTEND=noninteractive PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    apt-get update -qq
    apt-get install -qq --yes --no-install-recommends python3 python3-pip python3-venv git >/dev/null
    python3 -m venv /tmp/venv && . /tmp/venv/bin/activate
    pip -q install torch numpy safetensors >/dev/null
    python3 -m tools.convert.qwen3_8_27b.convert \
        --model /w/ckpt --out "/w/out/'"$OUT_FILE"'" --device "'"$DEVICE"'"'

echo "=== DONE"
ls -la out/
sha256sum "out/${OUT_FILE}"
[ -f "out/${OUT_FILE}.conversion.json" ] && echo "report: out/${OUT_FILE}.conversion.json"
