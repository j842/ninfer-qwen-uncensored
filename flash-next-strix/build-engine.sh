#!/usr/bin/env bash
#
# Build the llama.cpp engine image that serves Qwen3.8-Flash-Next on an AMD
# Strix Halo APU: the LaurentZuijdwijk fork at a pinned commit, Vulkan only,
# baked into a small Ubuntu image with a Strix-capable RADV.
#
# One-time, 10-20 minutes. Nothing is installed on the host. The result is a
# local Docker image
#
#   llamacpp-qwen4exp-vulkan:<commit12>
#
# keyed on the fork commit, so the serving command in
# models/qwen3.8-flash-next-strix-halo.md names exactly the engine it was
# measured on.
#
# Requirements: Docker. The GPU is only needed to *serve*.
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ── Pins (override via environment) ─────────────────────────────────────────
QWEN4EXP_REPO_URL="${QWEN4EXP_REPO_URL:-https://github.com/LaurentZuijdwijk/llama.cpp.git}"
# Head of vulkan/qwen4exp-rocmfpx on 2026-08-31, the commit the quant card's
# numbers were measured on. Carries the Vulkan large-k TOP_K radix path and
# the QSA pooled-key cache that the older pins lacked.
QWEN4EXP_COMMIT="${QWEN4EXP_COMMIT:-5e085d123}"

command -v docker >/dev/null || { echo "build-engine: docker is required"; exit 1; }

ENGINE_IMAGE="${ENGINE_IMAGE:-llamacpp-qwen4exp-vulkan:${QWEN4EXP_COMMIT:0:12}}"

if docker image inspect "$ENGINE_IMAGE" >/dev/null 2>&1; then
    echo "Engine image $ENGINE_IMAGE already built."
    exit 0
fi

echo "=== $QWEN4EXP_REPO_URL @ $QWEN4EXP_COMMIT -> $ENGINE_IMAGE"
docker build \
    --build-arg "QWEN4EXP_REPO_URL=$QWEN4EXP_REPO_URL" \
    --build-arg "QWEN4EXP_REF=$QWEN4EXP_COMMIT" \
    -f "$HERE/Dockerfile" \
    -t "$ENGINE_IMAGE" \
    "$HERE"

echo
echo "Built $ENGINE_IMAGE"
echo "Serve it with the docker run command in models/qwen3.8-flash-next-strix-halo.md."
