#!/usr/bin/env bash
#
# Build the llama.cpp engine image for Qwen3.8-Flash-Next on an RTX 3090 in a
# multi-NUMA-node host (EPYC Naples class): mainline llama.cpp at a pinned
# commit plus the same maxspeed.patch the 5090 build uses (#28243 MTP, #28068,
# #28213, unsloth #144) plus #28330 (skip the unused V cache of the QSA
# indexer), compiled for sm_86 only, baked into a CUDA runtime image WITH
# numactl (the interleave entrypoint is what multi-node decode depends on):
#
#   llamacpp-qwen4exp-numactl:<commit12>-p<stacksha8>-sm86
#
# 30-60 minutes on an idle Zen 1 EPYC, once. Requirements: Docker, curl, tar,
# sha256sum, ~2 GB of disk.
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ── Pins (override via environment) ─────────────────────────────────────────
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
# b10819, 2026-09-05. Carries qwen4exp (#27742) and the merged follow-ups
# #27941, #28023, #28123, #28040, #27970.
LLAMACPP_COMMIT="${LLAMACPP_COMMIT:-74a7c897f049c17e7080423aa2111776eff6ebbf}"
# sm_86 = RTX 3090 (GA102) and RTX 3060 (GA106) alike.
CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_IMAGE="${BUILD_IMAGE:-nvidia/cuda:13.1.2-devel-ubuntu24.04}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-nvidia/cuda:13.1.2-runtime-ubuntu24.04}"
# numactl >= 2.0.19 baked into the runtime image; distro packages are older.
NUMACTL_VERSION="${NUMACTL_VERSION:-2.0.19}"
BUILD_CPUSET="${BUILD_CPUSET:-}"
# The patch STACK, applied in order. maxspeed.patch is shared with the 5090
# build — one copy in this repo, byte-identical to what runs there.
PATCHES=("${HERE}/../flash-next-5090/maxspeed.patch" "${HERE}/28330-indexer-vcache.patch")
WORK="${WORK:-$HERE/../work-flash-next-3090}"

for t in docker curl tar sha256sum; do
    command -v "$t" >/dev/null || { echo "build-engine: $t is required"; exit 1; }
done

PH=""
for p in "${PATCHES[@]}"; do
    [ -s "$p" ] && PH="${PH}$(sha256sum "$p" | cut -d' ' -f1)"
done
ENGINE_TAG="${LLAMACPP_COMMIT:0:12}"
[ -n "$PH" ] && ENGINE_TAG="${ENGINE_TAG}-p$(printf '%s' "$PH" | sha256sum | cut -c1-8)"
ENGINE_IMAGE="${ENGINE_IMAGE:-llamacpp-qwen4exp-numactl:${ENGINE_TAG}-sm${CUDA_ARCH}}"

if docker image inspect "$ENGINE_IMAGE" >/dev/null 2>&1; then
    echo "Engine image $ENGINE_IMAGE already built."
    exit 0
fi

echo "=== llama.cpp @ ${LLAMACPP_COMMIT:0:12} + ${#PATCHES[@]} patches -> $ENGINE_IMAGE"
mkdir -p "$WORK"
SRC="$(mktemp -d "$WORK/src.XXXXXX")"
cleanup() {
    docker run --rm -v "$SRC:/w" busybox:latest sh -c 'rm -rf /w/* /w/.[!.]*' >/dev/null 2>&1 || true
    rmdir "$SRC" 2>/dev/null || true
}
trap cleanup EXIT

echo "Fetching source tarball..."
curl -fsSL "${LLAMACPP_REPO%/}/archive/${LLAMACPP_COMMIT}.tar.gz" \
    | tar xz -C "$SRC" --strip-components=1

i=0
for p in "${PATCHES[@]}"; do
    [ -s "$p" ] && cp "$p" "$SRC/.patch-$i-$(basename "$p")"
    i=$((i+1))
done

CPUSET_FLAG=()
[ -n "$BUILD_CPUSET" ] && CPUSET_FLAG=(--cpuset-cpus "$BUILD_CPUSET")

docker pull -q "$BUILD_IMAGE" >/dev/null || true
docker run --rm "${CPUSET_FLAG[@]}" \
    -v "$SRC:/src" -w /src \
    -e DEBIAN_FRONTEND=noninteractive \
    -e CUDA_ARCH="$CUDA_ARCH" \
    "$BUILD_IMAGE" bash -ec '
        apt-get update -qq
        apt-get install -qq --yes --no-install-recommends \
            cmake ninja-build pkg-config git >/dev/null
        for p in .patch-*; do
            [ -s "$p" ] || continue
            git apply --check "$p" \
                || { echo "$p does not apply to this base; see models/qwen3.8-flash-next-3090-epyc.md before re-pinning"; exit 1; }
            git apply "$p"
            echo "Applied $p"
        done
        cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
            -DBUILD_SHARED_LIBS=OFF \
            -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=OFF
        cmake --build build --parallel --target llama-server
    '

echo "Baking runtime image $ENGINE_IMAGE (llama-server + numactl ${NUMACTL_VERSION})..."
rm -f "$SRC/.dockerignore" 2>/dev/null || docker run --rm -v "$SRC:/w" busybox:latest rm -f /w/.dockerignore
docker build \
    --build-arg RUNTIME_IMAGE="$RUNTIME_IMAGE" \
    --build-arg NUMACTL_VERSION="$NUMACTL_VERSION" \
    -f "$HERE/Dockerfile.runtime" \
    -t "$ENGINE_IMAGE" \
    "$SRC"

echo
echo "Built $ENGINE_IMAGE"
echo "Serve it with the docker run command in models/qwen3.8-flash-next-3090-epyc.md."
