#!/usr/bin/env bash
#
# Build the llama.cpp engine image that serves Qwen3.8-Flash-Next on an
# RTX 5090: mainline llama.cpp at a pinned commit, plus maxspeed.patch (the
# still-open upstream PRs this setup wants, merged onto that base), compiled
# for sm_120 only and baked into a small CUDA runtime image.
#
# One-time, about 15 minutes. Nothing is installed on the host: the compile
# runs inside a CUDA devel container, and the result is a local Docker image
#
#   llamacpp-qwen4exp:<commit12>-p<patchsha8>
#
# whose tag is keyed on the base commit and the patch hash, so bumping either
# rebuilds and the serving command in models/qwen3.8-flash-next-5090.md always
# names exactly the engine it was measured on.
#
# Requirements: Docker (the NVIDIA runtime is only needed to *serve*), curl,
# tar, sha256sum. ~2 GB of disk for the source tree and build.
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ── Pins (override via environment) ─────────────────────────────────────────
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
# b10705, 2026-08-30. Includes the merged qwen4exp PR #27742 and #28011.
LLAMACPP_COMMIT="${LLAMACPP_COMMIT:-2578138397d7b422bb0e160efdd429976c55fb55}"
# CUDA 13.1.2. sm_120 needs >= 12.8; if the pinned commit ever fails to
# compile against 13.x, drop BOTH images to 12.8.1.
BUILD_IMAGE="${BUILD_IMAGE:-nvidia/cuda:13.1.2-devel-ubuntu24.04}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-nvidia/cuda:13.1.2-runtime-ubuntu24.04}"
# Optional: pin the compile to a core set, e.g. BUILD_CPUSET=48-55, so a
# worker already serving on the same box is not starved.
BUILD_CPUSET="${BUILD_CPUSET:-}"
PATCH_FILE="${PATCH_FILE:-$HERE/maxspeed.patch}"
WORK="${WORK:-$HERE/../work-flash-next-5090}"

for t in docker curl tar sha256sum; do
    command -v "$t" >/dev/null || { echo "build-engine: $t is required"; exit 1; }
done

ENGINE_TAG="${LLAMACPP_COMMIT:0:12}"
[ -s "$PATCH_FILE" ] && ENGINE_TAG="${ENGINE_TAG}-p$(sha256sum "$PATCH_FILE" | cut -c1-8)"
ENGINE_IMAGE="${ENGINE_IMAGE:-llamacpp-qwen4exp:${ENGINE_TAG}}"

if docker image inspect "$ENGINE_IMAGE" >/dev/null 2>&1; then
    echo "Engine image $ENGINE_IMAGE already built."
    exit 0
fi

echo "=== llama.cpp @ ${LLAMACPP_COMMIT:0:12} + $(basename "$PATCH_FILE") -> $ENGINE_IMAGE"
mkdir -p "$WORK"
SRC="$(mktemp -d "$WORK/src.XXXXXX")"
# The build container runs as root, so its output is root-owned on the host;
# clean up through a container for the same reason.
cleanup() {
    docker run --rm -v "$SRC:/w" busybox:latest sh -c 'rm -rf /w/* /w/.[!.]*' >/dev/null 2>&1 || true
    rmdir "$SRC" 2>/dev/null || true
}
trap cleanup EXIT

echo "Fetching source tarball..."
curl -fsSL "${LLAMACPP_REPO%/}/archive/${LLAMACPP_COMMIT}.tar.gz" \
    | tar xz -C "$SRC" --strip-components=1

[ -s "$PATCH_FILE" ] && cp "$PATCH_FILE" "$SRC/.maxspeed.patch"

CPUSET_FLAG=()
[ -n "$BUILD_CPUSET" ] && CPUSET_FLAG=(--cpuset-cpus "$BUILD_CPUSET")

docker pull -q "$BUILD_IMAGE" >/dev/null || true
# -DCMAKE_CUDA_ARCHITECTURES=120  sm_120 only: every other arch triples the
#                                 compile for kernels a 5090 can never run.
# -DBUILD_SHARED_LIBS=OFF         one self-contained llama-server binary, so
#                                 the runtime image COPYs it and nothing else.
# -DLLAMA_CURL=OFF                the server loads only from /models.
docker run --rm "${CPUSET_FLAG[@]}" \
    -v "$SRC:/src" -w /src \
    -e DEBIAN_FRONTEND=noninteractive \
    "$BUILD_IMAGE" bash -ec '
        apt-get update -qq
        apt-get install -qq --yes --no-install-recommends \
            cmake ninja-build pkg-config git >/dev/null
        if [ -s .maxspeed.patch ]; then
            git apply --check .maxspeed.patch \
                || { echo "maxspeed.patch does not apply to this base; see models/qwen3.8-flash-next-5090.md before re-pinning"; exit 1; }
            git apply .maxspeed.patch
            echo "Applied maxspeed.patch."
        fi
        cmake -S . -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=120 \
            -DBUILD_SHARED_LIBS=OFF \
            -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=OFF
        cmake --build build --parallel --target llama-server
    '

echo "Baking runtime image $ENGINE_IMAGE..."
# llama.cpp's .dockerignore excludes build/, which would hide the binary
# from the runtime Dockerfile's COPY.
rm -f "$SRC/.dockerignore" 2>/dev/null || docker run --rm -v "$SRC:/w" busybox:latest rm -f /w/.dockerignore
docker build \
    --build-arg RUNTIME_IMAGE="$RUNTIME_IMAGE" \
    -f "$HERE/Dockerfile.runtime" \
    -t "$ENGINE_IMAGE" \
    "$SRC"

echo
echo "Built $ENGINE_IMAGE"
echo "Serve it with the docker run command in models/qwen3.8-flash-next-5090.md."
