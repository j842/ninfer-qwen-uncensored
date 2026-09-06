# llama-server for Qwen3.8-Flash-Next on AMD Strix Halo: the LaurentZuijdwijk
# llama.cpp fork (branch vulkan/qwen4exp-rocmfpx) at a pinned commit, Vulkan
# backend only, built from source. Two reasons mainline cannot serve this
# GGUF, either sufficient on its own:
#   1. its Q4_0_ROCMFP4 / Q3_0_ROCMFPX tensor types exist only in the ROCmFPX
#      fork family;
#   2. its per-head n-gram (PLE) table layout, "ple16", is read only by this
#      fork. qwen4exp itself is in mainline since PR #27742.
#
# Vulkan rather than HIP: the ROCmFPX quants are benchmarked faster on Vulkan
# on gfx1151, the image is ~730 MB instead of many GB of ROCm, and the GPU
# passthrough is one /dev/dri render node.
ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION} AS build

ARG QWEN4EXP_REPO_URL=https://github.com/LaurentZuijdwijk/llama.cpp.git
ARG QWEN4EXP_REF=vulkan/qwen4exp-rocmfpx

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential cmake ninja-build wget gnupg \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# LunarG Vulkan SDK, the same source the upstream llama.cpp Vulkan image uses.
# Noble's stock vulkan-headers/glslc are too old for ggml's cooperative-matrix
# shaders.
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/lunarg.gpg \
    && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list \
        https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list \
    && apt-get update && apt-get install -y --no-install-recommends vulkan-sdk \
    && rm -rf /var/lib/apt/lists/*

RUN git clone "${QWEN4EXP_REPO_URL}" /opt/qwen4exp \
    && cd /opt/qwen4exp \
    && git checkout "${QWEN4EXP_REF}"

# The fork's README builds with just -DGGML_VULKAN=ON; the rest trims extras.
# GGML_NATIVE=OFF so an image built on another box still runs on the APU.
RUN cmake -S /opt/qwen4exp -B /opt/qwen4exp/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DGGML_HIP=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_WEBUI=OFF \
        -DLLAMA_USE_PREBUILT_WEBUI=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_BUILD_TESTS=OFF \
        -DLLAMA_CURL=ON \
    && cmake --build /opt/qwen4exp/build -j "$(nproc)" --target llama-server

# "*.so*", not "*.so": the fork's libraries carry versioned sonames
# (libllama-common.so.0) and the plain pattern misses them, which shows up
# at runtime as "cannot open shared object file".
RUN mkdir -p /app/lib \
    && find /opt/qwen4exp/build -name "*.so*" -exec cp -P {} /app/lib/ \; \
    && cp /opt/qwen4exp/build/bin/llama-server /app/

FROM ubuntu:${UBUNTU_VERSION}

# kisak-mesa: noble's stock Mesa predates Strix Halo, so its RADV never lists
# the GPU at all. The fork also gates its LDS-stride prefill path on
# RADV >= 25.3; kisak ships 26.1.x for noble. If the first start logs
# "no Vulkan devices found", run
#   docker run --rm --device /dev/dri/renderD128 --group-add <render gid> \
#       --entrypoint vulkaninfo <image> --summary
# and it must list "Radeon 8060S Graphics (RADV STRIX_HALO)".
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository -y ppa:kisak/kisak-mesa \
    && apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers vulkan-tools libgomp1 libcurl4t64 curl \
    && apt-get remove -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/lib/ /usr/local/lib/
COPY --from=build /app/llama-server /usr/local/bin/llama-server
ENV LD_LIBRARY_PATH=/usr/local/lib

STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/llama-server"]
