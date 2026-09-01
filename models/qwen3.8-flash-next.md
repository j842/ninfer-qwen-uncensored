# Qwen3.8-Flash-Next (SGLang, RTX PRO 6000)

Not NInfer, not a 5090, and there is no artifact to download. This page is here
because Flash-Next on one RTX PRO 6000 is the fastest thing we serve, and
because getting it there needed a seven-piece patch stack that is not obvious
from any single upstream page.

**257 tok/s prose and 324 tok/s code at shallow depth, 236 / 288 at 250K tokens
of context**, single stream, on one card, at the full 262,144-token native
window. Measured 2026-08-31. Before the patch stack, the same card and the same
checkpoint did 139.60 / 186.48 and could not exceed 196,608 tokens of context.

## The model

Qwen3.8-Flash-Next is the open-weight preview of the Qwen4 architecture.

| | |
|---|---|
| **Size** | 180B total: 125B core (6B active), 51B PLE n-gram embedding table, 4B MTP head |
| **Experts** | 512 routed, 10 firing per token, `moe_intermediate_size` 640 |
| **Layers** | 48: 36 gated-DeltaNet (linear attention) + 12 QSA sparse-attention |
| **Context** | 262,144 native. Only the 12 QSA layers carry KV, at 2 KV heads x 256 head_dim |
| **Modality** | Text + vision (image; video encoder disabled here to keep its worst-case reservation out of the KV budget) |

The KV geometry is why this fits at all: 12 of 48 layers carrying KV at fp8
costs 12 KiB/token, so a full 262,144-token sequence is 3.0 GB.

## The checkpoint

[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4),
135.3 GB. Only the routed experts are quantised (NVFP4 W4A4, group 16, FP8 E4M3
block scales); attention, QSA, GDN, mHC, shared experts, routers, embeddings,
lm_head, the vision tower and all 31 MTP tensors stay BF16 and byte-identical
to first-party.

It is the most-validated quant of this model on the hub, and validated in
public rather than asserted: GSM8K 97.27% against a 97.12-97.50 BF16 reference
band, AIME26 98.75 pass@1 and majority@8 100, plus structural (294,912 routed +
1,562 unchanged + 31 MTP tensors), scale (221,184 finite positive scales) and
byte-equality audits over 1,562 tensors and 118.4 GB. Its card notes one
behavioural caveat: single-turn accuracy is preserved, but long agentic
generations tend to run longer than BF16.

The alternatives, for the record:

- `lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8`, 132.6 GB, same three-way split,
  far less validation.
- `Qwen/Qwen3.8-Flash-Next-FP8`, 185.6 GB first-party, 125.2 GiB GPU-resident.
  Does not fit a 96 GB card.
- `Qwen/Qwen3.8-Flash-Next`, 360 GB BF16, reference only.

vLLM cannot serve this model at all on this card: it loads only the
first-party FP8 build, and it has no FP8-PLE loader for the 4-bit ones.

## What fits where

Every figure below was read off the server log on first boot, not estimated.

```
GPU   81.35 GB  weights at load (68.0 NVFP4 routed experts + BF16 dense)
    +  0.51 GB  MTP draft head (NVFP4; the 5.21 GB of BF16 mtp.* on disk is
                not what lands in VRAM)
    +  2.06 GB  GDN/mamba state, 36 slots
    +  3.00 GB  KV pool at 262,144 tokens, fp8: exactly 12 KiB/token
    +  0.26 GB  draft KV
    +  1.44 GB  workspace/misc
    +  2.23 GB  CUDA graphs (target-verify 1.33 + draft decode 0.66 + extend 0.24)
    =  3.14 GB  free (available_gpu_mem after graph capture), on a 95.6 GiB card
HOST 51.2 GB    FP8 n-gram (PLE) table, pinned, via --ple-offload-embedding
```

Two things in there are worth knowing before you tune anything.

The **1.33 GB target-verify CUDA graph** is the biggest new consumer, and it
exists only because patch `0001b` admits sm_120 to the FlashInfer MTP verify
path. Without the patch the card fell back to the Triton verifier and never
captured that graph. So fp8 KV handed back 1.5 GB and 65,536 extra tokens of
context, and the verify graph spent most of the 1.5 GB again.

Patch `0005` is **not** a memory optimisation. It builds per-row-scaled fp8
copies of every dense weight over 4 MB and serves those; the bf16 originals and
the NVFP4 checkpoint are untouched. Upstream states the copies cost ~3.6 GB of
VRAM, but on this box they do not appear at weight load (`Load weight end`
reports the same 81.35 GB as the unpatched engine), so they are either built
lazily or accounted elsewhere, most likely inside that 1.44 GB of workspace.
All the memory headroom here comes from fp8 KV alone.

## The engine: stock image plus seven patches

The base is the stock `lmsysorg/sglang` image, **pinned by digest**
(`sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`, tag
`qwen38flashnext`) rather than by tag, because a moving tag can be repointed
under a seven-piece patch stack.

| Patch | What it does | Why it matters |
|---|---|---|
| PR [#36556](https://github.com/sgl-project/sglang/pull/36556) | Full sm_120 QSA fix: widens the TRTLLM sparse-decode gate to accept SM120 *and* routes the fallback to SGLang's own FA4 dispatcher | Load-bearing. See the corruption warning below. Also fixes the `qwen3_coder` tool parser looping on token 0 with thinking on ([#36537](https://github.com/sgl-project/sglang/issues/36537)) |
| `0001b` | Ports jpezzulli's RecoverSSM + WY output-only MTP verify to sm_120 | Stock gate is `supports_target_verify = sm_major in (9, 10)`, so a major-12 card took the Triton GDN verify kernel. Now one launch over all T draft tokens instead of a per-token state kernel |
| `0002` | Dequantises fp8 tiles for QSA sparse prefill | The whole reason this setup exists. Fixes [#36545](https://github.com/sgl-project/sglang/issues/36545), so `--kv-cache-dtype fp8_e4m3` works and KV halves to 12 KiB/token |
| `0003` | fp32 prefill state for the sm_120 FlashInfer GDN kernel | flashinfer's `gdn_prefill.py` permits bf16 only when compute capability == 10 exactly, and 12.0 is not 10. Retires the triton-prefill / flashinfer-decode split |
| `0004` | Routes low-M bf16 dense GEMMs to a tuned Triton split-K kernel | cuBLAS under CUDA-graph capture serves the decode projections at 20-75% of DRAM bandwidth on sm_120; this reaches ~90%. Verify graph 14.43 -> 13.79 ms/replay |
| `0005` | W8A16 fp8 weight-only serving for low-M dense GEMMs | ~85% of per-step traffic read at half width |
| `0006` | fp8 weight-only for the HyperConnection mix and the decode-size lm_head | HC mix 14.5 -> 10.8 us. Also halves every MTP draft step's logits GEMV |

Patches `0001b` and `0003`-`0006` come from
[gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120),
which in turn ports the sm_120 RecoverSSM/WY and fp8-QSA groundwork from
[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000).
`0002` is jpezzulli's. PR #36556 is separate from all six and composes with
them, because it touches a different file.

### Why patching beats building

Every patch is pure Python under `python/sglang/**`, and the one new kernel is
Triton, which JITs at runtime. So there is no source build, no CUDA toolchain,
and no 47 GB image push to a private registry. A small script edits the
installed tree in place at container start, then the entrypoint execs SGLang.

Three details that cost real time to find:

- `0002`, `0004` and `0005` apply strictly with `git apply`. `0001b`, `0003`
  and `0006` need `patch -F3`, one hunk each, all pure context drift, because
  the patches were cut against `qwen4-main-squashed` on 2026-08-30/31 and the
  pinned image was pushed 2026-08-26. `0006`'s rejected hunk is literally a
  line wrap of `out = torch.empty((rows, hs), ...)`.
- `git apply -3` is not available. The image's `.git` is stale: HEAD is an
  unrelated AMD CI commit and the Qwen4 files are not in the index at all. Only
  working-tree application works.
- GNU `patch` must be invoked with `--batch --forward` and closed stdin.
  Without them it *prompts* on an already-applied hunk, and in a container
  entrypoint that is not a failure, it is a hang.

Because fuzzy application can in principle land a hunk in the wrong place, it
is never trusted on its own. The patch script ends every run by asserting ten
postconditions, the specific symbols each patch should have produced, and
refuses to start the server if any is unmet. Run that verification in a **fresh
interpreter**: if the script imported the modules earlier (to test
idempotency), every `hasattr` is answered from a stale `sys.modules` entry
while every source check reads the new bytes off disk, and that split brain
reports a correctly-patched tree as broken.

The container runs with `--restart unless-stopped`, so a restart re-runs the
script against the same, already-patched writable layer. If all ten
postconditions already hold, it is a no-op.

## A green health check is not sufficient on this card

This is the most dangerous failure mode here, and it survives every smoke test
you would think to run.

The stock image's Qwen Sparse Attention decode path is broken on sm_120. The
*obvious* fix, rerouting only the varlen fallback, boots, serves, and passes a
short smoke test, and then silently emits token ID 0 (`!`) past the first KV
page, at a fixed *absolute* position that tracks `--page-size` (~65 at page 64,
~193 at page 128).

```
"The capital of France is"  ->  "The capital of France is **Paris**."      ✓ fine
"Explain tidal power..."    ->  "...which drive the rise and fall of!!!!!!!!!"
```

PR #36556 is the full fix. After any install or image change, drive a
generation well past the page boundary and grep for the token-0 signature. That
is the only way to tell a healthy worker from a poisoned one.

## Launching it

Four things have to exist on the host before this command works:

- the checkpoint, downloaded from Hugging Face into a directory you mount at
  `/models`;
- `patches/`, holding the six `.patch` files from
  [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120);
- `sm120-patch.py`, the script that applies those six plus PR #36556 and then
  asserts the ten postconditions described above. It is about 400 lines and
  specific to our deployment, so it is not vendored here; the section above is
  the full specification if you write your own;
- `hot_tokens_64k.pt`, the 64K FR-Spec token map. Drop
  `--speculative-token-map` if you do not have one, at the cost of about 9% of
  decode throughput.

```bash
docker run -d --name qwen38-flash-next \
    --gpus '"device=0"' \
    --network host --ipc host \
    --shm-size 16g --memory 112g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_PTRACE \
    -v /path/to/Qwen3.8-Flash-Next-NVFP4:/models:ro \
    -v "$PWD/patches:/patches:ro" \
    -v "$PWD/patches/hot_tokens_64k.pt:/hot_tokens_64k.pt:ro" \
    -v "$PWD/sm120-patch.py:/sm120-patch.py:ro" \
    -v qwen38fn_jit_cache:/root/.cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
    -e SGLANG_SM120_LOWM_FP8_WEIGHT=1 \
    -e SGLANG_SM120_LM_HEAD_FP8=1 \
    -e XDG_CACHE_HOME=/root/.cache \
    -e MAX_JOBS=4 -e FLASHINFER_NVCC_THREADS=2 \
    -e OMP_NUM_THREADS=8 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint bash \
    lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1 \
    -c 'python3 /sm120-patch.py && exec python3 -m sglang.launch_server \
        --model-path /models --served-model-name default --tp 1 \
        --trust-remote-code \
        --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
        --kv-cache-dtype fp8_e4m3 \
        --context-length 262144 --max-total-tokens 262144 \
        --mem-fraction-static 0.95 \
        --page-size 64 --chunked-prefill-size 4096 \
        --ple-offload-embedding \
        --linear-attn-prefill-backend flashinfer \
        --linear-attn-decode-backend flashinfer \
        --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer \
        --mamba-track-interval 64 --max-mamba-cache-size 36 \
        --gdn-mtp-cache-mode none \
        --max-running-requests 6 --cuda-graph-max-bs-decode 6 \
        --speculative-algorithm NEXTN --speculative-num-steps 3 \
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
        --speculative-accept-threshold-single 0.3 \
        --speculative-accept-threshold-acc 0.3 \
        --speculative-token-map /hot_tokens_64k.pt \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
        --sampling-defaults model \
        --enable-metrics --enable-request-time-stats-logging \
        --watchdog-timeout 1800 \
        --host 0.0.0.0 --port 8001'
```

Notes on the flags that are not self-explanatory:

- `--ple-offload-embedding` is what puts the 51.2 GB n-gram table in pinned
  host RAM instead of VRAM. `--ulimit memlock=-1` is mandatory because of it,
  not hygiene, and `--shm-size` matters because the PLE gather runs host-side
  on the decode critical path (hence `OMP_NUM_THREADS=8`).
- `--tp 1` is not a choice. `moe_intermediate_size` is 640, and split across
  TP>1 the per-expert shards go below the kernels' minimum tile.
- `--memory 112g` on a 125 GB box is a cage against a JIT storm taking the host
  down, given the 51 GB pinned table. `MAX_JOBS=4` is part of the same
  argument: unbounded `cicc` compilers at ~3 GB each will exhaust host RAM.
- Autotune is on and costs about 20 minutes on a cold JIT cache, on top of
  loading 135 GB of weights and filling the pinned table. That is why
  `/root/.cache` is a named volume with Triton, TorchInductor, FlashInfer and
  SGLang's JIT all steered into it. Destroy the volume and you pay it again.
  `--disable-flashinfer-autotune` skips it.
- `--watchdog-timeout 1800` exists because the predecessor died silently.
- A vendored chat template is optional. Drop `--chat-template` to use the one
  in the checkpoint. We carry a fork only because the upstream template raises
  on `reasoning_effort: high|max`, which our clients send.

## Measured performance

At shallow depth, against the same card and checkpoint before the patch stack:

| | unpatched | patched | change |
|---|---|---|---|
| prose decode | 139.60 | **257.09** | +84% |
| code decode | 186.48 | **323.58** | +74% |
| TTFT | ~106 ms | **94 ms** | -11% |

At depth, with the worker otherwise idle so the acceptance figures are clean:

| achieved depth | prose decode | prose accept | code decode | code accept | prefill tok/s |
|---|---|---|---|---|---|
| 4,996 | 248.67 | 2.69 | 309.40 | 3.36 | ~12,100 |
| 59,941 | 249.28 | 2.73 | 303.10 | 3.32 | ~12,600 |
| 189,896 | 243.85 | 2.74 | 297.59 | 3.33 | ~11,600 |
| **249,857** | **236.24** | 2.68 | **287.78** | 3.25 | ~11,000 |

Acceptance is out of a possible 4. Decode holds to 95% (prose) and 93% (code)
of the shallowest reading out to 250K, which is QSA doing at decode what the
architecture claims. The unpatched engine managed prose 156.1 / 163.8 / 153.2
and code 213.4 / 207.6 / 210.7 at the first three depths, and could not reach
the last row at all.

Upstream on this card reports 231 tok/s C1 median at temp 0.6 (243 best), 203
greedy, and 620-657 at C4; their stated baseline is 171. We are ahead of that
on a Workstation Edition at a 400 W cap, but their figure is a single blended
C1 number and these are per-shape, so treat the comparison as indicative.

If you benchmark this yourself, watch for cross-traffic. SGLang's acceptance
counters are worker-global, so any other request during a measurement window
contaminates the row. That is not hypothetical: an earlier sweep reported code
acceptance 2.69-2.97 against a true 3.26-3.37, which is exactly the margin a
`--speculative-num-steps` decision turned on.

## Speculation, and the one lossy setting

`--speculative-accept-threshold-single` and `-acc` are shipped at **0.3**,
which force-accepts a draft token the target model gives at least 30%
probability instead of doing exact rejection sampling on it. The upstream
ladder, C1 median at temp 0.6:

| threshold | tok/s | |
|---|---|---|
| 1.0 | 179 | lossless |
| 0.5 | 203 | |
| **0.3** | **231** | shipped |

Greedy is 203 regardless, because the threshold cannot change an argmax. So
"speculation is lossless by construction" stops being true below 1.0: it is
exact at temperature 0 and lossy above it, sharpening sampling toward
high-probability tokens. This model's card specifies temperature 1.0 for
thinking and 0.7 for non-thinking, so production traffic sits firmly in the
lossy regime.

The trap: a quality benchmark that grades at temperature 0 cannot see this
setting at all. It reports the same score at 0.3 as at 1.0 while real traffic
gets the sharpened distribution. If answers start feeling flat or
over-confident, set both thresholds to 1.0 before looking anywhere else.

Two related settings are not lossy:

- **FR-Spec** (`--speculative-token-map`) restricts the draft head to a 64K
  hot-token subset of the 248K vocab, making draft logits about 4x cheaper.
  Verification stays exact against the full target distribution, so a token
  outside the map can only fail to be *drafted*, never wrongly accepted. The
  only thing at risk is acceptance length, and upstream measured that
  unchanged: C1 200.8 -> 219.4, C4 541 -> 592.
- **Draft-head quantisation** is left to the checkpoint's own quant, which puts
  the MTP head in VRAM at 0.51 GB with measured acceptance 3.27-3.95 of 4.
  Upstream instead passes `unquant` for a bf16 head several GB larger. An A/B
  is unresolved; memory is the scarce resource here.

`--speculative-num-steps 3` is an architectural ceiling, not a preference:
`num_draft_tokens = steps + 1` must be at most the model's
`indexer_compress_ratio` of 4. Asking for 4 loads all 135 GB and *then* dies at
scheduler start with "Qwen QSA requires speculative_num_draft_tokens <= the QSA
compress ratio (4)". Check it before launch. Going the other way, 2 was
measured at depth and lost: code decode -9.0% (210.6 -> 191.7), prose a tie,
because the head nearly saturates its draft budget at 3.

## Capacity and stability

`--max-total-tokens` is pinned to exactly the context length. With fp8 KV the
memory fraction would allow a far larger pool (upstream reports ~572K tokens at
this context) and you should not let it. The pool and the context are the same
memory, so headroom comes from pinning the pool *below* what the fraction
allows. Pinning pool == context also preserves the invariant that a caller can
never send a request longer than the pool, and banks the entire fp8-KV saving
as free VRAM. Lowering `--mem-fraction-static` alone does not buy headroom, it
just shrinks the context.

`--max-mamba-cache-size 36` is six times `--max-running-requests`, and that
ratio is load-bearing: below it the speculative CUDA graphs quietly cap at
bs=4. SGLang also **silently clamps** `max_running_requests` when the mamba
state pool cannot back it, so always re-read `max_running_requests=N` from the
server log after changing either number. Asking for 8 with 32 slots quietly
yields 6.

GDN recurrent state is the expensive axis and it scales with concurrency, not
context: 36 linear layers x 48 V heads x 128 x 128 is ~113 MB per slot at fp32
and ~57 MB at bf16. Raising concurrency costs more here than raising context.
Hence `--mamba-ssm-dtype bfloat16`, which halves the pool. Do not go below
bf16; field reports have fp8 SSM state corrupting output.

Two upstream reports look like they condemn bf16 SSM state and neither applies
to this card. [#36611](https://github.com/sgl-project/sglang/issues/36611) /
[#36627](https://github.com/sgl-project/sglang/issues/36627) (MTP target-verify
asserting `initial_state must be float32`) is SM90 only, because
`FlashInferGDNKernels` installs the bf16 MTP adapter under `use_state_pool =
sm_major >= 10`, so Hopper misses it and Blackwell gets it.
[#36701](https://github.com/sgl-project/sglang/issues/36701) issue 2 is a
flashinfer *prefill* constraint, lifted by patch `0003`. That same issue
concludes "the flashinfer linear-attn backend is unreachable on sm_120" and
recommends triton/triton/float32, which is correct for an unpatched tree and
wrong for a patched one; taking its advice here costs both the faster kernel
and the half-size state pool.

**3.14 GB free is thin.** The unpatched predecessor died every 6 to 17 hours at
3.03 GB reported free, because in service the real figure fell to 0.24-0.90 GiB
and a lazily-loaded Triton kernel could not allocate. This configuration is
0.11 GB above that line. A 250K-depth sweep and a shallow benchmark both passed
cleanly, which is not evidence of multi-day stability; the predecessor also
benchmarked fine and then died overnight. Watch the logs for
`available_gpu_mem`, `device-loaded after serving`, `SIGQUIT` and `watchdog`.

If it proves unstable, the levers in increasing order of what they cost:

1. `--cuda-graph-max-bs-decode` 6 -> 4 frees ~0.7-0.9 GB of the 2.23 GB of
   graphs. Requests above bs=4 run eager rather than failing.
2. `--gdn-mtp-cache-mode` `none` -> `full` frees the whole 1.33 GB
   target-verify graph, at the cost of patch `0001b` (back to the Triton
   verifier).
3. Context and pool 262144 -> 196608 frees 0.75 GB and gives back the headline
   context gain. Last resort.

Not `--mem-fraction-static`, for the reason above.

## The same model on an RTX 5090

If a 96 GB card is not on the table, llama.cpp will serve Flash-Next on a
32 GB 5090 with every routed expert streaming from system RAM. It is about a
sixth of the speed and needs ~100 GB of free RAM, but it is the same weights
and the same 262K window.

At `--n-cpu-moe 48`, all experts on the CPU, which is the floor:

```
GPU   5.2 GiB  non-expert tensors (attn/GDN/QSA/routers, shared experts, embed)
   +  0.9 GiB  vision ViT (mmproj)
   +  6.0 GiB  KV at 262,144 (24 KiB/token, 12 of 48 layers)
   +  6.0 GiB  compute buffers + CUDA context/graphs (ubatch 2048)
   = 18.1 GiB  measured, of 31.8
HOST 71.7 GiB  routed experts, all 48 layers (mmap page cache)
   + 26.8 GiB  PLE n-gram table, IQ4_NL (host-side by design)
```

- Weights: unsloth **UD-Q4_K_XL** (103.7 GiB, 4 shards) from
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
  Not UD-IQ4_XS: it drops 94 of 144 expert tensors to IQ3_S, a 3.4-bit expert
  build wearing a 4-bit name, and on a model with 6B active parameters per
  token that is the wrong end of the trade. Decode here is not
  bandwidth-saturated anyway (~51 GB/s of a ~205 GB/s bus). Our deployed file
  is an imatrix requant of the same recipe with the MTP head grafted in as
  blk.48; the stock file serves fine, since speculation is off either way.
- Engine: llama.cpp PR
  [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) merged to master
  on 2026-08-27, so the base is now mainline. We run b10705 plus four
  correctness and depth cherry-picks: #27941 (QSA blocks keyed per sequence and
  rank order, the fix for a silent logit-drift class of bug), #28011 (kv-cells
  scan early exit, which dominated decode cost at depth), #28023 (indexer
  head-sum by slices) and #27977 (QSA gather-window decode split).
- `--n-cpu-moe 43` puts the first 43 layers' experts in RAM and keeps the tail
  five on the card, which is 30.0 GiB of the 32 and worth about 4 tok/s over
  the all-experts-on-CPU floor.
- Measured: 42-43 tok/s on short natural prompts, 40.9 at 6K depth, 38.2 at
  24K, prefill 936 / 873 / 848 tok/s at 2K / 8K / 32K. Decode is essentially
  flat with depth, which is the GDN+QSA architecture doing what it promises.
- **Speculation is off.** The model's own MTP head can be grafted into the GGUF
  as blk.48 and #27836's `draft-mtp` will load it, but on every patch set tried
  it corrupts output, progressively and mid-answer, while acceptance collapses
  from 63-66% to 3-11%. Upstream #28019 explains why: the recurrent-state
  rollback flag fails llama.cpp's own rollback test with a logits diff of 9.25,
  because PLE conv and QSA indexer state lack snapshot coverage. n-gram
  speculation measured about 0% on honest traffic here.
- Bench until consecutive runs agree. The first deploy read 23.7 tok/s at 8K
  and 16.1 at 32K on pass 2 and 84.1 / 82.0 on pass 3: what looked like a depth
  cliff was page-cache warming.

## Sources

- [SGLang cookbook: Qwen3.8-Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120), the six-patch stack
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000), the sm_120 groundwork it ports
- SGLang issues [#36531](https://github.com/sgl-project/sglang/issues/36531) (QSA corruption), [#36537](https://github.com/sgl-project/sglang/issues/36537) (tool parser token-0 loop), [#36545](https://github.com/sgl-project/sglang/issues/36545) (fp8 KV)
