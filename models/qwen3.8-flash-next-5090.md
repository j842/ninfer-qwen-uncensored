# Qwen3.8-Flash-Next (llama.cpp, RTX 5090)

The same 180B model as the [RTX PRO 6000 page](qwen3.8-flash-next.md), served
on a 32 GB RTX 5090 by llama.cpp with every routed expert streaming from
system RAM. Same weights, same 262,144-token window, about a sixth of the
speed. Not NInfer, and there is no artifact to download: you build a small
engine image from a pinned llama.cpp commit plus one vendored patch, and pull
a public GGUF.

**42-43 tok/s single-stream decode on short prompts, 40.9 at 6K tokens of
context and 38.2 at 24K, prefill 850-940 tok/s**, measured 2026-08-31, with
speculation off. Decode is nearly flat with depth, which is the GDN+QSA
architecture doing what it promises.

| | |
|---|---|
| **Engine** | llama.cpp b10705 (`2578138397d7`) + [`flash-next-5090/maxspeed.patch`](../flash-next-5090/maxspeed.patch), built by [`build-engine.sh`](../flash-next-5090/build-engine.sh) |
| **Weights** | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-Q4_K_XL, 103.7 GiB in 4 shards, at revision `c8b5954a` |
| **Vision** | `mmproj-BF16.gguf` from the same repo, 865 MiB |
| **Card** | one RTX 5090, 30.0 GiB of 31.8 used at `--n-cpu-moe 43` |
| **Host** | ~100 GiB of free RAM for the mmap page cache, as many physical cores as you can give it (48 here), NVMe |

## Why llama.cpp and not vLLM or SGLang

The model is 180B total: a 125B core with 6B active, a 51B n-gram (PLE)
embedding table, and a 4B MTP head. vLLM and SGLang keep the whole core
GPU-resident, so their floor is ~78 GiB at NVFP4, which is the 96 GB card on
the other page. llama.cpp's per-tensor placement (`--n-cpu-moe`) is the only
engine mechanism that puts the routed experts in system RAM. With 10 of 512
experts firing per token the DDR4 stream is about 1.4 GiB per token, not the
71.7 GiB the experts occupy, which is what makes this viable at all.

## What fits where

Read off the GGUF tensor directory, with the two GPU-side constants
calibrated so the arithmetic reproduces the measured first load.

```
GPU   5.2 GiB  non-expert tensors (attn/GDN/QSA/routers, shared experts, embed)
   +  0.9 GiB  vision ViT (mmproj)
   +  6.0 GiB  KV at 262,144 tokens (24 KiB/token f16; 12 of 48 layers carry KV)
   +  6.0 GiB  compute buffers + CUDA context/graphs (ubatch 2048)
   = 18.1 GiB  measured at --n-cpu-moe 48 (every expert on the CPU)
HOST 71.7 GiB  routed experts, all 48 layers (mmap page cache)
   + 26.8 GiB  PLE n-gram table, IQ4_NL (host-side by design)
```

`--n-cpu-moe N` keeps layers 0..N-1's experts in RAM and puts N..47 on the
card. The expert layers are not uniform: 1.46 GiB each, except layers 2, 4,
30, 46 and 47 at 1.71-1.90 GiB, and the tail that moves to the GPU contains
two of the fat ones.

| `--n-cpu-moe` | experts on GPU | VRAM | margin of 31.8 | |
|---|---|---|---|---|
| 48 | 0 | 18.1 GiB | 13.7 | floor, first deploy |
| **43** | 7.9 GiB | **30.0 GiB** | 2.6 | **shipped**, 43.4 tok/s |
| 42 | 9.3 GiB | 31.7 GiB peak at 32K | 0.9 | 43.8 tok/s, too thin |
| 40 | 12.2 GiB | 30.3 GiB at load | 1.5 | untested at depth |
| 38 | 15.1 GiB | 33.2 GiB | -1.4 | does not fit |

Watch **peak** VRAM through a deep-context run, not the load-time line: 42
loads fine and then peaks 0.9 GiB from the ceiling at 32K. Moving the tail
onto the card cuts DDR4 expert traffic (1.41 GiB/token at 48, 1.16 at 40) but
decode here is not bandwidth-saturated (~51 GB/s of a ~205 GB/s bus), so the
gain is smaller than the byte count suggests: 48 to 43 was worth about
4 tok/s on the same engine.

## The quant

UD-Q4_K_XL, from the tensor directories rather than the size labels:

| | UD-Q4_K_XL | UD-IQ4_XS |
|---|---|---|
| routed experts | **71.7 GiB**: Q4_K x94, Q5_1 x43, Q8_0 x5, Q5_K x2 | **55.5 GiB**: **IQ3_S x94**, IQ4_NL x43, Q8_0 x5, IQ4_XS x2 |
| PLE n-gram table | 26.8 GiB, IQ4_NL | 26.8 GiB, IQ4_NL |
| dense + shared + embed | 5.1 GiB | 4.9 GiB |
| total | 103.7 | 87.3 |

The PLE table is IQ4_NL in every quant on the ladder. The whole 16.4 GiB gap
between the two is in the routed experts, the tensors that stream over DDR4
every token, so the smaller file is +29% host traffic, not RAM you happen to
have spare. And UD-IQ4_XS drops 94 of 144 expert tensors to IQ3_S, a 3.4-bit
expert build wearing a 4-bit name. On a model with 6B active parameters per
token that is the wrong end of the trade, and since decode is not
bandwidth-bound the bytes it saves were never the lever.

Pin the revision. Unsloth republished UD-Q4_K_XL on 2026-08-28 as an imatrix
requant with different shard bytes; `c8b5954a88c2775c546b92593eda40ea041d3176`
is that one, and it is what every number on this page was measured on.

The open quant question is the PLE table: 51.2B parameters, 28% of the model,
at 4.5 bits here against FP8 on the RTX PRO 6000 build. It is the one tensor
materially below the other page. Raising it to Q8_0 costs about +24 GiB of
host RAM and near-zero decode, because the PLE is a per-token gather of a few
rows rather than a streamed matmul, but it needs a local requant
(`llama-quantize --tensor-type per_layer_token_embd=Q8_0`). That experiment
is in progress and has no verdict yet.

## The engine: mainline plus one patch

PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (the `qwen4exp`
architecture) merged to llama.cpp master on 2026-08-27, so the base is
mainline: release b10705, commit `2578138397d7`, which also carries the merged
[#28011](https://github.com/ggml-org/llama.cpp/pull/28011) (kv-cells scan
early exit, the dominant decode cost at depth before it).

On top, [`maxspeed.patch`](../flash-next-5090/maxspeed.patch) is a plain diff
of the still-open upstream work this setup wants, merged onto that base.
llama.cpp and its PRs are MIT, so unlike the SGLang patch stack it is vendored
here rather than fetched.

| Piece | Head | What it does | Status |
|---|---|---|---|
| [#27941](https://github.com/ggml-org/llama.cpp/pull/27941) | `868e2f52` | QSA blocks keyed per sequence and rank order, sequence-copy indexer keys, M-RoPE image blocks kept separate, metadata asserts turned into throws, the `gridDim.y` overflow fix at full KV depth, kv-unified NaN fix | **Load-bearing.** Fixes a silent logit-drift class of bug that only shows on long thinking chains (see below) |
| [#27879](https://github.com/ggml-org/llama.cpp/pull/27879) | `a7fc7e40` only | GDN QK normalisation as `rsqrt(sum+eps)`, matching the reference | One cherry-pick. Its rollback flag (`edb6dec`) is deliberately **not** taken: [#28019](https://github.com/ggml-org/llama.cpp/pull/28019) showed it corrupts multi-sequence recurrent state |
| [#28023](https://github.com/ggml-org/llama.cpp/pull/28023) | `ead00aed` | QSA indexer head-sum by slices | Prefill perf |
| [#27977](https://github.com/ggml-org/llama.cpp/pull/27977) | `db40b22d` | QSA gather-window decode split, w0-bounded predecessor scan, bitmap used-cells | Depth decode perf; author measured 44 to 63 tok/s at depth. Draft PR: first hunk to suspect if depth behaviour regresses |
| [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) | `1d8de7c1` | MTP/NextN draft head, `--spec-type draft-mtp` | Compiled in, **switched off**. Corrupts output on this build |
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861) | `bccbacdb` | GPU-resident LRU cache for host-offloaded experts, `--moe-expert-cache N` | Compiled in, off. Measured a loss here (see below) |

The image tag is keyed on the base commit plus the patch's sha256, so the
serving command below names exactly the engine that was measured. As pieces
merge upstream, drop their hunks; once the patch is empty, the source build
can be retired for the stock `ghcr.io/ggml-org/llama.cpp:server-cuda` image.

## Building it

```bash
./flash-next-5090/build-engine.sh   # → llamacpp-qwen4exp:2578138397d7-p50825275
```

That fetches the pinned commit as a tarball, applies the patch with
`git apply --check` first so a base bump that breaks it dies loudly, compiles
`llama-server` for `sm_120` only inside `nvidia/cuda:13.1.2-devel`, and bakes
the static binary into a `13.1.2-runtime` image. About 15 minutes; no CUDA
toolchain on the host. `BUILD_CPUSET=48-55` pins the compile away from
anything already serving on the box. If the pinned commit ever fails to
compile against CUDA 13, drop both images to 12.8.1; `sm_120` needs at least
12.8, so lower is not an option.

Download the four UD-Q4_K_XL shards at the pinned revision and the mmproj into
a directory you will mount at `/models`:

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
    --revision c8b5954a88c2775c546b92593eda40ea041d3176 \
    --include 'UD-Q4_K_XL/*' 'mmproj-BF16.gguf' \
    --local-dir /path/to/flash-next-gguf
```

## Launching it

```bash
docker run -d --name qwen38-flash-next-5090 \
    --restart unless-stopped \
    --gpus '"device=0"' \
    --ulimit memlock=-1 \
    --cpuset-cpus 0-47 \
    -p 8001:8080 \
    -v /path/to/flash-next-gguf:/models:ro \
    -e LLAMA_ATTN_ROT_DISABLE=1 \
    llamacpp-qwen4exp:2578138397d7-p50825275 \
    --model /models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --mmproj /models/mmproj-BF16.gguf \
    --alias Qwen3.8-Flash-Next \
    --host 0.0.0.0 --port 8080 \
    -ngl 999 --n-cpu-moe 43 \
    --ctx-size 262144 --parallel 1 \
    --flash-attn on --cache-type-k f16 --cache-type-v f16 \
    --threads 48 --threads-batch 48 \
    --batch-size 2048 --ubatch-size 2048 \
    --load-mode mmap \
    --cont-batching --metrics --slots --jinja
```

`/health` answers in under a minute, once the ~10 GiB of dense tensors reach
VRAM. The ~100 GiB of experts and PLE table fault in from disk on demand, so
the first requests are far slower than steady state. Notes on the flags that
are not self-explanatory:

- `--cpuset-cpus 0-47` and `--threads 48`: physical cores only, no SMT. SMT
  siblings were measured as a loss for expert streaming, and llama.cpp's own
  `--cpu-mask` / `--cpu-range` did not place threads where the docker cpuset
  does. Adjust both to your core count. Anything else that streams from
  DDR4 on the same box will roughly halve this worker's decode and its own.
- `--n-cpu-moe 43`: see the table above. If the container dies at load with
  a CUDA OOM, raise it; each tail layer is 1.46-1.90 GiB. At 48 there is
  nothing left to shift and the budget is over from the dense side, so lower
  `--ctx-size` (each 32K of KV is ~0.75 GiB), `--ubatch-size` (2048 to
  1024), or drop `--mmproj`.
- `--load-mode mmap`: mlock would populate 100 GiB from the loading thread
  and pin it against everything else on the host; direct I/O bypasses the
  page cache the lazy expert faulting depends on.
- `--ubatch-size 2048`: op-offloaded prefill copies CPU-resident expert
  weights to the GPU per batch and runs the batch there, priced in PCIe
  bytes. That is what makes prefill 850-940 tok/s rather than CPU speed, and
  a big ubatch amortises it. It is also most of the 6.0 GiB of buffers.
- `--parallel 1`: a second active slot cost each stream ~29% when measured
  (two concurrent at 24.1 / 24.2 tok/s each against 33.5 single-stream on
  the same engine, aggregate 1.42x). `--ctx-size` is the total split across
  slots, so 2 slots at the native window means 524288. Safe again since
  #27941; the concurrency crashes had the same `gridDim.y` root cause.
- `LLAMA_ATTN_ROT_DISABLE=1`: the `qwen4exp` attention path does not support
  the quantised-KV activation rotation, and with it active the server aborts
  at load. KV is f16 here so this is belt and braces.
- A vendored chat template is optional. The checkpoint's own template raises
  on `reasoning_effort: high|max`; if your clients send those, carry a fork
  that maps them to `xhigh` and pass `--chat-template-file`.

## Speculation is off, and must stay off on this build

The model's own MTP head can be grafted into the GGUF as `blk.48` (the
converter's `--mtp` export of the 5.2 GB of `mtp.*` tensors, merged as a 49th
block with `nextn_predict_layers = 1`), and #27836's `--spec-type draft-mtp`
will load and self-draft from it. Do not.

On every patch composition tried it **corrupts output**: one prompt was token
salad from the first token, another started coherent and degraded mid-answer,
which is the recurrent-state-corruption signature. Acceptance collapsed from
63-66% to 3-11% within a run, decode fell to 15-24 tok/s against 43 with it
off, VRAM redlined at 32.1 GiB, and on one build the server wedged after a
single request. #28019 explains why: the recurrent-state rollback flag fails
llama.cpp's own rollback test with a logits diff of 9.25, because the PLE
convolution and QSA indexer state have no snapshot coverage. Spec-off serving
does not touch that path and is unaffected.

Even before the corruption, `draft-mtp` was a net loss on natural prompts
(63-66% accept, yet 15-34 tok/s and decaying in-run as the draft context
rebuild grew). n-gram speculation measured about 0% gain on honest traffic
and serialised multi-slot decode. Synthetic benchmarks disagree, loudly:
repetitive filler accepts ~100% of n-gram drafts and ~96% of MTP drafts, which
is how a first-day bench read 86 tok/s at 2K. Benchmark on real prompts.

`--moe-expert-cache 64` (the #27861 LRU) read 41.7 tok/s warm against 39.8
without, but 1.7-5.6 tok/s cold and 31.3 GiB of VRAM. Spending the same VRAM
on `--n-cpu-moe 43` instead gave 43.4 with no cold phase.

## Measured performance

The ladder, in the order it was built, idle box, natural prompts, 512
generated tokens, temperature 0:

```
34.0   original PR fork, n-gram spec, --n-cpu-moe 48          (2026-08-27)
39.8   mainline b10679 + patch v1, spec off                   (+17%, engine alone)
41.7   + expert LRU cache, warm only                          (rejected, see above)
43.8   + --n-cpu-moe 42                                        (31.7 GiB peak, too thin)
43.4   + --n-cpu-moe 43                                        (30.0 GiB)  <- shipped
42-43  + #27941/#27879 correctness picks (patch v2)            (no regression)
42-43  + b10705, #27977, #28023 (patch v3, this page)          (no regression short;
                                                               depth much better)
```

At depth on the shipped build:

| context | decode tok/s | prefill tok/s |
|---|---|---|
| short (~500 tokens) | 42.3-43.0 | |
| ~2K | | 936 |
| 6K | 40.9 | |
| 8K | | 873 |
| 24K | 38.2 | |
| 32K | | 848 |

The previous patch read 31.4 tok/s at 32K on synthetic filler; the depth work
in #27977 and #28011 is what flattened the curve.

Two things about measuring this:

- **Bench until consecutive runs agree.** The first deploy read 23.7 tok/s at
  8K and 16.1 at 32K on the second pass, and 84.1 / 82.0 on the third. What
  looked like a depth cliff was the page cache warming. Cold NVMe is pass
  one; deep-context rows are still cold on pass two.
- **Synthetic filler can read decode 0 on this build.** The correctness
  hunks changed model behaviour so that it EOSes on gibberish at token 1
  instead of continuing it, which is a legitimate answer, not a broken decode
  path. [`flash-next-5090/probe-long.sh`](../flash-next-5090/probe-long.sh)
  exists for exactly this: it prefills varied English and asks a question
  about it, so a healthy engine must generate, and you can read whether the
  answer is coherent.

```bash
PORT=8001 ./flash-next-5090/probe-long.sh 24000 200
```

## Quality: what the patch fixed, and what it could not

The engine went through three patch revisions in a week, and the reason was
a quality signal, not a speed one. On our in-house benchmark the first build
matched the RTX PRO 6000 NVFP4 build exactly with thinking off, and cratered
with thinking on: long reasoning chains came out fluent and wrong. That is
the silent logit-drift signature #27941 documents (QSA block selection keyed
without the sequence, so long contexts read the wrong indexer keys), and
taking #27941 whole fixed it. Mid-difficulty thinking now scores at parity
with the NVFP4 build.

Two traps in reading such a result:

- **A rebuilt engine can inherit a stale quality score.** If your evaluation
  caches graded answers under a model identity that does not include the
  engine build, a fixed llama.cpp re-derives the broken build's verdict in
  seconds without generating a token. Change the served alias on every
  output-affecting engine change.
- **A time limit masquerades as a quality gap.** With a six-minute answer
  deadline, 44 of the hardest questions timed out at 41 tok/s where the
  151 tok/s card finished them. After the depth fixes that fell to 1, and the
  whole thinking pass ran in ~90 minutes instead of ~5 hours.

What survives every engine fix and the deadline: the hardest tier scores about
half of the NVFP4 build (19 of 55 against 39). With the misses now genuine
wrong answers, that points at the quant, and specifically at the 4.5-bit PLE
table on the longest chains, which is why the Q8_0 PLE requant above is the
next experiment.

## Sources

- [ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742), the `qwen4exp` architecture, merged
- [#27941](https://github.com/ggml-org/llama.cpp/pull/27941), [#27879](https://github.com/ggml-org/llama.cpp/pull/27879), [#28023](https://github.com/ggml-org/llama.cpp/pull/28023), [#27977](https://github.com/ggml-org/llama.cpp/pull/27977), [#27836](https://github.com/ggml-org/llama.cpp/pull/27836), [#27861](https://github.com/ggml-org/llama.cpp/pull/27861), the vendored pieces
- [#28019](https://github.com/ggml-org/llama.cpp/pull/28019), why the rollback flag is dropped and why MTP self-draft corrupts
- [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), the quant
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the model card
