# Qwen3.8-Flash-Next (llama.cpp, RTX 3090 + EPYC Naples)

The 180B model on a 24 GB RTX 3090 riding a first-gen EPYC 7551P (32 Zen 1
cores, FOUR NUMA nodes, 512 GB DDR4-2400 on all eight channels): llama.cpp
with every routed expert in system RAM, MTP self-drafting on, the full
262,144-token window, vision. Same engine and weights recipe as the
[5090 page](qwen3.8-flash-next-5090.md); what this page adds is the
multi-NUMA host machinery, without which the same worker runs at roughly
HALF speed with nothing in any log to say why.

| | |
|---|---|
| **Engine** | llama.cpp b10819 (`74a7c897f049`) + [`maxspeed.patch`](../flash-next-5090/maxspeed.patch) + [#28330](../flash-next-3090/28330-indexer-vcache.patch), image `llamacpp-qwen4exp-numactl:74a7c897f049-p7e2b5c0e-sm86` |
| **Weights** | `Qwen3.8-Flash-Next-UD-Q4_K_XL-imx-MTP-PLE8-v1.gguf`, 130.1 GiB — built exactly per the [5090 page](qwen3.8-flash-next-5090.md#the-weights) (MTP head grafted as `blk.48`, PLE table IQ4_NL → Q8_0) |
| **Vision** | `mmproj-BF16.gguf`, 865 MiB, from `unsloth/Qwen3.8-Flash-Next-GGUF` |
| **Card** | RTX 3090 24 GB on Gen3 x16, `--n-cpu-moe 48` (every trunk expert in RAM; `blk.48`'s pinned by regex), q8_0 KV, ctx 262144 |
| **Host** | ~132 GiB resident (anonymous, interleaved over 4 nodes), 32 physical cores |
| **Decode** | 26–30 tok/s on natural prompts (MTP n-max 1, 77–94% accept); 24–25 spec-off |
| **Prefill** | ~500–610 tok/s at 2K–32K depth; ~305 tok/s sustained at 220K depth |

## Why this box works at all

10 of 512 experts fire per token, so the DDR4 stream is ~1.4 GiB/token —
except decode on this host is NOT bandwidth-bound (a measured 97% rise in
the memory ceiling bought 9% of decode); it is bound by per-layer compute
and synchronisation across four Zen 1 NUMA nodes. Consequences, each
measured on this hardware:

- **`--load-mode none` + a numactl interleave entrypoint.** Weights load
  into anonymous memory placed by the faulting thread, and llama.cpp
  populates them from ONE loader thread — unwrapped, the lot lands on one
  or two nodes: 8.35 tok/s vs 16.09 wrapped, same binary. The policy must
  be the container's *entrypoint* (`numactl --interleave=all llama-server
  …`): mempolicy is inherited across fork/exec, and wrapping `docker run`
  only policies the CLI. Needs `--cap-add SYS_NICE` (docker's seccomp gates
  `set_mempolicy`) and numactl ≥ 2.0.19 baked into the image.
- **`vm.zone_reclaim_mode=1` + free pages at fault time.** Setting the
  policy is not getting the placement: with the default `0`, an interleave
  allocation whose turn lands on a node full of stale page cache silently
  spills to a remote node. Measured: policy applied perfectly, weights at
  62.6/27.7/0.07/20.3 GiB over the four nodes, 10.8 tok/s. Drop caches
  before loading; placement is fixed at fault time and only a restart
  repairs it.
- **THP=never.** llama.cpp splits expert rows across threads in ~160 KiB
  chunks; one 2 MiB huge page spans ~13 threads' rows and first-touch
  places it for one of them. Worth 43% of decode on this host.
- **Physical cores only** (`--cpuset-cpus 0-31`, docker-enforced —
  llama.cpp's own `--cpu-mask` did not place threads in this image),
  performance governor, `kernel.numa_balancing=0`, `vm.swappiness=0`.
- **ubatch 2048.** Only 12 of 48 layers carry full attention, so the
  ubatch-scaled buffer is cheap: +210 MiB bought 2x prefill and a third
  more decode on this lineage.

None of this applies on a single-node host — the 5090 page runs none of it.

## VRAM budget (24 GB, measured by full-width probes)

The compute buffer scales with attended KV depth and is the largest elastic
term — bigger than the KV itself. Certify any change with a probe that
prefills the REAL window depth; shallow probes have shipped two service
OOMs on this lineage.

At ctx 262144 with q8_0 KV (13,824 B/token; only 12 layers carry KV), the
card holds: ~5.2 GiB dense + 3.4 GiB KV + ~9-10 GiB depth-scaled compute
buffer + the MTP draft context (~2.5 GiB) + vision (~0.9 GiB). That is why
`--n-cpu-moe` stays at 48 here: the draft head pays back several times what
one expert layer (~2% decode) costs. [#28330](https://github.com/ggml-org/llama.cpp/pull/28330)
(in the patch stack) stops allocating the indexer's unused V cache, which
funds part of the draft context.

## The engine

`./flash-next-3090/build-engine.sh` — the 5090's `maxspeed.patch`
(byte-identical, one copy in this repo) plus `28330-indexer-vcache.patch`,
compiled for sm_86 only, baked with numactl 2.0.19. See the
[5090 page](qwen3.8-flash-next-5090.md#the-engine) for the PR ledger.
Speculation config: `--spec-type draft-mtp --spec-draft-n-max 1`, plus

```
--override-tensor 'blk\.48\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU'
```

because `--n-cpu-moe N` covers blocks `0..N-1` and cannot reach the grafted
head's own 512-expert MoE at `blk.48`. The head's 2-token verify batches
stay under `GGML_OP_OFFLOAD_MIN_BATCH` (32), so expert verification runs on
the CPU — do not raise that threshold.

⚠ **`--spec-draft-n-max 1` is load-bearing on this class of host, and it is
the one number NOT to copy from the 5090 page.** Measured here (natural
prompts, idle box, n_predict 384, code / prose):

| speculation | decode tok/s | accept |
|---|---|---|
| off | 24.3 / 25.0 | — |
| **draft-mtp, n-max 1** | **28.6 / 26.7** | 93% / 79% |
| draft-mtp, n-max 2 | 12.0–14.7 | 65–93% |
| ngram-mod, spans 4–8 | 24.8 / 24.2 | — / 25% |

One extra draft token per cycle flips +15% into −45% at unchanged
acceptance: a Zen 1 multi-node CPU pays for the wider verify batch (and its
GDN state bookkeeping) far more than the extra accepted token pays back.
The 5090's Zen 3 host runs n-max 2 profitably. Judge any speculation change
on natural prompts — synthetic fillers flatter every drafter.

## Launch

```bash
docker run -d --name qwen38-flash-next-3090 \
    --restart unless-stopped --init \
    --gpus '"device=0"' \
    --cap-add SYS_NICE \
    --cpuset-cpus 0-31 \
    --ulimit memlock=-1 \
    -p 8092:8080 \
    -v /path/to/models:/models:ro \
    --entrypoint /usr/local/bin/numactl \
    llamacpp-qwen4exp-numactl:74a7c897f049-p7e2b5c0e-sm86 \
    --interleave=all /usr/local/bin/llama-server \
    --model /models/Qwen3.8-Flash-Next-UD-Q4_K_XL-imx-MTP-PLE8-v1.gguf \
    --mmproj /models/mmproj-BF16.gguf \
    --host 0.0.0.0 --port 8080 \
    --load-mode none \
    -ngl 999 --n-cpu-moe 48 \
    --override-tensor 'blk\.48\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU' \
    --spec-type draft-mtp --spec-draft-n-max 1 \
    --ctx-size 262144 --parallel 1 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 32 --threads-batch 32 \
    --batch-size 2048 --ubatch-size 2048 \
    --cont-batching --metrics --jinja
```

Before the first start, on the host (survives via a systemd unit in our
deployment; re-assert after any reboot):

```bash
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
sysctl -w vm.zone_reclaim_mode=1 kernel.numa_balancing=0 vm.swappiness=0
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done
sync && echo 3 > /proc/sys/vm/drop_caches    # free pages on every node for the interleave
```

## Notes

- Load takes ~5 minutes: 130 GiB read once into anonymous memory. `mlock`
  and mmap-preload tricks are WRONG under this load mode (they defeat the
  interleave).
- The 3090 replaced a 3060 12 GB in the same box (both sm_86 — no rebuild).
  The 24 GB went to context + the draft head, not expert layers: at the
  native window the depth-scaled compute buffer eats what the experts would
  have taken.
- Judge speculation on natural prompts only. Synthetic repetitive fillers
  flatter both the drafters and the PLE table; draft-mtp once won a
  synthetic bench while losing on real prompts (earlier, broken engine
  composition — but the lesson stands).
- Zen 1's fabric is the decode ceiling; the same recipe on a one-node Milan
  (the 5090 page) does ~2x the decode.
