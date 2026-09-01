# ninfer-qwen-uncensored

Build recipes and serving notes for the local models we run, one page per model.

Two of them are reproducible [NInfer](https://github.com/Neroued/ninfer) builds
for the RTX 5090: one command each, from pinned public inputs, to a single
`.ninfer` file. The third is Qwen3.8-Flash-Next on SGLang, which has nothing to
do with NInfer and is here because it is the fastest thing we serve and the
setup took a while to get right.

| Model | Engine / card | Speed | How to get it |
|---|---|---|---|
| [Qwen3.8-27B Uncensored](models/qwen3.8-27b-uncensored.md) | NInfer, RTX 5090 | dense 27B, not benchmarked here | build it: `./build.sh` |
| [Ornith-1.5-35B-A3B](models/ornith-1.5-35b-a3b.md) | NInfer, RTX 5090 | ~593 tok/s single stream | [download](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer) or `./build-ornith.sh` |
| [Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md) | SGLang, RTX PRO 6000 | 236-324 tok/s single stream | `./flash-next/fetch-patches.sh`, then the stock image |

## [Qwen3.8-27B Uncensored](models/qwen3.8-27b-uncensored.md)

An abliterated Qwen3.8-27B packed into a 16.96 GiB NInfer artifact: text and
vision, thinking mode, MTP speculation, 262K context on one 5090. "Abliterated"
means the refusal direction has been removed from the weights, so it does not
decline requests the way the instruct model does. That work was already done
and published by someone else; this repo only re-packages those weights.
Deliberately not redistributed, so you build it yourself. Takes about ten
minutes of GPU time on top of a 55 GB download.

## [Ornith-1.5-35B-A3B](models/ornith-1.5-35b-a3b.md)

shisa-ai's 35B MoE (256 experts, 8 active) with an MTP head distilled for it,
plus the z-lab DFlash draft model, in a 21.22 GiB artifact. Not abliterated. It
is in this repo because the checkpoint is an exact drop-in for NInfer's
registered `qwen3_6_35b_a3b` target, so the same build pattern applies with one
extra download step. Every input is Apache-2.0 or MIT, so the built artifact is
published: **[huggingJDE/Ornith-1.5-35B-A3B-NInfer](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer)**.
By a distance the faster of the two 5090 builds.

## [Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md)

Not NInfer, not a 5090, and there is no artifact to download. The Qwen4
architecture preview (180B total, 6B active, 512 experts, 262K context) served
by SGLang on one RTX PRO 6000 Blackwell, with a seven-piece patch stack applied
to the stock image at container start. 257 tok/s prose and 324 tok/s code at
shallow depth, still 236 / 288 at 250K tokens of context. The page covers the
patch stack, the full launch command, the measured VRAM budget, and the silent
output corruption that a health check will not catch. It also has the RTX 5090
fallback: llama.cpp with all 512 experts streaming from system RAM, about 43
tok/s.

## Building the NInfer artifacts

Both builds have the same requirements.

- **Docker** with the **NVIDIA container runtime** (`--gpus` support). Every
  heavy step runs in a pinned container; the host itself only needs `curl` and
  `python3`. No Docker? [`get-docker.sh`](get-docker.sh) is the official
  convenience script from get.docker.com, vendored so the build has no other
  host dependency: `sudo sh get-docker.sh`.
- A CUDA GPU with **~11 GB of free VRAM** for the quantise step. Any card, not
  necessarily the 5090 you serve on (the reference build used a 5060 Ti). No
  GPU? `DEVICE=cpu` works: slower, but it cannot OOM and needs no NVIDIA
  runtime.
- Free disk: about 90 GB for Qwen3.8-27B, about 110 GB for Ornith.
- To *serve* the result you need an actual **RTX 5090**. NInfer targets
  `sm_120a` only.

NInfer is a C++/CUDA inference engine built exclusively for the 5090
(Blackwell, `sm_120a`). It does not load Hugging Face checkpoints. It serves a
`.ninfer` artifact: one file carrying the quantised weights, the MTP
speculation head, the vision tower, and the tokenizer/chat-template frontend.
Both builds run NInfer's own converter over public weights, deterministically
and from scratch, so you can rebuild either yourself from pinned inputs.

## Responsible use

The base model of the Qwen3.8-27B build is already abliterated and already
public. This repo does not create that capability, it converts existing public
weights into a different serving format for self-hosting. An uncensored model
will follow instructions a safety-trained one refuses, which makes **you**
responsible for what you ask it to do and for what you expose it to. Don't put
an unfiltered model in front of untrusted users or the public without your own
guardrails, and comply with the base model's and Qwen's licences and with the
law where you operate. Provided as-is, for research and self-hosting.

## Layout

```
README.md                          this file
models/
  qwen3.8-27b-uncensored.md        NInfer build 1: the abliterated dense 27B
  ornith-1.5-35b-a3b.md            NInfer build 2: the 35B MoE, published on HF
  qwen3.8-flash-next.md            SGLang on an RTX PRO 6000 (not NInfer)
build.sh                           Qwen3.8-27B uncensored build (four steps)
build-ornith.sh                    Ornith-1.5-35B-A3B MoE build (five steps)
upload-ornith.sh                   publish the Ornith artifact to Hugging Face
hf-ornith/                         HF model card, LICENCE, NOTICE for that upload
verify_frontend.py                 sha256 pin check for the grafted frontend
frontend.sha256                    the six official Qwen3.8-27B frontend pins
frontend-qwen3_6_35b_a3b.sha256    the six official Qwen3.6-35B-A3B frontend pins
flash-next/
  fetch-patches.sh                 fetch + verify the sm_120 patch stack
  patches.sha256                   its seven sha256 pins, at a pinned commit
  sm120-patch.py                   apply the stack inside the SGLang container
get-docker.sh                      vendored get.docker.com installer
```

`flash-next/patches/` is fetched, not committed: the upstream patch repository
carries no licence, so this repo pins the files by sha256 instead of
redistributing them.

## Credits

- [NInfer](https://github.com/Neroued/ninfer) for the 5090 engine, converter,
  and groupwise-int recipes.
- [Qwen](https://huggingface.co/Qwen) for the Qwen3.8-27B, Qwen3.6-35B-A3B and
  Qwen3.8-Flash-Next base models and frontend resources.
- [`JonathanColetti/Qwen3.8-27B-Uncensored`](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored)
  for the abliterated base weights.
- [`shisa-ai/Ornith-1.5-35B-A3B-MTP`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP)
  for Ornith with the distilled MTP head, merged into the Qwen3.6-35B-A3B
  checkpoint layout.
- [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
  for the DFlash draft model.
- [SGLang](https://github.com/sgl-project/sglang),
  [gabrielolympie](https://github.com/gabrielolympie/sglang-flashnext-sm120)
  and [jpezzulli](https://github.com/jpezzulli/sglang-rtxpro6000) for the
  Flash-Next engine and the sm_120 patch stack.
