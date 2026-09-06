# Local LLM setups, September 2026

Build recipes and serving configs for the local models we run, one page per
setup: pins, commands, flags, measured numbers.

| Model | Engine / hardware | Decode, 1 stream | Prefill | Get it |
|---|---|---|---|---|
| [Qwen3.8-27B Uncensored](models/qwen3.8-27b-uncensored.md) | NInfer, RTX 5090 | not benchmarked | not benchmarked | `./build.sh` |
| [Ornith-1.5-35B-A3B](models/ornith-1.5-35b-a3b.md) | NInfer, RTX 5090 | 415–429 tok/s via a router; 650–675 engine-reported | ~13,700 tok/s | [download](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer) or `./build-ornith.sh` |
| [Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md) | SGLang, RTX PRO 6000 | 236–324 tok/s | 11,000–12,600 tok/s | `./flash-next/fetch-patches.sh` + stock image |
| [Qwen3.8-Flash-Next, 5090](models/qwen3.8-flash-next-5090.md) | llama.cpp, RTX 5090 + system RAM, MTP on | 46–57 tok/s | 830–950 tok/s | `./flash-next-5090/build-engine.sh` + a grafted GGUF |
| [Qwen3.8-Flash-Next, Strix Halo](models/qwen3.8-flash-next-strix-halo.md) | llama.cpp Vulkan, Ryzen AI MAX+ 395 | 22–24 tok/s | 190–350 tok/s | `./flash-next-strix/build-engine.sh` + public GGUF |
| [Ling-3.0-tiny](models/ling-3.0-tiny-a750.md) | llama.cpp Vulkan, Arc A750 8 GB | 40.5 tok/s | ~1,200 tok/s | pinned stock image + public GGUF |

The two NInfer rows are one-command builds to a single `.ninfer` file. The
three Flash-Next rows are the same 180B model at three price points. Ling is
the cheap end.

## Building the NInfer artifacts

- Docker with the NVIDIA container runtime. The host needs only `curl` and
  `python3`. [`get-docker.sh`](get-docker.sh) is the vendored get.docker.com
  installer: `sudo sh get-docker.sh`.
- A CUDA GPU with ~11 GB free for the quantise step, any card. `DEVICE=cpu`
  works, slower.
- Disk: ~90 GB for Qwen3.8-27B, ~110 GB for Ornith.
- Serving needs an RTX 5090: NInfer targets `sm_120a` only.

NInfer is a C++/CUDA engine for the 5090. It serves a `.ninfer` file
carrying quantised weights, the MTP head, the vision tower and the
tokenizer/chat template. Both builds run NInfer's converter (pinned at
`ad0f3d38`, 2026-09-04) over public weights from pinned inputs.

## Responsible use

The Qwen3.8-27B build starts from weights that are already abliterated and
public; this repo converts them to another serving format. An uncensored
model follows instructions a safety-trained one refuses. You are responsible
for what you ask it and who you expose it to. Do not put it in front of
untrusted users without your own guardrails. Comply with the base model's and
Qwen's licences and local law. Provided as-is.

## Layout

```
README.md                          this file
models/
  qwen3.8-27b-uncensored.md        NInfer: abliterated dense 27B
  ornith-1.5-35b-a3b.md            NInfer: 35B MoE, published on HF
  qwen3.8-flash-next.md            SGLang on an RTX PRO 6000
  qwen3.8-flash-next-5090.md       llama.cpp on an RTX 5090
  qwen3.8-flash-next-strix-halo.md llama.cpp Vulkan on a Strix Halo APU
  ling-3.0-tiny-a750.md            llama.cpp Vulkan on an Intel Arc A750
build.sh                           Qwen3.8-27B uncensored build
build-ornith.sh                    Ornith build
upload-ornith.sh                   publish the Ornith artifact to Hugging Face
hf-ornith/                         HF model card, LICENSE, NOTICE for that upload
verify_frontend.py                 sha256 pin check for the grafted frontend
frontend.sha256                    Qwen3.8-27B frontend pins
frontend-qwen3_6_35b_a3b.sha256    Qwen3.6-35B-A3B frontend pins
flash-next/
  fetch-patches.sh                 fetch + verify the sm_120 patch stack
  patches.sha256                   its sha256 pins
  sm120-patch.py                   apply the stack inside the SGLang container
  chat_template.jinja              Qwen3.8-Flash-Next template that accepts reasoning_effort high|max
flash-next-5090/
  build-engine.sh                  pinned llama.cpp + patch into a CUDA image
  maxspeed.patch                   #28243, #28068, #28213, unsloth #144 on b10819
  Dockerfile.runtime               the CUDA runtime image
  fetch_mtp.py                     range-fetch the MTP head tensors from the HF checkpoint
  graft_mtp.py                     graft the converted head into the GGUF as blk.48
  graft_ple.py                     swap the PLE table for unsloth's Q8_0 copy
  probe-chat.sh                    deep-context probe through /v1/chat/completions (any llama-server)
flash-next-strix/
  build-engine.sh                  pinned Vulkan fork into an image
  Dockerfile                       LunarG SDK build stage, kisak Mesa runtime
get-docker.sh                      vendored get.docker.com installer
```

`flash-next/patches/` is fetched, not committed: upstream carries no licence.
`flash-next-5090/maxspeed.patch` is committed: llama.cpp and its PRs are MIT.

## Credits

- [NInfer](https://github.com/Neroued/ninfer): engine, converter, recipes.
- [Qwen](https://huggingface.co/Qwen): base models and frontends.
- [JonathanColetti/Qwen3.8-27B-Uncensored](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored): abliterated weights.
- [shisa-ai/Ornith-1.5-35B-A3B-MTP](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP) and [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash).
- [SGLang](https://github.com/sgl-project/sglang), [gabrielolympie](https://github.com/gabrielolympie/sglang-flashnext-sm120), [jpezzulli](https://github.com/jpezzulli/sglang-rtxpro6000): the sm_120 patch stack.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and the authors of PRs #27742, #27941, #28023, #28123, #28040, #27970 (merged) and #28243, #28068, #28213 (vendored); [unsloth](https://huggingface.co/unsloth) for the UD-Q4_K_XL and Q8_0 GGUFs and fork PR #144.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), [agentionai](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF), [kisak-mesa](https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa): the Strix Halo stack.
- [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny), llama.cpp PR [#26608](https://github.com/ggml-org/llama.cpp/pull/26608), [bloomer010/Ling-3.0-tiny-GGUF](https://huggingface.co/bloomer010/Ling-3.0-tiny-GGUF).
