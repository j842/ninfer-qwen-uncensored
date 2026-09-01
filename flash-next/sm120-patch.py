#!/usr/bin/env python3
"""Apply the sm_120 patch stack to the SGLang tree inside the container, at start.

WHAT THIS APPLIES
-----------------
Two independent things, in this order:

  A. The six-patch stack from gabrielolympie/sglang-flashnext-sm120, fetched
     into ./patches/ by fetch-patches.sh and mounted at /patches. Three unblock
     sm_120, three are the speed work:

       0001b  RecoverSSM + WY output-only MTP verify on FlashInfer for sm_120.
              Without it `supports_target_verify = sm_major in (9, 10)` means
              this card (major 12) falls back to the Triton GDN verify kernel.
       0002   FP8-KV tile dequant for the QSA sparse prefill. THIS is what makes
              --kv-cache-dtype fp8_e4m3 usable at all: unpatched, the QSA FA4
              decode call gets BF16 queries against FP8 K/V and asserts that all
              three dtypes must match (SGLang #36545). It halves KV from
              24 KiB/token to 12, which is the entire reason this setup can run
              the full 262,144-token window.
       0003   fp32 prefill state for the sm_120 FlashInfer GDN kernel. flashinfer's
              gdn_prefill.py permits bf16 only when compute capability == 10
              EXACTLY, which is why an unpatched tree has to prefill on Triton.
              With this, prefill and decode both run on FlashInfer.
       0004   Triton low-M GEMM for the decode projections. cuBLAS under CUDA
              graph capture runs them at 20-75% of DRAM bandwidth on sm_120;
              this kernel reaches ~90%.
       0005   W8A16 fp8 weight-only serving of the dense bf16 stack (~85% of
              per-step traffic; the NVFP4 checkpoint is never modified).
              NOTE this COSTS ~3.6 GB of VRAM — it makes fp8 copies alongside
              the bf16 originals. It buys bandwidth, not memory.
       0006   Same fp8 treatment for the HyperConnection mix and the lm_head,
              which also halves every MTP draft step's logits.

  B. SGLang PR #36556 (fix(qsa): support SM120 sparse decode paths), applied
     below as anchor-based edits rather than as a patch file. It is NOT in the
     six-patch stack and is still required: it targets
     qwen_sparse_attn_backend.py, where 0002 targets qsa/sparse_attn.py.

     Without #36556 the server answers short prompts perfectly and then emits
     token ID 0 ('!') past the first KV page, at a fixed absolute position that
     tracks --page-size (~65 at page 64, ~193 at page 128). It also fixes
     SGLang #36537 (thinking + qwen3_coder tool parser looping on token 0).

WHY strict-THEN-fuzzy
---------------------
The patches were cut against sgl-project/sglang `qwen4-main-squashed` as it
stood on 2026-08-30/31. The pinned image was pushed 2026-08-26, so it sits
slightly behind. Measured against image digest sha256:12d3392b...:

    0002 0004 0005   apply strict (git apply)
    0001b 0003 0006  need fuzz — ONE hunk each, pure context drift
                     (0006's is literally a black line-wrap of
                      `out = torch.empty((rows, hs), ...)`)

`git apply -3` is NOT available: the image's .git is stale (HEAD is an unrelated
AMD CI commit and the Qwen4 files are not in the index at all), so only
working-tree application works.

Fuzzy application can in principle place a hunk in the wrong spot, so fuzz is
never trusted on its own — every run ends in POSTCONDITIONS below, which assert
the specific symbols each patch is supposed to have produced. A patch that
"applied" without producing them fails the start.

DESIGN CONTRACT, and the reason this script exists
--------------------------------------------------
Idempotent and self-verifying. The `qwen38flashnext` image tag MOVES, so the
tree can change underneath this script at any time. It must therefore either
produce a fully patched tree or exit non-zero, never a partial one. A silent
half-patch on an sm_120 card means corrupt tokens from a server that looks
healthy.

Idempotency matters concretely: the container runs with `--restart
unless-stopped`, so a restart re-executes this script against the SAME writable
layer, which is already patched. If every postcondition already holds on entry,
this is a no-op.
"""

import json
import os
import subprocess
import sys

SGLANG_ROOT = os.environ.get("SGLANG_ROOT", "/sgl-workspace/sglang")
PATCH_DIR = os.environ.get("PATCH_DIR", "/patches")

# ── Part B: SGLang PR #36556, as anchor-based edits ────────────────────────
# (name, anchor-to-replace, replacement, already-applied-marker)
QSA_TARGET = os.path.join(
    SGLANG_ROOT,
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py",
)
QSA_HUNKS = [
    (
        "trtllm-sparse-decode SM120 gate",
        """    from sglang.srt.utils import is_sm100_supported

    if not is_sm100_supported():
        return None
""",
        """    from sglang.srt.utils import is_sm100_supported, is_sm120_supported

    if not (is_sm100_supported() or is_sm120_supported()):
        return None
""",
        "if not (is_sm100_supported() or is_sm120_supported()):",
    ),
    (
        "varlen fallback SM120 dispatcher",
        """    Classic flash_attn (FA2, Ampere/Hopper) is preferred when installed;
    flash-attn-4's cute interface serves the same call shape on Blackwell.
    \"\"\"
    try:
        from flash_attn import flash_attn_varlen_func
""",
        """    SM120 uses SGLang's architecture-owned FA4 dispatcher. Other platforms
    prefer classic flash_attn (FA2) before flash-attn-4's cute interface.
    \"\"\"
    from sglang.srt.utils import is_sm120_supported

    if is_sm120_supported():
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
        )

        return flash_attn_varlen_func
    try:
        from flash_attn import flash_attn_varlen_func
""",
        "from sglang.kernels.ops.attention.flash_attention_v4 import (",
    ),
]

# ── Postconditions ─────────────────────────────────────────────────────────
# (label, module, expression). The expression is evaluated with `m` bound to the
# imported module and `s` to its source text. Each asserts something a specific
# patch is supposed to have produced — these are what make fuzzy application
# safe to rely on.
#
# They are expression STRINGS rather than lambdas because verification runs in a
# FRESH INTERPRETER (see check_postconditions). That is not stylistic: this
# script imports the modules once up front to test idempotency, so by the time
# the patches land those modules are already in sys.modules and every
# `hasattr(m, ...)` would be answered from the stale pre-patch object while
# every `inspect.getsource` check read the new bytes off disk. That split brain
# reports a correctly-patched tree as broken. Clearing __pycache__ does not help;
# only a new process does.
POSTCONDITIONS = [
    # 0001b
    (
        "0001b: WY output-only availability helper",
        "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer",
        "hasattr(m, 'is_flashinfer_gdn_wy_output_only_available')",
    ),
    (
        "0001b: FlashInferGDNKernel.can_target_verify",
        "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer",
        "hasattr(m.FlashInferGDNKernel, 'can_target_verify')",
    ),
    (
        "0001b: supports_none_mode_target_verify (the drifting hunk)",
        "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer",
        "'supports_none_mode_target_verify' in s",
    ),
    (
        "0001b: --gdn-mtp-cache-mode server arg",
        "sglang.srt.server_args",
        "'gdn_mtp_cache_mode' in s",
    ),
    # 0002 — the fp8-KV enabler. The patch's whole effect on this file is four
    # `.to(q_values.dtype)` casts on the k/v tile loads in _sparse_gqa_prefill
    # and _sparse_gqa_chunk_prefill; none exist pre-patch. Counting them is the
    # exact assertion, where matching on "dequant"/"fp8" would be vacuous — those
    # words appear only in the commit subject, never in the source.
    (
        "0002: QSA fp8 tile dequant (4 casts)",
        "sglang.srt.layers.attention.qsa.sparse_attn",
        "s.count('.to(q_values.dtype)') >= 4",
    ),
    # 0004 / 0005 / 0006
    (
        "0004: sm120 low-M GEMM module",
        "sglang.kernels.ops.gemm.sm120_lowm_bf16_gemm",
        "True",
    ),
    (
        "0005: fp8 weight-only entry point",
        "sglang.kernels.ops.gemm.sm120_lowm_bf16_gemm",
        "'SGLANG_SM120_LOWM_FP8_WEIGHT' in s or 'fp8_weight' in s",
    ),
    (
        "0006: fp8 HC mix (the drifting hunk)",
        "sglang.srt.layers.hc_mix_triton",
        "hasattr(m, '_maybe_fp8_weights')",
    ),
    (
        "0006: fp8 lm_head in the logits processor",
        "sglang.srt.layers.logits_processor",
        "'maybe_sm120_fp8_lm_head' in s",
    ),
    # Part B
    (
        "#36556: QSA SM120 decode gate + varlen dispatcher",
        "sglang.srt.layers.attention.qwen_sparse_attn_backend",
        "'is_sm120_supported' in s",
    ),
]

# Runs in a child interpreter so imports are never served from a stale
# sys.modules. Emits one JSON line on stdout; warnings go to stderr and are
# ignored, which is why only the last non-empty stdout line is parsed.
_VERIFY_CHILD = r"""
import importlib, inspect, json, sys
results = []
for label, modname, expr in json.loads(sys.argv[1]):
    try:
        m = importlib.import_module(modname)
        s = inspect.getsource(m)
        results.append([label, bool(eval(expr, {"m": m, "s": s})), ""])
    except Exception as exc:
        results.append([label, False, repr(exc)])
print(json.dumps(results))
"""


def log(msg):
    print(f"[sm120-patch] {msg}", flush=True)


def check_postconditions(quiet=False):
    """Return the list of FAILED postcondition labels.

    Always evaluated in a fresh interpreter — see the note on POSTCONDITIONS.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _VERIFY_CHILD, json.dumps(POSTCONDITIONS)],
        cwd=SGLANG_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        results = json.loads(lines[-1])
    except (IndexError, ValueError):
        if not quiet:
            log("  FATAL: verification subprocess produced no parsable result")
            for line in proc.stderr.strip().splitlines()[-8:]:
                log(f"    {line}")
        return [label for label, _, _ in POSTCONDITIONS]

    failed = []
    for label, ok, err in results:
        if not ok:
            failed.append(label)
        if not quiet:
            log(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({err})" if err else ""))
    return failed


def apply_patch_files():
    """Apply /patches/*.patch: git apply strict, then `patch -F3` as fallback."""
    if not os.path.isdir(PATCH_DIR):
        log(f"FATAL: patch directory not found: {PATCH_DIR}")
        return False

    patches = sorted(
        os.path.join(PATCH_DIR, f)
        for f in os.listdir(PATCH_DIR)
        if f.endswith(".patch")
    )
    if not patches:
        log(f"FATAL: no .patch files in {PATCH_DIR}")
        return False

    ok = True
    for path in patches:
        name = os.path.basename(path)
        strict = subprocess.run(
            ["git", "apply", "-p1", "--exclude=test/*", path],
            cwd=SGLANG_ROOT,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if strict.returncode == 0:
            log(f"  strict  OK   {name}")
            continue

        # --batch and --forward are NOT optional, and stdin is closed on purpose.
        # Without them GNU patch PROMPTS ("Reversed (or previously applied)
        # patch detected! Assume -R? [n]") whenever it meets an already-applied
        # hunk. In a container start that is not a failure, it is a HANG — the
        # entrypoint blocks forever on a tty that will never answer, and the
        # worker never becomes healthy. Fail fast, never ask.
        fuzzy = subprocess.run(
            ["patch", "-p1", "-F3", "-l", "--batch", "--forward",
             "--no-backup-if-mismatch", "--reject-file=-", "-s", "-i", path],
            cwd=SGLANG_ROOT,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if fuzzy.returncode == 0:
            log(f"  fuzz    OK   {name}")
        else:
            ok = False
            log(f"  FAILED       {name}")
            for line in (fuzzy.stdout + fuzzy.stderr).strip().splitlines()[-6:]:
                log(f"                 {line}")
    return ok


def apply_qsa_hunks():
    """Apply PR #36556 to qwen_sparse_attn_backend.py. Returns True on success."""
    if not os.path.isfile(QSA_TARGET):
        log(f"FATAL: QSA target not found: {QSA_TARGET}")
        return False

    with open(QSA_TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()

    applied, already, failed = [], [], []
    for name, anchor, replacement, marker in QSA_HUNKS:
        if marker in src:
            already.append(name)
            continue
        if anchor not in src:
            failed.append(name)
            continue
        src = src.replace(anchor, replacement, 1)
        applied.append(name)

    if failed:
        for name in failed:
            log(
                f"FATAL: #36556 hunk {name!r} did not match and is not already "
                "applied. On SM120 an unpatched QSA decode path silently "
                "corrupts output past the first KV page (SGLang #36531)."
            )
        return False

    if applied:
        with open(QSA_TARGET, "w", encoding="utf-8") as fh:
            fh.write(src)
        _drop_pycache(QSA_TARGET)
        log(f"  applied #36556: {', '.join(applied)}")
    if already:
        log(f"  #36556 already present: {', '.join(already)}")
    return True


def _drop_pycache(path):
    """Stale bytecode would shadow an in-place edit."""
    cache = os.path.join(os.path.dirname(path), "__pycache__")
    stem = os.path.basename(path).split(".")[0] + "."
    if os.path.isdir(cache):
        for entry in os.listdir(cache):
            if entry.startswith(stem):
                try:
                    os.remove(os.path.join(cache, entry))
                except OSError:
                    pass


def main() -> int:
    if not os.path.isdir(os.path.join(SGLANG_ROOT, "python", "sglang")):
        log(f"FATAL: no SGLang source tree at {SGLANG_ROOT}")
        return 1

    # Idempotency: a container restart re-runs this against an already-patched
    # writable layer. If everything already holds, do nothing.
    log("checking whether the tree is already patched...")
    if not check_postconditions(quiet=True):
        log("already fully patched — nothing to do.")
        return 0

    log(f"applying patch stack from {PATCH_DIR}")
    patches_ok = apply_patch_files()
    log("applying SGLang PR #36556 (QSA sm_120 decode)")
    qsa_ok = apply_qsa_hunks()

    # Bytecode from the pre-patch import above must not shadow the new source.
    for dirpath, dirnames, filenames in os.walk(
        os.path.join(SGLANG_ROOT, "python", "sglang")
    ):
        if os.path.basename(dirpath) == "__pycache__":
            for f in filenames:
                try:
                    os.remove(os.path.join(dirpath, f))
                except OSError:
                    pass

    log("verifying postconditions")
    failed = check_postconditions()

    if failed and (not patches_ok or not qsa_ok):
        log("FATAL: patch application failed AND postconditions are unmet.")
    if failed:
        log("")
        log("REFUSING TO START. Unmet postconditions:")
        for label in failed:
            log(f"    - {label}")
        log("")
        log(
            "The image tag moves, and the tree has changed shape underneath "
            "this patch stack. Serving anyway risks silently corrupt output "
            "(SGLang #36531) or a wrong-dtype QSA assert (#36545). Re-cut the "
            "patches against the new image, or pin the image back to the "
            "digest recorded in models/qwen3.8-flash-next.md."
        )
        return 2

    log("patch stack applied and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
