#!/usr/bin/env bash
#
# Upload the built Ornith artifact to Hugging Face, with the model card and
# licence notices from hf-ornith/.
#
# Runs in a container for the same reason the build does: under rootful Docker
# work-ornith/out is root-owned, so a host-side upload cannot read it (and the
# host stays free of python deps).
#
# Requires HF_TOKEN with write scope (https://huggingface.co/settings/tokens).
# The target repo defaults to <your-hf-username>/Ornith-1.5-35B-A3B-NInfer and
# is created if it does not exist.
#
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────────────
W="${W:-$(pwd)/work-ornith}"                 # build-ornith.sh working dir
export OUT_FILE="${OUT_FILE:-ornith_1_5_35b_a3b.ninfer}"
export HF_REPO="${HF_REPO:-}"                # e.g. you/Ornith-1.5-35B-A3B-NInfer
export PRIVATE="${PRIVATE:-false}"           # true = create the repo private
: "${HF_TOKEN:?set HF_TOKEN to a write-scope token (hf.co/settings/tokens)}"
export HF_TOKEN                              # bare -e flags only pass exported vars

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -f "$W/out/$OUT_FILE" ] \
    || { echo "missing $W/out/$OUT_FILE — run ./build-ornith.sh first"; exit 1; }

docker run --rm -e HF_TOKEN -e HF_REPO -e PRIVATE -e OUT_FILE \
    -v "$W/out:/out:ro" -v "$REPO_DIR/hf-ornith:/card:ro" \
    python:3.12-slim bash -ec '
    pip -q install huggingface_hub >/dev/null
    export HF_XET_HIGH_PERFORMANCE=1
    python3 - << "PY"
import os
from huggingface_hub import HfApi

api = HfApi()
repo = os.environ["HF_REPO"] or api.whoami()["name"] + "/Ornith-1.5-35B-A3B-NInfer"
api.create_repo(repo, repo_type="model",
                private=os.environ["PRIVATE"] == "true", exist_ok=True)

out = os.environ["OUT_FILE"]
files = [
    ("/card/README.md", "README.md"),
    ("/card/LICENSE", "LICENSE"),
    ("/card/NOTICE", "NOTICE"),
    (f"/out/{out}.conversion.json", f"{out}.conversion.json"),
    (f"/out/{out}", out),  # ~21 GiB last, so metadata lands even if this drops
]
for src, dst in files:
    if not os.path.exists(src):
        print(f"WARNING: skipping {dst} ({src} not found)")
        continue
    print(f"uploading {dst} ({os.path.getsize(src):,} bytes)")
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo)

print(f"done: https://huggingface.co/{repo}")
PY'
