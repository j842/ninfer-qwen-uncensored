#!/usr/bin/env bash
#
# Fetch the sm_120 patch stack that sm120-patch.py applies.
#
# Downloads six .patch files and the FR-Spec token map from
# gabrielolympie/sglang-flashnext-sm120 at a pinned commit, into ./patches/,
# and verifies every one against patches.sha256.
#
# They are fetched rather than committed here because that repository carries
# no licence, so there is no grant to redistribute its files. The hashes in
# patches.sha256 are what makes the fetch reproducible: a re-pointed branch or
# an edited file fails the check instead of silently changing what you serve.
#
# Only curl and sha256sum are needed. See models/qwen3.8-flash-next.md for what
# each patch does and how they are applied.
#
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────────────
PATCH_REPO="${PATCH_REPO:-gabrielolympie/sglang-flashnext-sm120}"
PATCH_COMMIT="${PATCH_COMMIT:-67d2f9234fa45ae1339f0d53cd37cb695e9c6493}"
DEST="${DEST:-$(cd "$(dirname "$0")" && pwd)/patches}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PIN_FILE="${PIN_FILE:-$HERE/patches.sha256}"
BASE="https://raw.githubusercontent.com/${PATCH_REPO}/${PATCH_COMMIT}"

# hot_tokens_64k.pt lives at the repo root; the six patches live in patches/.
_remote_path() {
    case "$1" in
        *.patch) echo "patches/$1" ;;
        *)       echo "$1" ;;
    esac
}

command -v curl >/dev/null || { echo "fetch-patches: curl is required"; exit 1; }
command -v sha256sum >/dev/null || { echo "fetch-patches: sha256sum is required"; exit 1; }

echo "=== ${PATCH_REPO} @ ${PATCH_COMMIT:0:12}"
mkdir -p "$DEST"

# Pin file lines are "<sha256>  <name>", with # comments and blank lines.
while read -r want name; do
    [ -n "${want:-}" ] || continue
    case "$want" in \#*) continue ;; esac

    target="$DEST/$name"
    if [ -f "$target" ] && [ "$(sha256sum "$target" | cut -d' ' -f1)" = "$want" ]; then
        echo "  have  $name"
        continue
    fi

    url="${BASE}/$(_remote_path "$name")"
    curl -fsSL "$url" -o "$target.part" \
        || { echo "  FAILED to download $name from $url"; exit 1; }

    got="$(sha256sum "$target.part" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        rm -f "$target.part"
        echo "  MISMATCH $name"
        echo "    got  $got"
        echo "    want $want"
        echo
        echo "Upstream has changed under the pin. Re-read the patch-application"
        echo "notes in models/qwen3.8-flash-next.md before updating $PIN_FILE:"
        echo "which patches apply strictly and which need fuzz depends on the"
        echo "SGLang image you pin alongside them."
        exit 1
    fi
    mv "$target.part" "$target"
    echo "  OK    $name"
done < "$PIN_FILE"

echo "patch stack verified in $DEST"
