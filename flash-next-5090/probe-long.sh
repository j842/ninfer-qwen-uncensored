#!/usr/bin/env bash
#
# Deep-prefill decode probe: does generation still work, and stay coherent,
# after a multi-thousand-token prefill of real English?
#
#   ./flash-next-5090/probe-long.sh [approx_prompt_tokens] [n_predict]
#   PORT=8001 API_KEY=... ./flash-next-5090/probe-long.sh 24000 200
#
# Why this exists: synthetic-filler benchmarks can read "decode 0 tok/s" on
# this engine because the model EOSes at token 1 rather than continue
# gibberish, which is a legitimate response, not a broken decode path. This
# probe prefills varied prose and asks a question about it, so a healthy
# engine MUST generate. Run it after any engine or patch change, and read the
# answer: a fast engine can still be wrong.
#
set -euo pipefail
command -v jq >/dev/null || { echo "probe-long: jq is required"; exit 1; }

TOKENS="${1:-6000}"
NPRED="${2:-200}"
BASE="http://localhost:${PORT:-8001}"
AUTH=()
[ -n "${API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer ${API_KEY}")

PROMPT_FILE="$(mktemp)"
trap 'rm -f "$PROMPT_FILE"' EXIT
python3 - "$TOKENS" > "$PROMPT_FILE" <<'PY'
import sys, random
random.seed(42)
subjects = ["the reactor", "the pipeline", "the caching layer", "the scheduler",
            "the migration", "the telemetry service", "the archive index",
            "the billing engine", "the render farm", "the consensus module"]
verbs = ["was redesigned", "failed intermittently", "scaled linearly",
         "required manual intervention", "outperformed expectations",
         "was deprecated", "recovered gracefully", "consumed excess memory"]
reasons = ["after the third quarter review", "when the load doubled",
           "because the vendor changed the API", "during the winter freeze",
           "once the team adopted the new protocol", "despite the added caching",
           "following the security audit", "as the dataset grew past a billion rows"]
out = ["Engineering log, collected notes:"]
words = 0
i = 0
target_words = int(int(sys.argv[1]) / 1.33)
while words < target_words:
    i += 1
    s = f"Entry {i}: {random.choice(subjects)} {random.choice(verbs)} {random.choice(reasons)}, and the on-call engineer documented the exact sequence of events in the incident tracker."
    out.append(s)
    words += len(s.split())
out.append("\nQuestion: In two or three sentences, what kind of document is the text above, and what recurring elements does it contain?")
print(" ".join(out))
PY

RESP="$(jq -n --rawfile p "$PROMPT_FILE" --argjson n "$NPRED" \
    '{prompt:$p, n_predict:$n, cache_prompt:false, temperature:0}' \
    | curl -sf -m 1800 -X POST "${BASE}/completion" "${AUTH[@]}" \
        -H "Content-Type: application/json" -d @- )" \
    || { echo "probe-long: request to ${BASE} failed"; exit 1; }

echo "$RESP" | jq -r '"prompt_n: \(.timings.prompt_n)  prefill: \(.timings.prompt_per_second | round) tok/s  n_gen: \(.timings.predicted_n)  decode: \(.timings.predicted_per_second | .*10 | round / 10) tok/s"'
echo "---"
echo "$RESP" | jq -r '.content'
