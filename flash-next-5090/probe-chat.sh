#!/usr/bin/env bash
#
# Deep-context probe through the chat endpoint: prefill a real document, ask a
# question about it, print the engine's timings and the answer.
#
#   ./flash-next-5090/probe-chat.sh <document.txt> [question] [max_tokens]
#   PORT=8001 API_KEY=... ./flash-next-5090/probe-chat.sh notes.txt "Summarise the decisions." 200
#
# Uses /v1/chat/completions with cache_prompt=false, so the prefill figure is
# for the whole prompt, not the few tokens a cached re-send processes.
# Synthetic filler is not a valid probe on the qwen4exp builds: the model
# EOSes at token 1 on gibberish. Read the answer as well as the numbers.
#
set -euo pipefail
command -v jq >/dev/null || { echo "probe-chat: jq is required"; exit 1; }

DOC="${1:?usage: probe-chat.sh <document.txt> [question] [max_tokens]}"
QUESTION="${2:-In two or three sentences, what is this document about and what does it decide?}"
MAXTOK="${3:-200}"
BASE="http://localhost:${PORT:-8001}"
AUTH=()
[ -n "${API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer ${API_KEY}")

RESP="$(jq -n --rawfile d "$DOC" --arg q "$QUESTION" --argjson n "$MAXTOK" \
    '{model:"default", stream:false, temperature:0, max_tokens:$n, cache_prompt:false,
      chat_template_kwargs:{enable_thinking:false},
      messages:[{role:"user", content:($d + "\n\n" + $q)}]}' \
    | curl -sf -m 1800 -X POST "${BASE}/v1/chat/completions" "${AUTH[@]}" \
        -H "Content-Type: application/json" -d @- )" \
    || { echo "probe-chat: request to ${BASE} failed"; exit 1; }

echo "$RESP" | jq -r '"prompt_n: \(.timings.prompt_n)  prefill: \(.timings.prompt_per_second | round) tok/s  n_gen: \(.timings.predicted_n)  decode: \(.timings.predicted_per_second | .*10 | round / 10) tok/s  finish: \(.choices[0].finish_reason)"'
echo "---"
echo "$RESP" | jq -r '.choices[0].message.content'
