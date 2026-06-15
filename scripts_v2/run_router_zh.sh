#!/usr/bin/env bash
# Round-2 MAIN METHOD (B) — Weibo21 (zh). Net-utility dual-head + regime gate.
# Run AFTER the uncertainty step exists (scripts_v2/run_uncertainty_zh.sh).
# UNC_TAG lets you point at a calibrated uncertainty version from step A, e.g.
#   UNC_TAG=__temp bash scripts_v2/run_router_zh.sh
# Run from repo root (Router_Exp1/).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=weibo21
UNC_TAG=${UNC_TAG:-}      # "" = base pilot uncertainty; "__temp" / "__p0.3" after A

python -m src_v2.net_utility_router \
  --name ${NAME} \
  --small_val  outputs/preds/${NAME}_roberta_val.json \
  --large_val  outputs/preds/${NAME}_gpt54_val.json \
  --small_test outputs/preds/${NAME}_roberta.json \
  --large_test outputs/preds/${NAME}_gpt54.json \
  --unc_val  outputs_v2/uncertainty/${NAME}_val${UNC_TAG}.json \
  --unc_test outputs_v2/uncertainty/${NAME}_test${UNC_TAG}.json \
  --out outputs_v2/router/${NAME}

echo "[run_router_zh] done -> outputs_v2/router/${NAME}/net_utility.json"
