#!/usr/bin/env bash
# Round-2 MAIN METHOD (B) — GossipCop (en). Net-utility dual-head + regime gate.
# The decisive test: does the gate CLOSE on GossipCop so the curve never drops
# below all-SLM (76.37)?  Run AFTER scripts_v2/run_uncertainty_en.sh.
#   UNC_TAG=__temp bash scripts_v2/run_router_en.sh   # use calibrated uncertainty
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=gossipcop
UNC_TAG=${UNC_TAG:-}

python -m src_v2.net_utility_router \
  --name ${NAME} \
  --small_val  outputs/preds/${NAME}_roberta_val.json \
  --large_val  outputs/preds/${NAME}_gpt54_val.json \
  --small_test outputs/preds/${NAME}_roberta.json \
  --large_test outputs/preds/${NAME}_gpt54.json \
  --unc_val  outputs_v2/uncertainty/${NAME}_val${UNC_TAG}.json \
  --unc_test outputs_v2/uncertainty/${NAME}_test${UNC_TAG}.json \
  --out outputs_v2/router/${NAME}

echo "[run_router_en] done -> outputs_v2/router/${NAME}/net_utility.json"
