#!/usr/bin/env bash
# End-to-end diagnostic for GossipCop (English).
# Edit the paths below, then run from the project root: bash scripts/run_en.sh
set -e

DATA=data/en            # official ARG GossipCop (full text + shipped GPT-3.5 rationales)
PRED=outputs/preds
CACHE=outputs/cache
DIAG=outputs/diagnostic/gossipcop
mkdir -p "$PRED" "$CACHE" "$DIAG"

# --- 1. Regenerate ARG rationales with GPT-5.4 advisor (IN-PLACE) -------------
# Overwrites td_*/cs_* in data/en/*.json so ARG's advisor == GPT-5.4 (matches the
# step2 large model). Makes a one-time *.bak backup; resumable via the cache.
for split in train val test; do
  python -m src.step1_generate_rationales \
    --input ${DATA}/${split}.json --in_place \
    --cache ${CACHE}/en_${split}_rat.jsonl --language en
done

# --- 2. GPT-5.4 direct judge (large model) -----------------------------------
python -m src.step2_llm_direct \
  --input ${DATA}/test.json \
  --output ${PRED}/gossipcop_gpt54.json \
  --cache  ${CACHE}/en_test_direct.jsonl --language en

# --- 3. RoBERTa small model --------------------------------------------------
python -m src.step3_train_roberta \
  --train ${DATA}/train.json --val ${DATA}/val.json --test ${DATA}/test.json \
  --language en --model_name roberta-base \
  --output ${PRED}/gossipcop_roberta.json \
  --ckpt_dir outputs/ckpt/gossipcop_roberta

# --- 4. ARG (small + LLM advisor): train in the ARG repo, then dump ----------
# This is a MANUAL step done inside the cloned ARG repo (see
# src/arg_integration/README.md). It produces ${PRED}/gossipcop_arg.json.
# Until that file exists, step5 below simply runs without the ARG leg.
ARG_PRED=${PRED}/gossipcop_arg.json

# --- 5. Diagnostic -----------------------------------------------------------
PRED_ARGS="--pred RoBERTa=${PRED}/gossipcop_roberta.json --pred GPT-5.4=${PRED}/gossipcop_gpt54.json"
if [ -f "$ARG_PRED" ]; then
  PRED_ARGS="$PRED_ARGS --pred ARG=${ARG_PRED}"
else
  echo "[run_en] NOTE: $ARG_PRED not found — running diagnostic WITHOUT the ARG leg."
  echo "[run_en]       Train+dump ARG (src/arg_integration/README.md), then re-run step5."
fi
python -m src.step5_diagnostic --dataset GossipCop $PRED_ARGS --outdir ${DIAG}
