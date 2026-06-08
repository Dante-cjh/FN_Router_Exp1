#!/usr/bin/env bash
# Weibo21 diagnostic with ONLY two legs: GPT-5.4 (LLM) vs RoBERTa (SLM).
# Skips ARG entirely -> no step1 rationale generation needed (saves API cost).
# Run from the project root: bash scripts/run_zh_llm_slm.sh
set -e

DATA=data/zh
PRED=outputs/preds
CACHE=outputs/cache
DIAG=outputs/diagnostic/weibo21_llm_slm
mkdir -p "$PRED" "$CACHE" "$DIAG"

# --- LLM: GPT-5.4 direct judge ----------------------------------------------
python -m src.step2_llm_direct \
  --input ${DATA}/test.json \
  --output ${PRED}/weibo21_gpt54.json \
  --cache  ${CACHE}/zh_test_direct.jsonl --language zh

# --- SLM: RoBERTa ------------------------------------------------------------
python -m src.step3_train_roberta \
  --train ${DATA}/train.json --val ${DATA}/val.json --test ${DATA}/test.json \
  --language zh --model_name hfl/chinese-roberta-wwm-ext \
  --output ${PRED}/weibo21_roberta.json \
  --ckpt_dir outputs/ckpt/weibo21_roberta

# --- Diagnostic: two legs only ----------------------------------------------
python -m src.step5_diagnostic \
  --dataset Weibo21 \
  --pred RoBERTa=${PRED}/weibo21_roberta.json \
  --pred GPT-5.4=${PRED}/weibo21_gpt54.json \
  --outdir ${DIAG}
