#!/usr/bin/env bash
# GossipCop diagnostic with ONLY two legs: GPT-5.4 (LLM) vs RoBERTa (SLM).
# Skips ARG entirely -> no step1 rationale generation needed (saves API cost).
# Run from the project root: bash scripts/run_en_llm_slm.sh
set -e

DATA=data/en
PRED=outputs/preds
CACHE=outputs/cache
DIAG=outputs/diagnostic/gossipcop_llm_slm
mkdir -p "$PRED" "$CACHE" "$DIAG"

# --- LLM: GPT-5.4 direct judge ----------------------------------------------
python -m src.step2_llm_direct \
  --input ${DATA}/test.json \
  --output ${PRED}/gossipcop_gpt54.json \
  --cache  ${CACHE}/en_test_direct.jsonl --language en

# --- SLM: RoBERTa ------------------------------------------------------------
python -m src.step3_train_roberta \
  --train ${DATA}/train.json --val ${DATA}/val.json --test ${DATA}/test.json \
  --language en --model_name roberta-base \
  --output ${PRED}/gossipcop_roberta.json \
  --ckpt_dir outputs/ckpt/gossipcop_roberta

# --- Diagnostic: two legs only ----------------------------------------------
python -m src.step5_diagnostic \
  --dataset GossipCop \
  --pred RoBERTa=${PRED}/gossipcop_roberta.json \
  --pred GPT-5.4=${PRED}/gossipcop_gpt54.json \
  --outdir ${DIAG} --strict-split

# --- Step B: cost-quality Pareto (routing motivation figure) -----------------
python -m src.step6_routing \
  --small ${PRED}/gossipcop_roberta.json \
  --large ${PRED}/gossipcop_gpt54.json \
  --name  gossipcop --out ${DIAG}
