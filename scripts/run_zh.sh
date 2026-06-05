#!/usr/bin/env bash
# End-to-end diagnostic for Weibo21 (Chinese).
# Edit the paths below, then run from the project root: bash scripts/run_zh.sh
set -e

DATA=data/zh            # official ARG Weibo21 (content + shipped GPT-3.5 rationales)
PRED=outputs/preds
CACHE=outputs/cache
DIAG=outputs/diagnostic/weibo21
mkdir -p "$PRED" "$CACHE" "$DIAG"

# Data: using the official ARG release (already in data/zh/, rationales included).
# step1 (GPT-5.4 advisor regeneration) is intentionally skipped -- ARG is trained
# on the shipped GPT-3.5 rationales.

# --- 2. GPT-5.4 direct judge (large model) -----------------------------------
python -m src.step2_llm_direct \
  --input ${DATA}/test.json \
  --output ${PRED}/weibo21_gpt54.json \
  --cache  ${CACHE}/zh_test_direct.jsonl --language zh

# --- 3. RoBERTa small model --------------------------------------------------
python -m src.step3_train_roberta \
  --train ${DATA}/train.json --val ${DATA}/val.json --test ${DATA}/test.json \
  --language zh --model_name hfl/chinese-roberta-wwm-ext \
  --output ${PRED}/weibo21_roberta.json \
  --ckpt_dir outputs/ckpt/weibo21_roberta

# --- 4. ARG (small + LLM advisor): train in the ARG repo, then dump ----------
# See src/arg_integration/README.md. Produces ${PRED}/weibo21_arg.json.

# --- 5. Diagnostic -----------------------------------------------------------
python -m src.step5_diagnostic \
  --dataset Weibo21 \
  --pred RoBERTa=${PRED}/weibo21_roberta.json \
  --pred ARG=${PRED}/weibo21_arg.json \
  --pred GPT-5.4=${PRED}/weibo21_gpt54.json \
  --outdir ${DIAG}
