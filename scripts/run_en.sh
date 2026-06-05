#!/usr/bin/env bash
# End-to-end diagnostic for GossipCop (English).
# Edit the paths below, then run from the project root: bash scripts/run_en.sh
set -e

DATA=data/en            # official ARG GossipCop (full text + shipped GPT-3.5 rationales)
PRED=outputs/preds
CACHE=outputs/cache
DIAG=outputs/diagnostic/gossipcop
mkdir -p "$PRED" "$CACHE" "$DIAG"

# Data: using the official ARG release (already in data/en/, rationales included).
# step1 (GPT-5.4 advisor regeneration) is intentionally skipped -- ARG is trained
# on the shipped GPT-3.5 rationales.

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
# See src/arg_integration/README.md. Produces ${PRED}/gossipcop_arg.json.

# --- 5. Diagnostic -----------------------------------------------------------
python -m src.step5_diagnostic \
  --dataset GossipCop \
  --pred RoBERTa=${PRED}/gossipcop_roberta.json \
  --pred ARG=${PRED}/gossipcop_arg.json \
  --pred GPT-5.4=${PRED}/gossipcop_gpt54.json \
  --outdir ${DIAG}
