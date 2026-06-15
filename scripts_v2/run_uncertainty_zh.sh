#!/usr/bin/env bash
# Round-2 pilot — Weibo21 (zh). Epistemic/aleatoric decoupling go/kill test.
# Reuses v1 checkpoint outputs/ckpt/weibo21_roberta/best.pt; writes only outputs_v2/.
# Run from the repo root (Router_Exp1/), same env as v1 step3.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=weibo21
LANG=zh
MODEL=hfl/chinese-roberta-wwm-ext
CKPT=outputs/ckpt/weibo21_roberta
T=${T:-30}

# --- step A: MC-Dropout uncertainty on val + test ---
for SPLIT in val test; do
  python -m src_v2.mc_dropout_uncertainty \
    --input data/${LANG}/${SPLIT}.json --language ${LANG} \
    --model_name ${MODEL} --ckpt_dir ${CKPT} \
    --name ${NAME} --split ${SPLIT} \
    --out outputs_v2/uncertainty --T ${T}
done

# --- step B: go/kill diagnostic on test (joined with v1 SLM/LLM preds) ---
python -m src_v2.uncertainty_diagnostic \
  --name ${NAME} \
  --unc   outputs_v2/uncertainty/${NAME}_test.json \
  --small outputs/preds/${NAME}_roberta.json \
  --large outputs/preds/${NAME}_gpt54.json \
  --out   outputs_v2/diagnostic/${NAME}

echo "[run_zh] done -> outputs_v2/diagnostic/${NAME}/uncertainty_bands.json"
