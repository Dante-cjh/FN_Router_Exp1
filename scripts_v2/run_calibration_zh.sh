#!/usr/bin/env bash
# Round-2 calibration experiments (A) — Weibo21 (zh).
# A1: amplify inference dropout (p=0.3).  A2: temperature scaling (T* fit on val).
# Also a combined config. Recomputes uncertainty + diagnostic for each, then a
# side-by-side comparison of the reducible-fraction headline.
# Run from repo root (Router_Exp1/), same env as v1 step3.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=weibo21; LANG=zh; MODEL=hfl/chinese-roberta-wwm-ext
CKPT=outputs/ckpt/weibo21_roberta
T=${T:-30}; DP=${DP:-0.3}

# --- A2: fit temperature on val ---
python -m src_v2.fit_temperature \
  --val data/${LANG}/val.json --language ${LANG} \
  --model_name ${MODEL} --ckpt_dir ${CKPT} \
  --name ${NAME} --out outputs_v2/calibration
TEMP=$(python -c "import json;print(json.load(open('outputs_v2/calibration/${NAME}_temperature.json'))['temperature'])")
echo "[cal] using T*=${TEMP}"

# --- recompute uncertainty under 3 configs (test split is what the diagnostic uses) ---
run_unc () {  # $1=tag  $2=extra args
  python -m src_v2.mc_dropout_uncertainty \
    --input data/${LANG}/test.json --language ${LANG} \
    --model_name ${MODEL} --ckpt_dir ${CKPT} \
    --name ${NAME} --split test --out outputs_v2/uncertainty \
    --T ${T} --tag "$1" $2
}
run_unc "p${DP}"        "--dropout_p ${DP}"
run_unc "temp"          "--temperature ${TEMP}"
run_unc "p${DP}_temp"   "--dropout_p ${DP} --temperature ${TEMP}"

# --- diagnostic for each config (base already exists from the first pilot run) ---
run_diag () {  # $1=tag
  python -m src_v2.uncertainty_diagnostic \
    --name ${NAME} \
    --unc   outputs_v2/uncertainty/${NAME}_test__$1.json \
    --small outputs/preds/${NAME}_roberta.json \
    --large outputs/preds/${NAME}_gpt54.json \
    --out   outputs_v2/diagnostic/${NAME}__$1
}
run_diag "p${DP}"; run_diag "temp"; run_diag "p${DP}_temp"

# --- side-by-side comparison ---
python -m src_v2.compare_calibration \
  --cfg base=outputs_v2/diagnostic/${NAME}/uncertainty_bands.json \
  --cfg p${DP}=outputs_v2/diagnostic/${NAME}__p${DP}/uncertainty_bands.json \
  --cfg temp=outputs_v2/diagnostic/${NAME}__temp/uncertainty_bands.json \
  --cfg p${DP}_temp=outputs_v2/diagnostic/${NAME}__p${DP}_temp/uncertainty_bands.json

echo "[run_calibration_zh] done"
