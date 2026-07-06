#!/usr/bin/env bash
# Round-3 full run. 无 GPU 也能跑 frugal/dict/emb 三档与主 pipeline；
# enc/full/transfer 三项需要先在 GPU/联网机器上跑 run_encode.sh。
set -e
cd "$(dirname "$0")"

bash run_arbiter_zh.sh
bash run_arbiter_en.sh
bash run_pipeline_zh.sh
bash run_pipeline_en.sh

if [ -d "../outputs_v3/enc" ]; then
  ENC=../outputs_v3/enc bash run_arbiter_zh.sh
  ENC=../outputs_v3/enc bash run_arbiter_en.sh
  FEAT=full ENC=../outputs_v3/enc bash run_pipeline_zh.sh
  FEAT=full ENC=../outputs_v3/enc bash run_pipeline_en.sh
  bash run_transfer.sh
else
  echo "[skip] outputs_v3/enc 不存在：先跑 scripts_v3/run_encode.sh 再补 enc/full/transfer"
fi
