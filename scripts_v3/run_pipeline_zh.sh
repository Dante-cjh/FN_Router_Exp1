#!/usr/bin/env bash
# RA^3 end-to-end (Weibo21): net-utility pre-route x {swallow, arb_f1, arb_crc}.
# FEAT 默认 emb（pilot 配置）；编码就绪后 FEAT=full ENC=outputs_v3/enc 重跑。
set -e
cd "$(dirname "$0")/.."

FEAT="${FEAT:-emb}"
ALPHA="${ALPHA:-0.2}"

python -m src_v3.pipeline --name weibo21 --lang zh --features "$FEAT" --alpha "$ALPHA" ${ENC:+--enc "$ENC"}
# 对照：最弱预路由信号（熵）+ 仲裁，隔离 stage-2 的贡献
python -m src_v3.pipeline --name weibo21 --lang zh --features "$FEAT" --alpha "$ALPHA" --stage1 entropy \
  --out_root outputs_v3/pipeline_entropy ${ENC:+--enc "$ENC"}
