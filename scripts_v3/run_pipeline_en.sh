#!/usr/bin/env bash
# RA^3 end-to-end (GossipCop): net-utility pre-route x {swallow, arb_f1, arb_crc}.
set -e
cd "$(dirname "$0")/.."

FEAT="${FEAT:-emb}"
ALPHA="${ALPHA:-0.2}"

python -m src_v3.pipeline --name gossipcop --lang en --features "$FEAT" --alpha "$ALPHA" ${ENC:+--enc "$ENC"}
python -m src_v3.pipeline --name gossipcop --lang en --features "$FEAT" --alpha "$ALPHA" --stage1 entropy \
  --out_root outputs_v3/pipeline_entropy ${ENC:+--enc "$ENC"}
