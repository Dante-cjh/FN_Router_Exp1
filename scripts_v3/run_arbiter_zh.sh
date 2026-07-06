#!/usr/bin/env bash
# Round-3 arbiter, Weibo21. Feature-tier ladder = ablation table in one run.
# ENC=outputs_v3/enc 时追加 enc/full 两档（先跑 scripts_v3/run_encode.sh）。
set -e
cd "$(dirname "$0")/.."

TIERS="frugal dict emb"
[ -n "$ENC" ] && TIERS="$TIERS enc full"

for t in $TIERS; do
  echo "=== weibo21 arbiter tier=$t ==="
  python -m src_v3.arbiter --name weibo21 --lang zh --features "$t" ${ENC:+--enc "$ENC"}
done
