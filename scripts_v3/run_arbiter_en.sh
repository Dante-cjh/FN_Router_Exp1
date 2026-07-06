#!/usr/bin/env bash
# Round-3 arbiter, GossipCop. Feature-tier ladder = ablation table in one run.
set -e
cd "$(dirname "$0")/.."

TIERS="frugal dict emb"
[ -n "$ENC" ] && TIERS="$TIERS enc full"

for t in $TIERS; do
  echo "=== gossipcop arbiter tier=$t ==="
  python -m src_v3.arbiter --name gossipcop --lang en --features "$t" ${ENC:+--enc "$ENC"}
done
