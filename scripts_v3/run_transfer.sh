#!/usr/bin/env bash
# M5 teaser: zero-shot arbiter transfer zh<->en.
# 检验命题："幻觉式佐证是 LLM 的性质而非数据集的性质"（reports/08 §6 M5）。
# 需要先跑 run_encode.sh（迁移只用可跨语言的特征：标量+词典+enc 交互；SLM emb 自动丢弃）。
set -e
cd "$(dirname "$0")/.."

ENC="${ENC:-outputs_v3/enc}"

echo "=== train on Weibo21(zh) -> eval on GossipCop(en) ==="
python -m src_v3.arbiter --name gossipcop --lang en --features enc --enc "$ENC" \
  --train_name weibo21 --train_lang zh

echo "=== train on GossipCop(en) -> eval on Weibo21(zh) ==="
python -m src_v3.arbiter --name weibo21 --lang zh --features enc --enc "$ENC" \
  --train_name gossipcop --train_lang en
