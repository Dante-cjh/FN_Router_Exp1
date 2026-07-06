#!/usr/bin/env bash
# Frozen multilingual encodings of (news, rationale) for the enc/full tiers
# and the zh<->en transfer test. Needs: pip install sentence-transformers
# GPU 机器上跑；模型可换 BAAI/bge-m3（更强更慢）: MODEL=BAAI/bge-m3 bash scripts_v3/run_encode.sh
set -e
cd "$(dirname "$0")/.."

MODEL="${MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"

python -m src_v3.encode_rationales --name weibo21   --lang zh --splits val test --model "$MODEL"
python -m src_v3.encode_rationales --name gossipcop --lang en --splits val test --model "$MODEL"
