# Hybrid LLM baseline (Ding et al., ICLR 2024)

- Paper: arXiv:2404.14618 — *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*
- Official code: https://github.com/m365-core/hybrid_llm_routing (DeBERTa-v3-large router, BCE loss, relaxation `t`)

## Why a re-implementation rather than the official repo

The official repo trains a 300M DeBERTa-v3-large router on **generative** quality
gaps (BART score, 10× sampling per model). Our task is binary fake-news
classification on top of **frozen RoBERTa features**, so:

- the quality gap degrades to discrete correctness `H = 1[SLM right] − 1[LLM right] ∈ {−1,0,+1}`;
- predictions are deterministic, so `r_prob`'s 10× sampling is unnecessary;
- the router is a cheap logistic head on `[RoBERTa emb | prob | entropy]` — the
  same pre-generation feature set step7 uses.

This keeps the **algorithm** (predict "is the small model good enough", threshold
sweep, relaxation `t` to rebalance labels) while matching this repo's I/O and the
step6 Pareto axes (x = % routed to LLM, y = macro-F1).

## Variants

- `r_det`  — hard label `1[SLM not worse than LLM]` (paper's deterministic router).
- `r_trans` — relaxation `1[H ≥ −t]`; `t` auto-picked to maximise label spread
  (paper Eq.3 in spirit). `t=0` ≡ `r_det`; `t=1` is the permissive
  "SLM right OR LLM wrong" label.

## Run

```bash
cd Router_Exp1
# Weibo21 (zh) — emb already merged into the *_roberta*.json via src/merge_emb.py
python -m baselines.hybrid_llm.hybrid_llm_router \
  --small_val  outputs/preds/weibo21_roberta_val.json \
  --large_val  outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json \
  --large_test outputs/preds/weibo21_gpt54.json \
  --variant r_trans --name weibo21 \
  --out outputs/diagnostic/weibo21/baselines

# GossipCop (en)
python -m baselines.hybrid_llm.hybrid_llm_router \
  --small_val  outputs/preds/gossipcop_roberta_val.json \
  --large_val  outputs/preds/gossipcop_gpt54_val.json \
  --small_test outputs/preds/gossipcop_roberta.json \
  --large_test outputs/preds/gossipcop_gpt54.json \
  --variant r_trans --name gossipcop \
  --out outputs/diagnostic/gossipcop/baselines
```

Also run `--variant r_det` for the deterministic-label ablation.

Output: `outputs/diagnostic/<name>/baselines/routing_hybridllm_<variant>.json`
(`fractions` + `router_f1` overlay directly on step6's plot).
