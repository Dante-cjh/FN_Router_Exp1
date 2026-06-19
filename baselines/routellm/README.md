# RouteLLM baseline (Ong et al., 2024)

- Paper: arXiv:2406.18665 — *RouteLLM: Learning to Route LLMs with Preference Data*
- Official code: https://github.com/lm-sys/RouteLLM (four routers: SW-ranking, MF, BERT, causal-LLM)

## Why a re-implementation rather than the official repo

The official repo trains routers on **Chatbot-Arena human preference data** and
serves a strong/weak chat-model pair. We need a pre-generation router over
**frozen RoBERTa features** for binary fake-news classification, evaluated on the
step6 Pareto axes. We port the two cheap, most-transferable routers:

- **`bert`** — a classification head on the contextual embedding → `P(win_strong)`.
  Here it consumes `[RoBERTa emb | prob | entropy]` directly.
- **`mf`** — matrix factorisation, the paper's **best** router: bilinear score
  `s(M,q)=⟨v_M, Wq⟩+b_M`, `P(win_strong)=σ(s(strong,q)−s(weak,q))`. SLM/LLM are
  the two "models" with learned embeddings; the query features are projected to
  the latent rank.

Preference labels become "which model was right":
`win_strong = 1` if LLM right & SLM wrong, `0` if SLM right & LLM wrong, `0.5`
for ties — trained with soft-target BCE. Sweeping the route threshold reproduces
the paper's α sweep; the script also reports **APGR** and **CPT(50/90%)** so the
numbers line up with both papers' terminology (see `baselines/common.py`).

## Run

```bash
cd Router_Exp1
# Weibo21 (zh), matrix-factorisation router (paper's best)
python -m baselines.routellm.routellm_router \
  --small_val  outputs/preds/weibo21_roberta_val.json \
  --large_val  outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json \
  --large_test outputs/preds/weibo21_gpt54.json \
  --variant mf --name weibo21 \
  --out outputs/diagnostic/weibo21/baselines

# BERT-classifier router
python -m baselines.routellm.routellm_router \
  --small_val  outputs/preds/weibo21_roberta_val.json \
  --large_val  outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json \
  --large_test outputs/preds/weibo21_gpt54.json \
  --variant bert --name weibo21 \
  --out outputs/diagnostic/weibo21/baselines

# GossipCop (en): swap weibo21_->gossipcop_ and name/out accordingly.
```

Output: `outputs/diagnostic/<name>/baselines/routing_routellm_<variant>.json`.
