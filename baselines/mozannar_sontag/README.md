# Mozannar & Sontag (2020) naive learning-to-defer baseline

- Paper: arXiv:2006.01862 — *Consistent Estimators for Learning to Defer to an Expert* (ICML 2020, PMLR v119)
- Official code: https://github.com/clinicalml/learn-to-defer

## What this is

The naive consistent surrogate `L_CE` (α=1), ported to this repo as the
**learning-to-defer ablation** the research report calls for. Augment the label
space to `Y ∪ {⊥}`; a `K+1=3`-logit head on the frozen RoBERTa features is
trained with

```
L_CE = -log softmax(g)[y]  -  1[m == y] · log softmax(g)[⊥]
```

where the expert `m` = GPT-5.4 (LLM) prediction. Test rule: predict
`h(x)=argmax_{y∈Y} g_y`; defer when `g_⊥ ≥ max_y g_y`. **Consistency holds only
at α=1** — keep `--alpha 1.0` for the honest baseline; any other α is a separate,
non-consistent variant.

## Notes that matter for the curve

- On **non-deferred** samples the system prediction is the jointly-learned `h(x)`,
  **not** the standalone RoBERTa pred (faithful to the `(h, r)` system). So the
  0%-routed endpoint is `macro-F1(h)`; the script also prints standalone RoBERTa
  for reference.
- The curve sweeps a threshold on the defer margin `g_⊥ − max_y g_y` (deferring
  highest-margin first); `τ=0` is the naive argmax deferrer, also reported.

## Ablation narrative (per the report)

Naive `L_CE` has **no class weighting** (hurts macro-F1 on imbalanced GossipCop),
**no calibration**, is **not realizable-consistent** (Mozannar et al. 2023), and
has **no profitability gate** (defers whenever the expert is "probably right",
ignoring whether the SLM was already right). Each missing piece maps onto one
component of your upgraded method.

## Run

```bash
cd Router_Exp1
# Weibo21 (zh)
python -m baselines.mozannar_sontag.l2d_router \
  --small_val  outputs/preds/weibo21_roberta_val.json \
  --large_val  outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json \
  --large_test outputs/preds/weibo21_gpt54.json \
  --name weibo21 --out outputs/diagnostic/weibo21/baselines

# GossipCop (en): swap weibo21_->gossipcop_ and name/out accordingly.
```

Output: `outputs/diagnostic/<name>/baselines/routing_mozannar_sontag.json`.
