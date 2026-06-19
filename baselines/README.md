# Router baselines

Faithful re-implementations of three published routers, **ported onto this
repo's setting** (binary fake-news classification, frozen RoBERTa features,
train-on-val / eval-on-test) so their cost-quality curves overlay directly on
the step6 Pareto plot and use the same macro-F1 axis.

| Folder | Method | Paper | Official code |
|---|---|---|---|
| `hybrid_llm/` | Hybrid LLM (r_det / r_trans) | arXiv:2404.14618 (ICLR'24) | github.com/m365-core/hybrid_llm_routing |
| `routellm/` | RouteLLM (BERT classifier + matrix factorisation) | arXiv:2406.18665 | github.com/lm-sys/RouteLLM |
| `mozannar_sontag/` | naive learning-to-defer (`L_CE`, α=1) | arXiv:2006.01862 (ICML'20) | github.com/clinicalml/learn-to-defer |

Each official repo targets **generative** tasks with heavy backbones
(DeBERTa-v3-large, Arena preference data, BART-score labels). Rather than run
those unchanged, we port the **core algorithm** onto the frozen RoBERTa features
this repo already produces — see each subfolder's README for the exact mapping
and the paradigm note for the paper.

## Shared feature contract (`common.py`)

Identical to `src/step7_learned_router.py`:

- `small_*`: `[{"id","label","pred","prob","emb":[..768..]?}]` — RoBERTa (+ optional emb)
- `large_*`: `[{"id","label","pred"}]` — GPT-5.4 direct judge
- inner-joined on integer `id`; features `= [emb | prob | entropy]` if every small
  row has `emb`, else scalar-only.
- **train on val, eval on test.** Each script writes a curve JSON with
  `fractions` (x = % routed to LLM) + `router_f1` (y = macro-F1), plus `apgr`
  and CPT in the printout — overlay `router_f1` on step6's plot.

## Prerequisite — embeddings already merged

The baselines use the frozen RoBERTa `emb` feature. In the current
`outputs/preds/*_roberta*.json` the `emb` key is **already merged in** (verified:
all four files carry it), so nothing to do. If you ever regenerate the preds and
the `emb` key is missing, re-merge from the `*_emb.npz` sidecars (the scripts
otherwise silently fall back to scalar-only features):

```bash
cd Router_Exp1
for f in weibo21_roberta weibo21_roberta_val gossipcop_roberta gossipcop_roberta_val; do
  python -m src.merge_emb --preds outputs/preds/$f.json --emb outputs/preds/${f}_emb.npz
done
```

## Run all baselines (Weibo21 example)

```bash
cd Router_Exp1
OUT=outputs/diagnostic/weibo21/baselines

# Hybrid LLM
python -m baselines.hybrid_llm.hybrid_llm_router \
  --small_val outputs/preds/weibo21_roberta_val.json --large_val outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json    --large_test outputs/preds/weibo21_gpt54.json \
  --variant r_trans --name weibo21 --out $OUT
python -m baselines.hybrid_llm.hybrid_llm_router \
  --small_val outputs/preds/weibo21_roberta_val.json --large_val outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json    --large_test outputs/preds/weibo21_gpt54.json \
  --variant r_det --name weibo21 --out $OUT

# RouteLLM (mf = paper's best, plus bert)
python -m baselines.routellm.routellm_router \
  --small_val outputs/preds/weibo21_roberta_val.json --large_val outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json    --large_test outputs/preds/weibo21_gpt54.json \
  --variant mf --name weibo21 --out $OUT
python -m baselines.routellm.routellm_router \
  --small_val outputs/preds/weibo21_roberta_val.json --large_val outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json    --large_test outputs/preds/weibo21_gpt54.json \
  --variant bert --name weibo21 --out $OUT

# Mozannar-Sontag naive L_CE (alpha=1)
python -m baselines.mozannar_sontag.l2d_router \
  --small_val outputs/preds/weibo21_roberta_val.json --large_val outputs/preds/weibo21_gpt54_val.json \
  --small_test outputs/preds/weibo21_roberta.json    --large_test outputs/preds/weibo21_gpt54.json \
  --name weibo21 --out $OUT
```

For **GossipCop (en)** replace `weibo21_` → `gossipcop_` and
`--name weibo21` → `--name gossipcop`, `outputs/diagnostic/weibo21` →
`outputs/diagnostic/gossipcop`.

All commands run from the `Router_Exp1/` root (the `-m` module form needs it).
Outputs land in `outputs/diagnostic/<name>/baselines/routing_*.json`.
