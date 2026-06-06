# 协同路由诊断实验 (Collaborative-Routing Diagnostic)

A small-but-decisive experiment to test whether your collaborative-routing
thesis has a learning-theoretic basis **before** you commit to it.

## The question

Three models judge fake news with different error profiles:

| Leg | Model | Role |
|---|---|---|
| Small | **RoBERTa** (`roberta-base` / `chinese-roberta-wwm-ext`) | content-only small model |
| Small + advisor | **ARG** (Hu et al., AAAI'24) with **GPT-5.4** as advisor | small model guided by LLM rationales |
| Large | **GPT-5.4** direct judge | LLM classifies the news directly |

We measure how much their mistakes **overlap** and how high a **perfect router**
could reach:

- **Error Jaccard** — `|both wrong| / |≥1 wrong|`. ~1 ⇒ same mistakes (World A,
  routing useless). ~0 ⇒ disjoint mistakes (World B, routing gold).
- **Oracle router upper bound** — accuracy of a perfect router = fraction of
  test samples where *at least one* model is correct.
- **Routing headroom** = oracle − best single model.

**Decision rule (from the brief):** headroom ≥ 5 pts → the routing direction is
well-founded; < 1 pt → pivot; 1–5 pts → grey zone.

## Pipeline

```
data (GossipCop / Weibo21, ARG format)
        │
        ├─ step1_generate_rationales.py   GPT-5.4 advisor → td/cs rationale+pred+acc
        │
   ┌────┼─────────────────────────────────────────────┐
   │    │                                              │
 step3  step2_llm_direct.py                    ARG repo + arg_integration/
RoBERTa  GPT-5.4 direct judge                  dump_arg_predictions.py
   │    │                                              │
   └────┴──────────────► step5_diagnostic.py ◄─────────┘
                         Jaccard + oracle + report + figures
```

Each model writes a **unified prediction file**:
`[{"id":..., "label": <gold 0/1>, "pred": <0/1>}, ...]`, all keyed on the same
`source_id`. step5 inner-joins them and produces `report.md`, `summary.json`,
and two figures.

## Setup

```bash
pip install -r requirements.txt          # install the CUDA torch build for your 4090
cp .env.example .env                     # then paste your real OPENAI_API_KEY
```

## Data

We use the **official ARG release** (full text + GPT-3.5 rationales), already
placed under `data/`:

```
data/en/{train,val,test}.json     GossipCop : 3884 / 1274 / 1258   (full article text)
data/zh/{train,val,test}.json     Weibo21   : 5203 / 1951 / 1951
```

Each item carries the complete ARG schema —
`content, label, source_id (numeric), td_rationale, cs_rationale, td_pred,
cs_pred, td_acc, cs_acc` — so it loads directly in both the official ARG repo
and this pipeline.

**Advisor = GPT-5.4.** The data ships GPT-3.5 rationales, but **step1
regenerates them in-place with GPT-5.4** so all three diagnostic legs share the
same large model (ARG advisor + step2 direct judge). step1 overwrites only the
six fields `td_rationale, cs_rationale, td_pred, cs_pred, td_acc, cs_acc`
(everything else preserved), makes a one-time `*.bak` backup, and is resumable
via its cache. It runs automatically as step 1 of `scripts/run_*.sh`; to do it
manually:

```bash
for s in train val test; do
  python -m src.step1_generate_rationales \
    --input data/zh/$s.json --in_place \
    --cache outputs/cache/zh_${s}_rat.jsonl --language zh
done   # ~18k calls for zh, ~13k for en; restartable any time
```

Train ARG **after** step1 so its advisor is GPT-5.4. To revert to the shipped
GPT-3.5 rationales, restore the `*.bak` files.

> `src/prepare_data.py` is a **fallback only** — for rebuilding splits from raw
> sources (Weibo21 `.pkl`, FakeNewsNet GossipCop CSVs) when the official ARG
> release isn't available. It is not used in the current setup.

## Run (on the 4090 server)

```bash
bash scripts/run_en.sh     # GossipCop
bash scripts/run_zh.sh     # Weibo21
```

Steps are independent and resumable; the two GPT-5.4 steps cache every call to
`outputs/cache/*.jsonl`, so an interrupted/expensive API run picks up where it
left off. ARG training happens inside the cloned ARG repo —
see `src/arg_integration/README.md`.

## Outputs

```
outputs/preds/<dataset>_<model>.json   unified per-sample predictions
outputs/diagnostic/<dataset>/
    report.md          human-readable verdict + tables + figures
    summary.json       machine-readable metrics
    accuracy_vs_oracle.png
    error_jaccard_heatmap.png
```

The `report.md` files are exactly what you can drop into the **motivation**
section of the paper.

## Files

```
src/common/llm_client.py        async OpenAI-compatible client (.env, retries, concurrency)
src/common/prompts.py           ARG td/cs advisor prompts + direct-judge prompts (EN/ZH)
src/common/io_utils.py          IO + label conventions (real=0, fake=1)
src/step1_generate_rationales.py  build ARG rationale data with GPT-5.4 advisor
src/step2_llm_direct.py         GPT-5.4 direct judge → preds
src/step3_train_roberta.py      RoBERTa SLM baseline → preds
src/arg_integration/            run ARG + dump aligned ARG preds
src/step5_diagnostic.py         Jaccard + oracle + report
scripts/run_en.sh, run_zh.sh    end-to-end drivers
```

## Notes & caveats

- **Advisor = GPT-5.4.** ARG ships GPT-3.5 rationales; step1 regenerates them
  with GPT-5.4 so the advisor matches your design. If you accept GPT-3.5
  rationales to save cost, skip step1 and train ARG on the shipped data.
- **ARG's backbone is BERT**, not RoBERTa. The "RoBERTa" leg is the separate
  standalone small model; ARG is the "small + advisor" system. The backbone
  choice doesn't affect the complementarity question.
- **Prompts** in `src/common/prompts.py` follow the ARG two-perspective method
  but are paraphrased; align them to the ARG paper appendix for a 1:1 repro.
- **Cost control.** Run the diagnostic on the test split only for GPT-5.4
  direct; step1 (rationales) is the expensive part since it covers train+val+test.
```
