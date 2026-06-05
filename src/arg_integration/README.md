# ARG integration

The diagnostic reuses the **official ARG implementation**
(<https://github.com/ICTMCG/ARG>, Hu et al., AAAI 2024) to train the
small-model-plus-LLM-advisor leg. We do not re-implement ARG; we only add a
script that dumps aligned per-sample test predictions.

## One-time setup

```bash
git clone https://github.com/ICTMCG/ARG.git
cd ARG
pip install -r requirements.txt
```

Download `bert-base-uncased` (EN) and `chinese-bert-wwm-ext` (ZH) and note
their local paths. The ARG datasets require an application form linked in the
ARG README; once approved you receive `train/val/test.json`.

> Note: ARG's small backbone is BERT. The "RoBERTa" leg of our diagnostic is a
> separate standalone model (`src/step3_train_roberta.py`). ARG here is the
> "small + LLM advisor" system, which is the leg that matters for the routing
> argument; its internal backbone choice does not change the diagnostic.

## Rationales

We use the **shipped GPT-3.5 rationales** as-is (step1 is skipped). The official
ARG data already lives at `Router_Exp1/data/en` and `Router_Exp1/data/zh` with
all rationale fields, so point ARG's `--root_path` straight at those folders.

> If you later decide you want a GPT-5.4 advisor instead, run
> `python -m src.step1_generate_rationales` to overwrite the
> `td_*/cs_*` fields, then retrain. Not needed for the current setup.

## Train ARG

```bash
cd ARG
# edit run_en.sh: set --root_path /path/to/Router_Exp1/data/en  and  --bert_path
bash run_en.sh        # -> ./param_model/ARG_en-arg/1/parameter_bert.pkl
```

## Dump aligned test predictions

Copy `dump_arg_predictions.py` into the ARG repo root and run:

```bash
python dump_arg_predictions.py \
    --root_path /path/to/Router_Exp1/data/en \
    --bert_path /path/to/bert-base-uncased \
    --ckpt ./param_model/ARG_en-arg/1/parameter_bert.pkl \
    --language en \
    --output gossipcop_arg.json
```

Move `gossipcop_arg.json` to `Router_Exp1/outputs/preds/` and feed it to step5.

**ID alignment:** all three prediction files key on `source_id`. ARG's
dataloader casts ids to a tensor, so source_ids must be numeric (GossipCop /
Weibo21 ARG data already are). Use the *same* `test.json` for all three models.
