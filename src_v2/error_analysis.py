# -*- coding: utf-8 -*-
"""错误样本深度分析（导师要求）：
1) 四象限 × 类别 × 置信度/不确定性 结构
2) LLM 错误样本的 rationale（thinking）失败模式挖掘
3) SLM 失败模式（类别、置信度、文本长度）
4) U_epi/U_ale 对监督特征的冗余度诊断（解释消融为何≈0）
输出: outputs_v2/error_analysis/<ds>/{summary.json, llm_errors.jsonl, slm_only_wrong.jsonl}
用法: python -m src_v2.error_analysis --name gossipcop --lang en
"""
import argparse, json, math, re, os
import numpy as np

def entropy(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))

# rationale 模式词典：三种论证类型
PAT = {
    "en": {
        "memory_match": r"\b(match(es|ed)?|widely reported|indeed|accurate(ly)?|confirmed|was (a )?real|actually (happened|occurred)|documented|verifiable)\b",
        "plausibility": r"\b(plausible|reads like|style|tone|typical|natural|consistent with|realistic|genuine)\b",
        "skepticism":   r"\b(lack(s|ing)?|no (reliable|credible|verifiable)|unverified|rumor|fabricat|implausible|sensational|clickbait|exaggerat|hoax|satir|unsubstantiated|misleading)\b",
    },
    "zh": {
        "memory_match": r"(属实|对应|确实|相符|吻合|一致|真实事件|确有|已证实|报道过)",
        "plausibility": r"(合理|符合常识|符合.{0,4}风格|通顺|自然|可信度较高)",
        "skepticism":   r"(缺乏|谣|夸大|未经证实|移花接木|不实|捏造|失实|误导|耸动|猎奇|拼接|冒用|无可靠|无权威|难以核实|可信度(很|较)?低)",
    },
}

def classify_rationale(r, lang):
    tags = [k for k, pat in PAT[lang].items() if re.search(pat, r, re.I)]
    return tags or ["other"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--unc_tag", default="__p0.3")
    args = ap.parse_args()
    ds, lang = args.name, args.lang

    data = {d["source_id"]: d for d in json.load(open(f"data/{lang}/test.json"))}
    slm = {d["id"]: d for d in json.load(open(f"outputs/preds/{ds}_roberta.json"))}
    llm = {d["id"]: d for d in json.load(open(f"outputs/preds/{ds}_gpt54.json"))}
    unc = np.load(f"outputs_v2/uncertainty/{ds}_test{args.unc_tag}.npz")
    umap = {int(i): (float(e), float(a)) for i, e, a in zip(unc["ids"], unc["U_epi"], unc["U_ale"])}

    rows = []
    for i, s in slm.items():
        l, d = llm[i], data[i]
        y = s["label"]
        rows.append(dict(
            id=i, y=y, slm=s["pred"], llm=l["pred"], prob_fake=s["prob_fake"],
            ent=entropy(s["prob_fake"]), rat=l.get("rationale", ""),
            text=d["content"], tlen=len(d["content"]), time=d.get("time"),
            U_epi=umap.get(i, (np.nan,)*2)[0], U_ale=umap.get(i, (np.nan,)*2)[1],
        ))
    for r in rows:
        so, lo = r["slm"] == r["y"], r["llm"] == r["y"]
        r["band"] = "both_correct" if so and lo else "harm(only_SLM)" if so else "gain(only_LLM)" if lo else "both_wrong"

    S = {"n": len(rows)}

    # ---- 1. 四象限结构 ----
    bands = {}
    for b in ["both_correct", "harm(only_SLM)", "gain(only_LLM)", "both_wrong"]:
        g = [r for r in rows if r["band"] == b]
        if not g: continue
        bands[b] = dict(
            n=len(g), fake_share=round(np.mean([r["y"] for r in g]), 3),
            slm_ent_mean=round(np.mean([r["ent"] for r in g]), 3),
            slm_conf_mean=round(np.mean([max(r["prob_fake"], 1 - r["prob_fake"]) for r in g]), 3),
            U_epi_mean=float(np.nanmean([r["U_epi"] for r in g])),
            U_ale_mean=float(np.nanmean([r["U_ale"] for r in g])),
            tlen_median=int(np.median([r["tlen"] for r in g])),
        )
    S["bands"] = bands

    # ---- 2. LLM / SLM 混淆结构（按真类）----
    def confusion(who):
        out = {}
        for y, yname in [(0, "real"), (1, "fake")]:
            g = [r for r in rows if r["y"] == y]
            err = [r for r in g if r[who] != y]
            out[yname] = dict(n=len(g), err=len(err), err_rate=round(len(err) / max(len(g), 1), 3))
        return out
    S["slm_confusion"] = confusion("slm")
    S["llm_confusion"] = confusion("llm")

    # ---- 3. LLM rationale 失败模式 ----
    def rat_stats(sub):
        cats = {}
        for r in sub:
            for t in classify_rationale(r["rat"], lang):
                cats[t] = cats.get(t, 0) + 1
        return cats
    llm_wrong = [r for r in rows if r["llm"] != r["y"]]
    llm_right = [r for r in rows if r["llm"] == r["y"]]
    S["rationale_cats_llm_wrong"] = rat_stats(llm_wrong)
    S["rationale_cats_llm_right"] = rat_stats(llm_right)
    # 每种论证类型的错误率 + 按 LLM 预测方向拆
    cat_err = {}
    for cat in list(PAT[lang]) + ["other"]:
        hit = [r for r in rows if cat in classify_rationale(r["rat"], lang)]
        if not hit: continue
        w = [r for r in hit if r["llm"] != r["y"]]
        cat_err[cat] = dict(n=len(hit), err_rate=round(len(w) / len(hit), 3),
                            pred_real_share=round(np.mean([1 - r["llm"] for r in hit]), 3))
    S["rationale_cat_error_rate"] = cat_err
    # LLM 错误的方向分解
    S["llm_error_direction"] = dict(
        fake_missed=len([r for r in llm_wrong if r["y"] == 1]),   # 假新闻判成真
        real_flagged=len([r for r in llm_wrong if r["y"] == 0]),  # 真新闻判成假
    )

    # ---- 4. U 特征冗余度：CV-AUC 增量测试（在 test 内 5 折）----
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score
        emb = {d["id"]: d["emb"] for d in json.load(open(f"outputs/preds/{ds}_roberta.json"))}
        ok = [r for r in rows if not np.isnan(r["U_epi"])]
        y_gain = np.array([1 if r["band"] == "gain(only_LLM)" else 0 for r in ok])
        E = np.array([emb[r["id"]] for r in ok])
        base = np.array([[r["prob_fake"], r["ent"]] for r in ok])
        U = np.array([[r["U_epi"], r["U_ale"]] for r in ok])
        def cv_auc(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
            return round(roc_auc_score(y, p), 3)
        corr = np.corrcoef(np.log(U[:, 0] + 1e-12), base[:, 1])[0, 1]  # log U_epi vs entropy
        S["redundancy"] = dict(
            corr_logUepi_entropy=round(float(corr), 3),
            corr_Uale_entropy=round(float(np.corrcoef(U[:, 1], base[:, 1])[0, 1]), 3),
            auc_gain_scalar=cv_auc(base, y_gain),
            auc_gain_scalar_plus_U=cv_auc(np.hstack([base, U]), y_gain),
            auc_gain_emb=cv_auc(E, y_gain),
            auc_gain_emb_plus_U=cv_auc(np.hstack([E, base, U]), y_gain),
            auc_gain_U_only=cv_auc(U, y_gain),
        )
    except Exception as e:
        S["redundancy"] = {"error": str(e)}

    out = f"outputs_v2/error_analysis/{ds}"
    os.makedirs(out, exist_ok=True)
    json.dump(S, open(f"{out}/summary.json", "w"), ensure_ascii=False, indent=2)

    # ---- 5. 供人工精读的错误样本 dump ----
    def dump(fn, sub, key):
        with open(f"{out}/{fn}", "w") as f:
            for r in sorted(sub, key=key):
                f.write(json.dumps(dict(
                    id=r["id"], band=r["band"], y=r["y"], slm=r["slm"], llm=r["llm"],
                    prob_fake=round(r["prob_fake"], 3), U_epi=r["U_epi"], U_ale=r["U_ale"],
                    rationale=r["rat"], cats=classify_rationale(r["rat"], lang),
                    text=r["text"][:500]), ensure_ascii=False) + "\n")
    dump("llm_errors.jsonl", llm_wrong, lambda r: (r["band"], r["y"]))
    dump("slm_only_wrong.jsonl", [r for r in rows if r["band"] == "gain(only_LLM)"], lambda r: r["y"])
    print(json.dumps(S, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__":
    main()
