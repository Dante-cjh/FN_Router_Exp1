# -*- coding: utf-8 -*-
"""Pilot: rationale-aware 仲裁器（谁的标签该被采纳）。
动机（错误样本分析）：LLM 的错误方向可由其 rationale 论证类型预测
（memory_match→压倒性判real；GossipCop 的 harm 带=LLM用'matches real events'背书假八卦）。
故在分歧样本上训练 P(LLM 对 | SLM置信度, 分歧方向, rationale特征)，val 训、test 评。
再评估两级流水线：熵 top-k 预路由 → 仲裁器决定采纳谁（vs 直接吞 LLM 标签）。
用法: python -m src_v2.arbitration_pilot --name gossipcop --lang en
"""
import argparse, json, math, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from src_v2.error_analysis import PAT, classify_rationale, entropy

def macro_f1(y, p):
    f1s = []
    for c in [0, 1]:
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
    return 100 * float(np.mean(f1s))

def load(ds, split):
    suf = "" if split == "test" else "_val"
    slm = {d["id"]: d for d in json.load(open(f"outputs/preds/{ds}_roberta{suf}.json"))}
    llm = {d["id"]: d for d in json.load(open(f"outputs/preds/{ds}_gpt54{suf}.json"))}
    return slm, llm

def feats(slm_d, llm_d, lang, use_rat=True):
    ids = sorted(slm_d)
    X, y, sp, lp, yy = [], [], [], [], []
    for i in ids:
        s, l = slm_d[i], llm_d[i]
        pf = s["prob_fake"]; ent = entropy(pf)
        cats = classify_rationale(l.get("rationale", ""), lang)
        onehot = [1.0 if c in cats else 0.0 for c in list(PAT[lang]) + ["other"]]
        row = [pf, ent, float(s["pred"]), float(l["pred"]), float(s["pred"] != l["pred"]),
               len(l.get("rationale", "")) / 100.0]
        if use_rat: row += onehot
        X.append(row); yy.append(s["label"]); sp.append(s["pred"]); lp.append(l["pred"])
    return np.array(X), np.array(yy), np.array(sp), np.array(lp), ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--lang", required=True)
    args = ap.parse_args(); ds, lang = args.name, args.lang
    out = {}
    for use_rat in [True, False]:
        Xv, yv, spv, lpv, _ = feats(*load(ds, "val"), lang, use_rat)
        Xt, yt, spt, lpt, _ = feats(*load(ds, "test"), lang, use_rat)
        # 仲裁器只在分歧样本上定义（一致时无决策必要）
        mv = spv != lpv; mt = spt != lpt
        yv_arb = (lpv[mv] == yv[mv]).astype(int)  # 1 = LLM 对
        yt_arb = (lpt[mt] == yt[mt]).astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(Xv[mv], yv_arb)
        p = clf.predict_proba(Xt[mt])[:, 1]
        auc = roc_auc_score(yt_arb, p) if len(set(yt_arb)) > 1 else float("nan")
        # 全量仲裁（成本=100% LLM 调用）：一致取共识，分歧取仲裁
        pred = spt.copy()
        adopt = p > 0.5
        idx = np.where(mt)[0]
        pred[idx[adopt]] = lpt[idx[adopt]]
        key = "with_rationale" if use_rat else "no_rationale"
        out[key] = dict(
            n_disagree_val=int(mv.sum()), n_disagree_test=int(mt.sum()),
            arb_auc_test=round(float(auc), 3),
            full_arbitration_f1=round(macro_f1(yt, pred), 2),
            adopt_llm_rate=round(float(adopt.mean()), 3),
        )
        if use_rat:
            # 两级流水线：熵 top-k 预路由，路由到的样本 (a) 直接吞LLM标签 vs (b) 仲裁
            ent_t = Xt[:, 1]
            order = np.argsort(-ent_t)
            curve = {}
            for b in [0.05, 0.1, 0.2, 0.3, 0.5]:
                k = int(round(b * len(yt))); sel = order[:k]
                pa = spt.copy(); pa[sel] = lpt[sel]  # 吞标签
                pb = spt.copy()
                selm = np.array([i for i in sel if mt[i]])  # 分歧且被路由
                if len(selm):
                    padopt = clf.predict_proba(Xt[selm])[:, 1] > 0.5
                    pb[selm[padopt]] = lpt[selm[padopt]]
                curve[f"{int(b*100)}%"] = dict(swallow=round(macro_f1(yt, pa), 2),
                                               arbitrated=round(macro_f1(yt, pb), 2))
            out["entropy_route_curve"] = curve
    out["endpoints"] = dict(all_slm=round(macro_f1(yt, spt), 2), all_llm=round(macro_f1(yt, lpt), 2))
    # oracle 上界（全量）
    orc = spt.copy(); orc[(lpt == yt) & (spt != yt)] = lpt[(lpt == yt) & (spt != yt)]
    out["endpoints"]["oracle_full"] = round(macro_f1(yt, orc), 2)
    json.dump(out, open(f"outputs_v2/error_analysis/arbitration_{ds}.json", "w"), indent=2)
    print(ds, json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
