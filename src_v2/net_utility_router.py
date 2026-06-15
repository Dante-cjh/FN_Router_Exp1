#!/usr/bin/env python
"""
src_v2/net_utility_router.py — Round-2 MAIN METHOD (supervised version).

The fix for v1's failure (reports/02 §2, reports/05 §5): replace the v1
"single gain score -> top-k -> adopt LLM label" router with a RISK-AWARE,
class-weighted, gated selective-upgrade router. Four pieces, each tied to an
observed failure:

  1. DUAL HEAD          predict p_gain(x)=P(LLM-right & SLM-wrong) AND
                        p_harm(x)=P(SLM-right & LLM-wrong). v1 only had gain.
  2. CLASS-WEIGHTED     decide on NET macro-F1 utility, not a gain ranking:
     UTILITY              u(x) = w[ŷ_llm]*p_gain(x) − w[ŷ_slm]*p_harm(x)
                        w_c = marginal macro-F1 value of one correction in class c
                        (estimated on val), so the fake/real imbalance is handled.
  3. CALIBRATION        isotonic/Platt on val so p_gain/p_harm are usable at the
                        low base rates (GossipCop gain rate 4.5%).
  4. REGIME GATE        dataset-level go/no-go from a VAL-ONLY rescue-asymmetry
                        statistic G = R_up − R_down with a bootstrap CI. Gate
                        closed -> route nothing -> provably == all-SLM floor.

  + BUDGET = UPPER BOUND: only route samples with u(x) > 0, up to fraction b
    (never forced to spend the whole budget). In an unfavourable regime the
    curve auto-caps at all-SLM.

Features = [emb(768), prob, entropy(prob), U_epi, U_ale]. Train on VAL, eval on
TEST, never train on test. Output overlays directly on v1 step7's Pareto.

ISOLATION: reads v1 preds (outputs/preds/*) + step-A uncertainty
(outputs_v2/uncertainty/*); writes only outputs_v2/router/.

USAGE:
  python -m src_v2.net_utility_router \
    --name weibo21 \
    --small_val outputs/preds/weibo21_roberta_val.json \
    --large_val outputs/preds/weibo21_gpt54_val.json \
    --small_test outputs/preds/weibo21_roberta.json \
    --large_test outputs/preds/weibo21_gpt54.json \
    --unc_val  outputs_v2/uncertainty/weibo21_val.json \
    --unc_test outputs_v2/uncertainty/weibo21_test.json \
    --out outputs_v2/router/weibo21
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ROOT)


# ---------------------------------------------------------------- io / features
def load_rows(path):
    return {int(r["id"]): r for r in json.loads(Path(path).read_text())}


def binary_entropy(p1, eps=1e-12):
    p1 = np.clip(p1, eps, 1 - eps)
    return -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))


def get_emb(rows_by_id, ids, small_path):
    """Penultimate [CLS] emb: prefer inline 'emb', else the *_emb.npz sidecar."""
    if all("emb" in rows_by_id[i] for i in ids):
        return np.array([rows_by_id[i]["emb"] for i in ids], dtype=np.float32)
    base, _ = os.path.splitext(small_path)
    npz = base + "_emb.npz"
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        idx = {int(k): j for j, k in enumerate(z["ids"])}
        return np.array([z["emb"][idx[i]] for i in ids], dtype=np.float32)
    raise SystemExit(f"no emb inline and no sidecar {npz}")


def build_split(small_path, large_path, unc_path):
    S, L, U = load_rows(small_path), load_rows(large_path), load_rows(unc_path)
    ids = sorted(set(S) & set(L) & set(U))
    if not ids:
        raise SystemExit(f"no overlapping ids for {small_path}")
    y = np.array([S[i]["label"] for i in ids])
    sp = np.array([S[i]["pred"] for i in ids])
    lp = np.array([L[i]["pred"] for i in ids])
    prob = np.array([float(S[i].get("prob", 0.5)) for i in ids])
    U_epi = np.array([U[i]["U_epi"] for i in ids])
    U_ale = np.array([U[i]["U_ale"] for i in ids])
    emb = get_emb(S, ids, small_path)
    scal = np.column_stack([prob, binary_entropy(prob), U_epi, U_ale])
    feats = np.column_stack([emb, scal])
    gain = ((lp == y) & (sp != y)).astype(int)
    harm = ((sp == y) & (lp != y)).astype(int)
    return dict(ids=ids, y=y, sp=sp, lp=lp, feats=feats, gain=gain, harm=harm)


# ---------------------------------------------------------------- metrics
def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro") * 100.0


def class_weights_macro_f1(y, sp, cap=300, rng=None):
    """Marginal macro-F1 value of correcting one currently-wrong gold-c sample.

    Finite-difference on val: for class c, average ΔmacroF1 from flipping one
    mispredicted gold-c sample to correct. Normalised to mean 1 across classes.
    Captures why a minority-class (fake) correction is worth more under macro-F1.
    """
    rng = rng or np.random.default_rng(0)
    base = macro_f1(y, sp)
    w = {}
    for c in (0, 1):
        wrong = np.where((y == c) & (sp != c))[0]
        if len(wrong) == 0:
            w[c] = 1.0
            continue
        if len(wrong) > cap:
            wrong = rng.choice(wrong, cap, replace=False)
        deltas = []
        for j in wrong:
            sp2 = sp.copy(); sp2[j] = c
            deltas.append(macro_f1(y, sp2) - base)
        w[c] = float(np.mean(deltas))
    m = (w[0] + w[1]) / 2.0 or 1.0
    return {0: w[0] / m, 1: w[1] / m}


def fit_head(Xtr, ytr, Xte, calib="sigmoid"):
    """Calibrated probability that a sample is in the (gain|harm) band."""
    from sklearn.linear_model import LogisticRegression
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    if ytr.sum() < 10 or calib == "none":
        base.fit(Xtr, ytr)
        return base.predict_proba(Xte)[:, 1]
    try:
        from sklearn.calibration import CalibratedClassifierCV
        n_pos = int(ytr.sum())
        cv = max(2, min(5, n_pos // 5))
        clf = CalibratedClassifierCV(base, method=calib, cv=cv)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    except Exception as e:  # pragma: no cover
        print(f"  [head] calibration failed ({e}); falling back to raw LogReg")
        base.fit(Xtr, ytr)
        return base.predict_proba(Xte)[:, 1]


def regime_gate(val, w, n_boot=2000, alpha=0.05, rng=None):
    """Dataset-level go/no-go from val rescue-asymmetry G = R_up − R_down.

    R_up   = Σ_c w_c · #{val: only-LLM-correct, gold=c}     (升级能净捞)
    R_down = Σ_c w_c · #{val: only-SLM-correct, gold=c}     (升级会误伤)
    Bootstrap over val samples -> CI. Gate OPENS iff CI_lower > 0.
    """
    rng = rng or np.random.default_rng(0)
    y, sp, lp = val["y"], val["sp"], val["lp"]
    up = ((lp == y) & (sp != y))     # only-LLM-correct
    dn = ((sp == y) & (lp != y))     # only-SLM-correct
    wv = np.array([w[int(c)] for c in y])

    def G_of(idx):
        return (wv[idx] * up[idx]).sum() - (wv[idx] * dn[idx]).sum()

    n = len(y)
    point = G_of(np.arange(n))
    boots = np.array([G_of(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "G_point": float(point),
        "G_ci_low": float(lo), "G_ci_high": float(hi),
        "R_up": float((wv * up).sum()), "R_down": float((wv * dn).sum()),
        "n_only_llm": int(up.sum()), "n_only_slm": int(dn.sum()),
        "gate_open": bool(lo > 0),
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    for k in ("name", "small_val", "large_val", "small_test", "large_test",
              "unc_val", "unc_test", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--calib", default="sigmoid",
                    choices=["sigmoid", "isotonic", "none"])
    ap.add_argument("--gate", default="off", choices=["off", "ci", "sign"],
                    help="dataset-level regime gate on top of the per-sample u>0 "
                         "filter. off=rely on per-sample net-utility (default; the "
                         "per-sample u>0 filter already gives no-harm). "
                         "ci=require val G bootstrap CI_lower>0. sign=require val "
                         "G point>0. NOTE: on temporal-split data (Weibo21) the val "
                         "gate can be non-representative — see reports/05b.")
    ap.add_argument("--drop_uncertainty", action="store_true",
                    help="ablation: drop the U_epi/U_ale features (keep emb+prob+"
                         "entropy) to measure the marginal value of decoupling")
    ap.add_argument("--points", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(args.seed)
    val = build_split(args.small_val, args.large_val, args.unc_val)
    test = build_split(args.small_test, args.large_test, args.unc_test)

    if args.drop_uncertainty:   # ablation: drop trailing U_epi,U_ale columns
        val["feats"] = val["feats"][:, :-2]
        test["feats"] = test["feats"][:, :-2]
        print("[ablation] dropped U_epi/U_ale features")

    # ---- standardize features on val ----
    sc = StandardScaler().fit(val["feats"])
    Xtr, Xte = sc.transform(val["feats"]), sc.transform(test["feats"])

    # ---- class weights (val), regime gate (val) ----
    w = class_weights_macro_f1(val["y"], val["sp"], rng=rng)
    gate = regime_gate(val, w, rng=rng)

    # ---- dual heads ----
    p_gain = fit_head(Xtr, val["gain"], Xte, args.calib)
    p_harm = fit_head(Xtr, val["harm"], Xte, args.calib)
    auc_gain = (float(roc_auc_score(test["gain"], p_gain))
                if test["gain"].sum() else float("nan"))
    auc_harm = (float(roc_auc_score(test["harm"], p_harm))
                if test["harm"].sum() else float("nan"))

    # ---- net utility on test ----
    y, sp, lp = test["y"], test["sp"], test["lp"]
    w_gain_cls = np.array([w[int(c)] for c in lp])   # class that BECOMES correct
    w_harm_cls = np.array([w[int(c)] for c in sp])   # class that WAS correct
    u = w_gain_cls * p_gain - w_harm_cls * p_harm    # net expected macro-F1 utility
    n = len(y)
    all_small, all_large = macro_f1(y, sp), macro_f1(y, lp)

    # ---- per-sample net-utility filter: route only u>0, highest-u first ----
    # This is the PRIMARY no-harm mechanism (a sample is upgraded only when its
    # expected class-weighted macro-F1 gain is positive). Budget = UPPER bound.
    order_ps = np.argsort(-u); order_ps = order_ps[u[order_ps] > 0]

    def curve(order):
        out = []
        for f in fr:
            k = min(int(round(f * n)), len(order))
            mask = np.zeros(n, bool); mask[order[:k]] = True
            comb = sp.copy(); comb[mask] = lp[mask]
            out.append(macro_f1(y, comb))
        return out

    fr = np.linspace(0, 1, args.points)
    # optional dataset-level gate on TOP of the per-sample filter
    if args.gate == "ci":
        gate_allows = gate["gate_open"]
    elif args.gate == "sign":
        gate_allows = gate["G_point"] > 0
    else:  # "off"
        gate_allows = True
    order = order_ps if gate_allows else np.array([], dtype=int)
    n_pos = int(len(order))
    learned = curve(order)
    learned_ps = curve(order_ps)   # pure per-sample reference (gate-independent)

    peak = max(learned); peak_i = int(np.argmax(learned))
    # realized test G (did the gate decision match reality?)
    test_gate = regime_gate(test, w, rng=rng)

    result = {
        "name": args.name, "n": n,
        "all_small_f1": all_small, "all_large_f1": all_large,
        "class_weights": {str(k): v for k, v in w.items()},
        "regime_gate_val": gate,
        "regime_gate_test_realized": test_gate,
        "head_auc": {"gain": auc_gain, "harm": auc_harm},
        "gate_mode": args.gate, "gate_allows": bool(gate_allows),
        "n_routed_positive_utility": n_pos,
        "fractions": fr.tolist(),
        "learned_f1": learned,
        "learned_f1_persample": learned_ps,
        "peak_f1": peak, "peak_budget": float(fr[peak_i]),
        "calib": args.calib,
    }
    odir = Path(args.out); odir.mkdir(parents=True, exist_ok=True)
    (odir / "net_utility.json").write_text(json.dumps(result, indent=2))

    # ---- readout ----
    print(f"\n===== [{args.name}] net-utility router (n={n}) =====")
    print(f"  class weights w: real(0)={w[0]:.3f}  fake(1)={w[1]:.3f}")
    print(f"  regime gate (val): G={gate['G_point']:.2f} "
          f"CI[{gate['G_ci_low']:.2f},{gate['G_ci_high']:.2f}] "
          f"-> {'OPEN' if gate['gate_open'] else 'CLOSED'}  "
          f"(R_up={gate['R_up']:.1f} R_down={gate['R_down']:.1f})")
    print(f"  test realized G={test_gate['G_point']:.2f} "
          f"-> gate decision {'MATCHED' if (gate['gate_open']==(test_gate['G_point']>0)) else 'MISMATCH'}")
    print(f"  head AUC: gain={auc_gain:.3f}  harm={auc_harm:.3f}")
    print(f"  gate mode={args.gate} -> allows routing={gate_allows}")
    print(f"  all-SLM={all_small:.2f}  all-LLM={all_large:.2f}  "
          f"#(u>0 per-sample)={int(len(order_ps))}")
    print(f"  peak learned={peak:.2f} @ {fr[peak_i]*100:.0f}% budget")
    for f in (0.05, 0.1, 0.2, 0.5):
        i = int(round(f * (args.points - 1)))
        print(f"    budget {int(f*100):>2}% -> learned {learned[i]:.2f} "
              f"(per-sample {learned_ps[i]:.2f})")
    floor_ok = all(v >= all_small - 1e-9 for v in learned_ps)
    print(f"  NO-HARM (per-sample curve >= all-SLM everywhere): {floor_ok}")
    print(f"  wrote {odir/'net_utility.json'}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4.2))
        plt.plot(fr, learned, "-o", ms=3, label=f"net-utility router (gate={args.gate})")
        plt.plot(fr, learned_ps, "--", label="per-sample u>0 reference")
        plt.axhline(all_small, color="gray", lw=1, label="all-SLM floor")
        plt.scatter([0, 1], [all_small, all_large], zorder=5,
                    label="all-SLM / all-LLM")
        plt.xlabel("fraction routed to LLM (cost proxy)"); plt.ylabel("macro-F1")
        ttl = "OPEN" if gate["gate_open"] else "CLOSED"
        plt.title(f"Net-utility router — {args.name} (gate {ttl})")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(odir / "net_utility_pareto.png", dpi=150); plt.close()
        print(f"  wrote {odir/'net_utility_pareto.png'}")
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
