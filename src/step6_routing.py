#!/usr/bin/env python
"""
step6_routing.py — Cost-quality Pareto for SLM<->LLM routing.

Builds the routing motivation figure: macro-F1 vs. fraction-of-samples-sent-to-LLM,
for several routing policies, on top of the same per-sample predictions step5 used.

INPUT FILES (unified per-sample preds, inner-joined on `id`)
  --small : [{"id", "label", "pred", "prob"}]   SLM (RoBERTa). `prob` = P(label==1).
                                                 (optional: "emb" for later learned router)
  --large : [{"id", "label", "pred"}]            LLM direct judge (GPT-5.4).

POLICIES (x = fraction routed to LLM, a cost proxy; y = macro-F1 of combined preds)
  all_small / all_large   the two endpoints you already have
  random(frac)            floor baseline, averaged over --seeds random subsets
  conf_threshold(frac)    escalate the most-UNCERTAIN SLM samples (by entropy) to LLM
  oracle_frontier(frac)   escalate by true per-sample gain ordering -> upper bound

USAGE
  python step6_routing.py \
      --small outputs/preds/weibo21_roberta.json \
      --large outputs/preds/weibo21_gpt.json \
      --name  weibo21 \
      --out   outputs/diagnostic/weibo21

Repeat with the en files for GossipCop. The conf_threshold curve REQUIRES `prob`
in the small file; if it is missing the script still runs random + oracle and warns.
"""
import json, argparse, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score


def load(path):
    rows = json.loads(Path(path).read_text())
    return {int(r["id"]): r for r in rows}


def macro_f1(y, p):
    return f1_score(y, p, average="macro") * 100.0


def entropy(p1):
    # binary entropy of [p1, 1-p1]; higher = more uncertain
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    return -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))


def combine(small_pred, large_pred, route_mask):
    # route_mask[i]=True -> trust the LLM for sample i, else the SLM
    out = small_pred.copy()
    out[route_mask] = large_pred[route_mask]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--points", type=int, default=21)  # fractions 0..1
    args = ap.parse_args()

    S, L = load(args.small), load(args.large)
    ids = sorted(set(S) & set(L))
    n = len(ids)
    y = np.array([S[i]["label"] for i in ids])
    sp = np.array([S[i]["pred"] for i in ids])
    lp = np.array([L[i]["pred"] for i in ids])

    def _prob(r):
        # accept either `prob` (new step3 dump) or `prob_fake` (legacy alias)
        v = r.get("prob", r.get("prob_fake"))
        return None if v is None else float(v)

    have_prob = all(_prob(S[i]) is not None for i in ids)
    if have_prob:
        prob = np.array([_prob(S[i]) for i in ids])
        unc = entropy(prob)              # SLM uncertainty signal
    else:
        warnings.warn("small preds have no `prob`/`prob_fake` -> skipping "
                      "conf_threshold. Add P(label==1) to step3's dump.")

    fr = np.linspace(0, 1, args.points)
    f1_small = macro_f1(y, sp)
    f1_large = macro_f1(y, lp)
    rng = np.random.default_rng(0)

    # --- random routing (averaged over seeds) ---
    f1_random = []
    for f in fr:
        k = int(round(f * n))
        vals = []
        for s in range(args.seeds):
            mask = np.zeros(n, bool)
            mask[rng.choice(n, k, replace=False)] = True
            vals.append(macro_f1(y, combine(sp, lp, mask)))
        f1_random.append(float(np.mean(vals)))

    # --- confidence-threshold routing (escalate most uncertain) ---
    f1_conf = None
    if have_prob:
        order = np.argsort(-unc)          # most uncertain first
        f1_conf = []
        for f in fr:
            k = int(round(f * n))
            mask = np.zeros(n, bool)
            mask[order[:k]] = True
            f1_conf.append(macro_f1(y, combine(sp, lp, mask)))

    # --- oracle frontier: the genuine macro-F1 UPPER BOUND at each budget ---
    # A perfect router only escalates samples where the LLM fixes an SLM error
    # (large-right & small-wrong); escalating anything else is at best neutral
    # (both wrong / both right -> identical pred in binary) or harmful
    # (small-right & large-wrong). So with a budget of k routes, the best you can
    # do is correct up to min(k, #gains) of those error-fixes.
    #
    # Which gains to pick matters for macro-F1 (a minority-class fix is worth more
    # than a majority-class one), so we pick greedily by marginal macro-F1.
    # Because each class's F1 is concave in #corrections and the two classes are
    # independent unit-cost items, greedy-by-marginal-gain is optimal at *every*
    # prefix k -> a monotone non-decreasing frontier, the true ceiling.
    s_ok, l_ok = (sp == y), (lp == y)
    gain_mask = l_ok & ~s_ok                       # the only useful escalations
    n_gain = int(gain_mask.sum())

    def f1_from_C(C):
        tp1, fp1, fn1 = C[1, 1], C[0, 1], C[1, 0]
        d1 = 2 * tp1 + fp1 + fn1
        f1_1 = (2 * tp1 / d1) if d1 else 0.0
        tp0, fp0, fn0 = C[0, 0], C[1, 0], C[0, 1]
        d0 = 2 * tp0 + fp0 + fn0
        f1_0 = (2 * tp0 / d0) if d0 else 0.0
        return 50.0 * (f1_0 + f1_1)                # *100/2 -> percent macro-F1

    # confusion of the all-SLM starting point: C[gold, pred]
    C = np.zeros((2, 2), float)
    for g in (0, 1):
        for p in (0, 1):
            C[g, p] = float(((y == g) & (sp == p)).sum())

    # available gain samples per gold class (each currently mispredicted as 1-gold)
    avail = {0: int((gain_mask & (y == 0)).sum()),
             1: int((gain_mask & (y == 1)).sum())}
    frontier = [f1_from_C(C)]                       # k = 0
    while avail[0] or avail[1]:
        best_c, best_f1 = None, -1.0
        for c in (0, 1):
            if not avail[c]:
                continue
            C[c, c] += 1; C[c, 1 - c] -= 1          # correct one class-c gain
            f = f1_from_C(C)
            C[c, c] -= 1; C[c, 1 - c] += 1          # revert
            if f > best_f1:
                best_f1, best_c = f, c
        C[best_c, best_c] += 1; C[best_c, 1 - best_c] -= 1
        avail[best_c] -= 1
        frontier.append(best_f1)                     # frontier[k] after k corrections

    # map each budget fraction to the best achievable F1 with <= k routes
    f1_oracle = []
    for f in fr:
        k = min(int(round(f * n)), n_gain)
        f1_oracle.append(frontier[k])

    out = {
        "name": args.name, "n": n,
        "fractions": fr.tolist(),
        "all_small_f1": f1_small, "all_large_f1": f1_large,
        "random_f1": f1_random,
        "conf_threshold_f1": f1_conf,
        "oracle_frontier_f1": f1_oracle,
    }
    odir = Path(args.out); odir.mkdir(parents=True, exist_ok=True)
    (odir / "routing_pareto.json").write_text(json.dumps(out, indent=2))

    # readout: best F1 reachable at a few budgets (nearest grid point)
    print(f"[{args.name}] n={n}  all-SLM={f1_small:.2f}  all-LLM={f1_large:.2f}  "
          f"(#gain={n_gain})")
    for f in (0.1, 0.2, 0.3, 0.5):
        i = int(np.argmin(np.abs(fr - f)))
        line = f"  budget {int(f*100):>2}% -> oracle {f1_oracle[i]:.2f}"
        if f1_conf is not None:
            line += f" | conf {f1_conf[i]:.2f}"
        line += f" | random {f1_random[i]:.2f}"
        print(line)

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4.2))
        plt.plot(fr, f1_oracle, "-", lw=2, label="oracle frontier (upper bound)")
        if f1_conf is not None:
            plt.plot(fr, f1_conf, "-o", ms=3, label="confidence threshold")
        plt.plot(fr, f1_random, "--", label="random routing")
        plt.scatter([0, 1], [f1_small, f1_large], zorder=5,
                    label="all-SLM / all-LLM")
        plt.xlabel("fraction routed to LLM  (cost proxy)")
        plt.ylabel("macro-F1")
        plt.title(f"Routing cost-quality Pareto — {args.name}")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(odir / "routing_pareto.png", dpi=150)
        print(f"  wrote {odir/'routing_pareto.png'}")
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
