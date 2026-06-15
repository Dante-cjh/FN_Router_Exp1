#!/usr/bin/env python
"""
src_v2/compare_calibration.py — Round-2 calibration experiment, comparison view.

Reads several uncertainty_bands.json files (one per calibration config) and
tabulates the two numbers that decide whether the reports/05 headline holds up:

  - reducible fraction  U_epi / (U_epi + U_ale)  per band  (esp. only_llm_GAIN)
  - check-(a) AUC gain-vs-bothwrong for U_epi (does the ranking signal survive?)

So you can see, side by side, whether amplifying inference dropout / applying
temperature changes the "epistemic ~1% on GossipCop vs ~50% on Weibo21" story.

USAGE:
  python -m src_v2.compare_calibration \
      --cfg base=outputs_v2/diagnostic/gossipcop/uncertainty_bands.json \
      --cfg p0.3=outputs_v2/diagnostic/gossipcop__p0.3/uncertainty_bands.json \
      --cfg temp=outputs_v2/diagnostic/gossipcop__temp/uncertainty_bands.json \
      --cfg p0.3_temp=outputs_v2/diagnostic/gossipcop__p0.3_temp/uncertainty_bands.json
"""
from __future__ import annotations

import argparse
import json


def red_frac(d, band):
    rf = d.get("reducible_fraction_by_band")
    if rf is not None:
        return rf.get(band)
    epi = d["U_epi_by_band"][band]["mean"]
    ale = d["check_b_U_ale_by_band"][band]["mean"]
    return epi / (epi + ale + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", action="append", required=True,
                    help="LABEL=path/to/uncertainty_bands.json (repeatable)")
    args = ap.parse_args()

    cfgs = []
    for c in args.cfg:
        label, path = c.split("=", 1)
        cfgs.append((label, json.load(open(path))))

    name = cfgs[0][1].get("name", "?")
    bands = ["both_correct", "only_slm_HARM", "only_llm_GAIN", "both_wrong"]
    print(f"\n===== calibration comparison — {name} =====")
    hdr = f"{'config':<14}" + "".join(f"{b.split('_')[-1][:6]:>8}" for b in bands)
    hdr += f"{'AUC_a_epi':>11}{'AUC_a_tot':>11}{'U_ale_gain':>11}"
    print("  reducible fraction (U_epi share) by band:")
    print("  " + hdr)
    for label, d in cfgs:
        row = f"{label:<14}" + "".join(f"{red_frac(d, b):>8.3f}" for b in bands)
        aa = d["check_a_auc_gain_vs_bothwrong"]
        row += f"{aa['U_epi']:>11.3f}{aa['U_tot']:>11.3f}"
        row += f"{d['check_b_U_ale_by_band']['only_llm_GAIN']['mean']:>11.4f}"
        print("  " + row)
    print("\n  read: if reducible-frac(GAIN) stays tiny on GossipCop and ~0.5 on "
          "Weibo21 ACROSS configs, the headline is robust to calibration.\n")


if __name__ == "__main__":
    main()
