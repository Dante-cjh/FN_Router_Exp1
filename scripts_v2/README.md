# Round-2 实验（src_v2 / scripts_v2 / outputs_v2）

第二轮验证，与第一轮**完全隔离**：只**读取** v1 的检查点与预测（`outputs/ckpt/*`、`outputs/preds/*`、`data/*`），所有新产物写到 `outputs_v2/`。代码里复用 v1 的稳定基础设施（tokenizer / dataset / 数据读取），新实验逻辑全部在 `src_v2/`。

对应方案：`reports/04_uncertainty_routing_plan.md` §3 的零 API 成本去风险小试验——基于 MC-Dropout 的认知/偶然不确定性解耦。

## 跑法（在 GPU 服务器，与 v1 step3 同环境，repo 根目录 = Router_Exp1/）

```bash
bash scripts_v2/run_uncertainty_zh.sh     # Weibo21
bash scripts_v2/run_uncertainty_en.sh     # GossipCop
# 想多采样几次： T=50 bash scripts_v2/run_uncertainty_zh.sh
```

## 两步

1. `src_v2/mc_dropout_uncertainty.py` — 载入 `best.pt`，推理时保持 dropout 开启，做 T 次随机前向，按 BALD/互信息分解算每条样本的 `U_tot / U_ale / U_epi`。输出 `outputs_v2/uncertainty/<ds>_<split>.json(.npz)`。
2. `src_v2/uncertainty_diagnostic.py` — 与 v1 的 SLM/LLM 预测内连，切成 4 个误差带，跑三个 go/kill 判据，画箱线图 + Pareto。输出 `outputs_v2/diagnostic/<ds>/uncertainty_bands.json` + 两张图。

## 三个 go/kill 判据（见 `uncertainty_bands.json` 的 `flags` 与 `verdict`）

- **(a) 解耦有效性**：`U_epi` 区分 gain 带 vs both-wrong 带的 AUC 是否 > `U_tot`（旧 conf 信号）。
- **(b) 偶然不确定性挡门**：both-wrong 带的 `U_ale` 是否高于 gain 带。
- **(c) 兑现**：`U_epi` 门控路由（`epi>ale` 无参数门）在低预算（5/10/20%）是否仍 ≥ all-SLM。

> 预判：Weibo21 大概率三项通过；GossipCop 的 (c) 可能仍过不了（LLM 是弱腿）——若如此，这本身是论文要的正面结论（可约性信号更优，但仍需 regime gate 才能转盈利），不是失败。
