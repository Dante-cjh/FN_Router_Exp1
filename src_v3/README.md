# src_v3 — Round-3 主实验：RA³（Rationale-Aware Arbitrated Routing with Risk Control）

> 设计依据：`reports/08_error_analysis_and_pivot.md`（错误样本分析 + 仲裁 pilot 87.27/77.38）。
> 隔离约定：只读 `outputs/preds/*`（v1 产物）与 `data/*`；只写 `outputs_v3/`。
> 全部命令在 `Router_Exp1/` 根目录运行；一键 `bash scripts_v3/run_all.sh`。

---

## 1. 方法（三个决策，三个模块，各有出处）

```
            ┌────────── Stage-1 预路由（pre-hoc，只用 SLM 特征）──────────┐
 x ──SLM──► │ u(x) = w[1-ŷ_s]·p_gain(x) − w[ŷ_s]·p_harm(x)   （净效用双头）│
            └──── u≤0 或超预算：输出 SLM 标签 ────┬──── u>0：调用 LLM ────┘
                                                  │
                          ┌───────────────────────▼───────────────────────┐
                          │ Stage-2 仲裁（post-hoc，零额外 LLM 成本）      │
                          │ s(x)=P(LLM对 | SLM置信, 分歧方向, rationale, 文本)│
                          │ s(x) > τ ？ 采纳 LLM 标签 ：保留 SLM 标签      │
                          └───────────────────────────────────────────────┘
                             τ 的两种取法：val macro-F1 调优（arb_f1）
                                          保形风险控制 CRC@α（arb_crc，带保证）
```

| 决策 | 模块 | 借鉴出处 | 对应我们的失败证据 |
|---|---|---|---|
| 花不花钱 | 净效用双头 + 预算上界 | v2 资产（对位 Hybrid LLM ICLR'24 / RouteLLM 的前置路由） | v1"单分数吞标签"结构性亏损（reports/03） |
| 信不信它 | **rationale 仲裁器** | **ARG (AAAI'24)** 的 rationale-usefulness 思想 + **SBERT (EMNLP'19)** 交互特征 `[u,v,|u−v|,u⊙v,cos]` | LLM 幻觉式佐证可从 rationale 读出（reports/08 §2，70% 漏检带 memory_match 论证） |
| 多保守 | **Conformal Risk Control** | **Angelopoulos et al. (ICLR'24)**，把误采纳率控制在 α 以下（分布无关、有限样本） | v2 的 no-harm 是经验现象；CRC 把它变成可证保证，替代 bootstrap gate |
| 消融对照 | 无 rationale 的 post-hoc scorer | **FrugalGPT (arXiv:2305.05176)** 级联打分器 | 量化"读 thinking"的净增量 |

关键成本论证：**仲裁不增加任何 LLM 调用**——它只对"已付费"的路由样本做一次 CPU 级打分。同预算下 arb ≥ swallow 是免费的 Pareto 改进（bootstrap CI 验证）。

## 2. 文件

| 文件 | 作用 |
|---|---|
| `common.py` | 数据装载（preds+rationale+emb+enc 对齐）、rationale 论证词典、macro-F1、bootstrap、类权重 |
| `crc.py` | CRC 阈值选择 + α 扫描（val 校准、test 兑现风险） |
| `encode_rationales.py` | 冻结多语言 encoder 编码 (news, rationale) → `outputs_v3/enc/`（GPU 机器跑，一次性） |
| `arbiter.py` | 仲裁器：5 档特征阶梯（frugal→dict→emb→enc→full）、两种训练范围、zh↔en 迁移、CRC 扫描、bootstrap CI |
| `pipeline.py` | 端到端：stage-1（netutil/entropy）× stage-2（swallow/arb_f1/arb_crc）× 预算格点 → Pareto + 图 |

特征阶梯即消融表：`frugal`（≈FrugalGPT，无 rationale 内容）→ `dict`（+4 维论证词典）→ `emb`（+SLM emb，= pilot 87.27/77.38 配置）→ `enc`（+冻结编码交互，可跨语言迁移）→ `full`。

## 3. 跑法（顺序）

```bash
# ① 无 GPU 即可：三档仲裁 + 主 pipeline（emb 档）
bash scripts_v3/run_arbiter_zh.sh && bash scripts_v3/run_arbiter_en.sh
bash scripts_v3/run_pipeline_zh.sh && bash scripts_v3/run_pipeline_en.sh

# ② GPU/联网机器（一次性，~分钟级）：编码 rationale
pip install sentence-transformers
bash scripts_v3/run_encode.sh          # 可 MODEL=BAAI/bge-m3

# ③ enc/full 档 + 零样本迁移
ENC=outputs_v3/enc bash scripts_v3/run_arbiter_zh.sh
ENC=outputs_v3/enc bash scripts_v3/run_arbiter_en.sh
FEAT=full ENC=outputs_v3/enc bash scripts_v3/run_pipeline_zh.sh
FEAT=full ENC=outputs_v3/enc bash scripts_v3/run_pipeline_en.sh
bash scripts_v3/run_transfer.sh
```

## 4. 判读（go/kill，对应 reports/08 §6）

1. **主张 (1) 仲裁是免费改进**：`pipeline.json::boot_arbf1_minus_swallow` 各预算 CI 下界 > 0（GossipCop 必须，Weibo21 期望）。
2. **主张 (2) GossipCop 翻正**：`curves.arb_*` 单调不降且峰值 > 77（swallow 对照单调降）。
3. **主张 (3) 有证书的 no-harm**：`floor_check.arb_crc.no_harm == true` 且 `crc_sweep` 中 test 兑现风险 ≤ α（Weibo21 时间切分是对交换性假设的压力测试，兑现偏差本身写进论文）。
4. **rationale 增量**：arbiter 特征阶梯上 dict/enc 相对 frugal 的 AUC 与 F1 增量 CI > 0（GossipCop 是主战场）。
5. **迁移**：`run_transfer.sh` 的 AUC 显著 > 0.5 ⇒ "幻觉式佐证是 LLM 性质"成立，直接支撑泛化叙事（02 §4 网格的先导）。
6. **kill 条件**：GossipCop 上 arb 相对 swallow 的增益 CI 含 0 → 退回 v2 叙事（no-harm + 负结果），仲裁降级为 analysis 小节。

## 5. 与论文的对应

主图 = 两数据集 `pipeline/pareto.png`（三策略曲线 + 端点）；主表 = 特征阶梯 × {AUC, 全量仲裁 F1, Δvs floor CI}；安全表 = CRC α 扫描（α vs 兑现风险 vs F1）；迁移表 = zh↔en AUC/F1。基线复用 `baselines/`（07 报告口径），RouteLLM-bert 是要压过的那条线（86.78 / 77.09）。

## 5b. 烟测结果（2026-07-06，emb 档、无 enc、n_boot=200，正式跑请用默认 1000）

仲裁器特征阶梯（test 分歧 AUC / 全量仲裁 F1）：

| tier | GossipCop | Weibo21 |
|---|---|---|
| frugal（≈FrugalGPT） | 0.665 / 76.31 | 0.553 / 82.02 |
| dict（+论证词典） | 0.703 / 75.93 | 0.541 / 80.43 |
| emb（pilot 配置） | **0.727 / 77.09** | **0.803 / 87.81** |

端到端（stage1=netutil）：`arb_relax` 两边登顶且全程 no-harm——GossipCop 77.09@100%（swallow 76.79）、Weibo21 87.81@100%（swallow 87.01）；对照 `swallow_relax`（大胆路由但无仲裁）在 GossipCop 崩到 70.51、Weibo21 100% 退回 82.97。**即：仲裁把"大胆预路由"从不安全变成占优——这就是主图故事。** GossipCop 同预算 arb−swallow 的 CI 尚含 0（小分歧样本 121 条），这正是 M2（enc 档 + 多 seed + bge-m3）要解决的。

## 6. 已知限制（写论文前要处理）

- GossipCop 分歧样本少（val 187 / test 121）：所有 GossipCop 仲裁结论必须带 bootstrap CI（脚本已内置）+ 多 seed。
- `enc` 档用冻结编码器 + 线性头，是 ARG 交互模块的轻量替身；若增益明显可升级为可训练 cross-encoder（数据量小，慎防过拟合）。
- CRC 的交换性假设在 Weibo21 时间切分下不严格成立——把"兑现风险 vs α"作为实验结果报告，不当作理所当然。
- Stage-1 的 u(x) 权重用 w[1−ŷ_s] 保证严格 pre-hoc（v2 曾用 ŷ_llm，部署时不可得），数字与 v2 可能有微小差异。
