# KLGA 双桶策略研究报告

- 状态：研究完成，暂不推广为核心生产策略
- 站点：KLGA
- 研究范围：2025-04-01 至 2026-03-31 historical OOF；2026-04 和 2026-05 冻结诊断
- 报告日期：2026-07-19
- 主要比较基准：normalized synthetic midpoint
- 生产影响：无；本研究未修改生产概率源、交易策略或风险配置

## 1. 执行摘要

双桶思路本身是成立的：由模型选择“众数桶 + 模型概率更高的相邻桶”，历史上既会选左邻桶，也会选右邻桶，并且联合区间包含最终结算桶的比例为 `57.36%`。这证明模型分布可以用于定向扩宽区间，而不是随机拼桶。

但现有证据不足以认定它具有稳定、可部署且有经济意义的 edge：

1. 无条件 taker 亏损 `$90.96`，说明“命中率超过 50%”不等于扣除价格和费用后盈利。
2. 严格 earlier-fold 校准与 LCB gate 后，历史 OOF 为 `+$165.87`、ROI `11.90%`，但 Event-day 90% CI 为 `[-$0.694, +$1.931]`，跨 fold 也有一个显著亏损阶段，因此只能判为 **inconclusive**。
3. April 冻结诊断很强，May 则失效。April 为 `+$136.80`，May 为 `-$7.84`；同一 LCB 在 May 高估命中率约 16 个百分点。该差异说明当前概率校准存在 regime 风险。
4. 历史 `30-42h` 表现最好，但 May 同一时段为负，因此它只能作为 shadow 候选区间，不能作为已验证的硬 gate。
5. normalized midpoint 的概率质量整体优于 MC。加入市场概率后 proper score 明显优于 MC-only，但模型在已知市场后只提供很小且统计不确定的增量信息；静态概率融合还会改变入场时点，April PnL 因此明显变差。
6. maker 没有解决容量问题：完整双腿成交仅 7 次，而 orphan/partial 共 355 次，存在严重腿风险。
7. 单次等份 taker 的实际投入远低于 `$50/Event` 上限。历史平均仅 `$6.48/Trade`，历史最佳 `30-42h` 区间平均 `$4.41/Trade`。即便统计 edge 最终成立，单 KLGA 的绝对收益容量也很薄。

**当前决策：** 保留为低成本 shadow/机会型候选，不作为值得单独建设的核心生产策略。下一步优先寻找更高频或更高容量的信号；只有真实前瞻 CLOB shadow 同时证明稳定 edge 和足够年化净利润容量时，才重新考虑推广。

## 2. 原始问题与结论台账

| 原始问题 | 状态 | 结论与主要证据 |
|---|---|---|
| 相邻桶是否由模型定向选择，而非固定选左/右或随机拼接？ | **supported** | 333 个 Event day 中选择左邻 185 次、右邻 148 次。 |
| 模型选择的双桶是否更可能包含最终结算桶？ | **supported** | 历史联合命中率 `57.36%`，高于 50%；但模型平均预测为 `69.21%`，存在过度自信。 |
| 无条件双桶 taker 是否盈利？ | **inconclusive** | 点估计亏损 `$90.96`、ROI `-5.68%`，Event-day CI 跨零。没有正 edge 证据。 |
| earlier-fold 校准 + LCB gate 能否形成可部署历史 edge？ | **inconclusive** | 历史 `+$165.87`、ROI `11.90%`，但 CI 跨零；5 个 fold 中 4 正 1 负。 |
| April 的冻结 edge 是否成立？ | **supported** | 29 个 eligible Event，24 次交易，PnL `+$136.80`，Event-day CI `[+$2.460,+$7.071]`。 |
| May 的冻结 edge 是否成立？ | **inconclusive** | 28 个 eligible Event，25 次交易，PnL `-$7.84`，CI `[-$2.164,+$1.695]`。 |
| 历史 `30-42h` 优势是否跨 April/May 持续？ | **inconclusive** | April `+$72.54`，May `-$18.38`；May 与历史方向相反。 |
| `42-48h` 是否应稳定禁用？ | **inconclusive** | 历史 `-$65.83`，April `+$44.23`，May `-$21.28`，方向不稳定。 |
| LCB90 是否在入场事件上保持保守？ | **inconclusive** | April 实际命中率比平均 LCB 高 `24.70pp`；May 低 `16.03pp`。 |
| `gap < 0.15` 是否是已经验证的入场规则？ | **not measured** | gap 只做描述性分层，未参与拟合或 gate；小 gap 在 April 很好，但 May 仅 3 笔，不能冻结为规则。 |
| MC 在市场概率已知后是否仍提供稳定增量信息？ | **inconclusive** | 历史 model+market 相对 market-only 的 Brier 改善仅 `0.00110`，CI 跨零；log loss 改善仅 `0.00027`，CI 跨零。 |
| 静态 model+market 融合能否改善交易 PnL？ | **inconclusive** | April 相对 MC-only 少赚 `$113.97`，该月为 contradicted；May 多赚 `$12.87` 但 CI 跨零，跨月结论不稳定。 |
| maker 是否比 taker 更合适？ | **inconclusive** | maker-taker 配对均值 `+$0.997/Event`，CI `[-$1.881,+$4.123]`；完整双腿成交过少。 |
| 多次交易能否把每个 Event 的实际投入扩到 `$50`？ | **not measured** | 当前只验证 first qualifying、单次、等 shares、full-fill-or-skip；没有真实 L2 多次回放。 |
| 单 KLGA 双桶是否已有足够绝对利润容量，值得建设独立生产策略？ | **inconclusive** | 最佳历史 `30-42h` 仅 87 笔、总投入 `$383.61`；当前证据不支持投入建设，但真实 L2 多次交易容量和业务利润门槛尚未测定。 |

这里的状态遵循项目实验规范：单月点估计、描述性分层或跨零置信区间都不能升级为稳定机制结论。

## 3. 策略定义与计算口径

### 3.1 双桶选择

对每个可交易 snapshot：

1. 找到 MC 日高 bucket PMF 的众数桶。
2. 比较众数桶左、右两个合法相邻桶的模型概率。
3. 选择概率更高的一侧，与众数桶组成连续双桶。
4. 两桶联合概率为两个 bucket 概率之和；结算结果为实际桶落在任一桶内时 `1`，否则 `0`。

因此该实验测量的是“模型定向选择的连续双桶”，不是任意两个最高概率桶，也不是事后选择包含真值的一侧。

### 3.2 概率与 frozen edge gate

历史 frozen 版本用严格 earlier-fold、Event-day 等权 logistic calibration 将 MC union probability 校准为二元命中概率，再通过 Event-day bootstrap 取单侧 90% 下界 `LCB90`。

入场条件为：

```text
conservative_edge = LCB90 - all_in_unit_cost
enter when conservative_edge > edge_threshold
```

主结果的 `edge_threshold=0`；`0.02` 和 `0.04` 只是敏感性分析，不能解释为已独立验证的最优阈值。

### 3.3 Taker 成本、shares 与 PnL

两个 YES leg 必须购买相同 shares。共同 shares 受较薄一腿深度和 `$50` 总风险上限共同约束：

```text
all_in_cost = lower_leg_VWAP_notional
            + upper_leg_VWAP_notional
            + lower_leg_taker_fee
            + upper_leg_taker_fee

all_in_unit_cost = all_in_cost / common_shares

realized_pnl = common_shares * union_outcome - all_in_cost
```

若任一腿无法完整成交，则整篮子跳过。主回放每个 Event 只取第一笔满足 edge 的 5 分钟 tick。

总 PnL 使用每笔交易的实际 shares 结算，并非 Event 等额下注；均值与 bootstrap CI 则按 Event day 等权，未交易 Event 以零 PnL 进入整体部署评估。

### 3.4 市场价格口径

主 comparator 是 **normalized synthetic midpoint**。执行成本来自 5 分钟 synthetic quote book，并包含配置中的 taker fee。它适合统一比较概率与策略机制，但不等同于真实历史 L2 可执行回放。

因此本报告不能证明：

- 真实盘口上一定能取得同样价格和深度；
- 多次交易可以无冲击地复用同一流动性；
- maker 两腿会同步成交；
- 扣除真实竞争、延迟和额外滑点后仍有同等 PnL。

## 4. Historical OOF 结果

### 4.1 选择质量

| 指标 | 数值 |
|---|---:|
| Event days | 333 |
| 选择左邻桶 | 185 |
| 选择右邻桶 | 148 |
| 平均模型 union probability | 69.21% |
| 平均 normalized midpoint union probability | 51.60% |
| 实际 union hit rate | 57.36% |

模型确实找到了具有方向性的相邻区间，但平均概率比实际命中率高约 `12.85pp`。所以模型适合提供 pair selection 信息，不足以直接把原始 union probability 当作下注概率。

### 4.2 无条件与 frozen taker

| 规则 | Event days | Trades | Turnover | PnL | ROI | Event-day mean 90% CI |
|---|---:|---:|---:|---:|---:|---:|
| 无条件首 tick | 333 | 333 | $1,600.60 | -$90.96 | -5.68% | [-$1.095, +$0.590] |
| earlier-fold calibration + LCB90 | 280 | 215 | $1,393.48 | +$165.87 | 11.90% | [-$0.694, +$1.931] |

Frozen gate 显著改善了点估计，但不确定性仍覆盖亏损；这不是“已确认盈利”，而是值得继续 shadow 的弱证据。

### 4.3 跨 fold 稳定性

| Fold | Trades | PnL | ROI |
|---|---:|---:|---:|
| fold-01 | 38 | -$96.02 | -47.50% |
| fold-02 | 28 | +$85.97 | 53.29% |
| fold-03 | 50 | +$66.01 | 22.98% |
| fold-04 | 49 | +$49.78 | 19.34% |
| fold-05 | 50 | +$60.14 | 12.39% |

4/5 folds 点估计为正，但 fold-01 的损失足以说明策略对 regime 敏感。不能只按正 fold 数量忽略损失幅度。

### 4.4 Horizon 分层

| Lead band | Trades | Turnover | PnL | ROI | Event-day mean 90% CI |
|---|---:|---:|---:|---:|---:|
| 6-12h | 6 | $101.74 | -$85.13 | -83.67% | [-$27.151, -$2.045] |
| 12-24h | 34 | $426.45 | -$46.05 | -10.80% | [-$7.063, +$4.252] |
| 24-30h | 28 | $132.66 | +$91.22 | 68.76% | [-$0.693, +$9.402] |
| 30-36h | 51 | $198.71 | +$136.87 | 68.88% | [+$0.376, +$5.974] |
| 36-42h | 36 | $184.90 | +$134.79 | 72.90% | [+$0.003, +$8.343] |
| 42-48h | 60 | $349.01 | -$65.83 | -18.86% | [-$3.449, +$1.105] |

历史中心集中在 `30-42h`，但这属于看过历史后的风险分层。April/May 没有证明其跨 regime 稳定，因此只能作为候选 shadow 范围。

### 4.5 模型与市场分歧分层

| `model_union_p - midpoint_union_p` | Trades | PnL | ROI | Event-day mean 90% CI |
|---|---:|---:|---:|---:|
| `<0.15` | 44 | +$18.03 | 6.83% | [-$1.404, +$2.266] |
| `0.15-0.30` | 70 | +$23.15 | 5.79% | [-$1.557, +$1.988] |
| `0.30-0.45` | 41 | +$307.94 | 123.80% | [+$2.484, +$13.193] |
| `>=0.45` | 60 | -$183.24 | -38.09% | [-$6.931, +$0.907] |

这里的 gap 是有符号差值，不是绝对值：`>=0.45` 表示模型 union probability 比市场高至少 45 个百分点；`<0.15` 同时包含模型低于市场和仅略高于市场的情况。历史上正向中大分歧 `0.30-0.45` 最强，极端正向分歧 `>=0.45` 最差。这支持把“模型远高于市场”作为过度自信风险警报候选，却不支持简单地认为“分歧越大越有利”。由于 May 的 `0.30-0.45` 同样为负，不能把历史最佳 band 直接写成生产 gate。

## 5. April/May 冻结诊断

### 5.1 总体表现

| 月份 | Eligible events | Trades | Turnover | PnL | ROI | Hit rate | Event-day mean 90% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | 29 | 24 | $136.63 | +$136.80 | 100.12% | 79.17% | [+$2.460, +$7.071] |
| 2026-05 | 28 | 25 | $124.31 | -$7.84 | -6.31% | 40.00% | [-$2.164, +$1.695] |

May 的平均入场 LCB 为 `56.03%`，实际命中率仅 `40%`；April 的平均 LCB 为 `54.47%`，实际命中率为 `79.17%`。这不是流动性单独能解释的现象，而是 union probability 校准在不同 regime 下方向相反。

April 和 May 使用不同冻结 forecast lineage，而且两个月此前均已被检查，因此这里只能做已知诊断，不能称为独立 holdout。

### 5.2 Horizon 持续性

| Lead band | April PnL | May PnL | 结论 |
|---|---:|---:|---|
| 30-42h | +$72.54 | -$18.38 | 历史优势未在 May 持续 |
| 42-48h | +$44.23 | -$21.28 | 历史风险在 April 反向，月份间不稳定 |

### 5.3 Gap 持续性

April 的 `<0.15` 为 15 笔、`+$118.96`、命中率 `93.3%`；May 的 `<0.15` 虽为 `+$15.59` 且全中，但只有 3 笔。May 的 `0.15-0.30`、`0.30-0.45` 和 `>=0.45` 均为负。

因此：

- `gap < 0.15` 没有足够 May 样本，不能作为已验证必要条件；
- `0.30-0.45` 的历史优势不是稳定机制；
- `>=0.45` 在 April/May 都是 0 命中，但各只有 2 笔，可作为保守 veto 候选，仍需前瞻验证。

## 6. 市场概率栈实验

### 6.1 三个概率源

- `current_model_only`：intercept + MC union probability logit。
- `market_only`：intercept + normalized midpoint logit + `36-42h` indicator。
- `model_plus_market`：同时使用 MC logit、midpoint logit 和 `36-42h` indicator。

冻结至 2026-03-31 的 model+market 点估计为：

```text
logit(p_union) = 0.205608
               + 0.162769 * logit(MC)
               + 0.729610 * logit(midpoint)
               - 0.151998 * I(36-42h)
```

市场权重显著高于 MC 权重。`gap < 0.15` 没有进入特征或 gate。

### 6.2 Historical OOF proper scores

| Probability source | Event days | Brier | Binary log loss |
|---|---:|---:|---:|
| current_model_only | 280 | 0.22828 | 0.64846 |
| market_only | 280 | 0.18515 | 0.55211 |
| model_plus_market | 280 | 0.18405 | 0.55183 |

model+market 相对 market-only：

- Brier delta `-0.00110`，90% CI `[-0.00488,+0.00247]`；
- log-loss delta `-0.00027`，90% CI `[-0.01303,+0.01366]`。

所以 midpoint 已包含绝大多数可预测信息；MC 条件增量为正向点估计，但很小且不确定。历史 5 个 fold 的 MC 系数均为正，说明它不是完全无信息，只是尚未证明具有稳定可交易增量。

### 6.3 Historical execution

| Probability source | Trades | Turnover | PnL | ROI | Event-day mean 90% CI |
|---|---:|---:|---:|---:|---:|
| current_model_only | 215 | $1,393.48 | +$165.87 | 11.90% | [-$0.694,+$1.931] |
| market_only | 255 | $1,274.34 | +$151.08 | 11.86% | [-$0.532,+$1.647] |
| model_plus_market | 252 | $1,373.38 | +$255.12 | 18.58% | [-$0.231,+$2.085] |

model+market 相对 current 的平均差为 `+$0.319/Event`，90% CI `[-$0.289,+$0.989]`。点估计更好，但不能认定部署改善。

### 6.4 Frozen April/May execution

| 月份 | current MC | market-only | model+market |
|---|---:|---:|---:|
| April PnL | +$136.80 | +$60.70 | +$22.83 |
| May PnL | -$7.84 | -$23.54 | +$5.03 |

April 中 model+market 相对 current 少赚 `$113.97`，并非因为同一笔交易的 payoff 公式发生变化，而是概率源改变了交易集合和入场时点：23 个共同交易事件中有 14 个 entry timestamp 改变，该部分贡献 `-$81.19`；同一 timestamp 的 PnL delta 恰为 0。

这说明静态市场融合虽能改善平均概率质量，却可能破坏原策略的入场时机。市场信息更适合作为 benchmark、风险状态或动态 entry-control 输入，而不是不经时序验证直接覆盖 MC 概率。

## 7. Maker、流动性与容量

### 7.1 Maker 结果

| Fill status | Count |
|---|---:|
| full | 7 |
| none | 667 |
| orphan | 295 |
| partial | 60 |

Maker 相对 taker 的配对均值为 `+$0.997/Event`，90% CI `[-$1.881,+$4.123]`。完整双腿成交极少，而 orphan/partial 风险很高。因此 maker 当前应关闭，除非以后实现原子化/腿风险控制并用真实订单簿前瞻验证。

### 7.2 单次 taker 容量

历史 frozen MC-only：

- 215 笔交易，总投入 `$1,393.48`，平均 `$6.48/Trade`；
- `30-36h`：51 笔，投入 `$198.71`，平均 `$3.90/Trade`；
- `36-42h`：36 笔，投入 `$184.90`，平均 `$5.14/Trade`；
- `30-42h` 合计：87 笔，投入 `$383.61`，平均 `$4.41/Trade`，实际 PnL `+$271.67`。

在保持这些已观测单次 fills 且假设 87 笔全部命中的不现实条件下，`30-42h` 的 perfect-hit PnL 上界约为 `$855.90`。这只是固定成交记录下的机械容量上界，不是期望收益，也没有计入真实竞争、滑点或盘口复用。

April 平均投入约 `$5.69/Trade`，May 约 `$4.97/Trade`。较薄一腿通常先耗尽，因此 `$50` 是风险上限，不是可实现的目标仓位。

### 7.3 经济性判断

策略价值应按下面的乘积判断，而不能只看 ROI：

```text
annual_net_value = robust_edge * executable_capacity * opportunity_frequency
                   - engineering_and_operational_cost
```

当前统计 edge 尚未稳定，单 KLGA 的容量和频率又低。即使最终证明有正 edge，其绝对收益也很可能不足以支撑独立的生产开发、监控与腿风险系统。若该逻辑可以低成本复用现有执行设施，或扩展到多个独立站点/Event，它仍可能作为机会型 overlay 有价值。

## 8. 当前可保留的 Shadow 配置

以下是研究候选，不是已批准生产参数：

- Pair：MC 众数桶 + 模型概率更高的合法相邻桶。
- Probability：checkpoint-aligned current MC，经严格 earlier Event-day calibration，并使用单侧 `LCB90`。
- Execution：taker、equal shares、两腿 full-fill-or-skip、包含费用、每 Event 最多一次入场、风险上限 `$50`。
- Coverage：forecast-only 为主；partial-day 仅 shadow，因为冻结 April/May 几乎没有足够 partial 交易证据。
- Lead：优先观察 `30-42h`，但不得把它描述为跨 regime 已验证规则。
- Gap：不强制 `<0.15`；`>=0.45` 可作为保守 veto 候选，但必须前瞻确认。
- Market：用于 normalized midpoint benchmark、gap/风险状态和监控；暂不使用静态 model+market 概率覆盖 MC。
- Maker：关闭。

特别注意：`lead 30-42h + gap veto + 某个 edge_threshold` 的最终组合尚未做独立、端到端 frozen replay。不能把从多个分层表挑出的条件拼接后，宣称为已经验证的策略。

## 9. 何时值得重启

在继续优化 MC 或双桶概率前，先完成真实前瞻 shadow。至少同时满足以下条件才考虑生产化：

1. 使用此前未消费结果的前瞻月份，不回调该月阈值。
2. 使用真实 timestamp 对齐的 L2/可执行盘口，不只用 synthetic midpoint/quote。
3. 报告 Event-day 等权 edge、实际 shares PnL、费用、滑点、拒单和两腿失败。
4. 单独报告 `30-42h`、其他 horizon、forecast-only 与 partial-day，防止聚合掩盖 regime 失败。
5. 验证 `>=0.45` veto、预测稳定性和市场分歧风险，但阈值只能由更早数据冻结。
6. 做多次交易容量回放时，必须逐时推进订单簿并消耗已用深度，不得重复使用同一 synthetic 流动性。
7. 在实验前写明最低可接受年化净 PnL；若容量上界低于该门槛，即使统计显著也停止开发。

## 10. 支撑产物与代码

### 10.1 冻结产物

| 产物 | 用途 | `report.json` SHA256 |
|---|---|---|
| [`model_selected_v1`](../../data_store/backtest/two_bucket_union/KLGA/model_selected_v1/report.json) | Historical A2 选择、frozen taker/maker、fold 与粗 horizon | `7eabb72ced8f3ea142c9c892069d0f4619cbd06bf0004895c220654d6cb4375f` |
| [`frozen_april_may_v3`](../../data_store/backtest/two_bucket_union/KLGA/frozen_april_may_v3/report.json) | April/May 冻结诊断、细 horizon、gap、LCB calibration | `dd11db24558066e0ca4932c9ecd3eb6842b8d132e3e0e43386ed77349bc85804` |
| [`historical_oof_v2`](../../data_store/backtest/two_bucket_probability_stack/KLGA/historical_oof_v2/report.json) | 三概率源 historical OOF proper score 与 execution | `f18a4bcacf3eabf47a0bf33890932bfcc317240523c04ec993ac32845e68c161` |
| [`frozen_april_may_v2`](../../data_store/backtest/two_bucket_probability_stack/KLGA/frozen_april_may_v2/report.json) | 三概率源 April/May frozen proper score、PnL 与 entry transition | `4aca0d13d33923dbdf1273ff430566187e2af4b9de488448e423086de4d7aed2` |

每个目录的 `manifest.json` 保存输入 lineage 和 parquet hashes。若报告被重跑或替换，应先比对 manifest，不应只按目录名复用本文结论。

### 10.2 主要实现

- [`backtest/two_bucket_union_study.py`](../../backtest/two_bucket_union_study.py)：pair selection、Event-equal calibration、LCB、taker/maker replay 与报告。
- [`backtest/two_bucket_union_frozen_diagnostic.py`](../../backtest/two_bucket_union_frozen_diagnostic.py)：April/May frozen diagnostic。
- [`backtest/two_bucket_probability_stack.py`](../../backtest/two_bucket_probability_stack.py)：historical expanding probability stack。
- [`backtest/two_bucket_probability_stack_frozen.py`](../../backtest/two_bucket_probability_stack_frozen.py)：frozen April/May probability stack。
- [`core/pricing/quote.py`](../../core/pricing/quote.py)：执行报价与 shares-based quote 计算。

## 11. 最终结论

双桶不是一个无效想法：模型能有方向地选择相邻桶，历史 frozen 回放和 April 都显示过可观的点估计收益。问题在于该收益没有跨 regime 稳定，市场概率已吸收多数信息，静态融合会扰动入场时机，而单站点真实可执行容量又很薄。

所以最准确的表述是：

> KLGA 双桶存在值得保留的候选信号，但尚无稳定、独立且具有足够经济容量的生产 edge 证据。

在真实前瞻容量验证之前，不应继续围绕历史最佳 gap、horizon 或 edge threshold 做适应性调参，也不应为它单独建设复杂生产链路。
