# V2 Slice Prompts

Use these prompts one at a time. Start a fresh request for each slice so the agent
keeps scope tight and does not pull future-slice behavior forward.

For every implementation prompt, the agent should use `.github/skills/tdd/SKILL.md`.
For every review prompt, the agent should use
`.github/skills/improve-codebase-architecture/SKILL.md`.

Before using any implementation prompt:
- Confirm the slice's `Blocked by` entries in `docs/issues/proposed-slices-v2.md`
  are complete.
- If a blocker is incomplete, run that blocker prompt first.
- If a test failure is unclear after two reasonable attempts, switch to the
  Diagnose Prompt at the end of this file.

---

## Batch 1 -- Contract Foundations

### V2F01 -- Schema Expansion

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2F01 Schema Expansion。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2F01
- V2F01 的 Source refs
- `docs/grill-tasks-v2.md`

严格只实现 V2F01：
- 不实现 V2F05 的 Position/PnL 计算
- 不实现 V2V08 的 TradingLoop signals 写入
- 不实现 V2V12 的 redeem processing
- 不实现 V2V01 的 discovery 行为

按 TDD 执行：先写失败测试，再最小实现。完成后运行 V2F01 相关测试和相邻 schema/db 测试，并总结结果。
```

### V2F03 -- L2 Quote Primitive

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2F03 L2 Quote Primitive。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2F03
- V2F03 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0011

严格只实现 V2F03：
- 只新增/修改 L2 quote primitive 和 `quote_size_usd`
- 不改 EdgeStrategy
- 不改 TradingLoop
- 不改 exchange/paper fill 行为
- 不改 backtest

按 TDD 执行：先写失败测试，再最小实现。完成后运行 quote/pricing/settings 相关测试，并总结结果。
```

### V2F04 -- ProbabilitySnapshot + Feature Contract

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2F04 ProbabilitySnapshot + Feature Contract。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2F04
- V2F04 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0004 以及 V2 相关 feature-contract 决策

严格只实现 V2F04：
- 修改 ProbabilityStore 数据结构到 `ProbabilitySnapshot`
- 新增/集中 `model/features.py`
- 更新 `model/tft.py` 和 `scripts/train_tft.py` 的 feature list 来源
- 不实现 ModelPipeline 的输入校验
- 不实现 ForecastLoop 的写入逻辑

按 TDD 执行：先写失败测试，再最小实现。完成后运行 ProbabilityStore、feature contract、TFT/train import 相关测试，并总结结果。
```

### Batch 1 Architecture Review

```text
请使用 `.github/skills/improve-codebase-architecture/SKILL.md` review 刚完成的 V2 Batch 1 slices：
- V2F01
- V2F03
- V2F04

请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md`
- 相关 Source refs in `docs/grill-tasks-v2.md`
- ADRs
- 本批次改动的代码和测试

重点检查：
- schema / ProbabilitySnapshot / quote / feature contract 边界是否清晰
- `core/pricing/quote.py` 是否保持 leaf module
- V2F01 是否没有提前实现 V2F05/V2V08/V2V12
- V2F04 是否没有提前实现 V2V05/V2V07

不要修改代码或创建 issue。输出 architecture review notes 和建议的后续调整。
```

---

## Batch 2 -- Client + Accounting Foundations

### V2F02 -- Polymarket SDK Client Split

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2F02 Polymarket SDK Client Split。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2F02
- V2F02 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0009

严格只实现 V2F02：
- 替换旧 Polymarket credential settings
- 按 live/paper 分离 SecureClient/PublicClient wiring
- 更新 `data/clob/orderbook.py`
- 更新 live CLOB adapter 到新 SDK 方法
- 不实现 VWAP fill simulation
- 不实现 TradingLoop arbitration

按 TDD 执行：先写失败测试，再最小实现。完成后运行 settings、bot wiring、orderbook fetcher、exchange adapter 相关测试，并总结结果。
```

### V2F05 -- Position Accounting + PnL/Risk Primitives

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2F05 Position Accounting + PnL/Risk Primitives。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2F05
- V2F05 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0011

严格只实现 V2F05：
- 更新 PositionTracker 的 order accounting 字段
- 新增/更新 PnLCalculator
- 更新 risk scale-in 和 SELL sizing primitives
- 不实现 TradingLoop SELL YES vs BUY NO arbitration
- 不实现 exchange fill simulation
- 不实现 settlement/redeem

按 TDD 执行：先写失败测试，再最小实现。完成后运行 positions、PnL、risk 相关测试，并总结结果。
```

### Batch 2 Architecture Review

```text
请使用 `.github/skills/improve-codebase-architecture/SKILL.md` review 刚完成的 V2 Batch 2 slices：
- V2F02
- V2F05

请阅读 V2 IW、V2 slices、Source refs、ADRs、代码和测试。

重点检查：
- SDK live/paper client split 是否清楚
- paper mode 是否没有加载 private key
- Position/PnL 是否基于 signed_shares 和 cash_flow_usd
- risk 是否允许合法 scale-in
- 是否没有提前实现 V2V08/V2V09/V2V12

不要修改代码或创建 issue。输出 architecture review notes 和建议的后续调整。
```

---

## Batch 3 -- Data + Discovery Readiness

### V2V01 -- Discovery + Lifecycle Hardening

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V01 Discovery + Lifecycle Hardening。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V01
- V2V01 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0009

严格只实现 V2V01：
- 实现 SDK discovery call、station aliases、slug/token/outcome hardening
- 实现 unrecognized_slug 入库 + discovery_warn
- 实现 downstream hard gate
- 实现 local lifecycle advance
- 不实现 ForecastLoop invalid-bucket guard
- 不实现 settlement/redeem

按 TDD 执行：先写失败测试，再最小实现。完成后运行 discovery/lifecycle/db query 相关测试，并总结结果。
```

### V2V02 -- HRRR Source Hardening + Extrapolation Flags

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V02 HRRR Source Hardening + Extrapolation Flags。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V02
- V2V02 的 Source refs
- `docs/grill-tasks-v2.md`

严格只实现 V2V02：
- 实现 HRRR source hardening 和 `is_extrapolated` contract
- 实现 HrrrFill tail extrapolation / duplicate valid_time / floor-to-second / column union
- 实现 GRIB idx/range validation
- 实现 S3/NOMADS/Zarr config base URL 和 Zarr fail-fast
- 不改 ParquetStore
- 不改 ModelPipeline 的 extrapolated-row filter

按 TDD 执行：先写失败测试，再最小实现。完成后运行 HRRR source/HrrrFill 相关测试，并总结结果。
```

### V2V03 -- Obs + Parquet Hardening

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V03 Obs + Parquet Hardening。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V03
- V2V03 的 Source refs
- `docs/grill-tasks-v2.md`

严格只实现 V2V03：
- 修 ParquetStore dedup/read/schema/TODO
- 修 IEM :51 valid_time、skyc1、VRB、U/V fill order、排序
- 修 NOAA tick fallback scope 和 VRB
- 修 DataManager obs completeness
- 不实现 encoder readiness gate
- 不实现 production Bot 的 NOAA wiring

按 TDD 执行：先写失败测试，再最小实现。完成后运行 ParquetStore、IEM、NOAA、DataManager completeness 相关测试，并总结结果。
```

### Batch 3 Architecture Review

```text
请使用 `.github/skills/improve-codebase-architecture/SKILL.md` review 刚完成的 V2 Batch 3 slices：
- V2V01
- V2V02
- V2V03

请阅读 V2 IW、V2 slices、Source refs、ADRs、代码和测试。

重点检查：
- discovery_status gate 是否不会让下游看到 NULL bucket
- lifecycle advance 是否基于本地 non-terminal events
- HRRR `is_extrapolated` contract 是否所有 source 一致
- obs/parquet hardening 是否没有提前实现 readiness gate
- data 层是否没有依赖 model 层

不要修改代码或创建 issue。输出 architecture review notes 和建议的后续调整。
```

---

## Batch 4 -- Inference Contract

### V2V04 -- Encoder Readiness Gate

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V04 Encoder Readiness Gate。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V04
- V2V04 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0001

严格只实现 V2V04：
- 分离 cold_start burn-in 和 encoder window
- 实现 DataManager shared readiness gate
- 实现 fixed 72h encoder window defensive builder
- ForecastLoop 和 BacktestEngine 使用 readiness skip
- 不实现 TFT/Pipeline validation
- 不实现 ForecastLoop V2 ProbabilitySnapshot write path

按 TDD 执行：先写失败测试，再最小实现。完成后运行 DataManager readiness/window、ForecastLoop readiness、Backtest readiness 相关测试，并总结结果。
```

### V2V05 -- ModelPipeline Validation + Daily-High Target

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V05 ModelPipeline Validation + Daily-High Target。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V05
- V2V05 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0010

严格只实现 V2V05：
- 实现 TFT 输入/输出校验
- 实现 ModelPipeline source-column、decoder、quantile、sigma 校验
- 拒绝 `tft=None`
- 实现 `is_extrapolated` target row exclusion
- 实现 event local-calendar-day target selection
- 不实现 ForecastLoop event iteration/write behavior
- 不实现 MC covariance loading

按 TDD 执行：先写失败测试，再最小实现。完成后运行 TFT、ModelPipeline、ADR-0010 target-date 相关测试，并总结结果。
```

### V2V06 -- MC Covariance + Production Model Wiring

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V06 MC Covariance + Production Model Wiring。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V06
- V2V06 的 Source refs
- `docs/grill-tasks-v2.md`

严格只实现 V2V06：
- 实现 ModelRegistry covariance loading
- 实现 McSampler set_cov 和输入校验
- 修 Bot 的 McSampler 构造、NOAA fallback wiring、TFT load guard、covariance wiring
- 不实现 ForecastLoop probability writes
- 不实现 V2V09 的 exchange fill semantics / paper fill simulation

按 TDD 执行：先写失败测试，再最小实现。完成后运行 registry、McSampler、Bot construction 相关测试，并总结结果。
```

### Batch 4 Architecture Review

```text
请使用 `.github/skills/improve-codebase-architecture/SKILL.md` review 刚完成的 V2 Batch 4 slices：
- V2V04
- V2V05
- V2V06

请阅读 V2 IW、V2 slices、Source refs、ADRs、代码和测试。

重点检查：
- ForecastLoop 和 BacktestEngine 是否共用 readiness gate
- readiness 是否没有 double-count cold_start
- ModelPipeline 是否只接受 event target_date / station_config 语义
- `tft=None` 是否被明确拒绝
- covariance 是否由 Registry 加载、Bot wiring、McSampler 接收

不要修改代码或创建 issue。输出 architecture review notes 和建议的后续调整。
```

---

## Batch 5 -- Forecast + Trading Path

### V2V07 -- ForecastLoop V2 Write Path

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V07 ForecastLoop V2 Write Path。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V07
- V2V07 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0010

严格只实现 V2V07：
- ForecastLoop 使用 event station_id 和 event_end_date 调用 ModelPipeline
- `day_offset` 只作为 ProbabilityStore key
- filter invalid buckets
- empty bucket/probs 时不写 ProbabilityStore
- `run()` 顶层异常 guard
- 使用 V2 ProbabilitySnapshot shape
- 不实现 DataManager readiness 本身
- 不实现 ModelPipeline internals

按 TDD 执行：先写失败测试，再最小实现。完成后运行 ForecastLoop/ProbabilityStore 相关测试，并总结结果。
```

### V2V08 -- Directional Trading Arbitration

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V08 Directional Trading Arbitration。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V08
- V2V08 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0011

严格只实现 V2V08：
- 新增/使用 DirectionalSignal 和 OrderIntent
- EdgeStrategy 只产出 directional mispricing signals
- TradingLoop 做 SELL YES vs BUY NO arbitration
- TradingLoop 做 L2 quote checks 和 re-quote
- TradingLoop 更新 per-tick local risk state
- TradingLoop 写 signals observability rows
- 不修改 exchange adapter internals
- 不修改 SniperLoop

按 TDD 执行：先写失败测试，再最小实现。完成后运行 strategy、TradingLoop、risk/signal DB 相关测试，并总结结果。
```

### V2V09 -- Exchange Fill Semantics

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V09 Exchange Fill Semantics。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V09
- V2V09 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0009
- ADR-0011

严格只实现 V2V09：
- live exchange 只提交已验证 limit price
- exchange adapter 不 fetch books、不 re-quote、不 evaluate edge
- PaperExchange 用 compute_quote 做 BUY/SELL VWAP fill simulation
- depth insufficient 时 paper reject
- PaperExchange 不持久化 orders/positions
- TradingLoop 记录 paper/live order fields
- 不实现 TradingLoop arbitration 本身
- 不实现 PnL calculations

按 TDD 执行：先写失败测试，再最小实现。完成后运行 exchange、paper fill、TradingLoop persistence 相关测试，并总结结果。
```

### Batch 5 Architecture Review

```text
请使用 `.github/skills/improve-codebase-architecture/SKILL.md` review 刚完成的 V2 Batch 5 slices：
- V2V07
- V2V08
- V2V09

请阅读 V2 IW、V2 slices、Source refs、ADRs、代码和测试。

重点检查：
- ForecastLoop 是否只负责 forecast/probability writes
- TradingLoop 是否拥有 SELL/BUY-NO arbitration 和 re-quote
- exchange adapters 是否不拥有 pricing decisions
- signal observability 是否落 DB 而非乱发 AlertBus
- paper/live fill semantics 是否共用 quote vocabulary

不要修改代码或创建 issue。输出 architecture review notes 和建议的后续调整。
```

---

## Batch 6 -- Adjacent Runtime Paths

### V2V10 -- Backtest V2 Semantics

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V10 Backtest V2 Semantics。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V10
- V2V10 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0011

严格只实现 V2V10：
- 确认并保留 V2V04 已实现的 Backtest DataManager readiness skip
- Backtest L2 historical rows 使用 compute_quote
- mid-only rows 标记为 legacy_mid_price
- legacy_mid_price 不传入 compute_quote、不叫 clob_price
- 不重新实现 readiness gate
- 不实现 full historical L2 ingestion
- 不改 ForecastLoop readiness

按 TDD 执行：先写失败测试，再最小实现。完成后运行 Backtest pricing 测试，以及 readiness skip 的回归测试，并总结结果。
```

### V2V11 -- SniperLoop Alignment

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V11 SniperLoop Alignment。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V11
- V2V11 的 Source refs
- `docs/grill-tasks-v2.md`
- ADR-0011

严格只实现 V2V11：
- SniperLoop risk approve/size 调用带 open_positions
- SniperLoop 用 compute_quote pricing
- SniperLoop record_order 写 is_paper 和 executable_price
- SniperLoop 只用 get_tradeable_events
- 不实现 TradingLoop arbitration
- 不实现 discovery 本身

按 TDD 执行：先写失败测试，再最小实现。完成后运行 SniperLoop 相关测试，并总结结果。
```

### V2V12 -- Redeem Loop

```text
请使用 `.github/skills/tdd/SKILL.md` 实现 `docs/issues/proposed-slices-v2.md` 中的 V2V12 Redeem Loop。

开始前请阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 V2V12
- V2V12 的 Source refs
- `docs/grill-tasks-v2.md`

严格只实现 V2V12：
- SettlementLoop live winning condition enqueue redeem task
- paper mode skipped_paper 或显式不创建 pending task
- 新增 RedeemLoop pending/failed retry
- SecureClient.ctf_redeem_positions_call 每个 condition_id 调一次
- 失败记录 attempt_count/last_error 并下轮重试
- 新增 `redeem_loop_interval_s` settings 字段控制 sleep 节奏
- Bot live mode wiring RedeemLoop
- 不实现额外 on-chain settlement relayer

按 TDD 执行：先写失败测试，再最小实现。完成后运行 settlement/redeem/Bot wiring 相关测试，并总结结果。
```

### Batch 6 Final Architecture Review

```text
请使用 `.github/skills/improve-codebase-architecture/SKILL.md` review 刚完成的 V2 Batch 6 slices，以及整体 V2 overhaul：
- V2V10
- V2V11
- V2V12
- 全部已完成 V2 slices

请阅读 V2 IW、V2 slices、Source refs、ADRs、代码和测试。

重点检查：
- Backtest/live semantics 是否一致
- SniperLoop 是否不再绕过 tradeable events 和 risk/order contracts
- RedeemLoop 是否 condition-level、live-only、可重试
- 所有 V2 contracts 是否没有被 V1 旧语义污染
- 是否还有未覆盖的 source item 或 future-slice 泄漏

不要修改代码或创建 issue。输出 final architecture review notes 和建议的后续调整。
```

---

## Continue Prompt

```text
请根据 `docs/issues/implementation-workflow-v2.md` 和 `docs/issues/proposed-slices-v2.md`，在已完成 [COMPLETED_SLICE_IDS] 的前提下，找出下一个未阻塞的 V2 slice。

然后使用 `.github/skills/tdd/SKILL.md` 只实现这个 slice。开始前阅读该 slice、Source refs、`docs/grill-tasks-v2.md` 和相关 ADR。不要实现 out-of-scope 或未来 slice 行为。
```

## Diagnose Prompt

```text
请使用 `.github/skills/diagnose/SKILL.md` 诊断当前 V2 slice 的失败：[SLICE_ID]。

请先阅读：
- `docs/issues/implementation-workflow-v2.md`
- `docs/issues/proposed-slices-v2.md` 中的 [SLICE_ID]
- 该 slice 的 Source refs
- `docs/grill-tasks-v2.md`
- 相关 ADR

请从最小复现命令开始，说明失败现象，列出最可能的假设，只检查必要代码，并做最小修复。

不要扩大 slice scope。修复后保留或新增回归测试，并重跑最小失败测试；如果 public interface 变了，再跑相邻测试。
```
