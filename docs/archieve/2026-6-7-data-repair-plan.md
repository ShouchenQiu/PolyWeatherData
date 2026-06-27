# 历史下载复盘与优化实施计划

## Summary
本次暂停后的事实结论：

- HRRR f01 已完整：`1560` rows，覆盖 `2026-04-01 01Z` 到 `2026-06-05 00Z`。
- IEM obs 有 `1553/1560` rows，缺 7 个 canonical hour。
- IEM 缺口不是 IEM 源完全无数据，而是 downloader 只接受 exact `:51`；缺失小时源里有 `:50/:55` 等邻近 T-group 观测。
- Weather.com 当前只提供温度 truth，不能完整替代 IEM obs schema，不作为 canonical obs 补全主路径。
- decoder cache 已完整落盘 `948/1560` init，剩余 `612` init；慢点来自 zarr connect timeout 与部分 non-major `23Z` 触发 S3 fallback/splice。
- 后续策略锁定为：
  - IEM：用 IEM 邻近代表观测补缺。
  - decoder：保持 zarr primary/S3 fallback，但增强 zarr 稳健重试、统计和可暂停恢复。

## Key Changes

### IEM Obs 补缺策略
- 修改 `BatchIemDownloader`：不再只保留 exact `:51`，改为保留日窗口内所有候选观测，交给 `Cleaner.clean_iem()` 按现有规则选择每小时最佳候选。
- 在 `Cleaner.clean_iem()` 增加代表观测最大距离限制：同一 canonical hour 只接受距离 `valid_hour - 9min` 不超过 `20min` 的候选。
- 优先级：1. exact :51 可用时，优先选 exact :51
2. exact :51 缺失时，才在同小时邻近候选中选最接近 hour-9min 的记录
3. 邻近候选内部仍然 T-group 优先
4. 邻近候选必须在 20min tolerance 内。
- `BatchIemDownloader._already_done_dates()` 改为只把 `24` 个 canonical hours 都存在的日期视为 done；这样 20/23 小时的短缺日会被后续下载自动重试补齐。
- 下载后仍只落现有 IEM Parquet schema，不落 Weather.com 字段，不改变 TFT 输入 schema。

### Weather.com 角色
- Weather.com 只作为温度 truth 审计/对照源，不直接补 canonical IEM obs。
- 不用 Weather.com 补 `raw_tmpc_tenths`，因为它无法补齐风、云、降水、气压等 IEM obs 字段，会造成 mixed-source obs schema 和训练语义不一致。
- 可后续单独增加一个只读 data-health 报告：列出 IEM 缺失小时，并展示 Weather.com 温度是否可用，但不写入 obs store。

### Decoder Cache 优化
- 保持 `TrainingDecoderCache` 主路径：`HrrrZarrSource` primary，`HrrrS3Source` fallback。
- 为 zarr critical reads 增加有限重试：
  - TMP_2m shape probe 和 TMP_2m variable read 最多重试 2 次，指数退避。
  - 非关键变量仍允许 NaN，但记录 variable timeout 计数。
- 对 non-major fill 依赖读取增加更明确行为：
  - nearest/prior major zarr 读取失败时先重试。
  - 重试后仍失败才进入 S3 fallback。
  - 不把短窗口 non-major 结果静默返回给 decoder cache 验证。
- 增加 decoder prewarm 统计：
  - `warm_elapsed_seconds`
  - `fetch_elapsed_seconds`
  - `zarr_success_inits`
  - `fallback_success_inits`
  - `fallback_failed_inits`
  - `zarr_timeout_count`
  - `slow_init_samples`
- 增强暂停恢复：
  - `smoke_bootstrap` 捕获 `KeyboardInterrupt` 后写 partial summary，例如 `.demo/smoke_summary.partial.json`。
  - partial summary 写入当前阶段、已完成 init、错误数、已落盘 decoder complete count。
  - 保持当前月 parquet 断点复用语义，重跑同一命令会跳过已完整的 decoder init。

### Resume 操作建议
- 补完剩余 decoder 时优先使用现有 flags，避免重复 f01/IEM：

```bash
. .venv/bin/activate
python -m scripts.smoke_bootstrap \
  --smoke-dir .demo \
  --station KLGA \
  --start 2026-04-01 \
  --end 2026-06-04 \
  --workers 12 \
  --skip-polymarket \
  --skip-hrrr \
  --skip-iem
```

- IEM 补缺代码完成后，再单独重跑 IEM：

```bash
. .venv/bin/activate
python -m scripts.smoke_bootstrap \
  --smoke-dir .demo \
  --station KLGA \
  --start 2026-04-01 \
  --end 2026-06-04 \
  --workers 12 \
  --skip-polymarket \
  --skip-hrrr \
  --skip-decoder
```

## Test Plan
- IEM tests:
  - 缺 exact `:51` 但有 `:50/:55` T-group 时，输出对应 canonical hour。
  - 超过 `20min` 代表距离的候选不被用于补小时。
  - exact `:51` 存在时仍选择 exact `:51`。
  - 短缺日 `<24h` 不再被 `_already_done_dates()` 判定为 done。
  - 最终 obs Parquet columns 不包含 Weather.com 字段或内部排序字段。

- Decoder tests:
  - zarr TMP_2m 首次 timeout、第二次成功时不触发 S3 fallback。
  - zarr critical read 连续失败时触发 S3 fallback，并计入 fallback stats。
  - non-major zarr fill 依赖失败时不返回短窗口 decoder；必须重试或 fallback。
  - `prewarm()` summary 区分 warm/fetch elapsed，不再用 warm 时间污染 fetch ETA。
  - `KeyboardInterrupt` 时已完成 frames 被 flush，并写 partial summary。

- Integration checks:
  - 补缺后 `.demo` obs 应达到 `1560` canonical hours，除非 IEM 源在 `20min` tolerance 内确实无候选。
  - decoder resume 后 complete init 应从 `948` 增至 `1560`。
  - 用新 split 做 preflight：train/val/test 每个 init 都应为 `120` rows，无关键 TFT column 缺失或 NaN。

## Assumptions
- Canonical obs 仍以 IEM 为唯一训练 obs source；Weather.com 不进入 TFT 输入。
- IEM 代表时间仍沿用当前 ADR：canonical hour 对应 `hour - 9min` 附近观测。
- `20min` tolerance 用于历史 batch；live tick 仍可继续请求窄窗口 H-1 `:51`，不在本轮改变。
- Decoder 目标是减少误 fallback 和提升可观测性，不改变 HRRR f01-only production store 策略。
- 本轮不重新设计 HRRR feature schema，不移除 S3 fallback。

## 其他
当前下载暂停点也记住：
HRRR f01: 完整 1560
IEM obs: 1553/1560，缺 7 小时
decoder cache: 948/1560 complete init，剩 612HRRR f01: 完整 1560
IEM obs: 1553/1560，缺 7 小时
decoder cache: 948/1560 complete init，剩 612