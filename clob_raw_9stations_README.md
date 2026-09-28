# 数据包说明：clob_raw_9stations.zip

> 生成日期：2026-08-26
> 包内数据：9 个活跃站点（KLGA 除外）的 Polymarket 历史 CLOB 原始下载制品

---

## 1. 数据包基本信息

| 项目 | 内容 |
|---|---|
| 文件名 | `clob_raw_9stations.zip` |
| 位置 | 项目根目录（`d:\NN\PolyWeatherData\`） |
| 大小 | 约 520.8 MB（压缩后；解压后约 662 MB） |
| 站点数 | 9（LONDON, CHICAGO, MIAMI, SEATTLE, ATLANTA, TOKYO, SHANGHAI, SEOUL, TORONTO） |
| 数据源 | Polymarket 公开 API（`gamma-api.polymarket.com` + `data-api.polymarket.com`） |
| 数据性质 | **taker 成交（fill price）原始逐笔数据**，非盘口中价、非 K 线 |
| 时间跨度 | 各站系列创建日起至 2026-08（见第 3 节逐站清单） |
| 市场类型 | 每日最高温分桶市场（如 "London 1 月 22 日最高温是否在 36-37°F 之间"） |
| 排除项 | KLGA（纽约，数据单独维护）、DALLAS、DENVER（异常/已过期，不在此包） |

包内结构（zip 内层级，解压后即为各站目录）：

```
clob_raw_9stations.zip
├── LONDON/20250121_20260810/{events,markets,trades,coverage}.parquet + manifest.json
├── CHICAGO/20251204_20260810/{...}
├── MIAMI/20250101_20260812/{...}
├── SEATTLE/20250101_20260812/{...}
├── ATLANTA/20250101_20260812/{...}
├── TOKYO/20250101_20260818/{...}
├── SHANGHAI/20250101_20260818/{...}
├── SEOUL/20250101_20260818/{...}
└── TORONTO/20250101_20260818/{...}
```

---

## 2. 五个文件各自是什么

每个站点目录下有 5 个文件，均为**不可变制品**（下载完成后不再修改）：

### 2.1 `events.parquet` — 事件清单（每天一个事件）

| 列 | 类型 | 说明 |
|---|---|---|
| `event_id` | str | 事件唯一 ID |
| `event_slug` | str | 事件 slug，如 `highest-temperature-in-london-on-jan-22` |
| `event_title` | str | 事件标题 |
| `target_date` | str | 目标日期（该市场所预测的天气日期，YYYY-MM-DD） |
| `event_end_at` | datetime | 事件结束（结算）时间 |
| `station_id` | str | 站点代码 |

一个事件 = 某城市某一天的"最高温"市场，下面挂着多个分桶条件市场（见 `markets.parquet`）。

### 2.2 `markets.parquet` — 条件市场清单（每个温度桶一个市场）

| 列 | 类型 | 说明 |
|---|---|---|
| `event_id` | str | 所属事件 |
| `market_id` / `condition_id` | str | 市场 ID / 条件 ID（**trades 以 condition_id 关联**） |
| `market_slug` | str | 市场 slug，含温度桶定义，如 `...-between-36-37f-on-january-22` |
| `yes_token_id` / `no_token_id` | str | YES / NO 两个结果的 ERC-1155 token ID |
| `bucket_lo_f` / `bucket_hi_f` | float | 温度桶下界 / 上界（°F），开区间边界如 `(36, 38)` |
| `yes_outcome_price` / `no_outcome_price` | float | 结算后官方结果价（1.0 / 0.0 或中间值） |
| `resolved_yes` | bool | 是否已结算且 YES 中奖 |
| `fees_*` 系列 | — | 手续费配置（费率、类型、返点等） |
| `min_tick_size` / `min_order_size` | float | 最小变动价位 / 最小下单量 |
| `accepting_orders` / `active` / `closed` | bool | 市场状态 |
| `neg_risk` | bool | 是否负风险市场 |
| `market_start_at` / `market_end_at` / `closed_at` 等 | datetime | 时间戳 |

**桶划分规则**（由 slug 解析，`events/discovery.py::parse_slug`）：
- `35f-or-lower` → 下界 `(-inf, 36)`；`between-36-37f` → `(36, 38)`；半开区间 `[lo, hi+1)`；
- 国际站存在摄氏变体（`{X}c`、`{X}c-or-higher`、连字符形式），已统一转为 `*_f` 字段。

### 2.3 `trades.parquet` — 逐笔成交（核心数据，最大文件）

| 列 | 类型 | 说明 |
|---|---|---|
| `condition_id` | str | 关联 `markets.parquet` |
| `token_id` | str | YES 或 NO 的 token ID |
| `timestamp` | datetime(UTC) | 成交时间（UTC） |
| `outcome` | str | `YES` / `NO` |
| `side` | str | `BUY` / `SELL` |
| `price` | float | 成交价（0~1），**taker 成交的填充价** |
| `size` | float | 成交数量（股数） |
| `usdc` | float | 成交金额（USDC） |
| `transaction_hash` | str | 链上交易哈希 |
| `proxy_wallet` | str | 交易者钱包地址 |
| `source` | str | 数据来源（恒为 `data_api_trade`） |

校验规则（下载时已强制）：
- `price ∈ [0, 1]`、`size > 0`、`usdc = price × size`；
- 同一 (condition, token, tx_hash, price, size) 无重复；
- 该文件只含**公共 taker 成交**（API 参数 `takerOnly=true`），maker 单不在此列。

### 2.4 `coverage.parquet` — 逐日覆盖统计（数据质量审计表）

| 列 | 类型 | 说明 |
|---|---|---|
| `target_date` | str | 目标日期 |
| `event_count` | int | 该日事件数 |
| `market_count` | int | 该日市场数 |
| `token_count` | int | 该日 token 数 |
| `trade_count` | int | 该日成交笔数 |
| `complete` | bool | 该日是否完整（无缺失市场/被截断分页） |

用途：快速筛查哪天数据缺漏。`complete=false` 的日子与 `manifest.json` 的 `incomplete_target_dates` 对应。

### 2.5 `manifest.json` — 制品清单与完整性校验

| 键 | 说明 |
|---|---|
| `data_class` | 恒为 `raw_point_in_time`（逐点原始数据，未加工） |
| `provider` | `polymarket_public_api` |
| `requested_range` | [起始日, 结束日]（下载时请求的范围） |
| `row_counts` | events / markets / trades 三表行数 |
| `missing_target_dates` | 完全缺失的目标日期列表 |
| `incomplete_target_dates` | 部分缺失的目标日期列表 |
| `schema_version` | 表结构版本（本包为 2） |
| `sha256` | **每个 parquet 文件的 SHA256**（加载时校验用） |
| `fetched_at` | 下载完成时间 |

---

## 3. 逐站清单

| 站点 | 包内目录 | events | markets | trades | 数据范围 | 缺失日期数 |
|---|---|---|---|---|---|---|
| LONDON | 20250121_20260810 | 563 | 4,593 | 1,795,225 | 2025-01-22 ~ 2026-08-10 | 4 |
| CHICAGO | 20251204_20260810 | 203 | 2,073 | 584,165 | 2025-12-04 ~ 2026-08-11 | 49 |
| MIAMI | 20250101_20260812 | 204 | 2,084 | 695,543 | 2025-12-04 ~ 2026-08-12 | 49 |
| SEATTLE | 20250101_20260812 | 248 | 2,392 | 625,936 | 2025-12-06 ~ 2026-08-13 | 3 |
| ATLANTA | 20250101_20260812 | 251 | 2,421 | 692,462 | 2025-12-06 ~ 2026-08-15 | 3 |
| TOKYO | 20250101_20260818 | 161 | 1,759 | 570,916 | 2026-03-10 ~ 2026-08-17 | 1 |
| SHANGHAI | 20250101_20260818 | 158 | 1,732 | 882,451 | 2026-03-13 ~ 2026-08-17 | 1 |
| SEOUL | 20250101_20260818 | 255 | 2,457 | 1,252,836 | 2025-12-06 ~ 2026-08-17 | 1 |
| TORONTO | 20250101_20260818 | 255 | 2,465 | 620,582 | 2025-12-06 ~ 2026-08-17 | 1 |

合计：**1,144 个事件、9,461 个市场、约 720 万笔成交**。

注：
- "缺失日期"多为系列创建初期的空档或当日市场未上架（API 无记录），不影响主体数据；
- CHICAGO / MIAMI 缺失 49 天，是因为这两个系列 2025-12 中旬才创建，而请求范围覆盖到更早（目录名以请求起点命名，`requested_range` 以 manifest 为准）。

---

## 4. 数据获取方式（简要）

1. **事件枚举**：通过 `gamma-api.polymarket.com/events?series_id={id}&closed=true` 分页拉取某天气系列的**全部历史事件**（系列 ID 见 `scripts/download_historical_clob.py` 的 `DAILY_WEATHER_SERIES`）；
2. **成交下载**：对每个条件的每个市场，调用 `data-api.polymarket.com/trades?market={condition_id}&limit=1000&takerOnly=true` 分页拉取全部 taker 成交（每页 1000 条，撞到页尾恰满 1000 会二次确认完整性）；
3. **元数据**：市场 slug 解析出温度桶边界，结算结果取官方 `outcomePrices`；
4. **下载器**：`data/clob/history.py::HistoricalClobSource`，请求间隔 ≥0.4s、6 次重试、429 退避 `30s·2^attempt`；
5. **审计**：`scripts/audit_historical_clob.py` 逐日检查覆盖完整性、桶划分完整性、成交合法性。

---

## 5. 使用示例

```python
import pandas as pd
import json

base = "LONDON/20250121_20260810"   # 解压后的相对路径

trades  = pd.read_parquet(f"{base}/trades.parquet")
markets = pd.read_parquet(f"{base}/markets.parquet")
events  = pd.read_parquet(f"{base}/events.parquet")
cov     = pd.read_parquet(f"{base}/coverage.parquet")

# 完整性校验：对比 manifest 中的 SHA256
man = json.load(open(f"{base}/manifest.json", encoding="utf-8"))
print(man["sha256"])
```

---

## 6. 注意事项与已知问题

1. **价格定义**：本包是 **taker 成交填充价**。研究面板中的"某时距价格"= 达峰前 ±30 分钟窗口内全部成交的**均值**（见 `docs/icdm/code/05_rebuild_panel_argmax_30min.py`）；2026 年 7 月 KLGA 测试集用的是实时 L2 盘口中价（前瞻采集），两种口径不同，论文 §2.3 已注明。
2. **manifest 元数据小 bug**：个别站点（如 LONDON）的 `manifest.json` 中 `station_id` 字段误写为 `KLGA`（下载器硬编码遗留），**不影响数据本身**，请以目录名为准。
3. **本包不含**：KLGA（其制品单独维护于 `KLGA/` 下多段目录）、DALLAS / DENVER（异常/过期站点）、`.raw-cache` JSON 下载缓存、规范窗口副本（`20251205_20260810`）。
4. **空桶说明**：`bucket_lo_f`/`bucket_hi_f` 存在 `-inf`/`inf`（如 `35f-or-lower` 上界、`85f-or-higher` 下界），读取时注意处理。
5. **`source` 恒为 `data_api_trade`**：所有成交均来自 Data API 的 taker 数据源，无其他来源混入。
