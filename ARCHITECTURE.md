# PolyWeatherBot 架构设计

## 概述

PolyWeatherBot 是一个全自动气象事件交易系统，服务于 Polymarket 平台。
系统使用 HRRR 数值天气预报 + TFT 模型预测美国机场每日极端温度，通过
蒙特卡洛采样生成概率分布，与 CLOB 盘口比较后自动下单。

---

## 设计原则

1. **全面 OOP** - 除 config 文件和 `main.py` 外，每个文件只暴露一个主类；
   无模块级散装函数。
2. **接口先行** - 每个子系统有 ABC 基类定义契约；具体实现在同级文件。
3. **最少依赖** - 只引入无法用标准库替代的包（参见 DEPENDENCIES.md）。
   `herbie-data` 不是依赖；HRRR 取数用 `requests + eccodes` 直接完成。
   `catboost` 不移植（无活跃使用，仅为历史实验）。
   `slixmpp` 不移植（NWWS-OI 不含有效数据）。
4. **单模型多站点** - 共享一个 TFT 检查点；`station_id` 作为分类特征输入。
5. **两层存储** - 事务/状态数据用 SQLite；气象时序数据用按季度分区的 Parquet。
6. **优先级容灾** - 每个数据源有排序后的备选链；调用方无需知道最终由哪个源提供数据。
7. **纸单模拟一致性** - `PaperExchange` 实现与 `ClobExchange` 相同的接口；
   系统其余部分在两种模式下完全一致。
8. **代码入库 / 数据忽略** - 模型权重、Parquet、SQLite、凭证永不提交到 git。
9. **纯 venv 环境** - 目标系统为 Linux（开发: WSL2 Ubuntu 22.04，生产: Linux 服务器）。
   eccodes C 库通过 `apt install libeccodes-dev` 安装，其余全部 `pip install`。
   不使用 conda/mamba。

---

## 目录结构

```
PolyWeatherBot/
|
+-- config/                          # 静态配置（非类文件）
|   +-- __init__.py                  # 导出 Settings 和 STATIONS
|   +-- settings.py                  # Settings 数据类：路径、阈值、
|   |                                #   线程数、交易模式
|   +-- stations.py                  # StationConfig 数据类 + STATIONS 字典
|
+-- core/                            # 交易引擎
|   +-- __init__.py
|   +-- bot.py                       # BotManager：顶层协调器；
|   |                                #   拥有所有子管理器并串联各循环
|   |
|   +-- strategy/                    # 信号生成
|   |   +-- __init__.py
|   |   +-- base.py                  # BaseStrategy (ABC)：compute_signals()
|   |   +-- edge.py                  # EdgeStrategy：mc_prob - 盘口价格
|   |   |                            #   超过 min_edge 阈值时产生信号
|   |   +-- registry.py              # StrategyRegistry：名称 -> 类映射
|   |
|   +-- exchange/                    # 订单执行层
|   |   +-- __init__.py
|   |   +-- base.py                  # BaseExchange (ABC)：place_order(),
|   |   |                            #   cancel_order(), get_book(),
|   |   |                            #   get_positions()
|   |   +-- clob.py                  # ClobExchange：通过 py_clob_client
|   |   |                            #   发送实盘订单
|   |   +-- paper.py                 # PaperExchange：读取真实盘口，
|   |   |                            #   模拟成交，不实际发送订单
|   |   +-- settlement.py            # Settler：SettlementMode 枚举 +
|   |                                #   Relayer gasless / 0.99 抛售逻辑
|   |
|   +-- positions.py                 # PositionTracker：开仓/成交/平仓
|   |                                #   生命周期；持久化到 SQLite
|   |
|   +-- risk/                        # 仓位管理和风控
|       +-- __init__.py
|       +-- base.py                  # BaseRiskModel (ABC)：approve_signal(),
|       |                            #   compute_size()
|       +-- fixed.py                 # FixedFraction：固定 USDC / 笔，
|       |                            #   最大持仓和总量上限
|       +-- kelly.py                 # KellyRisk：预留 Kelly 公式计算
|
+-- data/                            # 数据获取与存储
|   +-- __init__.py
|   +-- manager.py                   # DataManager：协调所有数据源，
|   |                                #   ThreadPoolExecutor 并行取数，
|   |                                #   执行容灾切换
|   |
|   +-- forecast/                    # HRRR 预报数据
|   |   +-- __init__.py
|   |   +-- base.py                  # BaseForecastSource (ABC)：fetch()
|   |   +-- hrrr_s3.py               # HrrrS3Source：S3 Range Request
|   |   |                            #   主要生产数据源；requests + eccodes
|   |   +-- hrrr_nomads.py           # HrrrNomadsSource：NOAA NOMADS HTTP
|   |   |                            #   S3 的备选源；同 GRIB2+idx 格式
|   |   +-- hrrr_zarr.py             # HrrrZarrSource：Zarr 备份 + 批量下载；
|   |   |                            #   zarr + fsspec
|   |   +-- hrrr_herbie.py           # HrrrHerbieSource：herbie-data 库
|   |                                #   第四级备份，可选安装
|   |
|   +-- obs/                         # 地面实况观测
|   |   +-- __init__.py
|   |   +-- base.py                  # BaseObsSource (ABC)：fetch()
|   |   +-- iem.py                   # IemSource：Iowa Environmental Mesonet
|   |   |                            #   REST API（主要实况源）
|   |   +-- noaa_file.py             # NoaaFileSource：NOAA tgftp METAR
|   |                                #   文件轮询（备份实况源）
|   |
|   +-- clob/                        # CLOB 盘口数据
|   |   +-- __init__.py
|   |   +-- orderbook.py             # OrderbookFetcher：封装 ClobClient
|   |                                #   get_order_book()；按 token 缓存
|   |
|   +-- storage/                     # 持久化天气数据
|       +-- __init__.py
|       +-- parquet_store.py         # ParquetStore：去重追加、按季度分区、
|       |                            #   带日期范围的批量读取
|       +-- cleaner.py               # Cleaner：单位换算、异常值去除、
|                                    #   缺值插补；输出标准 schema
|
+-- events/                          # 事件生命周期管理
|   +-- __init__.py
|   +-- discovery.py                 # EventDiscovery：每日轮询 Gamma API；
|   |                                #   按 slug 模式 + 标签匹配；
|   |                                #   写入 SQLite events 表
|   +-- lifecycle.py                 # EventLifecycle：状态机
|                                    #   discovered -> active -> cutoff ->
|                                    #   pending_settle -> settled
|
+-- model/                           # 预测模型
|   +-- __init__.py
|   +-- tft.py                       # TftModel：加载检查点、推理；
|   |                                #   封装 pytorch-forecasting
|   +-- mc.py                        # McSampler：AR(1) 蒙特卡洛采样
|   |                                #   生成每日高温各桶概率
|   +-- pipeline.py                  # ModelPipeline：编排
|   |                                #   DataManager -> TftModel -> McSampler
|   |                                #   -> ProbDist（单站点单 init）
|   +-- store/
|       +-- __init__.py
|       +-- registry.py              # ModelRegistry：版本目录；
|                                    #   load/publish/rollback/list
|
+-- snipers/                         # 实时观测抓取（AWC + NOAA 双通道）
|   +-- __init__.py
|   +-- base.py                      # BaseSniperSource (ABC)：start(),
|   |                                #   stop(), on_observation 回调
|   +-- awc.py                       # AwcSniper：轮询 AWC METAR API
|   |                                #   正常 2s / 狙击 0.65s 间隔
|   +-- noaa.py                      # NoaaSniper：轮询 NOAA tgftp METAR
|   |                                #   正常 3s / 狙击 0.2s 间隔
|   +-- manager.py                   # SniperManager：并发启动 AWC + NOAA；
|                                    #   去重观测；喂入队列
|
+-- loops/                           # 调度 / 主循环
|   +-- __init__.py
|   +-- forecast_loop.py             # ForecastLoop：每小时；并行
|   |                                #   站点取数 -> 模型 -> 信号 ->
|   |                                #   风控 -> 交易 -> 持仓
|   +-- sniper_loop.py               # SniperLoop：事件驱动；消费
|   |                                #   SniperManager 队列；FOK 订单
|   +-- discovery_loop.py            # DiscoveryLoop：每日；调用
|   |                                #   EventDiscovery，更新 DB
|   +-- settlement_loop.py           # SettlementLoop：每日盘后；
|                                    #   调用 Settler 结算待处理头寸
|
+-- backtest/                        # 离线回测
|   +-- __init__.py
|   +-- engine.py                    # BacktestEngine：遍历历史 init；
|   |                                #   在 Parquet 数据上回放 Pipeline
|   +-- analysis.py                  # BacktestAnalysis：Brier 评分、
|                                    #   阈值/提前量/init 小时分组 ROI、
|                                    #   图表输出
|
+-- db/                              # 数据库层
|   +-- __init__.py
|   +-- schema.py                    # DbClient：首次运行自动建表；
|                                    #   所有表的 CRUD 工具方法；
|                                    #   表：stations, events, markets,
|                                    #   signals, orders, model_versions
|
+-- ui/                              # Streamlit Web 界面
|   +-- app.py                       # StreamlitApp：入口；侧栏导航；
|   |                                #   仅读取 SQLite + ParquetStore
|   +-- pages/
|       +-- dashboard.py             # DashboardPage：实时损益、持仓、
|       |                            #   各站点状态
|       +-- positions.py             # PositionsPage：订单/成交历史，
|       |                            #   纸单 vs 实盘切换
|       +-- events.py                # EventsPage：事件发现列表、
|       |                            #   生命周期状态、桶位价格
|       +-- backtest.py              # BacktestPage：运行 BacktestAnalysis；
|       |                            #   交互式阈值/筛选控件
|       +-- model.py                 # ModelPage：当前版本、各站 RMSE、
|                                    #   预测分布查看器
|
+-- tests/                           # pytest 测试（镜像 src 结构）
|   +-- conftest.py                  # 共享 fixtures：内存 SQLite、mock
|   |                                #   ClobClient、样本 DataFrame、临时目录
|   +-- test_config/
|   |   +-- test_stations.py         # StationConfig 校验
|   +-- test_core/
|   |   +-- test_strategy.py         # EdgeStrategy 信号正确性
|   |   +-- test_exchange.py         # BaseExchange 接口契约；
|   |   |                            #   PaperExchange 模拟成交
|   |   +-- test_positions.py        # PositionTracker 状态转换
|   |   +-- test_risk.py             # 风控上限、Kelly 计算（桩）
|   +-- test_data/
|   |   +-- test_hrrr_s3.py          # S3 idx 解析 + Range Request 组装
|   |   +-- test_hrrr_zarr.py        # Zarr 变量提取
|   |   +-- test_iem.py              # IEM JSON 解析、缺值处理
|   |   +-- test_parquet_store.py    # 去重追加、季度分区、日期读取
|   |   +-- test_cleaner.py          # 单位换算、异常值去除
|   +-- test_events/
|   |   +-- test_discovery.py        # Slug 拼接、DB 注册
|   |   +-- test_lifecycle.py        # 状态机转换
|   +-- test_model/
|   |   +-- test_mc.py               # AR(1) 统计量：均值、方差、桶概率和
|   |   +-- test_pipeline.py         # 端到端 mock：HRRR -> ProbDist
|   |   +-- test_registry.py         # 版本加载、最新选择、回滚
|   +-- test_snipers/
|   |   +-- test_manager.py          # 多源去重、队列消费
|   +-- test_loops/
|   |   +-- test_forecast_loop.py    # 单周期状态变更校验
|   +-- test_backtest/
|   |   +-- test_engine.py           # 固定输入损益正确性
|   +-- test_db/
|       +-- test_schema.py           # DDL、CRUD、外键约束
|
+-- scripts/                         # 一次性离线操作（非常驻循环）
|   +-- batch_download_zarr.py       # BatchZarrDownloader：批量下载历史
|   |                                #   HRRR 数据用于初始训练集
|   +-- batch_download_iem.py        # BatchIemDownloader：批量下载 IEM
|   |                                #   历史数据用于初始训练集
|   +-- train_tft.py                 # TftTrainer：读 Parquet、训练模型、
|   |                                #   发布到 ModelRegistry
|   +-- migrate_db.py                # DbMigrator：数据库版本升级
|
+-- data_store/                      # 运行时数据（git 忽略）
|   +-- polyweatherbot.db            # SQLite 数据库
|   +-- forecast/
|   |   +-- parquet/
|   |   |   +-- KLGA/
|   |   |   |   +-- 2025-Q4.parquet
|   |   |   |   +-- 2026-Q1.parquet
|   |   |   +-- KBOS/ ...
|   |   +-- raw_zarr/                # 仅离线批量下载时使用
|   +-- obs/
|       +-- parquet/
|           +-- KLGA/
|           +-- KBOS/ ...
|
+-- model/store/                     # 模型检查点（git 忽略）
|   +-- v1.0/
|   |   +-- tft_checkpoint.ckpt
|   |   +-- meta.json
|   +-- v1.1/ ...
|
+-- .github/
|   +-- instructions/
|   |   +-- copilot-instructions.md  # Copilot 编码规则
|   +-- workflows/
|       +-- ci.yml                   # push/PR 时跑 pytest
|
+-- main.py                          # CLI 入口（argparse，非类文件）
+-- pyproject.toml                   # 包元数据和依赖声明
+-- .env.example                     # 凭证模板（入库）
+-- .env                             # 实际凭证（git 忽略）
+-- .gitignore
```

---

## 与旧版差异

| 条目 | 旧版 | 新版 | 原因 |
|---|---|---|---|
| `snipers/nwws.py` | 存在 | **已删除** | NWWS-OI 不含有效温度数据 |
| `snipers/iem.py` | 存在 | **已删除** | 生产用 AWC + NOAA 双通道即可 |
| `snipers/awc.py` | 不存在 | **新增** | 从 `combined_sniper.py` 提取 AWC 通道 |
| `model/store/catboost_bias.cbm` | 存在 | **已删除** | catboost 仅为历史实验，TFT/MC 管线无引用 |
| `slixmpp` 依赖 | 存在 | **已删除** | 跟随 NWWS 移除 |
| `catboost` 依赖 | 存在 | **已删除** | 跟随模型移除 |
| HRRR NOMADS 数据源 | 不存在 | **新增** `hrrr_nomads.py` | S3 的备选；官方 NOAA 源 |
| 环境管理 | conda/mamba | **纯 venv + pip**（Linux `apt` 装 eccodes） | 彻底去除 conda |

---

## 类继承关系

```
BaseStrategy
  EdgeStrategy

BaseExchange
  ClobExchange
  PaperExchange

BaseRiskModel
  FixedFraction
  KellyRisk（桩）

BaseForecastSource
  HrrrS3Source       <- 主要生产数据源
  HrrrNomadsSource   <- S3 备选（NOAA NOMADS HTTP）
  HrrrZarrSource     <- 第三级备份 + 离线批量
  HrrrHerbieSource   <- 第四级备份（可选安装）

BaseObsSource
  IemSource          <- 主要实况（API 查询）
  NoaaFileSource     <- 备份实况（tgftp 文件）

BaseSniperSource
  AwcSniper          <- AWC METAR API（xx:51:30 狙击窗口）
  NoaaSniper         <- NOAA tgftp METAR（xx:50:45 狙击窗口）
```

---

## Sniper 架构说明

新版只保留 AWC + NOAA 双通道并发架构，直接对应参考代码
`combined_sniper.py` 的设计：

| 通道 | 端点 | 正常间隔 | 狙击间隔 | 狙击窗口 |
|---|---|---|---|---|
| **NOAA** | `tgftp.nws.noaa.gov/.../KLGA.TXT` | 3s | 0.2s | xx:50:45 ~ xx:56:00 |
| **AWC** | `aviationweather.gov/api/data/metar` | 2s | 0.65s | xx:51:30 ~ xx:53:59 |

`SniperManager` 在 daemon 线程中同时启动两个通道，用 `seen_obs` 集合
去重后将观测推入 `queue.Queue`。

---

## 数据流概览

### 1) 预报流（每小时）

```
HRRR S3 / NOMADS -> HrrrS3Source / HrrrNomadsSource -> ParquetStore
                              |
                              v
                        FeaturePipeline
                         /          \
                   TftModel      McSampler
                         \          /
                          ProbDist
                             |
                        EdgeStrategy
                             |
                    FixedFraction / Kelly
                             |
                    ClobExchange / PaperExchange
                             |
                        SQLite (orders)
```

### 2) 狙击流（实时）

```
AWC / NOAA METAR -> SniperManager -> queue.Queue
                                        |
                                   SniperLoop
                                        |
                                   EdgeStrategy -> Exchange -> SQLite
```

### 3) 结算流（每日盘后）

```
EventLifecycle detects resolved event
       |
  ObsDataManager (IEM / tgftp) -> 确定获胜桶
       |
  Settler: gasless Relayer 或 0.99-dump
       |
  SQLite (orders status = settled)
```

### 4) 离线训练流（按需）

```
scripts/batch_download_zarr.py -> ParquetStore (forecast)
scripts/batch_download_iem.py  -> ParquetStore (obs)
scripts/train_tft.py           -> ModelRegistry (checkpoint)
```
