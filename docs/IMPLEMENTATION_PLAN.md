# PolyWeatherBot 实施计划

> **本文档为 AI agent 执行手册。**
> 每个阶段 (Phase) 是一个独立交付单元，由若干任务 (Task) 组成。
> Agent 应按顺序执行，每完成一个 Task 后运行验证命令确认通过再继续。

---

## 约定

| 符号 | 含义 |
|---|---|
| `[创建]` | 新建文件 |
| `[迁移]` | 从 Herbie-main 参考代码重构为 OOP |
| `[测试]` | 编写 + 运行 pytest |
| `[验证]` | 运行命令确认产物正确 |
| `→` | 前置依赖 |

- **工作目录**: `/home/Code/PolyWeatherBot/`
- **参考代码 (Herbie-main)**: `/mnt/c/Codes/Polymarket/Herbie-main/`
- **参考代码 (py-clob-client-main)**: `/mnt/c/Codes/Polymarket/py-clob-client-main/`
  - 含 18 个自有脚本：合成盘口构建 (`build_synthetic_orderbook.py`)、
    Gamma API 事件发现 (`list_feb20_markets.py`)、sniper 检测
    (`analyze_snipers.py`)、市场生命周期 (`analyze_lifecycle.py`)、
    价格历史 (`probe_prices_history.py`) 等
- **CLOB SDK**: `py-clob-client`（pip install 或 editable install）
- 所有路径为 Linux 风格；Agent 应使用绝对路径调用工具。
- 每个 Phase 结束后整体运行 `pytest tests/ -q` 保证回归绿灯。

---

## Phase 0 — 项目脚手架

> 目标：创建目录结构、环境配置、基础设施文件。无业务代码。

### Task 0.1 [创建] 初始化仓库和目录

创建以下空目录结构（每个包含 `__init__.py`）：

```
PolyWeatherBot/
├── config/
├── core/
│   ├── strategy/
│   ├── exchange/
│   └── risk/
├── data/
│   ├── forecast/
│   ├── obs/
│   ├── clob/
│   └── storage/
├── events/
├── model/
│   └── store/
├── snipers/
├── loops/
├── backtest/
├── db/
├── ui/
│   └── pages/
├── tests/
│   ├── test_config/
│   ├── test_core/
│   ├── test_data/
│   ├── test_events/
│   ├── test_model/
│   ├── test_snipers/
│   ├── test_loops/
│   ├── test_backtest/
│   └── test_db/
└── scripts/
```

**输入**: 无
**输出**: 上述目录 + `__init__.py`
**验证**: `python3 -c "import config; import core; import data; import model; import snipers; import loops"` 无报错

### Task 0.2 [创建] pyproject.toml

文件: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "polyweatherbot"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "pandas",
    "pyarrow",
    "xarray",
    "cfgrib",
    "eccodes",
    "zarr",
    "fsspec",
    "requests",
    "torch",
    "pytorch-forecasting",
    "python-dotenv",
    "streamlit",
    "plotly",
]

[project.optional-dependencies]
clob = ["py-clob-client"]
herbie = ["herbie-data"]
dev = ["pytest", "pytest-cov", "black", "ruff"]
```

> 前置条件：`sudo apt install libeccodes-dev` 安装 eccodes C 库。

**验证**: `pip install -e ".[dev]"` 成功

### Task 0.3 [创建] 环境配置

1. 创建 `.env.example`:
```
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_PASSPHRASE=
POLYMARKET_FUNDER=
POLYMARKET_PROXY_ADDRESS=
CLOB_API_URL=https://clob.polymarket.com
GAMMA_API_URL=https://gamma-api.polymarket.com
CHAIN_ID=137
TRADING_MODE=paper
```

2. 创建 `.gitignore`:
```
__pycache__/
*.pyc
.env
data_store/
model/store/v*/
*.ckpt
*.parquet
*.db
```

**验证**: `.gitignore` 存在且包含 `data_store/`

### Task 0.4 [创建] conftest.py 基础设施

文件: `tests/conftest.py`

提供以下 fixtures:
- `tmp_db` — 返回内存 SQLite 连接（`":memory:"`）
- `tmp_dir` — `tmp_path` 别名
- `sample_hrrr_df` — 含 36 个 HRRR 变量 + `init_time`, `fxx`, `valid_time` 的小 DataFrame（3 个 init × 48 行）
- `sample_iem_df` — 含 14 个 IEM 变量 + `valid_time` 的小 DataFrame

参考: `Herbie-main/tft_step6_build_72h.py` 中的 `HRRR_COLS` 和 `IEM_COLS` 列定义。

**验证**: `pytest tests/conftest.py --co -q` 列出 fixtures 无报错

---

## Phase 1 — 配置层

> 目标：`config/` 包。所有硬编码常量集中在此。

### Task 1.1 [创建] config/settings.py

定义 `Settings` dataclass，字段包括：

```python
@dataclass
class Settings:
    # 路径
    project_root: Path
    data_store_dir: Path         # data_store/
    model_store_dir: Path        # model/store/

    # 交易
    trading_mode: str            # "paper" | "live"
    min_edge: float              # 默认 0.04
    max_position_usd: float      # 默认 50.0
    max_total_usd: float         # 默认 500.0

    # HRRR
    hrrr_s3_bucket: str          # "noaa-hrrr-bdp-pds"
    hrrr_zarr_bucket: str        # "hrrrzarr"
    klga_grid_row: int           # 698 (Zarr) 或 676 (S3 idx)
    klga_grid_col: int           # 1556 (Zarr) 或 1168 (S3 idx)

    # MC
    mc_n_samp: int               # 5000
    mc_rng_seed: int             # 42
    mc_rho: float                # 0.90
    mc_bias_corr_c: float        # 1.5206

    # 数据源 API
    iem_api_url: str             # "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    noaa_tgftp_url_tpl: str     # "https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT"
    awc_metar_url: str           # "https://aviationweather.gov/api/data/metar"
    hrrr_nomads_base: str        # "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
    gamma_api_url: str           # "https://gamma-api.polymarket.com"
    clob_api_url: str            # "https://clob.polymarket.com"

    # 调度
    forecast_interval_min: int   # 60
    clob_window_min: int         # 30
    cutoff_hour_utc: int         # 21 (= 16:00 EST)
    min_hours_before_cutoff: int # 8
```

使用 `python-dotenv` + `os.environ` 加载 `.env` 凭证。

**参考**: `Herbie-main/mc_stage2_clob.py` 中的常量定义、`run_hrrr_cache.py` 中的 S3 配置、
`combined_sniper.py` 中的 API URL 定义。
**验证**: `python3 -c "from config.settings import Settings; s = Settings(); print(s.trading_mode)"`

### Task 1.2 [创建] config/stations.py

定义 `StationConfig` dataclass:

```python
@dataclass
class StationConfig:
    station_id: str              # "KLGA"
    lat: float                   # 40.7769
    lon: float                   # -73.8740
    timezone: str                # "America/New_York"
    tz_offset: int               # -5 (EST, 无夏令时 Dec-Feb)
    hrrr_grid_row_zarr: int      # 698
    hrrr_grid_col_zarr: int      # 1556
    iem_network: str             # "NY_ASOS"
    slug_pattern: str            # "will-the-high.*lga"
```

导出: `STATIONS: dict[str, StationConfig]`，初始只含 `"KLGA"`。

**参考**: `Herbie-main/run_hrrr_zarr.py` (KLGA_ROW=698, KLGA_COL=1556)。
**验证**: `python3 -c "from config.stations import STATIONS; print(STATIONS['KLGA'].lat)"`

### Task 1.3 [测试] test_config/

- `test_stations.py`:
  - 断言 `STATIONS["KLGA"]` 存在且 lat/lon 在合理范围
  - 断言 `slug_pattern` 可被 `re.compile()`

**验证**: `pytest tests/test_config/ -q`

---

## Phase 2 — 数据库层

> 目标：`db/` 包。SQLite schema + CRUD。

### Task 2.1 [创建] db/schema.py

类 `DbClient`:
- `__init__(self, db_path: Path)` — 连接 SQLite，`WAL` 模式
- `_create_tables()` — 仅在首次运行时建表
- 表：

```sql
-- stations: 站点列表（手动添加）
CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY,
    lat REAL, lon REAL, timezone TEXT
);

-- events: Gamma API 发现的事件
CREATE TABLE IF NOT EXISTS events (
    condition_id TEXT PRIMARY KEY,
    question TEXT,
    event_slug TEXT,
    event_end_date TEXT,
    station_id TEXT REFERENCES stations(station_id),
    status TEXT DEFAULT 'discovered'
);

-- markets: 事件下的各桶位
CREATE TABLE IF NOT EXISTS markets (
    token_id TEXT PRIMARY KEY,
    condition_id TEXT REFERENCES events(condition_id),
    market_slug TEXT,
    bucket_lo_f REAL,
    bucket_hi_f REAL,
    outcome TEXT
);

-- signals: 策略产生的信号
CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    token_id TEXT REFERENCES markets(token_id),
    mc_prob REAL,
    clob_price REAL,
    edge REAL,
    side TEXT
);

-- orders: 已提交订单
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    signal_id INTEGER REFERENCES signals(signal_id),
    status TEXT DEFAULT 'pending',
    size_usd REAL,
    price REAL,
    filled_at TEXT
);

-- model_versions: 模型版本注册
CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    checkpoint_path TEXT,
    train_mae REAL,
    test_mae REAL,
    created_at TEXT
);
```

CRUD 方法: `upsert_event()`, `upsert_market()`, `insert_signal()`, `insert_order()`,
`update_order_status()`, `get_active_events()`, `get_open_orders()` 等。

**参考**: `Herbie-main/mc_stage2_clob.py` 中的 market_slug 解析逻辑。
**验证**: `python3 -c "from db.schema import DbClient; db = DbClient(':memory:'); print('ok')"`

### Task 2.2 [测试] test_db/test_schema.py

- 建表不报错
- upsert_event 插入 + 更新（幂等性）
- 外键约束（向具有不存在 condition_id 的 markets 插入应失败）
- get_active_events 过滤正确

**验证**: `pytest tests/test_db/ -q`

---

## Phase 3 — 数据存储层

> 目标：`data/storage/` 包。Parquet 时序存储。

### Task 3.1 [创建] data/storage/parquet_store.py

类 `ParquetStore`:

```python
class ParquetStore:
    def __init__(self, base_dir: Path):
        """base_dir 下按 station_id/YYYY-QN.parquet 分区"""

    def append(self, station_id: str, df: pd.DataFrame) -> int:
        """去重追加。返回实际新增行数。"""

    def read(self, station_id: str,
             start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """读取日期范围内的数据，拼接多个季度文件。"""

    def _quarter_key(self, ts: pd.Timestamp) -> str:
        """返回 '2026-Q1' 格式的分区键。"""
```

去重逻辑: 按 `(init_time, fxx)` 或 `(valid_time,)` 主键去重。

**参考**: `Herbie-main/run_hrrr_cache.py` 输出的 `hrrr_klga_forecast_48h.parquet` schema。
**验证**: `pytest tests/test_data/test_parquet_store.py -q`

### Task 3.2 [创建] data/storage/cleaner.py

类 `Cleaner`:

- `clean_hrrr()`: Kelvin→℃、风速 m/s→kt、异常值裁剪、列名标准化
- `clean_iem()`: IEM METAR 原始字段清洗

**参考**: `Herbie-main/clean_iem.py`, `Herbie-main/parse_iem_raw.py`,
`Herbie-main/tft_step2_3_merge_features.py` 中的清洗逻辑。
**验证**: `pytest tests/test_data/test_cleaner.py -q`

### Task 3.3 [测试] test_data/

- `test_parquet_store.py`: 追加 + 去重 + 季度读取
- `test_cleaner.py`: Kelvin→℃ 精度、异常值裁剪边界

**验证**: `pytest tests/test_data/ -q`

---

## Phase 4 — HRRR 数据源

> 目标：`data/forecast/` 包。四级容灾 HRRR 数据获取（S3 → NOMADS → Zarr → Herbie）。

### Task 4.1 [创建] data/forecast/base.py

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseForecastSource(ABC):
    @abstractmethod
    def fetch(self, init_time: pd.Timestamp, fxx_range: range,
              station: "StationConfig") -> pd.DataFrame:
        """返回 DataFrame，列: init_time, fxx, valid_time, + 36个HRRR变量名。"""
```

### Task 4.2 [迁移] data/forecast/hrrr_s3.py

类 `HrrrS3Source(BaseForecastSource)`:

核心逻辑：
1. 构造 S3 URL: `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/hrrr.t{HH}z.wrfsfcf{FXX}.grib2`
2. 下载 `.idx` 文件解析各变量字节偏移
3. HTTP Range Request 下载目标变量 GRIB2 分片
4. `eccodes` 解码提取 KLGA 网格点值

**参考文件**: `Herbie-main/run_hrrr_cache.py`（完整实现）。
关键函数映射:
- `build_s3_url()` → `HrrrS3Source._build_url()`
- `parse_idx()` → `HrrrS3Source._parse_idx()`
- `fetch_range()` → `HrrrS3Source._fetch_range()`
- `decode_grib_value()` → `HrrrS3Source._decode_grib()`
- `_find_klga_index()` → `HrrrS3Source._find_grid_index()`

**验证**: `python3 -c "from data.forecast.hrrr_s3 import HrrrS3Source; print('import ok')"`

### Task 4.3 [迁移] data/forecast/hrrr_nomads.py

类 `HrrrNomadsSource(BaseForecastSource)`:

核心逻辑与 S3 完全相同（GRIB2 + idx 解析 + Range Request + eccodes 解码），
仅 URL 模板不同：

- S3: `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/hrrr.t{HH}z.wrfsfcf{FXX}.grib2`
- NOMADS: `https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.YYYYMMDD/conus/hrrr.t{HH}z.wrfsfcf{FXX}.grib2`

可考虑提取共享基类 `BaseGribSource`，将 URL 构造方法留为抽象。

NOMADS 特点：
- 官方 NOAA 源，时效性略优于 S3（S3 同步有延迟）
- 仅保留近 2-3 天数据，不适合历史回填
- 无并发限制，但带宽不如 S3

容灾顺序：S3（主）→ NOMADS（备）→ Zarr（第三）→ Herbie（第四）

**参考文件**: 复用 `Herbie-main/run_hrrr_cache.py` 的解析逻辑，仅替换 URL 前缀。
**验证**: `python3 -c "from data.forecast.hrrr_nomads import HrrrNomadsSource; print('ok')"`

### Task 4.4 [迁移] data/forecast/hrrr_zarr.py

类 `HrrrZarrSource(BaseForecastSource)`:

**参考文件**: `Herbie-main/run_hrrr_zarr.py`。
关键：按 Zarr chunk 读取，无需下载完整 GRIB。

### Task 4.5 [测试] test_data/test_hrrr_s3.py

- `.idx` 文件解析测试（mock HTTP 响应）
- Range Request 的 `bytes=start-end` header 格式正确
- GRIB 解码出的值在合理物理范围内（温度 200-330K）

**验证**: `pytest tests/test_data/test_hrrr_s3.py -q`

---

## Phase 5 — 观测数据源

> 目标：`data/obs/` 包。IEM（主）+ NOAA tgftp（备）。

### Task 5.1 [创建] data/obs/base.py

```python
class BaseObsSource(ABC):
    @abstractmethod
    def fetch(self, station: "StationConfig",
              start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """返回 DataFrame，列: valid_time + IEM 14列观测变量。"""
```

### Task 5.2 [迁移] data/obs/iem.py

类 `IemSource(BaseObsSource)`:

IEM REST API:
- 端点: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
- 参数: `station=KLGA&data=all&tz=Etc/UTC&format=comma&latlon=no`
- 解析 CSV 响应

**参考文件**: `Herbie-main/iem_download_klga.py`, `Herbie-main/clean_iem.py`。

### Task 5.3 [迁移] data/obs/noaa_file.py

类 `NoaaFileSource(BaseObsSource)`:

NOAA tgftp 文件:
- 端点: `https://tgftp.nws.noaa.gov/data/observations/metar/stations/KLGA.TXT`
- HEAD 请求获取 `Last-Modified`，GET 请求获取 METAR 原文
- 解析 METAR 提取温度

**参考文件**: `Herbie-main/combined_sniper.py`（NOAA task 部分）。

### Task 5.4 [测试] test_data/

- `test_iem.py`: mock IEM CSV 响应解析
- `test_cleaner.py`: 更新测试覆盖 IEM 清洗

**验证**: `pytest tests/test_data/ -q`

---

## Phase 6 — TFT 模型

> 目标：`model/` 包。加载检查点 + 推理 + MC 采样。

### Task 6.1 [迁移] model/tft.py

类 `TftModel`:

```python
class TftModel:
    def __init__(self, checkpoint_path: Path):
        """加载 pytorch-forecasting TFT 检查点"""

    def predict(self, encoder_df: pd.DataFrame,
                decoder_df: pd.DataFrame) -> dict:
        """返回 {fxx: {"p10": float, "p50": float, "p90": float}}"""
```

**参考文件**:
- `Herbie-main/tft_step10_eval.py`（推理流程）
- `Herbie-main/tft_step7_dataloader.py`（KNOWN_REALS/UNKNOWN_REALS 列定义）
- `Herbie-main/tft_step9_train.py`（超参、Kelvin 转换逻辑）

关键要点:
- `KNOWN_REALS`: 5 个日历特征 + 36 个 HRRR 变量
- `UNKNOWN_REALS`: 14 个 IEM 观测变量（仅 encoder 可见）
- `TARGET`: `temp_error`（残差预测，非直接温度）
- 输出分位数: P10, P50, P90
- encoder: local_idx 0..23 (fxx -23..0，24 步)
- decoder: local_idx 24..71 (fxx 1..48，48 步)

> **窗口几何说明**:
> `tft_step6_build_72h.py` 构建了 120 行/init (fxx -71..+48)，但
> `TimeSeriesDataSet` 中 `MAX_ENCODER_LENGTH = 24`，实际只消费
> 最近 24 行作为 encoder。fxx -71..-24 的 48 行历史数据
> 虽在数据集中存在，但不进入模型。

### Task 6.2 [迁移] model/mc.py

类 `McSampler`:

```python
class McSampler:
    def __init__(self, settings: Settings):
        self.n_samp = settings.mc_n_samp
        self.rng_seed = settings.mc_rng_seed
        self.rho = settings.mc_rho
        self.bias_corr_c = settings.mc_bias_corr_c

    def sample_daily_high(self, mu_vec: np.ndarray,
                          sigma_vec: np.ndarray) -> np.ndarray:
        """AR(1) MC 采样；返回 (n_samp,) 每日最高温度样本（°F）。"""

    def bucket_probs(self, daily_highs: np.ndarray,
                     buckets: list[tuple[float, float]]) -> dict:
        """返回 {(lo, hi): probability} 各桶概率。"""
```

**参考文件**: `Herbie-main/mc_helpers.py` (`ar1_mc_sample` 函数)。
核心参数: `BEST_RHO=0.90`, `BIAS_CORR_C=1.5206`, `N_SAMP=5000`, `RNG_SEED=42`。

### Task 6.3 [迁移] model/pipeline.py

类 `ModelPipeline`:

```python
class ModelPipeline:
    def __init__(self, tft: TftModel, mc: McSampler,
                 data_mgr: "DataManager"):
        pass

    def run(self, station: StationConfig,
            init_time: pd.Timestamp) -> dict:
        """
        完整流程：
        1. data_mgr.fetch_hrrr(init_time) -> 48小时预报
        2. data_mgr.fetch_iem(encoder 时段) -> 24小时观测
        3. 组装 encoder_df + decoder_df
        4. tft.predict() -> 各小时 p10/p50/p90
        5. mc.sample_daily_high() -> 每日高温分布
        6. mc.bucket_probs() -> 各桶概率
        返回: {"station": str, "init_time": ts,
               "probs": {(lo,hi): float}, "p50_high_f": float}
        """
```

### Task 6.4 [创建] model/store/registry.py

类 `ModelRegistry`:

```python
class ModelRegistry:
    def __init__(self, store_dir: Path, db: DbClient):
        pass

    def publish(self, version: str, ckpt_path: Path,
                metrics: dict) -> None:
        """复制检查点到 store_dir/version/，写入 meta.json，注册到 DB。"""

    def load_latest(self) -> TftModel:
        """读 DB 找最新版本，返回加载好的 TftModel。"""

    def rollback(self, version: str) -> None:
        """标记指定版本为 active。"""
```

### Task 6.5 [测试] test_model/

- `test_mc.py`:
  - `ar1_mc_sample` 输出 shape = (n_samp, n_hours)
  - 均值 ≈ mu_vec（误差 < 0.5°C）
  - `bucket_probs` 所有桶概率之和 ≈ 1.0
- `test_pipeline.py`: 全 mock 端到端（HRRR mock df → 返回 probs dict）

**验证**: `pytest tests/test_model/ -q`

---

## Phase 7 — CLOB 盘口 + 事件发现

> 目标：`data/clob/`、`events/` 包。

### Task 7.1 [创建] data/clob/orderbook.py

类 `OrderbookFetcher`:

```python
class OrderbookFetcher:
    def __init__(self, clob_client):
        """clob_client: py_clob_client.ClobClient 实例"""

    def get_book(self, token_id: str) -> dict:
        """返回 {"bids": [...], "asks": [...], "mid_price": float}"""
```

**参考**: `py-clob-client-main` SDK 的 `get_order_book()` 方法。
另见 `py-clob-client-main/build_synthetic_orderbook.py`（合成盘口重建逻辑）、
`py-clob-client-main/probe_prices_history.py`（CLOB 价格历史 API 用法）。

### Task 7.2 [迁移] events/discovery.py

类 `EventDiscovery`:

```python
class EventDiscovery:
    def __init__(self, db: DbClient, settings: Settings):
        pass

    def discover(self) -> list[dict]:
        """
        1. GET Gamma API 事件列表
        2. 按 slug_pattern 过滤
        3. 解析各桶 (lo_f, hi_f)
        4. upsert 到 DB
        返回新发现的事件列表。
        """
```

桶解析逻辑参考: `Herbie-main/mc_helpers.py` (`parse_slug` 函数)。
Gamma API 端点: `https://gamma-api.polymarket.com/events`
另见 `py-clob-client-main/list_feb20_markets.py`（Gamma API 事件发现实例）。

### Task 7.3 [创建] events/lifecycle.py

类 `EventLifecycle`:

```python
class EventLifecycle:
    """
    状态机: discovered -> active -> cutoff -> pending_settle -> settled
    """
    def advance(self, event: dict, now_utc: pd.Timestamp) -> str:
        """根据当前时间和事件结束日期推进状态。"""
```

- `cutoff`: event_end_date 当天 21:00 UTC (16:00 EST) 之后
- `pending_settle`: event_end_date 已过
- `settled`: 已结算（Relayer 或 0.99 抛售）

### Task 7.4 [测试]

- `test_discovery.py`: mock Gamma API JSON → 正确解析 slug → DB 记录
- `test_lifecycle.py`: 状态转换矩阵全覆盖

**验证**: `pytest tests/test_events/ -q`

---

## Phase 8 — 交易引擎

> 目标：`core/` 包（strategy, exchange, risk, positions）。

### Task 8.1 [创建] core/strategy/base.py + edge.py

`BaseStrategy` ABC + `EdgeStrategy`:

```python
class EdgeStrategy(BaseStrategy):
    def compute_signals(self, probs: dict, book: dict,
                        markets: list[dict]) -> list[Signal]:
        """
        对每个桶: edge = mc_prob - clob_price
        edge > min_edge → 生成 BUY 信号
        edge < -min_edge → 生成卖出信号（若持仓）
        """
```

**参考**: `Herbie-main/_sniper_backtest.py` 中的 edge 计算逻辑。
阈值: `min_edge = 0.04`（可配置）。

### Task 8.2 [创建] core/exchange/base.py + clob.py + paper.py

`BaseExchange` ABC:

```python
class BaseExchange(ABC):
    @abstractmethod
    def place_order(self, token_id: str, side: str,
                    size: float, price: float) -> str: ...
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...
    @abstractmethod
    def get_positions(self) -> list[dict]: ...
```

- `ClobExchange`: 调用 `py_clob_client` 发送真实订单
- `PaperExchange`: 读取真实盘口模拟成交，不发送订单

### Task 8.3 [创建] core/exchange/settlement.py

类 `Settler`:

```python
class Settler:
    def settle(self, event: dict, exchange: BaseExchange) -> str:
        """
        结算策略:
        - 如 Relayer gasless 结算可用 → 调用 Relayer
        - 否则以 0.99 挂卖单抛售
        """
```

### Task 8.4 [创建] core/risk/

- `BaseRiskModel` ABC: `approve_signal()`, `compute_size()`
- `FixedFraction`: 固定每笔 USDC 金额，`max_position_usd` / `max_total_usd` 上限

### Task 8.5 [创建] core/positions.py

类 `PositionTracker`:

```python
class PositionTracker:
    def __init__(self, db: DbClient):
        pass
    def open(self, signal: Signal, order: Order) -> None: ...
    def fill(self, order_id: str, filled_price: float) -> None: ...
    def close(self, order_id: str) -> None: ...
    def get_open(self) -> list[dict]: ...
```

### Task 8.6 [测试] test_core/

- `test_strategy.py`: EdgeStrategy 信号方向 + 阈值校验
- `test_exchange.py`: PaperExchange 模拟成交
- `test_positions.py`: open → fill → close 生命周期
- `test_risk.py`: 超过上限时拒绝信号

**验证**: `pytest tests/test_core/ -q`

---

## Phase 9 — Sniper（仅 AWC + NOAA）

> 目标：`snipers/` 包。只移植 `combined_sniper.py` 中的 AWC 和 NOAA 两个通道。
> **不移植**: `nwws_sniper.py`（NWWS-OI 不含有效数据）、`iem_sniper.py`（延迟太高）。

### Task 9.1 [创建] snipers/base.py

```python
class BaseSniperSource(ABC):
    @abstractmethod
    async def start(self, on_observation: Callable) -> None:
        """启动轮询循环，捕获新观测时调用 on_observation(metar_text, source, timestamp)。"""

    @abstractmethod
    async def stop(self) -> None:
        """优雅关闭。"""
```

### Task 9.2 [迁移] snipers/noaa.py

类 `NoaaSniper(BaseSniperSource)`:

逻辑源自 `combined_sniper.py` 的 `noaa_task()`:
- HEAD 轮询 `tgftp.nws.noaa.gov/data/observations/metar/stations/{STATION}.TXT`
- `Last-Modified` 变化 → GET 获取内容
- 空闲间隔: 3.0s
- 狙击窗口: xx:50:45 ~ xx:56:00，间隔 0.2s

```python
class NoaaSniper(BaseSniperSource):
    IDLE_INTERVAL = 3.0
    SNIPER_INTERVAL = 0.2
    SNIPER_START = (50, 45)   # (minute, second)
    SNIPER_END   = (56, 0)

    async def start(self, on_observation): ...
    async def stop(self): ...
```

**参考**: `Herbie-main/combined_sniper.py` 第 67-175 行。

### Task 9.3 [迁移] snipers/awc.py

类 `AwcSniper(BaseSniperSource)`:

逻辑源自 `combined_sniper.py` 的 `awc_task()`:
- GET 轮询 `aviationweather.gov/api/data/metar?ids={STATION}&format=raw&taf=false&hours=1`
- 通过 METAR 文本变化检测新观测
- 正常间隔: 2.0s
- 狙击窗口: xx:51:30 ~ xx:53:59，间隔 0.65s

```python
class AwcSniper(BaseSniperSource):
    NORMAL_INTERVAL = 2.0
    SNIPER_INTERVAL = 0.65
    SNIPER_START = (51, 30)
    SNIPER_END   = (53, 59)

    async def start(self, on_observation): ...
    async def stop(self): ...
```

**参考**: `Herbie-main/combined_sniper.py` 第 185-270 行。

### Task 9.4 [创建] snipers/manager.py

类 `SniperManager`:

```python
class SniperManager:
    def __init__(self, sources: list[BaseSniperSource]):
        self.seen_obs: set[str] = set()    # 去重
        self.queue: asyncio.Queue = asyncio.Queue()

    async def start_all(self) -> None:
        """asyncio.gather 并发启动所有源。"""

    def _on_observation(self, metar_text: str, source: str,
                        timestamp: datetime) -> None:
        """去重后放入队列。"""
```

### Task 9.5 [测试] test_snipers/test_manager.py

- mock 两个源同时报同一 METAR → 队列只出现一次
- 验证狙击窗口时间判断逻辑

**验证**: `pytest tests/test_snipers/ -q`

---

## Phase 10 — 主循环 + DataManager

> 目标：`loops/`、`data/manager.py`。将所有组件连接为运行时循环。

### Task 10.1 [创建] data/manager.py

类 `DataManager`:

```python
class DataManager:
    def __init__(self, forecast_sources: list[BaseForecastSource],
                 obs_sources: list[BaseObsSource],
                 forecast_store: ParquetStore,
                 obs_store: ParquetStore):
        pass

    def fetch_hrrr(self, init_time: pd.Timestamp,
                   station: StationConfig) -> pd.DataFrame:
        """按优先级尝试各源；成功后追加到 ParquetStore。"""

    def fetch_iem(self, station: StationConfig,
                  start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """IEM 优先，NOAA tgftp 备份；追加到 ParquetStore。"""
```

### Task 10.2 [创建] loops/forecast_loop.py

类 `ForecastLoop`:

```python
class ForecastLoop:
    """每小时触发一次（或新 init 可用时）。"""

    def __init__(self, pipeline: ModelPipeline,
                 strategy: BaseStrategy,
                 exchange: BaseExchange,
                 risk: BaseRiskModel,
                 positions: PositionTracker,
                 db: DbClient):
        pass

    async def tick(self) -> None:
        """
        1. 对每个活跃站点 + 活跃事件：
           a. pipeline.run() → probs
           b. orderbook.get_book() → book
           c. strategy.compute_signals() → signals
           d. risk.approve_signal() → approved
           e. exchange.place_order() → order_id
           f. positions.open()
        """
```

### Task 10.3 [创建] loops/sniper_loop.py

类 `SniperLoop`:

```python
class SniperLoop:
    """事件驱动：消费 SniperManager 队列，快速更新模型并下单。"""

    async def run(self) -> None:
        """
        while True:
            obs = await sniper_mgr.queue.get()
            # 用最新 obs 更新 encoder
            # 重新运行 pipeline → 信号 → FOK 订单
        """
```

### Task 10.4 [创建] loops/discovery_loop.py + settlement_loop.py

- `DiscoveryLoop`: 每日调用 `EventDiscovery.discover()`
- `SettlementLoop`: 每日盘后调用 `Settler.settle()` 处理到期事件

### Task 10.5 [创建] core/bot.py

类 `BotManager`:

```python
class BotManager:
    """顶层协调器。拥有所有子管理器，启动所有循环。"""

    def __init__(self, settings: Settings):
        # 初始化所有组件
        pass

    async def run(self) -> None:
        """asyncio.gather 启动 forecast_loop, sniper_loop, discovery_loop, settlement_loop。"""
```

### Task 10.6 [创建] main.py

```python
"""CLI 入口。argparse 解析 --mode paper|live, --station KLGA。"""
```

### Task 10.7 [测试]

- `test_forecast_loop.py`: mock 单周期运行，验证信号 → 订单流转

**验证**: `pytest tests/test_loops/ -q`

---

## Phase 11 — 回测引擎

> 目标：`backtest/` 包。

### Task 11.1 [迁移] backtest/engine.py

类 `BacktestEngine`:

```python
class BacktestEngine:
    def __init__(self, pipeline: ModelPipeline,
                 strategy: BaseStrategy,
                 store: ParquetStore):
        pass

    def run(self, start_date: str, end_date: str,
            station: StationConfig) -> pd.DataFrame:
        """
        遍历日期范围内所有 init_time：
        1. pipeline.run() → probs
        2. 从历史 CLOB 快照（Parquet/CSV）获取当时盘口
        3. strategy.compute_signals()
        4. 计算 PnL（含半点差扣除）
        返回逐信号 DataFrame。
        """
```

**参考**: `Herbie-main/_sniper_backtest.py`（完整回测逻辑）。
关键参数: `HALFSPREAD_DRAG = -1.27` pts。

### Task 11.2 [迁移] backtest/analysis.py

类 `BacktestAnalysis`:

```python
class BacktestAnalysis:
    def __init__(self, results_df: pd.DataFrame):
        pass

    def summary(self) -> dict:
        """ROI、Brier Score、胜率、最大回撤。"""

    def by_threshold(self, thresholds: list[float]) -> pd.DataFrame:
        """按 edge 阈值分组的 ROI/Brier。"""

    def by_lead_time(self) -> pd.DataFrame:
        """按提前量（init 到事件的小时数）分组。"""

    def by_init_hour(self) -> pd.DataFrame:
        """按 init UTC 小时分组。"""
```

**参考**: `Herbie-main/_sniper_analysis.py`（6 种分析输出）。

### Task 11.3 [测试]

- `test_engine.py`: 固定输入 probs + 固定盘口 → PnL 计算正确

**验证**: `pytest tests/test_backtest/ -q`

---

## Phase 12 — UI

> 目标：`ui/` 包。Streamlit Web 界面。

### Task 12.1 [创建] ui/app.py

Streamlit 入口。侧栏导航：Dashboard / Positions / Events / Backtest / Model。

### Task 12.2 [创建] ui/pages/

各页面类（读取 SQLite + ParquetStore 展示数据）：
- `dashboard.py` — 实时损益、各站状态
- `positions.py` — 订单历史、纸单/实盘切换
- `events.py` — 事件列表、生命周期状态、桶位价格
- `backtest.py` — 交互式回测控件
- `model.py` — 当前模型版本、各站 RMSE

**验证**: `streamlit run ui/app.py` 能启动（可手动验证）

---

## Phase 13 — 离线脚本

> 目标：`scripts/` 包。批量数据下载 + 模型训练。

### Task 13.1 [迁移] scripts/batch_download_zarr.py

类 `BatchZarrDownloader`:
- 批量下载历史 HRRR Zarr 数据
- 输出到 `ParquetStore`

**参考**: `Herbie-main/run_hrrr_zarr.py`。

### Task 13.2 [迁移] scripts/batch_download_iem.py

类 `BatchIemDownloader`:
- 批量下载 IEM 历史 METAR
- 输出到 `ParquetStore`

**参考**: `Herbie-main/iem_download_klga.py`。

### Task 13.3 [迁移] scripts/train_tft.py

类 `TftTrainer`:
- 读取 `ParquetStore` 数据
- 执行 TFT 训练（步骤对应 `tft_step5` ~ `tft_step9`）
- 发布到 `ModelRegistry`

**参考**: `Herbie-main/tft_step9_train.py`（训练入口）、`tft_step7_dataloader.py`（数据集构建）。

**验证**: `python3 -c "from scripts.train_tft import TftTrainer; print('import ok')"`

---

## Phase 14 — CI 和文档

> 目标：GitHub Actions + Copilot Instructions。

### Task 14.1 [创建] .github/workflows/ci.yml

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: sudo apt-get install -y libeccodes-dev
      - run: pip install -e .[dev]
      - run: pytest tests/ -q --tb=short
```

> 注意：Linux 上 eccodes 可通过 `apt` 安装，无需 conda。

### Task 14.2 [创建] .github/instructions/copilot-instructions.md

Agent 编码规则要点：
- 每个文件一个主类（除 config 和 main.py）
- 使用 ABC 基类定义接口
- 类型提示所有公开方法
- 无模块级散装函数
- 私有方法以 `_` 前缀
- 测试文件必须覆盖所有公共方法
- 目标系统 Linux，不使用 Windows 特有路径或命令

### Task 14.3 [创建] README.md

项目简介、快速开始（环境安装 + 运行命令）、架构概览链接。

**验证**: `pytest tests/ -q` 全绿

---

## 参考文件迁移映射表

| Herbie-main 源文件 | → PolyWeatherBot 目标 | Phase |
|---|---|---|
| `run_hrrr_cache.py` | `data/forecast/hrrr_s3.py` + `hrrr_nomads.py` | 4 |
| `run_hrrr_zarr.py` | `data/forecast/hrrr_zarr.py` | 4 |
| `iem_download_klga.py` | `data/obs/iem.py` + `scripts/batch_download_iem.py` | 5, 13 |
| `clean_iem.py` + `parse_iem_raw.py` | `data/storage/cleaner.py` | 3 |
| `mc_helpers.py` | `model/mc.py` | 6 |
| `mc_stage1_fix.py` + `mc_stage2_clob.py` | `model/pipeline.py` + `backtest/engine.py` | 6, 11 |
| `mc_daily_high.py` | `model/pipeline.py` | 6 |
| `tft_step5_dataset.py` | `scripts/train_tft.py` | 13 |
| `tft_step6_build_72h.py` | `scripts/train_tft.py` | 13 |
| `tft_step6b_coldstart_fill.py` | `scripts/train_tft.py` | 13 |
| `tft_step7_dataloader.py` | `scripts/train_tft.py` | 13 |
| `tft_step8_model.py` | `scripts/train_tft.py` | 13 |
| `tft_step9_train.py` | `scripts/train_tft.py` | 13 |
| `tft_step10_eval.py` | `model/tft.py` (推理部分) | 6 |
| `combined_sniper.py` (NOAA) | `snipers/noaa.py` | 9 |
| `combined_sniper.py` (AWC) | `snipers/awc.py` | 9 |
| `_sniper_backtest.py` | `backtest/engine.py` | 11 |
| `_sniper_analysis.py` | `backtest/analysis.py` | 11 |

### py-clob-client-main 参考映射

| py-clob-client-main 源文件 | 参考用途 | Phase |
|---|---|---|
| `list_feb20_markets.py` | `events/discovery.py` 的 Gamma API 调用方式 | 7 |
| `build_synthetic_orderbook.py` | `data/clob/orderbook.py` 的盘口数据重建逻辑 | 7 |
| `probe_prices_history.py` | `data/clob/orderbook.py` 的价格历史 API 用法 | 7 |
| `analyze_snipers.py` | `backtest/analysis.py` 的 sniper 窗口检测 | 11 |
| `analyze_lifecycle.py` | `backtest/analysis.py` 的市场生命周期分析 | 11 |

### 不迁移的文件

| 文件 | 原因 |
|---|---|
| `nwws_sniper.py` | NWWS-OI 不含有效温度数据 |
| `iem_sniper.py` | 延迟太高，IEM 只作为历史数据源 |
| `emwin_sniper.py` | 独立实验 |
| `swim_sniper.py` | SWIM 数据源不移植 |
| `hrrr_bias_correction.py` | CatBoost 实验，已废弃 |
| `lga_daily_high.py` | 依赖 herbie-data；已被 pipeline.py 替代 |
| `synoptic_push.py` | Synoptic 数据源不移植 |

---

## 校验清单

Agent 完成所有 Phase 后，执行以下最终验证：

```bash
# 1. 全量测试
pytest tests/ -q --tb=short

# 2. 导入验证
python3 -c "
from config.settings import Settings
from config.stations import STATIONS
from db.schema import DbClient
from data.forecast.hrrr_s3 import HrrrS3Source
from data.forecast.hrrr_nomads import HrrrNomadsSource
from data.obs.iem import IemSource
from model.tft import TftModel
from model.mc import McSampler
from model.pipeline import ModelPipeline
from snipers.manager import SniperManager
from core.bot import BotManager
print('All imports OK')
"

# 3. Paper 模式快速冒烟测试
python3 main.py --mode paper --station KLGA --dry-run
```
