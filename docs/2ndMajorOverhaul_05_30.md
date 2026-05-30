1. polymarket的API重大更新：我们需要把整个项目的相关模块按新API进行调整，参考：[Python SDK - Polymarket Documentation](https://docs.polymarket.com/dev-tooling/python)；
2. probability_store.py：quantiles_by_fxx 不是“桶级别”数据，所有的桶的这个值是相同的，没必要挨个桶保留，桶里只保留 mc_prob + last_tick_utc。把它从 BucketPayload 挪到 ProbabilitySnapshot（key 级）上。
3. 对于window_start 的定义，这里面包含了cold_start。我觉得这地方不对。这里有2个问题需要澄清：1、只有头部encoder才需要剔除cold_start, 后期等数据完整了就不用再剔除了，或者说我们可以等数据完整后才启动模型；2、拼接encoder窗口应该是个固定长度例如72的窗口，例如从-1~-72，cold_start是等待数据变完整的时间，不应该影响窗口长短。
4. 这个day_offset的逻辑具体是什么，如何控制的？最好举例说明。
5. 当前 books schema 有完整 L2 数据，但 strategy/execution 只用 mid，导致信号定价、真实成交价格、paper fill 语义都偏简化。而且，无论实际交易还是paper run，都应该沿 bids/asks 往后算 VWAP（Volume‑Weighted Average Price）;
6. 问题：EdgeStrategy 把 SELL YES 和 BUY NO 当成两个独立的 token 信号处理。SELL 操作需要已有持仓，但策略当前并不会检查持仓。策略不会比较 YES 的 bid 深度 vs NO 的 ask 深度／有效价格。方向：策略层只表达方向性错误定价（directional mispricing）；由执行/组合层决定是平掉已有仓位，还是开对立 token 的新仓。待定：短期规则是——如果已有 YES 仓位 → SELL YES，否则 → BUY NO；还是要实现一个基于 orderbook 的仲裁器（arbiter）。另外，当前 open_positions_by_token() 是按 token 数订单，不理解同一 bucket 下 YES/NO 的净风险。
7. 当前按 token 的重复下单保护机制过于粗糙，限制了策略的扩展性与执行效率：由于 `max_open_per_token=1` 会阻止合法的 scale‑in 行为，而 `open_positions_by_token` 统计的是订单数量而非真实敞口或方向性仓位，加上 TradingLoop 在每个 tick 内只快照一次 exposure 与 open_positions 而不会在成功提交订单后即时更新，导致策略与风控无法在同一 tick 内连续下单。改进方向是：当敞口限制允许时，应允许策略/风控层重复下单，并在同一 tick 内每次成功下单后立即更新本地风险状态。
8. paper run的功能过于简单，甚至都没用DB，没有起到分析作用。
9. **SELL 语义还不完整**：EdgeStrategy 会生成 SELL 信号，但 FixedFraction 对 SELL 和 BUY 一样按新仓 sizing，PositionTracker 也没有真正的 position lot / inventory。PaperExchange 的 SELL 只是从内存 position 减 size_usd，但 DB 记录上仍是一个普通 order。
10. 系统需要增加一个pnl模块（主要给生产用，backtest不用），无论实际run还是paper run都应该统一用pnl模块用来分析，然后用dashboard来展示。“订单/成交语义”需要统一，例如**size_usd 语义有潜在大问题**，**TradingLoop 不记录 signal，交易归因断裂**，建议增加 PnL 纯计算模块，先基于现有 orders 表，写测试覆盖 settled realized PnL、open unrealized PnL、paper fill等，再引入 fills 表，支持 partial fill/live sync，最后改 dashboard 和 positions page 使用统一 PnL 模块。
11. **SniperLoop 和 TradingLoop 行为不一致**：SniperLoop 直接用 get_active_events()，绕过 get_tradeable_events()、staleness gate、open_positions 风控参数，且下单后也只 record_order()。如果 sniper 是实盘组件，它现在和主 trading 风控/生命周期不统一。
12. 运行时的可观测性仍然不足：虽然系统已经有了 alert bus，但交易核心缺少关键的可观测事件，包括订单提交事件、成交事件、取消/拒单事件、敞口快照、PnL 快照以及 tick 级别的决策日志；在缺少这些基础信号的情况下，一旦出现问题，就很难回答“为什么这笔单下了 / 没下 / 为什么亏了”这样的核心追责问题。
13. _infer_station() 比较脆。它只看 slug 里有没有 station id，否则默认第一个 station。多站点以后，这里容易把 Chicago/Miami 的 event 错归到 KLGA。建议做一个配置层映射或者在 config/stations.py 给每个 station 加 aliases。
14. _process_markets() 对 token_ids 和 outcomes 用 zip()，长度不一致时会静默丢 token。_sync_missing_markets() 反而有 fallback。两个方法行为不完全一致。
15. parse_slug() 没匹配时仍会把 market 写入 DB，只是 bucket 为 None, None。这对“不认识的新 slug 格式”比较宽容，但后面的 strategy/settlement 可能会跳过或行为异常。这里最好将来加更明显的 alert 或 discovery quality 状态。
16. **生命周期没有真正串起来**：**规则存在，但状态推进没有被系统性执行**。EventLifecycle.advance() 已经存在，但没有看到任何 loop 周期性调用它并写回 DB。也就是说，发现到的 event 可能一直停在 discovered。不需要cutoff状态。**缺少 transition logging**。
17. EventDiscovery.discover(params=None) 支持传 query params，DiscoveryLoop.run() 现在没有传：`{"active": "true"}`，这会导致它可能拉过宽的 Gamma events，依赖本地 slug 过滤兜底。现在还行，但以后事件多了会浪费，也可能拉到历史/关闭市场。
18. **未识别 slug 仍然入库**：现在 parse_slug() 失败时：然后仍然 upsert_market()。这个策略宽容，但有风险：下游 strategy/settlement 可能遇到 bucket 为 NULL 的 token。
19. **缺少 discovery 结果摘要**：DiscoveryLoop 现在只 log “completed”，没有记录本轮发现了几个 event、同步了几个 market、跳过几个、解析失败几个。
20. **slug filter 太粗**：建议从单字符串升级成 patterns/aliases：slug_patterns = ["temperature", "daily-high", "high-temp"]。
21. TFT 推理阶段的模式对齐：避免有害的零填充。保持 encoder/decoder 的 schema 对齐，但不要再使用基于并集的盲目零填充作为默认语义策略。最佳方案是：将实时推理帧对齐到**训练数据集的精确特征契约**，只填充那些在实时推理中本来就不存在的列；对于缺失的、必须存在的已知未来特征，应快速报错，而不是静默变成 `0.0`。
22. **station 参数没有按 event 使用**：ForecastLoop 初始化时只传了一个 self.station，而 event 本身有 station_id。如果未来同一套 bot 跑多个站点，现在 pipeline.run(self.station, ...) 会把所有 event 都用同一个 station 预测。  更稳的是按 event["station_id"] 找对应 StationConfig，再传给 pipeline。
23. **init_time 返回值没有统一成 UTC**：如果传入的是 naive 或非 UTC，返回值和实际计算用的时间不一致。建议后期统一返回 init_utc。
24. **bucket 时间目标可能不精确**：现在 pipeline 永远基于未来 48h 的最高温，但 Polymarket event 是某个 event_end_date 的日最高温。对于 day_offset=0/1/2，不一定应该都用完整未来 48h 的 max。更准确的做法是：按 event 的目标日、本地时区/UTC 边界，筛选对应 fxx 小时，再对那部分取 max。 
25. **decoder fallback 可能掩盖数据失败**：如果 hrrr_df.empty，代码会造一个只有 fxx/init_time/valid_time 的 decoder。这样很多 HRRR known reals 后面可能被补 0，模型会用极差输入继续推理。  实盘里更合理是 fail fast：HRRR decoder 不完整就跳过该 event，不要产概率。
26. **特征契约分散**：pipeline.py 自己补时间特征，tft.py 定 KNOWN_REALS/UNKNOWN_REALS，训练脚本又有一份类似列表。长期容易漂移。  建议后期把“训练/推理共同 schema”集中到一个模块，比如 model/features.py 或 model/schema.py。
27. `build_encoder_obs_window()` 本该只负责构造模型输入窗口，也就是严格取 `[init_time - 72h, init_time - 1h]`。但现在它把“历史不足 72 小时无法预测”这种 readiness 问题，变成了“每次都多读 72 小时再丢掉前 72 行”，导致正常滚动窗口被人为往前扩，最终取到更早的数据，语义被扭曲。更合理的行为是：DataManager 只读取 `[init_time - n_hours, init_time - 1h]`；HRRR 要求窗口内每个目标小时都有 TMP_2m；obs 要求每小时附近有观测（继续用 20 分钟 tolerance）；若不完整就抛出明确错误（如 incomplete HRRR/obs window）。`cold_start_hours` 仍可留给 backtest engine 用来跳过起始阶段，但不应在 live encoder 构造时由 DataManager 强制使用。
28. hrrr_fill.py：对 non-MAJOR init 来说，最后几小时，是无法用“已经存在的更早 MAJOR run”严格按 valid_time 补齐的。建议：nearest major 按 valid_time 对齐，对无法覆盖的 tail，用最近可用小时做短时平滑外推。而且，建议打上标识，目的在于对MC算最高温时把最后平滑的数据剔除。
29. data/forecast/_grib_source.py (line 132) 的 .idx 匹配太宽松：有静默误匹配风险。比如 search 是 "TMP:2 m above ground"，理论上可能匹配到类似 "MAXTMP:2 m above ground" 的行；"APCP:surface" 也可能匹配不同累计窗口的 APCP。更危险的是，如果同一个 colname 匹配到多行，后面并发下载时用 downloaded[colname] = ...，最终哪个覆盖哪个取决于线程完成顺序，结果会不确定。建议：.idx 应结构化解析字段，按 VAR + LEVEL + forecast descriptor 精确匹配；同一个目标列只允许一个 match，重复时要 deterministic 选择或直接 warning/error。
30. data/forecast/_grib_source.py (line 169) 接受 HTTP 200 作为 Range 成功 Range request 正常应该返回 206 Partial Content。如果服务端忽略 Range 返回 200 整个 GRIB 文件，这里会把整文件当成某个变量的 message 解码，可能读到第一个 GRIB message，然后被错误写进当前 colname。建议：默认只接受 206；如果保留 200，必须验证返回大小/Content-Range，并在解码时验证 GRIB metadata 确实对应 colname。
31. data/forecast/_grib_source.py (line 131) 最后一条 GRIB message 的 end = off + 5_000_000 是硬编码 。如果目标变量刚好是 .idx 最后一条，且 message 超过 5MB，会被截断，导致解码失败或数据不完整。虽然不一定常见，但这是脆弱点。建议：对最后一条用 open-ended Range：bytes={start}-，或先拿 Content-Length 后算真实 end。
32. data/forecast/_grib_source.py (line 155) _fetch_range() 失败没有日志上下文 ：  _fetch_idx() 最终会 log URL，但 _fetch_range() 三次失败后直接 None。上层只知道某列 NaN，不知道 URL、byte range、status code、异常原因。线上排障会很痛。建议：最后一次失败时 warning，带 grib_url/start/end/status/exception；单次 transient 可以继续静默或 debug。
33. hrrr_fill.py (line 100) 输出列只取 non_major_df.columns：如果 non_major_df 因为短窗口/下载问题缺某些 HRRR 列，而 nearest_major_df 里有这些列，输出会直接丢掉这些列。虽然 _grib_source._fetch_all_stations() 当前会补齐 GRIB_TO_COL，但 HrrrFill 自身不健壮。
34. hrrr_fill.py (line 89) _vt_index() 对重复 valid_time 静默覆盖 ，如果输入 DataFrame 有重复 valid_time，后面的 row 会静默覆盖前面的 row，没有 warning，也没有排序规则。这会让结果取决于输入行顺序。建议：进入 fill 前按 valid_time 排序并 drop_duplicates，规则明确，比如 keep first 或 keep last，并在发现重复时 warning。
35. hrrr_fill.py (line 88) 注释说 “rounded to second”，实际没有 round，没有 round/floor 到 second。  如果来源里一个是 03:00:00，另一个是 03:00:00.000001，就匹配不上。HRRR 通常是整点，所以不一定马上炸，但注释和实现不一致。建议：要么改注释，要么真的 .dt.round("s") / .dt.floor("s")。我更倾向于显式 floor/round 到秒，保证 splice 稳定。
36.  hrrr_s3_bucket、hrrr_nomads_base：配置项没有被使用：[config/settings.py] 里有类似 hrrr_s3_bucket、hrrr_nomads_base，但 hrrr_s3.py (line 12) 和 hrrr_nomads.py (line 12) 直接硬编码。
37. NOMADS 没有自己的 URL 单测 ：当前测试只覆盖了 HrrrS3Source._build_url()。HrrrNomadsSource._build_url() 没看到对应测试。这个文件虽然简单，但 NOMADS 是 fallback source，URL 拼错会让 fallback 形同虚设。
38. hrrr_zarr.py:**大量异常被静默吞掉，会生成看似成功但实际全 NaN 的数据**，在 hrrr_zarr.py (line 110) 和 hrrr_zarr.py (line 121)，每个变量读取失败都会直接填 NaN，问题是没有任何 log，也没有失败计数。batch 下载时如果 Zarr path 变了、网络挂了、row/col 越界了，最后可能写入一批大面积 NaN 数据，但脚本看起来“成功”。建议：至少记录 warning，包含 init_time、station_id、col、zarr_path、url_anl/url_fcst。如果关键变量如 TMP_2m 失败，应直接 raise，让 batch 失败而不是污染 store。
39.  hrrr_zarr.py: **n_fcst_steps 探测失败后 fallback 可能掩盖真实错误**hrrr_zarr.py (line 98)：建议：TMP_2m shape 探测失败应该 raise，或者至少在 batch 模式下 fail-fast。
40.  hrrr_zarr.py:  **非 major run 不做 HrrrFill，和其他 source 行为不一致**hrrr_zarr.py (line 131) 会按请求的 fxx_range 返回行。对于 non-MAJOR，如果请求 range(0,49)，超过 n_fcst_steps 的部分会填 NaN。这和 _grib_source.py 的 non-MAJOR 行为不同。GRIB source 会尝试 splice，而 Zarr source 不会。 建议：复用 HrrrFill，让行为和 GRIB source 对齐。
41. hrrr_zarr.py: **station grid index fallback 到 KLGA 默认值有误用风险**，如果某个新 station 缺配置，会静默用 KLGA 的 grid point。这个比直接报错危险，因为会拿到“看起来合理但属于错站”的天气。建议：去掉 fallback。station，缺 hrrr_grid_row_zarr / hrrr_grid_col_zarr 时直接 raise。
42. hrrr_zarr.py: **没有 row/col 边界校验**。如果 station 配置的 row/col 错了，现在会在每个变量读取时 exception，然后大量填 NaN。  建议：打开一个关键变量后先检查 shape，验证，失败就 raise 清晰错误。
43. hrrr_zarr.py: **DATA_COLS 和 ZARR_VARS 缺少一致性保护** 。 输出循环用 DATA_COLS (line 124)，读取循环用 ZARR_VARS。现在看起来基本一致，但没有测试保证。未来如果 DATA_COLS 新增变量，而 ZARR_VARS 忘了加，输出会静默 NaN。建议：加测试或运行时断言。
44. hrrr_zarr.py: **配置项没有被使用**。config/settings.py 里有 hrrr_zarr_bucket，但这里硬编码，这和 S3/NOMADS 那两个文件类似：配置看起来存在，但 source 不读配置。建议：允许构造函数注入 base_url，默认保持当前值。
45. parquet_store.py：去重 key 没有规范化时间类型，可能写入逻辑重复行。用 tuple 直接比较 existing/new key。如果第一次写入 valid_time 是 Timestamp('2026-01-01 00:00Z')，第二次是字符串 "2026-01-01 00:00:00+00:00"，逻辑上同一时刻，但 tuple 可能不相等，导致重复写入。HRRR 的 init_time 也有同样风险。建议 append 前把 dedup key 中的时间列统一 pd.to_datetime(..., utc=True)。
46. parquet_store.py：**read() 会创建目录，有读操作副作用。**  parquet_store.py (line 72) 调 _station_dir()，而 _station_dir() 在 104-107 (line 104) 总是 mkdir。所以读一个不存在的 station 也会创建空目录。这不影响功能，但存储目录会被“查询”污染。建议拆成 _station_path() 和 _ensure_station_dir()：读只看路径，写才创建。
47. parquet_store.py：**缺少 schema/key 显式校验，错误会晚爆且不友好。** _pick_dedup_key() 在 109-112 (line 109) 如果不是 HRRR，就默认 obs key。传入没有 valid_time 的坏 DataFrame，会在后面 df[ts_col] 或 new_rows[dedup_key] 抛 KeyError。建议显式判断：既不是 HRRR，也没有 valid_time 时 raise ValueError("ParquetStore requires ...")。
48. parquet_store.py：**写文件不是原子操作，也没有锁。** 直接 to_parquet(path) 覆盖。如果两个进程同时写同一个 station/quarter 文件，可能丢数据或写坏文件。单 bot 顺序执行没问题；如果会开多个 bot、回填脚本和 live 同时跑，建议至少加临时文件 + atomic replace，更进一步加文件锁。
49. iem.py：1. **:51 观测没有真正改写成代表的整点时间**，当前 Cleaner.clean_iem() 只是用 _rounded 去重，但最终 valid_time 仍保留原始 13:51。  这会导致 ParquetStore.read(start=14:00, end=...) 读不到本应代表 14:00 的 13:51 数据。建议改成：清洗后直接把 valid_time 设为最近整点。
50. iem.py： **IEM tick fallback 到 NOAA 时基本不可用**：IEM source 会用 now_utc 计算 H-1 :51 窗口，但 NoaaFileSource.fetch() 完全忽略 now_utc，只用原 start=end=now_utc 做过滤。所以 IEM 失败后，NOAA 最新 METAR 比如 13:51，会被 start=end=14:00 排除掉。这个 fallback 逻辑需要修：NOAA 在 now_utc 存在时也要使用同样的 tick 窗口。
51. iem.py： **skyc1 缺失/未知代码没有做 ffill().fillna(0)**
52. iem.py： **IEM 风向缺失/VRB 被提前当成 0° 计算 U/V**， **NOAA fallback 也有同样的 VRB/缺失风向问题**。
53. iem.py： **build_features_iem() 会把 U/V 的 NaN 直接填 0，可能掩盖问题**， 如果前面没有先做 ffill/bfill，这里会把缺测风直接当静风。  建议把 U/V 的时序补值放到 clean_iem()，build_features_iem() 只负责派生 obs_WS/obs_WD_*。
54. iem.py：7. **完整性检查太弱，只看温度** DataManager._obs_is_complete()，所以 IEM 即使风、云、降水等关键列大量缺失，只要温度有值，就不会切到 NOAA fallback。  如果目标是“数据不完整也触发第二源”，这里需要扩展完整性规则，至少检查核心列：温度、风速或 U/V、skyc1、p01i 等。
55. noaa_file.py： **NOAA 不能在大时间段请求里当成完整 fallback**，NoaaFileSource 只有最新单文件，最多 1 条 METAR。现在如果 IEM 挂了，而调用方请求一个较大窗口，比如 72 小时，只要 NOAA 最新 METAR 落在这个窗口里，就可能返回 1 行，然后 DataManager._obs_is_complete() 只看温度非空，会把它当成功。这不对。建议：NOAA 只用于 now_utc tick fallback，不用于历史/range fallback。
56. noaa_file.py： **_obs_is_complete() 仍然太弱**，这会让“只有 1 条温度”的 NOAA 数据通过完整性检查。
57. cleaner.py： **Cleaner.clean_iem() 应该先按时间排序，再做 ffill/bfill**；
58. **build_encoder_obs_window() 缺关键列时可能静默返回坏数据**：如果 TMP_2m 或 obs 温度列缺失，它不会报错，而是继续，建议这里强制检查：hrrr_raw 必须有 valid_time, TMP_2m，obs_raw 必须有 valid_time, raw_tmpc_tenths，merged 必须生成 temp_error，否则 RuntimeError；
59. **build_encoder_obs_window() 会直接修改 store 返回的 DataFrame**：如果 store 返回的是共享引用、mock 返回对象、或者未来有缓存，这种原地修改可能有副作用。建议读出后马上 .copy()；
60. /model/tft.py:  **decoder_df 必须强制校验 fxx=1..48**，如果缺了某个 fxx，result[fxx] 和预测数组行号可能错位。这个应该直接 raise ValueError；
61.  /model/tft.py:   **TMP_2m 缺失或 NaN 不应该默默填 0**，这很危险。因为 TFT 输出是 temp_error 残差，必须加回 HRRR TMP_2m 才是绝对温度。如果 TMP_2m 缺失，返回的就会变成“残差伪装成温度”。建议改成：decoder_df 必须有 TMP_2m， TMP_2m 必须 48 行全非空，否则报错；
62.  /model/tft.py:  **encoder_df / decoder_df 行数没有校验**，  **模型输出 shape 没有校验**；
63.  /model/tft.py:   **Pipeline 的“缺列补 0”会掩盖模型输入问题**，虽然这个在 pipeline.py，但会影响 TftModel，建议在 TftModel.predict() 里增加输入列校验；
64.  /model/tft.py: **predict() 应该保证输入按时间顺序**，建议在进入 TimeSeriesDataSet 前显式排序：encoder 按 time_idx 或 valid_time，decoder 按 fxx / time_idx；
65.  model/mc.py： **fit_cov() 是死功能，经验协方差没有接入生产链路**，需要新增一种接入方式，例如：ModelRegistry 加载模型时同时加载 covariance.npy，然后 mc.fit_cov 或直接 set_cov；
66. model/mc.py： **fit_cov() 没有输入校验**， _mc_sample() 没有校验 mu_vec 和 sigma_vec ，  bucket_probs() 对空样本会除以 0 ， bucket_probs() 不检查 buckets 是否覆盖/重叠；
67. model/mc.py： **core/bot.py 里构造方式疑似是 bug**，会把整个 Settings 对象当成 n_samp； 实际 obs source 配置也可能不符合 fallback 预期：没有 NoaaFileSource()，所以即使我们修了 NOAA fallback，Bot 默认也没用上。
68. model/pipeline.py：**tft=None 没有保护**：要么 core/bot.py 必须加载真实 TFT，要么 ModelPipeline 明确支持 MC-only/fallback 模式。现在两边语义不一致。
69. model/pipeline.py：**encoder/decoder 长度没有在 pipeline 层校验** ，建议在 tft.predict() 前加硬校验，**quantiles 没有校验**，**sigma 公式需要防御异常分位数**，**返回的 init_time 应该统一用 init_utc**；
70. model/pipeline.py：**decoder 从 hrrr_df 筛出来后没有排序**：如果 hrrr_df 顺序乱，decoder 也乱。  _add_model_features() 会按当前行顺序生成 time_idx=72..119，这可能导致 time_idx 和 fxx 对不上。
71. model/pipeline.py：_add_model_features() 遇到缺 valid_time 时填 0 太宽松：正常 encoder/decoder 都必须有 valid_time。现在缺了就给 calendar 特征填 0，这会隐藏上游坏数据。建议改成：缺 valid_time 直接报错。
72. forecast_loop.py：1.构造 buckets 时最好过滤/记录无效 bucket，比如 bucket_lo_f/bucket_hi_f 是 None。现在 discovery 解析失败会写入 (None, None)，后面 mc.bucket_probs() 会炸，然后整个 event 被跳过。2.如果 result["probs"] 是空 dict，现在会写一个空 snapshot；ProbabilityStore.read() 对空 snapshot 会 StopIteration。建议 bucket_map 为空时不要 write，打 warning 后跳过。3.run() 没包住 tick() 的顶层异常。单个 pipeline/write 已经 catch 了，但如果 db.get_forecast_events() 或别的 tick 顶层逻辑抛错，整个长期协程会退出。建议 run() 里 catch 普通 Exception，记录日志后继续睡到下个小时。
73. 没有实现redeem，可根据新polymarket进行实现。
