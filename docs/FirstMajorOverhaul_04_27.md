# 0. 背景
本工程PolyWeatherBot是基于两个原始工程的合并，详见文档“IMPLEMENTATION_PLAN.md”。
合并后发现很多问题，有的是整体设计问题，有的是局部功能性问题，有的是原始工程遗留问题。
本文是总纲领，所有涉及到的修改意见已整理成相应的文档附于此文件下。
请对所有材料进行梳理，先不要动代码或者给出代码示例，先形成一个可供vscode-copilot大模型分步执行的指导性文件。要求此文件务必做到步骤清晰、明确，尤其是问题的解决要有针对性地逐个实施并给出改正后的测试方案（将来形成测试代码）。

# 1 整体设计
目前已完成的整体框架设计得很不错，各模块各司其职，解耦性好，不需要大改动。
## 1.1 **目标生产流程：** <font color="#00b050"> ✓</font><font color="#808080"> [→L4: forecast_loop, trading_loop[NEW], probability_store[NEW]]</font>
生产端时间tick最小间隔是5分钟（或可设置），这5分钟执行以下操作：获取市场、获取订单薄、根据TFT+MC预测值（只有最近1小时的最多3天的市场桶概率）得到概率、策略判断、风险判断、下单操作等；
每隔1小时，做以下操作：获取天气数据（包括当前小时真值以及基于当前小时的未来预测数据），其中，数据分别预处理（包括特征筛选、温度、风向转换，真值解析、18小时HRRR补偿至48小时操作等等），并按1小时间隔保存（这里仅保存2部分数据，分别是真值、历史预测f01）。TFT拼装数据后（72小时历史+48小时未来）放入模型，模型输出结果送MC采样，采样结果按事件x天拆分，分别获取最多3天（由于48小时）的市场桶概率，交给上面5分钟流程处理。
<font color="#00b050">注意：目前工程以预测为主，已经实现的sniper流程为备选，可配置执行。</font>

## 1.2 **目标训练流程：** <font color="#00b050"> ✓</font><font color="#808080"> [→L1: settings(train/val/test_start/end)]</font>
与生产分离。训练可沿用现在的脚本下载。历史数据时间范围：UTC 2024年10月1日~2026年04月10日。在此基础上划分train、val、test。
针对每个时间点（小时），数据应包含以下内容：
- 过去72小时的每个小时的f01预测数据；
- 过去72小时的每个小时的真值数据；
- 未来48小时的预测数据；

← 过去72h（encoder）──────────┬── 未来48h（decoder）──────────→
			每小时一行：                  │      每小时一行：
		  -   f01 HRRR预报值              │        - HRRR全量预报值（fxx=1..48）
          -       IEM实测值                   │        - IEM实测值（作为target）

可扩展至多个节点，例如NYC, Toronto, Atlanta, Chicago, Seattle, Dallas, Miami

## 1.3 关于paper run： <font color="#00b050"> ✓</font><font color="#808080"> [→L5: strategy/base, edge, multi[NEW], risk/base, fixed_fraction, kelly[NEW]]</font>
- 对于paper run，需要更丰富的策略、风险控制、更广泛的交易时间等一起测试；
# 2 模块问题
各个有问题的模块如下（按字母排序），未列出的暂无问题：

## 2.1 backtest模块
- analisys.py：无；
- engine.py：是否有对于"cold-start 零值问题"的处理模块？具体地，由于数据不可能从一开始就能回测，我们可以放弃最初的一部分数据，之前代码是额外下载，这里不用，因为我们数据量够大，放弃开头一部分即可； <font color="#00b050"> ✓</font><font color="#808080"> [→L7: backtest/engine(ADD cold-start guard)]</font>
## 2.2 core模块
### 2.2.1 exchange子模块
- clob：加鉴权功能。 <font color="#00b050"> ✓</font><font color="#808080"> [→L5: core/bot(authentication wire-up)]</font>
- paper：不用mid-prices做交易，用真实CLob <font color="#00b050"> ✓</font><font color="#808080"> [→L5: core/exchange/paper]</font>
- settlement：这个模块目前没完成，需要进一步了解**Relayer 路径**的实现方式；<font color="#ffc000"> ⏸ deferred — ADR-0007; tasks·L8·settlement_loop[NEW]</font>
### 2.2.2 risk子模块
- fixed_fraction： 整个系统没有任何地方对单个 token/market 的重复下单做限制，只控制总下单金额和每次下单金额； <font color="#00b050"> ✓</font><font color="#808080"> [→L5: core/risk/fixed_fraction, kelly[NEW]]</font>
### 2.2.3 strategy子模块
- edge：策略非常不完善，目前用mid_price与MC概率的差做edge，过于简单，后期需要修改；<font color="#ffc000"> ⏸ deferred — tasks·L5·strategy/edge(stub+NO tokens), multi[NEW]</font>
### 2.2.4 bot
- 默认 paper 模式下 [DataManager]没有挂载任何真实数据源; <font color="#00b050"> ✓</font><font color="#808080"> [→L5: core/bot]</font>
## 2.3 data模块
### 2.3.1 clob子模块
- orderbook：暂无问题；
### 2.3.2 forecast子模块
_grib_source：1、没有用到station变量，我们需要对多个站点，你在BaseGribSource的init里明确指定KLGA（klga_idx）是不对的，而且，在后续下载、解码等过程中要体现多站点的情况，例如一次可获取到指定的多个站点的某个属性等等；2、这里_fetch_fxx有个问题，就是一天24小时内只有4个具有48小时预报时长，其他只有18小时，需要在这里或者[cleaner.py]做补全：之前采用的是拼接的方式补全，可参考之前代码，补偿逻辑建议封装成独立的类。3、整个文件容错做得不够，例如_fetch_fxx里，_fetch_idx 函数无法取到数据该如何处理？不能只填NAN了事，至少需要一方面换源，另一方面做log；同理，_fetch_range 函数也是，如果下载出现问题如何处置？解码 GRIB2 出现问题怎么办？觉得所有这些涉及到下载数据、分析数据、整合数据的地方都需要做容错，并且记录、通知。换源的目的不是让我选择哪个源，而是发现有问题时能自动切换并且解决问题，这才是换源的目的。4、数据的完整性处理：若HRRR的预报初始化（init_time）出现某些时次缺失（服务器超时、网络中断等），导致时间序列出现空洞，是否有处理机制（原始工程step0工作）；数据填充（把 MAJOR hours（0/6/12/18z）的 HRRR 场展开填充至每小时，原始step1工作）； <font color="#00b050"> ✓</font><font color="#808080"> [→L2: _grib_source(batch multi-stn, HrrrFill, fallback, gap-detect, MAJOR expand); hrrr_fill[NEW]]</font>
### 2.3.3 obs子模块
- iem：1、不能下载一个时段内数据，而只下载一个时间点数据即f00，整点数据一般是51分发出，但一般到达时间会晚3分钟多，这地方需要做分析判断；2、model需要的历史数据是由多个之前下载的及当前整点的数据拼成的，而不是每次都重新下载的；2、 数据源没起到互补作用。第二源的目的是当第一源出现问题，例如断网、数据不全等情况下自动切换为第二源并解决上述问题，从而可以返回正确数据的目的。目前仅在数据全为空时启用第二数据源，能否改为更优化的方案，尽量把数据下载全。3、 警告不够细致全面，例如：extract_temp_c 函数，当找不到高精度数据时应打印log及警告，因为一般情况都会有高精度数据。 <font color="#00b050"> ✓</font><font color="#808080"> [→L2: data/obs/iem]</font>
- noaa_file：1、与iem数据类似，只获取整点文件数据，往期不可能获取到。而且整点数据一般在51分放出，一般会有2分钟以上延迟。2、对数据解析容错度不够，例如：有些 METAR 会出现多个数字段等情况，或者数据不全等等，这些都需要做预案；2、告警不够细致、全面，解析异常、数据异常等都需产生告警并log。 <font color="#00b050"> ✓</font><font color="#808080"> [→L2: data/obs/noaa_file]</font>
### 2.3.4 storge子模块
- cleaner：目前是孤立的，没有插入数据流中； <font color="#00b050"> ✓</font><font color="#808080"> [→L2: data/storage/cleaner]</font>
- parquet_store：HRRR只保存f01数据即可，IEM不变； <font color="#00b050"> ✓</font><font color="#808080"> [→L2: data/storage/parquet_store]</font>
### 2.3.5 manager
- 注意是否有开尔文 → 摄氏度转换问题？HRRR温度数据单位是K，IEM的是F，高精度的是C吧，这个确认下，或存在问题； <font color="#00b050"> ✓</font><font color="#808080"> [→L2: cleaner(K→C/F→C conv)]</font>
- 所有数据二次处理放哪里了？都在[cleaner.py]吗？全面吗？可参考step9文件，包括HRRR 风场合并、观测风场合并、重尾变量 log1p 平滑、计算残差目标列等；还有那个iem.py里的_parse_csv函数、noaa_file.py里的_parse_txt等； <font color="#00b050"> ✓</font><font color="#808080"> [→L2: cleaner(feature engineering), data/manager]</font>
- 一定要排除cold-start部分，舍去数据，不然temp_error≈ IEM实测值不合理！ <font color="#00b050"> ✓</font><font color="#808080"> [→L1: settings(cold_start_hours); L2: cleaner(DROP rows); L7: backtest/engine]</font>
- 最后的target 'temp_error' 是否应该加到数据列中，所有的二次处理完的数据是否都应保存，替换掉原始数据； <font color="#00b050"> ✓</font><font color="#808080"> [→L2: cleaner(SAVE processed output)]</font>
- 原始Herbie数据存在偏差问题（由于时间覆盖小），处理时采用了消除偏差以及硬性扩大P10~P90覆盖范围做法，这些做法在新框架（大数据集）上不一定仍然采用； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: mc(max_bias), registry(calibrate_max_bias)]</font>
## 2.4 db模块
- schema：暂无。
## 2.5 events模块
### 2.5.1 discover子模块
- 获取events的时间间隔改为每小时一次，目前是每天一次； <font color="#00b050"> ✓</font><font color="#808080"> [→L4: discovery_loop(CHANGE interval)]</font>
- 是否有查重功能，对于数据库里已经存在的事件或市场无需添加； <font color="#00b050"> ✓</font><font color="#808080"> [→L4: discovery_loop(ADD dedup)]</font>
- 对于clob的市场解析，slug解析还应更宽泛、鲁棒些，将来可能会变；例如： [parse_slug]的解析缺陷：解析含 `or-below` 的 slug 行全部返回 NaN。 <font color="#00b050"> ✓</font><font color="#808080"> [→L4: events/discovery(FIX parse_slug)]</font>
### 2.5.2 lifecycle子模块
- 认定的事件甩5个状态是否合理？尤其其中_CUTOFF_HOUR_UTC = 21，即下午16点不再交易，这个过于武断； <font color="#00b050"> ✓</font><font color="#808080"> [→L1: settings(REMOVE cutoff); L6: lifecycle(REMOVE gating)]</font>
- "settled"这个状态到底是谁设置的？到底应该到什么时候才是settled合适？ <font color="#00b050"> ✓</font><font color="#808080"> [→L8: settlement_loop[NEW]/ADR-0007; L6: lifecycle(ADD set_settled)]</font>
## 2.6 loops模块
### 2.6.1 discovery_loop子模块
### 2.6.2 forecast_loop子模块
- 这里不应该只做YES的token，而是YES和NO都做。由于我们不做空（不拆对子Minting/Splitting），但我们仍然可以双向做单。理论上，只要我的预测足够准，我总能在这两端中找到一个正期望値的买入点； <font color="#00b050"> ✓</font><font color="#808080"> [→L4: trading_loop(YES+NO handling)]</font>
-  **init_time** 是HRRR 模型的初始化时刻，无参数时默认为当前时间，注意别搞错时区； <font color="#00b050"> ✓</font><font color="#808080"> [→L4: forecast_loop]</font>
 - 只有做YES合约，这个不完整，做空YES相当于做多YES等价于做多NO，这是需要实现； <font color="#00b050"> ✓</font><font color="#808080"> [→L5: strategy/edge(FIX NO tokens)]</font>
 - MC返回値仅含各个桶的概率，没有P10、P90等値； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: mc(FIX bucket_probs p10/p50/p90)]</font>
 - 其他协程都用run进行调用，唯独此协程用的tick，很奇怪； <font color="#00b050"> ✓</font><font color="#808080"> [→L4: forecast_loop(SPLIT)/ADR-0004]</font>
### 2.6.3 settlement_loop子模块
- 暂时不动，将来需要借鉴下第三方代码研究下是如何settle的； <font color="#00b050"> ✓</font><font color="#808080"> [→L8: settlement_loop[NEW]/ADR-0007; Relayer stub deferred]</font>
### 2.6.4 sniper_loop子模块
- 暂时不用，参考整体1.1 **目标生产流程**。 <font color="#00b050"> ✓</font><font color="#808080"> [→L1: settings(enable_sniper=False)]</font>
## 2.7 model模块
### 2.7.1 store子模块
- registry：`_db_latest_row()` 直接访问 db._conn，绕过了 DbClient公共接口直接用私有属性，违反了封装原则。 <font color="#00b050"> ✓</font><font color="#808080"> [→L3: model/store/registry(FIX+ADD max_bias methods)]</font>
### 2.7.2 mc
- AR(1)算法换掉，采用经验协方差矩阵，用残差做协方差矩阵捕掄48个时刻之间的关系，即协方差，共给MC采样用； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: mc(empirical cov)/ADR-0002; L1: settings(REMOVE mc_rho)]</font>
- 先不要用bias_corr_c这个参数，这个参数来自于之前训练结果，不一定符合现在情况； <font color="#00b050"> ✓</font><font color="#808080"> [→L1: settings(REMOVE mc_bias_corr_c); L3: mc]</font>
- n_samp默认値设置为10000； <font color="#00b050"> ✓</font><font color="#808080"> [→L1: settings(mc_n_samp→10000); L3: mc]</font>

### 2.7.3 pipeline
- 注意：原始工程Herbie的step7、step8没用上，step9重新在step6基础上建立的模型； <font color="#00b050"> ✓</font><font color="#808080"> [informational — no task]</font>
- `ModelPipeline` 调用 [fetch_hrrr] 时**只传了 [init_time]**（见 [pipeline.py:79]），但 [DataManager.fetch_hrrr] 的签名要求 **[station] 也必须传入**； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: pipeline(FIX fetch_hrrr station arg)]</font>
- 之前Herbie工程有bug，tft_step6_build_72h.py是按历史-72..0准备的，但tft_step7_dataloader.py却只用了-23..0，这是个bug，我们需要-72..0数据； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: pipeline(FIX encoder window -71..0)]</font>
- fetch_iem 的过去数据量不对，仅取f00时刻附近的，HRRR取未来0..48的（不足补全）；但输入给模型的还需要历史-72..0，而不是-23..0; <font color="#00b050"> ✓</font><font color="#808080"> [→L2: manager(build_encoder_obs_window); L3: pipeline]</font>
- assemble_frames 函数不对：encoder是72小时，不是24小时，而且容差 **30 分钟** 太大了，改为20分钟。另外，encoder的输入既有iem真值，也有HRRR预测数据，都是从-71 ~ 0，更重要的，HRRR的过去每一个时刻都是当时f01的值，参考文件"C:\Codes\Polymarket\Herbie-main\tft_step6_build_72h.py"Step 5（第140~170行）。 <font color="#00b050"> ✓</font><font color="#808080"> [→L3: pipeline(FIX assemble_frames 72h/20min)]</font>
- 由于这次数据会超过一年，因此时间数据要体现季节、月份、日期等周期性； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: pipeline(ADD seasonal features)]</font>
- encoder与decoder数据拼接前需要构建"超集"列名列表，避免依赖 `pd.concat` 自动对齐引入额外 NaN，且列顺序不受控问题； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: pipeline(ADD superset column alignment)]</font>
- 是否有'time_idx'字段？没有的话如何定位全局时间轴上的位置以及划分 encoder / decoder 边界？ <font color="#00b050"> ✓</font><font color="#808080"> [→L3: pipeline(ADD time_idx)]</font>
- **"数据陈旧度（Staleness）"**。如果你的 1 小时任务因为 Grib2 下载失败而延迟，5 分钟 Tick 必须知道它现在用的是"多久以前"的概率，从而自动调高 Margin 阈值或停止交易； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: pipeline(ADD staleness ts); L4: trading_loop(staleness gate)/ADR-0006]</font>
- 在处理MC数据时有个问题需要注意：**极值期望偏高**的问题。原始Herboe工程采用了**经验性偏差校正**方案，即对所有样本取平均，得到**稳定的系统性偏差估计**，然后通过**提前压低每小时的均值**，使取最大值之后的期望恰好对齐实际最高温的方式进行了修正； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: mc(ADD max_bias), registry(calibrate_max_bias)]</font>
- 在计算MC采样结果的P10/P50/P90时，要所有样本最大值一起合并后计算，不能按桶拆开算； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: mc(FIX bucket_probs)]</font>
### 2.7.4 tft
- MAX_ENCODER_LENGTH应该是72，不是24； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: tft(CHANGE MAX_ENCODER_LENGTH)/ADR-0001]</font>
- 不要在这个模块里做数据处理，目前函数_prep做的，但应改在数据模块做； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: tft(MOVE _prep → cleaner)]</font>
- 这个返回値277行的result是残差，应该还原为数据値本身； <font color="#00b050"> ✓</font><font color="#808080"> [→L3: tft(FIX un-residualise)]</font>
## 2.8 snipers模块
- awc：暂时无修改； <font color="#00b050"> ✓</font><font color="#808080"> [no task — no changes needed]</font>
- manager：与2.3.5 manager重名； <font color="#00b050"> ✓</font><font color="#808080"> [→L8: snipers/sniper_manager(RENAME)]</font>
- noaa：暂时无修改； <font color="#00b050"> ✓</font><font color="#808080"> [no task — no changes needed]</font>
## 2.9 tests模块
- 对上面改动做相应调整。<font color="#ffc000"> ⏸ pending — update after all layers implemented</font>

---
