这里是针对grill-task.md文件提出的问题，所有提问以涉及到的标题为头，紧跟问题：

1. config/settings.py：max_open_per_token: int = 1是啥意思？一个市场是可以多次下单的，这个指的是最大下单数量吗？
2. data/forecast/_grib_source.py：ADD MAJOR-hour hourly expansion: after HrrrFill, expand MAJOR-hour frames to fill  
    every intermediate hour by forward-filling from the nearest MAJOR-hour f01 grid  
    (equivalent to original step1 logic)：这段是什么意思？填充不是已经由`HrrrFill`完成了吗？为啥这里还有填充内容？
3. data/obs/iem.py：~90 min window：为啥是90分钟窗口？
4. data/obs/noaa_file.py：from current-hour-only to support fetching past hours：这个不对吧，NOAA根本就没有过去时间的数据。
5. Layer 2: data/：谁负责拼凑过去72小时数据给TFT？
6. loops/forecast_loop.py：- SPLIT into ForecastLoop (1h) and TradingLoop (5min) [ADR-0004]：这里两个loop间数据交换如何实现的？如果是ProbabilityStore，请指明；
7. loops/forecast_loop.py：后两个条目是trading_loop里的，却放到了forecast_loop这里；
8. loops/forecast_loop.py：如何处理3天同时存在的市场？
9. events/discovery.py：这地方的slug解析应该更未宽泛，以适应各种变化；
10. core/strategy：是否为将来扩展成一个更为成熟交易系统做好了应有的铺垫：Alpha 策略（生成机会）、过滤策略（否决不合理信号）、 执行策略（优化下单方式）、Meta 策略（策略组合、冲突解决）、监控策略（健康检查、暂停交易）等？
11. core/risk：目前风险管理过于简单，应适应未来扩展成一个更为成熟交易系统做好了应有的铺垫：仓位与敞口风险、市场风险、模型风险、执行风险、操作风险、事件风险等。
12. core/exchange：是否应该增加1、2级鉴权部分？另外，API key不应该放到settings里，不安全。
13. core/exchange/paper.py：这部分过于简单。应至少满足以下要求：同一行情 → 跑 N 个策略、每个策略有独立账户、每个策略可以组合不同风控、真实 CLOB 驱动成交、行为与实盘高度一致、但比实盘更“宽”，方便实验；
14. 考虑增加账户与结算模块，例如包含以下内容：**PortfolioManager**（组合管理）、**PositionTracker**（持仓跟踪）、**PnLCalculator**（盈亏计算）、**CashManager**（现金管理）、**ExposureAggregator**（敞口聚合）、**TradeLedger**（交易账本）；
15. 考虑增加监控与告警模块，例如包括以下内容：SystemHealthMonitor、DataHealthMonitor、ModelHealthMonitor、RiskAlerts、OrderFailureAlerts、UI／telegram 通知；
16. backtest部分：这部分没太想好该怎么处理，之前做回测实验的时候用的是自己构建的历史CLOB，但现在是生产和paper run部分，这部分代码没啥用处删了可惜比较鸡肋了。
17. 这些个改动都需要完成，那对应的测试应该如何设计？是每完成一个模块的修改就改动相应的测试吗？