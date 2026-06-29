# PolyWeatherBot 实验管理与 Optuna 集成计划

**状态**：最终方案，下一步按实施切片执行  
**适用机器**：首先在 `john_linux` 单 GPU 执行（RTX 2060，约 5.6 GiB VRAM）  
**首个数据 workspace**：`runs/klga_train_default_20241001_20260604`

## 1. 目标

建立一个可重复、可恢复、可审计的训练实验系统：

- Git 管理实验模板、搜索空间和预算，不再手写散落的超参数。
- 管理端负责选择已审计 workspace，并自动生成 study、trial、candidate 标识。
- 后台 worker 顺序执行 smoke、基线校准、Optuna 搜索和最终评估。
- UI 展示队列、进度、指标、参数、日志、失败原因和产物。
- 在 48 GPU-hour 预算内寻找较优超参数，同时严格锁住 test 集。
- 最终模型仍遵守“candidate 导入但不自动 promotion”的生产边界。

不在本计划范围内：

- 自动部署或自动 promotion。
- 多 GPU / 分布式训练。
- 用 Optuna 管理历史数据下载和 readiness audit。
- 让生产推理代码依赖 Optuna、训练 catalog 或训练数据库。

## 2. 已确定的设计决策

### 2.1 配置和 schema

1. `experiments/` 下的 JSON 是实验定义的唯一事实源，UI 不维护另一套隐藏默认值。
2. 首个正式契约使用显式 `schema_version: 1`。
3. 当前尚无落盘的历史 `experiment_config.json`、`train_result.json` 或 candidate，
   因此不实现旧草案兼容器、字段别名或 v1→v2 迁移。
4. schema v1 在第一次成功 smoke 落盘后冻结；以后只有不兼容变化才升级 v2。
5. 配置拆为三个独立契约，避免把模板、调度状态和 trial 结果混成一个文件：

   - `experiment template schema v1`：站点、数据 splits、固定模型契约、默认参数。
   - `study specification schema v1`：搜索空间、sampler、pruner、预算、筛选策略。
   - `resolved trial snapshot schema v1`：一次执行的不可变参数、指纹、环境、指标和产物。

`schema_version` 必填；读取器不得把缺失版本默认为 1。

### 2.2 身份和 workspace

- 现有 ready `training workspace` 继续拥有数据、audit evidence 和 audit frames。
- 一个 study 引用一个 ready workspace 和一个 experiment-template Git revision/hash。
- station 从模板读取，并必须与 `audit_summary.json` 一致；用户不在每条命令中重复输入。
- manager 自动生成：

  - `study_id`：`<station>-<template>-<UTC timestamp>-<short hash>`
  - `trial_number`：使用 Optuna 的单调序号
  - `candidate_version`：`tft_<station>_<study_id>_final`

- 同一 workspace 可承载多个 study，但 trial 产物必须按 study/trial 隔离。

建议目录：

```text
experiments/
└── klga/
    └── tft_optuna_v1.json              # Git 跟踪，模板 + study spec

runs/
├── _management/
│   ├── training_catalog.db             # 队列、状态、指标、审批、artifact 索引
│   └── optuna.db                       # Optuna study/trial storage
└── klga_train_default_20241001_20260604/
    ├── data_store/
    ├── audit_summary.json
    ├── artifacts/audit_frames/
    └── studies/
        └── <study_id>/
            ├── study_snapshot.json
            ├── logs/
            ├── smoke/
            ├── calibration/
            ├── trials/
            │   └── <trial_number>/
            │       ├── trial_snapshot.json
            │       ├── metrics.json
            │       └── checkpoints/
            └── final/
```

`training_catalog.db` 和 `optuna.db` 必须分开：前者是 PolyWeatherBot 的管理事实，
后者只是 Optuna sampler/pruner 的适配器状态。删除或替换 Optuna 不应破坏实验历史。

### 2.3 执行边界

执行被拆成两条明确路径：

```text
smoke / calibration / Optuna trial
ready audit → train frame + val frame → validation evidence
                                      └─ 不读 test、不发布 candidate

approved final evaluation
selected trial snapshot → train + val + 一次 test
                        → candidate artifacts → package
                        └─ 不自动 promotion
```

当前 `TftTrainer.run()` 会加载 test、计算 `test_mae` 并发布 candidate，实施时必须先
拆开，不能直接套 Optuna 循环。test-lock 规则见 ADR-0017。

## 3. Smoke 和基线校准门禁

48 GPU-hour 搜索开始前必须依次通过两个门禁。

### 3.1 Pipeline smoke

使用与正式训练相同的 audited train/val frames 和模型构建路径，但限制 batch：

```json
{
  "mode": "smoke",
  "max_epochs": 1,
  "batch_size": 4,
  "precision": "16-mixed",
  "num_workers": 0,
  "limit_train_batches": 20,
  "limit_val_batches": 5,
  "evaluate_test": false,
  "publish_candidate": false,
  "timeout_minutes": 30
}
```

通过条件：

- CUDA、模型构建、forward、backward 和 optimizer step 成功。
- train/val loss 均为有限值。
- 无 OOM；记录 GPU 型号、CUDA/Torch、峰值显存和 batch 耗时。
- checkpoint 能保存并重新加载。
- resolved snapshot、日志和指标完整落盘。
- 确认没有读取 test frame，也没有写 model registry/candidate。

Smoke 失败时停止 study，修复后创建新的 smoke attempt；不得自动进入搜索。
batch size 4 通过后可以额外探测 8，但探测结果必须记录，不能悄悄改变模板。

### 3.2 Baseline calibration

- 使用固定默认超参数跑完整 train/val 的 1 epoch。
- 不读取 test，不发布 candidate。
- 测量完整 epoch 时间、峰值显存、val quantile loss 和数据加载占比。
- 根据实测时间估算 48 小时可完成的 trial 数，再冻结 study 的
  `max_epochs`、pruner warmup 和搜索空间。
- 建议给 calibration 单独设置不超过 2 GPU-hour 的安全上限；它不计入
  Optuna 的 48 GPU-hour 预算，但计入 catalog 的总资源消耗。

只有 smoke 和 calibration 均成功并经 UI/CLI 确认后，study 才可从
`gated` 进入 `queued`。

## 4. Optuna 搜索方案

### 4.1 资源模型

- `john_linux` 单 GPU，worker 并发为 1，Optuna `n_jobs=1`。
- 搜索预算为累计 **48 GPU-hours**。
- 预算从 trial 开始 GPU 训练时计时，到训练/验证结束或失败为止。
- completed、pruned、OOM/异常 trial 实际消耗的 GPU 时间都计入预算。
- 排队、UI、smoke、calibration、下游 evidence 和人工等待不计入这 48 小时。
- 到达预算后不再领取新 trial；正在执行的 trial允许正常结束，然后 study
  转为 `budget_exhausted`。

### 4.2 固定项与搜索项

v1 固定以下模型/数据契约：

- station、train/val/test 日期范围及 audit `data_order_hash`
- encoder length 72、prediction length 48
- quantiles `[0.1, 0.5, 0.9]`
- target、feature 顺序、decoder weighting 规则
- seed policy、deterministic policy
- test-lock 和 candidate 发布规则

首轮建议搜索：

| 参数 | 建议空间 |
|---|---|
| `hidden_size` | categorical: 16, 32, 48, 64 |
| `attention_head_size` | categorical: 1, 2, 4 |
| `hidden_continuous_size` | categorical: 8, 16, 24, 32 |
| `dropout` | float: 0.10–0.40 |
| `learning_rate` | log float: 1e-4–3e-3 |
| `weight_decay` | log float: 1e-6–1e-2 |
| `batch_size` | 仅允许 smoke 证明可运行的 4/8 |
| `gradient_clip_val` | categorical: 0.05, 0.1, 0.5, 1.0 |

首轮不搜索 splits、quantiles、encoder/decoder 长度、target winsor 和特征集合。
这些变化会改变问题定义或数据契约，应由独立 study 比较。

### 4.3 Sampler、pruner 和目标

- sampler：有固定 seed 的 TPE。
- pruner：以 epoch 为 resource 的 Median Pruner；前若干 startup trials
  不剪枝，并设置 warmup epochs，具体数值由 calibration 冻结。
- 主优化目标：validation quantile loss，方向 `minimize`。
- 每个 epoch 向 Optuna report validation loss；pruned trial 保留完整原因和
  截止时指标。
- NaN、OOM 和不可恢复异常标为失败，不伪装成极差分数。

首版不做多目标优化。多目标 Pareto 会令“最终选谁”含糊，且 48 小时预算不足以
稳定覆盖更大的搜索面。

## 5. 分层评估与最终选择

### 5.1 所有 trial 的基础 evidence

- best/final validation quantile loss
- epoch 数、pruned/early-stop 状态
- 参数、seed、配置 hash、data hash
- elapsed/GPU time、峰值显存
- checkpoint 和日志位置

### 5.2 Top-K 下游 validation evidence

study 结束后，对 validation loss 最优的 `top_k=5` 个 completed trials 统一计算：

- p50 MAE、各 quantile pinball loss
- p10/p90 coverage 和区间宽度
- 按 fxx、UTC init hour/month、trusted horizon 的误差分层
- validation 上的 48h 路径和 daily-high 误差
- McSampler 后的 calibration/bucket probability evidence
- 可得到相同历史事件输入时的 trading/backtest evidence

Top-K evidence 只能使用 validation 时间范围。v1 中它用于人工复核、排除明显
失准或不稳定模型，不回写 Optuna objective，不用 test 来打破平局。

### 5.3 人工审批和唯一 final

UI 展示参数、主目标、下游 evidence、资源消耗和失败记录。操作者选择一个 trial
并写入审批记录后，系统才允许：

1. 用该 resolved config 执行完整最终训练；
2. 解锁并仅评估一次 test；
3. 生成 candidate artifacts 和 package；
4. 将 candidate 标为 `inactive/review_required`。

test 结果不得触发新的 trial、改变参数或重新排序后再测试。若最终结果不可接受，
应记录该 study 结论并设计新的 study；不能在同一 study 中反复窥视 test。
candidate 的 import 与 production promotion 继续使用现有显式流程。

## 6. 管理 catalog、队列和状态机

建议最小表：

- `workspaces`：ready audit、station、splits、data hash、路径。
- `templates`：模板路径、schema、Git/hash、解析结果。
- `studies`：workspace/template 引用、预算、状态、Optuna study name。
- `executions`：smoke/calibration/trial/final 的统一状态和 attempt。
- `trial_metrics`：step/epoch 指标和最终汇总。
- `artifacts`：类型、路径、hash、所属 execution。
- `approvals`：启动搜索、选择 final、执行 final 的操作者和时间。
- `events`：append-only 状态变化和失败原因。

状态：

```text
draft
  → smoke_queued → smoke_running → smoke_passed
  → calibration_queued → calibration_running → gated
  → queued → running → budget_exhausted/completed
  → awaiting_selection → final_approved → final_running
  → candidate_ready
```

任一步均可进入 `failed`；运行中支持 `cancel_requested → cancelled`。重启 worker 后：

- pending/queued execution 可继续领取。
- Optuna 从独立 SQLite storage 恢复。
- 被进程中断的 running execution 标为 `interrupted`，不得假装成功。
- 操作者显式 retry 时创建新 attempt，保留旧日志和失败证据。

UI 只提交命令和审批，不在 Streamlit 请求线程里训练。后台 worker 使用事务领取
一条 execution；单 GPU 锁保证同一时刻只有一个训练任务。

## 7. 目标工程结构

```text
experiments/
└── klga/tft_optuna_v1.json

training/
├── config.py                 # schema-v1 解析、验证、hash
├── experiment.py             # ID、resolved snapshot、状态模型
├── catalog.py                # training_catalog.db repository
├── trial_runner.py           # train/val-only 与 final execution seam
├── optuna_engine.py          # Optuna adapter、budget、sampler/pruner
├── worker.py                 # 后台队列、GPU lock、恢复
├── trainer.py                # TFT fit/evaluate primitives
├── artifacts.py
└── workspace.py

scripts/
├── manage_training.py        # 初始化、enqueue、approve、status、retry
└── training_worker.py        # 长驻 worker CLI

ui/
└── pages/console_training.py # 由当前只读页扩展为实验管理页

tests/
└── test_training/
    ├── test_experiment_config.py
    ├── test_catalog.py
    ├── test_trial_runner.py
    ├── test_optuna_engine.py
    ├── test_worker.py
    └── test_experiment_ui.py
```

`scripts/` 只做 CLI 适配；深层逻辑属于 `training/`。`model/` 仍只负责生产推理、
artifact contract 校验、candidate import 和显式 promotion。

## 8. 计划中的 CLI 和管理端操作

以下是实施完成后的目标接口：

```bash
# 注册/刷新 ready workspaces 和 Git 模板
./.venv/bin/python -m scripts.manage_training sync

# 创建 study；station、splits 和标识均由模板/workspace 推导
./.venv/bin/python -m scripts.manage_training create-study \
  --workspace runs/klga_train_default_20241001_20260604 \
  --template experiments/klga/tft_optuna_v1.json

# 排队 smoke、确认门禁、启动搜索
./.venv/bin/python -m scripts.manage_training enqueue-smoke --study <STUDY_ID>
./.venv/bin/python -m scripts.manage_training approve-search --study <STUDY_ID>

# 启动单 GPU 后台 worker
./.venv/bin/python -m scripts.training_worker \
  --catalog runs/_management/training_catalog.db \
  --optuna-storage sqlite:///runs/_management/optuna.db

# 选择 trial 并批准唯一 final
./.venv/bin/python -m scripts.manage_training approve-final \
  --study <STUDY_ID> \
  --trial-snapshot <TRIAL_SNAPSHOT_JSON> \
  --approved-by <OPERATOR>
```

最终应提供 systemd user service 示例，使 worker 可在退出终端后继续运行并自动恢复。

## 9. 实施切片和验收标准

按以下顺序实施；每个切片都采用测试先行并可独立验收。

### Slice 1：正式 schema v1 与模板

**实施状态：已完成（2026-06-29）**

- 重构 `training/config.py`，显式区分 template、study spec、trial snapshot。
- 新增首个 `experiments/klga/tft_optuna_v1.json`。
- 严格校验版本、未知字段、参数范围、station/split 和固定模型契约。
- 配置 canonical JSON hash 可重复。

验收：缺失/未知 schema、非法搜索空间和不匹配的 station 均 fail closed；同一模板
重复解析产生同一 hash。

### Slice 2：隔离 train/val 与 final，落实 test lock

**实施状态：已完成（2026-06-29）**

- 将 dataset loading、fit、validation evidence、test evaluation 和 publish 拆开。
- 普通 execution API 根本不接受 test loader/publish 参数。
- final API 要求 catalog 中有效审批。

验收：测试证明 smoke/calibration/trial 不读取 test、不写 registry；只有 approved
final 能产生 test metric 和 inactive candidate。

### Slice 3：Catalog、自动 ID 和 workspace 注册

**实施状态：已完成（2026-06-29）**

- 建立独立 `training_catalog.db` repository 和 migration 001。
- 实现 workspace/template sync、study 创建、自动 ID、状态事件和 artifact 索引。
- catalog 验证 ready audit、station、splits 和 data hash。

验收：重复/并发创建不碰撞；audit 失效后不能创建或继续 study；已有记录可完整查询。

### Slice 4：Smoke 与 calibration tracer bullet

**实施状态：已完成（2026-06-29）**

- 实现 execution limits、timeout、GPU/environment telemetry、checkpoint reload。
- 以当前 KLGA workspace 完成一次真实 smoke，再完成一次受限 calibration。
- 失败不会进入 Optuna。

真实 smoke 已验证 RTX 2060 / Torch 2.12 + CUDA 13.0 的 mixed-precision
执行链、逐 batch catalog 进度、显存证据和 checkpoint reload。诊断过程中发现并修复
`pytorch-forecasting` 默认 attention `mask_bias=-1e9` 的 FP16 溢出；当前 Torch
CUDA 的 `upsample_linear1d_backward` 无严格 deterministic 实现，因此执行证据明确
记录 `deterministic_mode=warn_only`，不将其表述为 bit-exact。

真实 smoke 完成 20/20 train batches 和 5/5 val batches，
`best_val_loss=0.655743`；真实 calibration 完成 3246/3246 train batches 和
180/180 val batches，耗费约 771 GPU-seconds，`best_val_loss=0.728890`。两者均
重载 checkpoint 成功，catalog 中的 study 状态为 `gated`，日志未出现 test loader、
test metric 或 candidate publish。长执行的 catalog 进度在首个、每 10 个和最后一个
batch 持久化，Lightning 控制台进度仍连续显示。

验收：管理端可看到实时阶段、batch/epoch 进度、显存、日志和明确失败原因；test 和
registry 保持未访问。

### Slice 5：后台队列和 UI

**实施状态：已完成（2026-06-29）**

- worker 事务领取 execution，单 GPU lock，支持 restart/interrupted/retry/cancel。
- 扩展 `ui/pages/console_training.py`：workspace、study、队列、详情、日志和审批。
- UI 不直接执行训练。

`TrainingWorker` 在跨进程 GPU 文件锁内恢复遗留 `running`、事务领取最早的 queued
execution，并将崩溃遗留任务标为 `interrupted`。显式 retry 创建新 attempt，保留旧
日志和错误；queued cancel 立即完成，running cancel 在下一进度检查点协作完成。
管理页只读写 catalog，可展示 workspace、study、execution、approval、进度、指标和
日志尾部，不导入或调用训练 runner。

验收：关闭并重启 worker 后 queued 任务继续；running 任务不会被误报成功；UI
刷新不会重复提交。

### Slice 6：Optuna adapter 与 48 GPU-hour budget

**实施状态：已完成（2026-06-29，未启动正式 48 小时运行）**

- 增加 Optuna 依赖、独立 storage、TPE、pruner、epoch reporting。
- 实现累计 GPU-time budget、OOM/NaN/失败分类和 resume。
- 使用 calibration 结果冻结首个 study spec。

实现使用 Optuna 4.x、固定 seed 的 TPE、MedianPruner 和独立 SQLite storage；
`n_jobs=1`。每个 sampled config 生成 immutable trial snapshot，并只进入
train/validation TFT adapter。每个 validation epoch 同时写 catalog progress 和
Optuna intermediate value。completed、pruned、OOM、NaN、异常及崩溃恢复产生的
实际/保守 GPU 秒都计入 catalog 预算；达到预算后 callback 停止领取新 trial，当前
trial 正常收口。Optuna RUNNING 与 catalog interrupted 会在 resume 时对账为失败，
随后使用下一个 trial number。

search snapshot 冻结 template/study spec、data hash、calibration execution、完整
epoch GPU 秒、max epochs 和预算内预计完整 trial 数。当前真实 calibration
约 `771.23s/epoch`，以 20 epochs 粗估单个未剪枝 trial 约 4.28 GPU-hours，48 小时
约容纳 11.2 个完整 trial；pruning 会提高实际覆盖数。

验收：小预算集成测试可完成、剪枝、停止和恢复；`n_jobs=1`；预算耗尽后不领取新 trial。

本切片只由 Codex 实现 engine 并运行小预算/非 48 小时验收。正式 48 GPU-hour study
由操作者手动执行 worker/启动命令；Codex 不持续轮询或监管长任务，只在运行结束后
读取 catalog、Optuna storage、日志和 Top-K evidence 做一次结果验证，以避免无意义
的对话 token 消耗。

### Slice 7：Top-K evidence、人工 final 和 candidate

- 生成 Top-5 validation evidence。
- UI/CLI 选择一个 trial、写审批、执行唯一 test、生成 inactive candidate package。
- candidate 记录 template/study/trial/config/data hashes。

验收：未审批、第二次 final、test 后回写 study、自动 promotion 均被拒绝；candidate
可由现有 import 流程导入且保持 inactive。

### Slice 8：运维、文档和真实 48 小时运行

- 更新 training runbook、故障恢复、磁盘/日志清理和 systemd user service。
- 完成真实 smoke 与 baseline 报告，人工确认搜索空间。
- 启动 48 GPU-hour study，结束后完成 Top-K 评审和一次 final。

验收：从模板到 candidate package 全链路可由文档复现；catalog、Optuna storage、
trial snapshots、日志和 candidate hashes 相互可追溯。

## 10. 总体验收不变量

实施完成必须同时满足：

1. 无需手写 run-id、station、trial version。
2. Git JSON 是唯一配置源；每次 execution 有不可变 resolved snapshot。
3. readiness audit/data hash 不匹配时 fail closed。
4. smoke 和 calibration 是 48 小时搜索的强制前置门禁。
5. 单 GPU 同时最多一个执行；后台任务可恢复、可取消、有进度。
6. 48 GPU-hour 预算可审计，失败消耗不会凭空消失。
7. Optuna trial 只看 validation；test 在人工批准前不可访问。
8. 每个 study 最多执行一次 approved final test。
9. 训练端只能生成 inactive candidate，不能 promotion。
10. 从 candidate 可反查 workspace、audit hash、template、study、trial、环境和指标。

## 11. 下一步

Slice 1 至 Slice 6 已完成。下一步实施 **Slice 7：Top-K evidence、人工 final 和
candidate**。真实 48 小时 search 仍未批准或启动，留给操作者在最终检查后手动执行。
