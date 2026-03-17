# 量化选股系统跨区域流程验证契约断言

## 文档信息
- **生成日期**: 2026-03-17
- **适用范围**: A股量化选股系统验证Mission
- **验证维度**: 端到端流程、数据流、一致性、边界条件

---

## 1. 端到端流程验证 (VAL-CROSS-001 ~ VAL-CROSS-020)

### VAL-CROSS-001: 完整流程可执行性
**标题**: 从数据到报告的完整流程必须可执行

**行为描述**:
- **PASS条件**: 执行 `run_automation.py` 或 `main.py auto-pilot` 时，9个Milestone全部完成且无致命错误
- **FAIL条件**: 任一Milestone抛出未捕获异常导致流程中断

**证据要求**:
- 日志文件显示 `[Step X/9]` 全部完成标记
- 最终报告文件生成于 `data/auto_reports/auto_report_YYYYMMDD_HHMMSS.md`
- 结果JSON文件生成于 `data/auto_reports/auto_results_YYYYMMDD_HHMMSS.json`

---

### VAL-CROSS-002: 数据准备到选股信号传递
**标题**: Milestone 1数据必须成功传递给Milestone 2选股分析

**行为描述**:
- **PASS条件**: `data/` 目录下CSV文件数量 > 0，且 `analyzer.py` 能读取这些文件
- **FAIL条件**: 数据文件缺失或格式损坏导致 `analyze_all_stocks()` 返回空结果

**证据要求**:
- 日志: `数据更新完成: {N} 个数据文件`
- 选股结果CSV文件生成: `selected_stocks_YYYYMMDD.csv`

---

### VAL-CROSS-003: 选股结果到回测的数据流
**标题**: 选股信号必须正确传递给回测模块

**行为描述**:
- **PASS条件**: `scan_today_signal()` 返回的信号格式包含必需字段: `code`, `date`, `close`, `buy_score`
- **FAIL条件**: 回测时因信号字段缺失导致 `KeyError` 或 `TypeError`

**证据要求**:
- 信号字典结构验证日志
- 回测结果DataFrame非空且包含所有统计列

---

### VAL-CROSS-004: 回测到参数优化的结果传递
**标题**: 回测统计结果必须正确用于参数优化目标函数

**行为描述**:
- **PASS条件**: `batch_backtest()` 返回的DataFrame包含 `sharpe`, `return_pct`, `max_drawdown`, `win_rate` 列
- **FAIL条件**: `compute_objective()` 函数因缺少必需列返回 `-999.0` 或报错

**证据要求**:
- 优化器日志: `基线得分: {X:.6f}, 最优得分: {Y:.6f}`
- `data/optimize_results/` 目录下的优化结果文件

---

### VAL-CROSS-005: 参数优化到配置更新的原子性
**标题**: 优化后的参数必须原子性地写回配置文件

**行为描述**:
- **PASS条件**: 优化完成后，`config.yaml` 中的参数值与优化器输出的 `best_params` 一致
- **FAIL条件**: 配置更新过程中断导致 `config.yaml` 处于半更新状态

**证据要求**:
- 优化前后配置文件的MD5哈希对比
- 日志: `最优参数已应用到 config.yaml`

---

### VAL-CROSS-006: 失败时的优雅回退
**标题**: 任何Milestone失败时必须触发优雅回退机制

**行为描述**:
- **PASS条件**: 失败时日志包含 `[ERROR]` 标记，系统释放资源并返回有意义的状态码
- **FAIL条件**: 失败时进程挂起、资源泄漏或静默忽略错误

**证据要求**:
- 错误日志: `自动化运行失败: {e}`
- 性能监控器 `monitor.stop_monitoring()` 被调用记录

---

### VAL-CROSS-007: AI模型加载失败回退
**标题**: AI模型加载失败时必须回退到纯规则引擎

**行为描述**:
- **PASS条件**: `_AI_MODEL_LOAD_ATTEMPTED = True` 且模型加载失败后系统继续运行
- **FAIL条件**: 模型加载失败导致整个选股流程终止

**证据要求**:
- 日志: `AI 模型文件不存在: {path}，将使用纯规则引擎。`
- 选股结果中存在 `ai_prob` 字段但值为默认值 `0.5`

---

### VAL-CROSS-008: 回测无交易信号处理
**标题**: 回测无交易信号时必须返回中性得分而非崩溃

**行为描述**:
- **PASS条件**: 当AI过滤所有信号时，`compute_objective()` 返回 `0.0` 而非 `-999.0`
- **FAIL条件**: 无信号时优化器崩溃或返回极端负值导致错误剪枝

**证据要求**:
- 代码注释: `Neutral score when AI filters all signals (not catastrophic)`
- 优化日志中无信号时的得分记录为 `0.0`

---

### VAL-CROSS-009: Web UI与CLI流程一致性
**标题**: Web UI执行的流程必须与CLI命令等价

**行为描述**:
- **PASS条件**: `app.py` 中的按钮操作调用的函数与 `main.py` 中的命令函数相同
- **FAIL条件**: Web UI和CLI产生不一致的结果或行为

**证据要求**:
- 代码审查: `app.py` 调用 `cmd_update_list()` 等CLI函数
- 相同输入下Web UI和CLI输出结果一致

---

### VAL-CROSS-010: 并行执行安全性
**标题**: 多进程并行执行时必须保证数据一致性

**行为描述**:
- **PASS条件**: `ProcessPoolExecutor` 和 `joblib.Parallel` 执行后结果无数据竞争或重复
- **FAIL条件**: 并行执行导致同一股票被处理多次或结果丢失

**证据要求**:
- 并行任务完成后结果数量与预期一致
- 无 `concurrent.futures` 相关异常日志

---

## 2. 数据流验证 (VAL-CROSS-021 ~ VAL-CROSS-040)

### VAL-CROSS-021: 选股结果到统计分析的数据完整性
**标题**: 选股结果CSV必须包含所有必需字段用于统计分析

**行为描述**:
- **PASS条件**: `selected_stocks_{date}.csv` 包含字段: `code`, `buy_score`, `total_score`, `rsi`, `signal_type`, `market_state`
- **FAIL条件**: 缺少必需字段导致 `reverse_validation.py` 无法读取

**证据要求**:
- CSV文件头验证
- 反向验证日志: `扫描完成，共发现 {N} 个买点信号`

---

### VAL-CROSS-022: 分析结果到参数优化的指标传递
**标题**: Walk-Forward验证得分必须正确传递给优化器

**行为描述**:
- **PASS条件**: `walk_forward_cv()` 返回的平均fold得分被 `objective_function()` 正确使用
- **FAIL条件**: fold得分在传递过程中被错误归一化或截断

**证据要求**:
- 优化器日志显示 `[Hyperband] Trial #{N}: Stage {X} - score: {Y}`
- 最终参数与fold得分趋势一致

---

### VAL-CROSS-023: 优化配置到最终验证的应用
**标题**: 优化后的策略参数必须在最终验证中被使用

**行为描述**:
- **PASS条件**: `apply_best_params()` 后新生成的选股结果使用了更新后的阈值
- **FAIL条件**: 最终验证仍使用旧配置导致结果不一致

**证据要求**:
- 配置更新时间戳早于最终验证开始时间
- 选股结果中的 `ai_threshold` 值与新配置一致

---

### VAL-CROSS-024: 中间结果持久化
**标题**: 关键中间结果必须持久化到磁盘

**行为描述**:
- **PASS条件**: 以下文件在流程中被创建:
  - 选股结果: `selected_stocks_YYYYMMDD.csv`
  - 回测结果: `historical_scan_YYYYMMDD.csv`
  - 优化结果: `data/optimize_results/*`
  - 风险报告: `data/risk_reports/*`
- **FAIL条件**: 关键中间结果仅存在于内存中，进程崩溃后丢失

**证据要求**:
- 文件系统存在上述文件且修改时间戳符合流程执行时间

---

### VAL-CROSS-025: 数据更新增量性
**标题**: 数据更新必须支持增量更新而非全量覆盖

**行为描述**:
- **PASS条件**: `update_history_data()` 检测到已有数据时，只拉取 `last_date + 1` 之后的数据
- **FAIL条件**: 每次更新都全量删除并重新拉取所有历史数据

**证据要求**:
- 日志: `增量更新: {code}, 从 {start_date} 到 {end_date}`
- 数据文件的时间范围连续无断层

---

### VAL-CROSS-026: 特征矩阵与模型输入一致性
**标题**: 训练时和预测时的特征矩阵维度必须一致

**行为描述**:
- **PASS条件**: `_AI_MODEL.num_feature()` 返回的特征数与 `extract_all_features()` 生成的特征数相等
- **FAIL条件**: 特征数不匹配导致 `predict()` 抛出维度错误

**证据要求**:
- 模型加载日志: `AI 模型已加载: {path} ({N} features)`
- 特征提取日志中的特征数与模型特征数一致

---

### VAL-CROSS-027: 日期索引对齐
**标题**: 多股票数据的时间索引必须对齐

**行为描述**:
- **PASS条件**: `walk_forward_cv()` 使用同一只股票的日期作为所有股票的时间基准
- **FAIL条件**: 不同股票的时间范围不一致导致截面分析错误

**证据要求**:
- 代码: `sample_file = os.path.join(data_dir, f"{codes[0]}.csv")` 作为日期基准
- fold分割的日期范围在日志中显示连续

---

### VAL-CROSS-028: 信号质量评分的跨模块一致性
**标题**: `buy_score` 的计算逻辑必须在所有模块中保持一致

**行为描述**:
- **PASS条件**: `scan_today_signal()` 中的 `buy_score = rule_score * ai_prob * confidence` 公式在选股和回测中一致
- **FAIL条件**: 不同模块使用不同的评分公式导致结果不一致

**证据要求**:
- 代码grep: `buy_score.*rule_score.*ai_prob` 在 `backtester.py` 和 `analyzer.py` 中一致
- 同一股票在选股和回测中的评分值相同

---

### VAL-CROSS-029: 配置参数的单点管理
**标题**: 所有模块必须引用同一配置源

**行为描述**:
- **PASS条件**: 所有模块使用 `from quant.infra.config import CONF` 而非硬编码参数
- **FAIL条件**: 存在模块使用本地硬编码配置导致不一致

**证据要求**:
- 代码审查: 模块头文件统一导入 `CONF`
- `config.yaml` 修改后所有模块行为一致变化

---

### VAL-CROSS-030: 风险报告数据的完整性
**标题**: 风险报告必须包含所有必需的风险指标

**行为描述**:
- **PASS条件**: 风险报告CSV包含: `VaR_95`, `max_drawdown`, `sharpe`, `sector_exposure`, `correlation_matrix`
- **FAIL条件**: 风险报告缺少关键指标导致无法评估风险状况

**证据要求**:
- `data/risk_reports/` 目录下的报告文件包含上述所有字段
- 风险报告生成日志无警告

---

## 3. 一致性验证 (VAL-CROSS-041 ~ VAL-CROSS-060)

### VAL-CROSS-041: 统计口径一致性
**标题**: 不同阶段的收益率计算必须使用相同的统计口径

**行为描述**:
- **PASS条件**: 回测、优化、最终验证都使用 `(exit_price - entry_price) / entry_price` 计算收益率
- **FAIL条件**: 不同阶段使用不同的收益计算方式（如对数收益 vs 简单收益）

**证据要求**:
- 代码grep: 所有模块的收益率计算公式一致
- 同一交易在不同阶段的收益率值相同

---

### VAL-CROSS-042: 滑点模型一致性
**标题**: 回测和验证必须使用相同的滑点模型

**行为描述**:
- **PASS条件**: `SlippageModel` 的实现在 `backtester.py` 和 `reverse_validation.py` 中引用同一类
- **FAIL条件**: 不同模块使用不同的滑点假设导致结果不可比

**证据要求**:
- 代码: `get_slippage_model()` 作为统一入口
- 滑点配置参数从 `config.yaml` 的 `risk.slippage` 读取

---

### VAL-CROSS-043: 市场环境分类一致性
**标题**: 市场环境状态定义必须在所有模块中一致

**行为描述**:
- **PASS条件**: `market_state` 枚举值 (`strong_bull`, `weak_bull`, `sideways`, `weak_bear`, `strong_bear`) 在所有模块中一致
- **FAIL条件**: 不同模块使用不同的状态命名或分类逻辑

**证据要求**:
- `config.yaml` 中的 `market_state_thresholds` 键名与代码中的状态判断一致
- 市场状态转换矩阵在日志中一致

---

### VAL-CROSS-044: 配置修改可追溯性
**标题**: 配置修改必须通过版本控制可追溯

**行为描述**:
- **PASS条件**: 每次 `config.yaml` 修改都通过 `git commit` 记录，包含修改原因
- **FAIL条件**: 配置修改无记录，无法追溯历史配置

**证据要求**:
- `git log --oneline -- config.yaml` 显示配置修改历史
- 提交信息包含修改原因（如"优化后参数更新"）

---

### VAL-CROSS-045: 结果可复现性
**标题**: 相同输入和随机种子必须产生相同结果

**行为描述**:
- **PASS条件**: 设置相同 `random.seed(42)` 和 `np.random.seed(42)` 后，重复执行产生相同选股列表
- **FAIL条件**: 结果随时间变化而无确定性的原因

**证据要求**:
- 两次执行结果文件的MD5哈希值相同
- 日志中的随机采样序列一致

---

### VAL-CROSS-046: 时间戳格式一致性
**标题**: 所有时间戳必须使用ISO 8601格式

**行为描述**:
- **PASS条件**: 日期字段格式为 `YYYY-MM-DD` (如 `2024-05-10`)
- **FAIL条件**: 混合格式如 `05/10/2024` 或 `2024年5月10日`

**证据要求**:
- 代码中统一使用 `strftime('%Y-%m-%d')` 或 `pd.to_datetime()`
- CSV文件中的日期列格式一致

---

### VAL-CROSS-047: 股票代码格式一致性
**标题**: 股票代码必须使用统一格式

**行为描述**:
- **PASS条件**: 代码格式为 `sh.600000` 或 `sz.000001` (交易所前缀.代码)
- **FAIL条件**: 混合格式如 `600000.SH` 或纯数字 `600000`

**证据要求**:
- 文件名使用 `{code}.csv` 格式
- 日志中的代码格式统一

---

### VAL-CROSS-048: 数值精度一致性
**标题**: 浮点数比较必须使用容差而非精确相等

**行为描述**:
- **PASS条件**: 浮点数比较使用 `abs(a - b) < epsilon` 或 `np.isclose()`
- **FAIL条件**: 直接使用 `==` 比较浮点数导致误判

**证据要求**:
- 代码grep: `isclose` 或 `epsilon` 或 `1e-` 的使用
- 无直接使用 `float_val == 0` 的硬编码

---

### VAL-CROSS-049: 参数空间定义一致性
**标题**: 优化参数空间必须在定义和使用处一致

**行为描述**:
- **PASS条件**: `CORE_PARAM_SPACE` 在 `strategy_params.py` 定义，在 `optimizer.py` 和 `validation_pipeline.py` 中使用
- **FAIL条件**: 不同模块定义重复的参数空间导致不一致

**证据要求**:
- 代码grep: `CORE_PARAM_SPACE` 的导入和使用
- 参数空间修改后所有优化器行为一致变化

---

### VAL-CROSS-050: 回测参数一致性
**标题**: Walk-Forward回测参数必须与实盘参数一致

**行为描述**:
- **PASS条件**: 优化时使用的 `StrategyParams` 默认值与实盘使用的参数一致
- **FAIL条件**: 优化和实盘使用不同的参数默认值

**证据要求**:
- `StrategyParams.from_app_config(CONF)` 同时用于优化和实盘
- `config.yaml` 的 `strategy` 节同时影响优化和实盘

---

## 4. 边界条件验证 (VAL-CROSS-061 ~ VAL-CROSS-080)

### VAL-CROSS-061: 无信号日期处理
**标题**: 无信号日期必须返回空结果而非崩溃

**行为描述**:
- **PASS条件**: 当 `scan_historical_date()` 无信号时返回 `[]` 且日志显示 `[OK] 历史扫描完成：未发现任何满足策略要求的买入标的`
- **FAIL条件**: 无信号时抛出 `IndexError` 或 `ValueError`

**证据要求**:
- 代码: `if not results: return []`
- 日志包含 "未发现任何满足策略要求的买入标的"

---

### VAL-CROSS-062: 数据缺失股票处理
**标题**: 数据缺失的股票必须被跳过而非导致流程中断

**行为描述**:
- **PASS条件**: 当CSV文件不存在或为空时，`scan_today_signal()` 返回 `None` 并继续处理下一股票
- **FAIL条件**: 单个股票数据缺失导致整个选股流程失败

**证据要求**:
- 代码: `if not stock_file.exists(): return None`
- 日志: `扫描 {stock_file.name} 失败：{e}` 后流程继续

---

### VAL-CROSS-063: 极端市场情况处理
**标题**: 市场暴跌情况必须有压力测试覆盖

**行为描述**:
- **PASS条件**: `stress_tester.py` 包含 `2015_stock_crash` 和 `black_monday` 场景
- **FAIL条件**: 压力测试场景缺失或无法模拟极端下跌

**证据要求**:
- `StressScenario` 定义包含 `market_drop >= 0.30` 的场景
- 压力测试报告生成且 `survived` 字段存在

---

### VAL-CROSS-064: 流动性危机模拟
**标题**: 流动性危机必须有专门的处理逻辑

**行为描述**:
- **PASS条件**: `StressScenario` 包含 `liquidity_crunch=True` 且 `slippage` 模型考虑流动性影响
- **FAIL条件**: 压力测试忽略流动性因素导致结果过于乐观

**证据要求**:
- `SlippageModel` 根据市值分层设置不同滑点率
- 压力测试场景 `black_monday` 设置 `liquidity_crunch=True`

---

### VAL-CROSS-065: 持仓天数边界
**标题**: 持仓天数必须在有效范围内

**行为描述**:
- **PASS条件**: `hold_days` 被限制在 `max(1, min(hold_days, max_hold_days))`
- **FAIL条件**: 持仓天数为0或负数导致计算错误

**证据要求**:
- 代码: `hold_days = max(1, min(hold_days, max_hold_days))`
- 回测结果中的实际持仓天数均为正数

---

### VAL-CROSS-066: 价格数据异常处理
**标题**: 零价格或负价格必须被过滤

**行为描述**:
- **PASS条件**: `if entry_price <= 0: return None` 在交易模拟中被检查
- **FAIL条件**: 零价格或负价格参与计算导致 `Inf` 或 `NaN`

**证据要求**:
- 代码中使用 `np.isfinite()` 检查价格有效性
- 回测结果中无 `Inf` 或 `NaN` 值

---

### VAL-CROSS-067: 空样本集处理
**标题**: Walk-Forward无有效样本时必须返回安全值

**行为描述**:
- **PASS条件**: 当所有folds都失败时，`walk_forward_cv()` 返回 `0.0` 而非 `-999.0`
- **FAIL条件**: 空样本导致优化器错误剪枝所有参数

**证据要求**:
- 代码: `if len(fold_test_scores) == 0: return 0.0`
- 优化日志中显示空样本时的中性得分

---

### VAL-CROSS-068: 数据长度不足处理
**标题**: 数据长度不足的股票必须被跳过

**行为描述**:
- **PASS条件**: `if len(df) < max(ma_long, bbands_length, atr_length, macd_slow): return df` 保护
- **FAIL条件**: 数据不足导致技术指标计算失败

**证据要求**:
- `calculate_indicators()` 中的长度检查
- 技术指标列无全 `NaN` 值

---

### VAL-CROSS-069: 极端夏普比率处理
**标题**: 极端夏普比率必须有上限和下限

**行为描述**:
- **PASS条件**: 夏普比率计算包含标准差为零的保护 `if std_sharpe == 0: sharpe = 0`
- **FAIL条件**: 标准差为零导致除以零错误

**证据要求**:
- 代码: `sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)` 包含长度检查
- 回测结果中夏普比率无 `Inf` 值

---

### VAL-CROSS-070: 止损/止盈同时触发处理
**标题**: 同时触及止损和止盈时必须使用悲观假设

**行为描述**:
- **PASS条件**: 当同一天可能同时触及止损和止盈时，代码优先触发止损
- **FAIL条件**: 优先触发止盈导致回测结果过于乐观

**证据要求**:
- 代码注释: `同一天同时触及止损/止盈: 悲观假设先触发止损`
- 回测中止损触发的优先级高于止盈

---

### VAL-CROSS-071: 缓存溢出保护
**标题**: 内存缓存必须有大小限制

**行为描述**:
- **PASS条件**: `LazyDataLoader` 的 `cache_size` 有上限且使用LRU淘汰策略
- **FAIL条件**: 缓存无限增长导致内存溢出

**证据要求**:
- `LazyDataLoader` 初始化参数 `memory_cache_size: int = 100`
- `_add_to_cache()` 中的LRU淘汰逻辑

---

### VAL-CROSS-072: 并行任务异常隔离
**标题**: 单个并行任务失败不得影响其他任务

**行为描述**:
- **PASS条件**: `ProcessPoolExecutor` 的 `future.result()` 在 `try-except` 块中执行
- **FAIL条件**: 单个股票处理失败导致整个并行批次失败

**证据要求**:
- 代码: `try: res = future.result(); except Exception as e: logger.debug(...)`
- 并行任务完成后部分失败但整体流程继续

---

### VAL-CROSS-073: 配置参数越界处理
**标题**: 配置参数越界时必须被钳制到有效范围

**行为描述**:
- **PASS条件**: 参数使用 `min(max(value, lower), upper)` 被钳制到有效范围
- **FAIL条件**: 越界参数导致模型预测失败或异常行为

**证据要求**:
- `StrategyParams` 的属性设置器包含范围检查
- 优化器输出的参数值均在有效范围内

---

### VAL-CROSS-074: 日期格式异常处理
**标题**: 非标准日期格式必须被正确解析或拒绝

**行为描述**:
- **PASS条件**: `pd.to_datetime()` 使用 `errors='coerce'` 将无效日期转为 `NaT`
- **FAIL条件**: 无效日期格式导致解析失败

**证据要求**:
- 代码: `df['date'] = pd.to_datetime(df['date'], errors='coerce')`
- 日期列无解析错误日志

---

### VAL-CROSS-075: 行业分类缺失处理
**标题**: 未知行业的股票必须被分配到默认分类

**行为描述**:
- **PASS条件**: `get_sector()` 函数对未知代码返回 `'other'`
- **FAIL条件**: 未知行业导致 `KeyError` 或分类失败

**证据要求**:
- 代码: `else: return "other"` 作为默认分支
- 行业暴露统计中包含 `'other'` 类别

---

### VAL-CROSS-076: 相关性矩阵异常处理
**标题**: 相关性计算必须有足够的样本量

**行为描述**:
- **PASS条件**: 相关性计算检查 `min_periods` >= 30
- **FAIL条件**: 样本量不足导致相关性估计不可靠

**证据要求**:
- `config.yaml` 中的 `correlation.min_periods: 30`
- 相关性矩阵计算前检查数据长度

---

### VAL-CROSS-077: AI置信度边界处理
**标题**: AI置信度必须在[0,1]范围内

**行为描述**:
- **PASS条件**: `ai_confidence` 被限制在 `max(0.0, min(1.0, confidence))`
- **FAIL条件**: 置信度越界导致信号质量计算错误

**证据要求**:
- 代码: `ai_prob = max(0.0, min(1.0, ai_prob))`
- 选股结果中的 `ai_prob` 值均在 [0,1] 范围内

---

### VAL-CROSS-078: 蒙特卡洛模拟边界
**标题**: 蒙特卡洛模拟必须有足够的样本路径

**行为描述**:
- **PASS条件**: `monte_carlo.n_paths >= 1000` 且 `block_size` 合理
- **FAIL条件**: 样本路径过少导致统计结果不可靠

**证据要求**:
- `config.yaml` 中的 `monte_carlo.n_paths: 1000`
- 蒙特卡洛报告的置信区间宽度合理

---

### VAL-CROSS-079: VaR计算边界条件
**标题**: VaR计算必须在有足够历史数据时执行

**行为描述**:
- **PASS条件**: VaR计算检查 `len(returns) >= history_window` (252天)
- **FAIL条件**: 历史数据不足导致VaR估计不可靠

**证据要求**:
- `config.yaml` 中的 `risk.var.history_window: 252`
- VaR计算前检查数据长度

---

### VAL-CROSS-080: 信号冷却期边界
**标题**: 信号冷却期必须在合理范围内

**行为描述**:
- **PASS条件**: `signal_cooldown_days` 在 [1, 30] 范围内
- **FAIL条件**: 冷却期为0或过大导致信号过于稀疏或密集

**证据要求**:
- `config.yaml` 中的 `strategy.signal_cooldown_days: 5`
- 选股结果中同一股票的信号间隔符合冷却期设置

---

## 附录: 验证执行检查清单

### 执行前准备
- [ ] 确认 `config.yaml` 配置正确
- [ ] 确认 `data/` 目录存在且有历史数据
- [ ] 确认 Python 环境依赖已安装
- [ ] 确认日志级别设置为 `INFO` 或 `DEBUG`

### 执行验证
- [ ] 运行 `python main.py auto-pilot` 检查完整流程
- [ ] 运行 `python main.py reverse-validate --date 2024-01-15` 检查反向验证
- [ ] 运行压力测试脚本检查极端情况处理
- [ ] 检查生成的报告文件完整性

### 验证后检查
- [ ] 确认所有日志无 `ERROR` 级别记录
- [ ] 确认生成的CSV文件格式正确
- [ ] 确认报告中的数值无 `NaN` 或 `Inf`
- [ ] 确认配置修改已正确应用

---

*文档版本: 1.0*
*最后更新: 2026-03-17*
