# 生产级 A股量化分析工程 (Production A-Share Quant System)

这是一个基于 `baostock`、`腾讯财经` 以及 `pandas-ta` 构建的现代化量化分析与可视化回测框架。

## 主要特性 (Features)
1. **交互式 Web UI (Gradio)**: 内置现代化可视化大屏，支持一键数据同步、执行打分选股，以及对指定个股进行 `pyecharts` 交互式 K线绘制。**图表支持鼠标悬停、缩放平移，并直观地在 K 线上方标记出了策略跑出的每个“买入(红色向上)”和“卖出(绿色向下)”信号点。**
2. **多因子综合打分模型**: 融合了趋势跟踪（MA/MACD）、均值回归（RSI/布林带）、量价监控（OBV强弱）和动态止损防守（ATR），每天给出量化高分优选股列表并落盘保存。
3. **资金与活跃度防御**: 智能调用腾讯实时接口深度精准查杀微盘股（<50亿）、僵尸股（极低成交金额）以及具有高度操盘嫌疑的暴走标的（极高换手率）。
4. **全自动增量更新**: 基于 `baostock` 的历史 K 线系统能自动识别文件的最后更新时间，支持断点续传（每日仅追加未更新的数据），大幅提效。
5. **向量化回测与盘后统计**: 基于轻量级框架 `backtesting.py` 进行信号买卖演练，**按单只股票自动统计出：最终净值、收益率、最大回撤、交易次数以及历史胜率等维度的核心数据。**

## 目录结构 (Structure)
- `quant/`: 核心计算与业务逻辑子包 (模块化设计)
  - `config.py`: 核心超参数配置反序列化引擎
  - `analyzer.py`: 多因子信号打分模型与 pandas-ta 指标集
  - `backtester.py`: 原生的历史策略买卖点复盘测试引擎
  - `stock_filter.py`: Baostock/Tencent 并发资金面过滤接口
  - `data_updater.py`: K线历史数据增量拉取模块
- `config.yaml`: 规则与因子权重的动态外部配置文件
- `main.py`: 后台工程调度控制台命令总入口
- `app.py`: Gradio 交互式前端视图
- `data/`: 存放所有 K 线与输出日志

## 环境与安装 (Installation)
1. 使用新型现代化工具 `uv` 管理依赖：
```bash
uv sync   # 或者手动执行: uv add baostock pandas requests pyyaml pandas-ta tqdm gradio plotly backtesting
```

## 使用说明 (Usage)

本系统不仅支持传统的命令行操作，还提供了对业务人员高度友好的前端 Web 操作端。

### 📊 推荐：通过可视 Web 操作台使用 (Gradio UI)
你可以通过网页图形化操作数据同步、自动每日选股以及对任意股票分析和叠加指标回测复盘：
```bash
uv run python main.py ui
```
启动后自动弹出的浏览器界面包含了完整的流程指引（默认地址为 `http://127.0.0.1:7860`）。

### 💻 传统命令行控制台流 (CLI)
当你需要把本工程串入服务器定时任务 (Cron) 时，可以使用控制台模式：

1. **更新有效活股票池**
```bash
uv run python main.py update-list
```
2. **增量拉取所有标的的 K 线历史数据**
```bash
uv run python main.py update-data
```
3. **运行全市场多因子计算与选股引擎**
```bash
uv run python main.py analyze
```
> 符合要求的强势股票将按时间戳保存在根目录：`selected_stocks_YYYYMMDD.csv`。

## 配置参考 (`config.yaml`)
所有的股票过滤要求（资金过滤上限限制、安全边际限制），和 `pandas-ta` 多因子打分引擎（短中长线均线周期、MACD权重、BBAND常数）等均可在 `config.yaml` 灵活修改配置，即时生效，无需更改代码。

## 更新日志 / 最近的功能升级
- **引入 Gradio 可视化引擎 (`app.py`)**: 
  - 支持数据拉取与进度监控。
  - 支持调用量化引擎并在页面直接输出当日高分结果池。
  - **强大的交互式行情复盘 (`pyecharts`)**: 用户可输入代码，不仅能看到自动生成的胜率与盈亏统计文本，下方还会由 `pyecharts` 渲染出一个支持缩放双图（包含主 K 线、双均线以及下方的特征成交量）。最重要的是，**回测策略所产生的买卖点将直接以醒目的图标形式精准标注在主图发生交易的 K 线上。**
- **策略引擎升级 (`quant/analyzer.py` & `quant/backtester.py`)**: 
  - 彻底抛弃了单指标轮动，改写为兼具**均指回归、资金势能、动态ATR止损与趋势跟踪的多因子模型。**
- **工程结构沉淀**:
  - 原本零散的 `fetchx` 脚本已被重构成严格解耦的量化领域模型（Data Updater -> Filter -> Config -> Backtester -> Analyzer）。
