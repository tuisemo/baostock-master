import gradio as gr
import pandas as pd
import os
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig

CurrentConfig.ONLINE_HOST = "https://cdn.staticfile.net/echarts/5.4.3/"
from pyecharts.charts import Kline, Bar, Grid, Scatter, Line
from backtesting import Backtest
import base64

from quant.stock_filter import update_stock_list
from quant.data_updater import update_history_data
from quant.analyzer import analyze_all_stocks
from quant.config import CONF
from quant.backtester import run_backtest, scan_today_signal
from quant.logger import logger


def ui_update_list():
    yield '开始获取并过滤 A 股股票池...请稍候。'
    try:
        update_stock_list()
        yield '✅ 股票池更新完成！您可以进入“数据更新”选项卡拉取历史数据。'
    except Exception as e:
        yield f'❌ 股票池更新失败：{e}'


def ui_update_data():
    yield '开始增量拉取最新历史 K 线数据...这可能需要一些时间，取决于网络和新增的日线数量。'
    try:
        update_history_data()
        yield '✅ 历史数据增量更新完成！您可以进入“每日量化选股”跑批模型。'
    except Exception as e:
        yield f'❌ 历史数据更新失败：{e}'


def ui_run_analyzer():
    yield '开始进行全市场多因子指标计算与打分...', None
    try:
        analyze_all_stocks()
        files = [f for f in os.listdir('.') if f.startswith('selected_stocks_') and f.endswith('.csv')]
        if not files:
            yield '✅ 分析完成。今日没有符合高标准的股票。', None
            return
        latest_file = max(files)
        df = pd.read_csv(latest_file)
        yield f'✅ 分析完成。找到强势标的并保存至 {latest_file}。', df
    except Exception as e:
        yield f'❌ 选股分析失败：{e}', None


def ui_backtest_stock(code):
    if not code:
        return '请输入股票代码，例如 sh.600000', None, None

    try:
        file_path = os.path.join(CONF.history_data.data_dir, f"{code}.csv")
        if not os.path.exists(file_path):
            return f'❌ 错误: 本地未找到 {code} 的数据，请先更新历史数据。', None, None

        result = run_backtest(code)
        if not result:
            return '❌ 错误: 回测失败，可能是数据量过少或无法计算指标。', None, None

        bt, stats = result

        stats_text = (
            f"**回测标的**: {code}\n\n"
            f"**起止时间**: {stats['Start'].strftime('%Y-%m-%d')} -> {stats['End'].strftime('%Y-%m-%d')}\n\n"
            f"**初始资金**: ￥100,000.00\n\n"
            f"**最终资金**: ￥{stats['Equity Final [$]']:,.2f}\n\n"
            f"**收益率 (Return)**: {stats['Return [%]']:.2f}%\n\n"
            f"**最大回撤 (Max Drawdown)**: {stats['Max. Drawdown [%]']:.2f}%\n\n"
            f"**交易次数**: {stats['# Trades']}\n\n"
            f"**胜率 (Win Rate)**: {stats['Win Rate [%]']:.2f}%\n\n"
        )

        df = bt._data
        df.reset_index(inplace=True)
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
        kline_data = df[['Open', 'Close', 'Low', 'High']].values.tolist()

        kline = (
            Kline()
            .add_xaxis(dates)
            .add_yaxis("K线", kline_data, itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True),
                title_opts=opts.TitleOpts(title=f"{code} 多因子策略回测分析"),
                datazoom_opts=[
                    opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 1], range_start=80, range_end=100),
                    opts.DataZoomOpts(is_show=True, xaxis_index=[0, 1], type_="slider", pos_top="95%", range_start=80, range_end=100)
                ],
                legend_opts=opts.LegendOpts(pos_top="5%", pos_left="center")
            )
        )

        ma_s_col = f"SMA_{CONF.analyzer.ma_short}"
        ma_l_col = f"SMA_{CONF.analyzer.ma_long}"
        line = Line().add_xaxis(dates)
        if ma_s_col in df.columns:
            line.add_yaxis(f"MA{CONF.analyzer.ma_short}", df[ma_s_col].tolist(), is_symbol_show=False, color="orange", label_opts=opts.LabelOpts(is_show=False))
        if ma_l_col in df.columns:
            line.add_yaxis(f"MA{CONF.analyzer.ma_long}", df[ma_l_col].tolist(), is_symbol_show=False, color="blue", label_opts=opts.LabelOpts(is_show=False))
        kline.overlap(line)

        buy_y = [None] * len(dates)
        sell_y = [None] * len(dates)

        if not stats['_trades'].empty:
            trades = stats['_trades']
            for _, row in trades.iterrows():
                entry_t = row['EntryTime']
                exit_t = row['ExitTime']
                if isinstance(entry_t, pd.Timestamp):
                    entry_idx = df[df['Date'] == entry_t].index[0]
                else:
                    entry_idx = int(entry_t)
                if isinstance(exit_t, pd.Timestamp):
                    exit_idx = df[df['Date'] == exit_t].index[0]
                else:
                    exit_idx = int(exit_t)
                buy_y[entry_idx] = row['EntryPrice']
                sell_y[exit_idx] = row['ExitPrice']

            if any(y is not None for y in buy_y):
                buy_scatter = (
                    Scatter()
                    .add_xaxis(dates)
                    .add_yaxis("买入", buy_y, symbol="triangle", symbol_size=15,
                               itemstyle_opts=opts.ItemStyleOpts(color="red"), label_opts=opts.LabelOpts(is_show=False))
                )
                kline.overlap(buy_scatter)

            if any(y is not None for y in sell_y):
                sell_scatter = (
                    Scatter()
                    .add_xaxis(dates)
                    .add_yaxis("卖出", sell_y, symbol="triangle-down", symbol_size=15,
                               itemstyle_opts=opts.ItemStyleOpts(color="green"), label_opts=opts.LabelOpts(is_show=False))
                )
                kline.overlap(sell_scatter)

        volumes = df['Volume'].tolist()
        bar = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis("成交量", volumes, label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="#8db6cd"))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category", grid_index=1, axislabel_opts=opts.LabelOpts(is_show=False)),
                yaxis_opts=opts.AxisOpts(is_scale=True, grid_index=1, axislabel_opts=opts.LabelOpts(is_show=False)),
                legend_opts=opts.LegendOpts(is_show=False)
            )
        )

        grid = (
            Grid(init_opts=opts.InitOpts(width="100%", height="800px"))
            .add(kline, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", height="65%"))
            .add(bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="80%", height="15%"))
        )

        html_file = "temp_backtest_chart.html"
        grid.render(html_file)
        with open(html_file, "r", encoding="utf-8") as f:
            raw_html = f.read()

        b64_html = base64.b64encode(raw_html.encode('utf-8')).decode('utf-8')
        iframe_html = f'<iframe src="data:text/html;base64,{b64_html}" width="100%" height="850px" frameborder="0"></iframe>'

        return stats_text, iframe_html, stats['_trades']
    except Exception as e:
        return f"绘制图表失败: {e}", None, None


def ui_scan_signals():
    yield '开始扫描全库最新买点 (这可能需要几分钟)...请稍候。', pd.DataFrame()
    files = [f for f in os.listdir('.') if f.startswith('selected_stocks_') and f.endswith('.csv')]
    if not files:
        yield '❌ 扫描失败。未找到任何有效股票池列表，请先在“数据同步中心”执行步骤一。', pd.DataFrame()
        return
    latest_file = max(files)
    df_stocks = pd.read_csv(latest_file)
    col_name = 'code' if 'code' in df_stocks.columns else 'Code' if 'Code' in df_stocks.columns else None
    if not col_name:
        yield f'❌ 扫描失败。股票池文件 {latest_file} 格式不正确。', pd.DataFrame()
        return

    stocks = df_stocks[col_name].dropna().tolist()
    results = []
    total = len(stocks)
    for i, code in enumerate(stocks):
        if i % 100 == 0:
            logger.info(f"Scanning progress: {i}/{total} ({code})")
        try:
            res = scan_today_signal(code)
            if res:
                results.append(res)
        except Exception as e:
            logger.error(f"Error scanning {code}: {e}")

    if not results:
        yield '✅ 扫描完成。今日全市场没有任何股票触发严苛的量化策略买入信号。', pd.DataFrame()
        return

    df = pd.DataFrame(results)
    yield f'✅ 扫描完成！共遍历了 {total} 只股票，发现 {len(results)} 只处于高胜率买入节点。', df


def ui_run_optimization(rounds, samples, objective):
    yield '开始多轮迭代优化...这可能需要较长时间，请耐心等待。', None
    try:
        from quant.config import CONF
        from quant.optimizer import run_optimization, save_results

        if rounds:
            CONF.optimizer.max_rounds = rounds
        if samples:
            CONF.optimizer.sample_count = samples
        if objective:
            CONF.optimizer.objective = objective

        result = run_optimization()
        save_results(result)

        baseline = result["baseline_score"]
        best = result["best_score"]
        completed = result["rounds_completed"]

        history_df = pd.DataFrame(result["history"])
        summary = (
            f"✅ 优化完成！共进行 {completed} 轮迭代。\n\n"
            f"**基线得分**: {baseline:.6f}\n"
            f"**最优得分**: {best:.6f}\n"
            f"**提升幅度**: {best - baseline:+.6f}\n\n"
            f"最优参数已自动写回 config.yaml，并在 `data/optimize_results` 目录保存详细报告。"
        )
        yield summary, history_df
    except Exception as e:
        logger.exception("优化过程中发生异常")
        yield f'❌ 优化失败：{e}', None


with gr.Blocks(title="生产级 A 股量化系统", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 📈 生产级 A 股多因子量化选股与回测系统")

    with gr.Tab("1️⃣ 数据同步中心"):
        gr.Markdown("### 步骤一：获取并清洗基础股票池")
        gr.Markdown("将查询全量 A 股字典，并按照 `config.yaml` 严格剔除微盘股（<50亿）、僵尸股（低成交量）、被操盘高危股（异常高换手）。")
        btn_update_list = gr.Button("🔄 1. 拉取/清洗最新有效股票池", variant="primary")
        txt_list_log = gr.Textbox(label="运行日志", lines=3, interactive=False)
        btn_update_list.click(fn=ui_update_list, outputs=txt_list_log)
        gr.Markdown("---")
        gr.Markdown("### 步骤二：增量拉取历史 K 线数据")
        gr.Markdown("增量模式：将为上述股票池拉取或从断点处续接（追加）最新的日均线数据，自动免除冗余抓取。")
        btn_update_data = gr.Button("📥 2. 增量更新所有股票历史 K 线", variant="secondary")
        txt_data_log = gr.Textbox(label="运行日志", lines=3, interactive=False)
        btn_update_data.click(fn=ui_update_data, outputs=txt_data_log)

    with gr.Tab("2️⃣ 每日量化多因子选股"):
        gr.Markdown("对当前所有本地数据集进行多因子矩阵（趋势因子、均值回归因子、量价因子、波幅因子）并行计算并给每一只股票综合打分。")
        btn_analyze = gr.Button("⚡ 运行今日高分优选", variant="primary")
        txt_analyze_log = gr.Textbox(label="运行状态", lines=2, interactive=False)
        df_selected = gr.Dataframe(label="🎯 今日强势选股结果 (按评分降序)", interactive=False)
        btn_analyze.click(fn=ui_run_analyzer, outputs=[txt_analyze_log, df_selected])

    with gr.Tab("3️⃣ 个股回测与买卖信号分析"):
        gr.Markdown("输入指定股票的代码（例如 `sh.600000` 或 `sz.000001`），系统将模拟该策略在过去区间的历史回测，并标记出买卖点。")
        with gr.Row():
            txt_code = gr.Textbox(label="股票代码", placeholder="例如: sh.600000", scale=4)
            btn_backtest = gr.Button("🔬 运行策略回测", variant="primary", scale=1)
        with gr.Row():
            txt_stats = gr.Markdown("等待执行...")
            plot_chart = gr.HTML(label="策略买卖点复盘")
        df_trades = gr.Dataframe(label="详细交易明细表", interactive=False)
        btn_backtest.click(fn=ui_backtest_stock, inputs=txt_code, outputs=[txt_stats, plot_chart, df_trades])

    with gr.Tab("4️⃣ 策略买点扫描 (特定股票池)"):
        gr.Markdown("对当前选股结果进行扫描，筛选出**最新交易日**符合量化策略买入条件的标的。")
        btn_scan = gr.Button("🔍 扫描最新买入信号", variant="primary")
        txt_scan_log = gr.Textbox(label="运行状态", lines=2, interactive=False)
        df_scan_result = gr.Dataframe(label="🎯 触发买入信号的标的", interactive=False)
        btn_scan.click(fn=ui_scan_signals, outputs=[txt_scan_log, df_scan_result])

    with gr.Tab("5️⃣ 策略参数自动优化"):
        gr.Markdown("### 多轮迭代优化引擎")
        gr.Markdown("基于海量数据回测，通过**策略制定 → 买卖回测 → 反哺策略优化 → 下一轮优化**的闭环模式，")
        gr.Markdown("自动调整 MA、MACD、RSI、布林带、ATR 等所有技术指标参数，以及买卖阈值、止盈止损系数，")
        gr.Markdown("寻找使**夏普比率调整值**最大化的最优参数组合，以实现盈利最大化、稳定性最高。")
        with gr.Row():
            sl_rounds = gr.Slider(label="最大优化轮数", minimum=1, maximum=10, value=5, step=1)
            sl_samples = gr.Slider(label="每轮采样股票数", minimum=50, maximum=500, value=200, step=50)
            sl_objective = gr.Dropdown(label="优化目标函数", choices=["sharpe_adj", "return", "win_rate"], value="sharpe_adj")
        btn_optimize = gr.Button("🚀 启动自动优化", variant="primary")
        txt_opt_log = gr.Textbox(label="运行状态", lines=2, interactive=False)
        df_opt_history = gr.Dataframe(label="📊 优化历史记录", interactive=False)
        btn_optimize.click(fn=ui_run_optimization, inputs=[sl_rounds, sl_samples, sl_objective], outputs=[txt_opt_log, df_opt_history])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
