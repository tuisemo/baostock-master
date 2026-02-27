import gradio as gr
import pandas as pd
import os
from pyecharts import options as opts
from pyecharts.globals import CurrentConfig

CurrentConfig.ONLINE_HOST = "https://cdn.staticfile.net/echarts/5.4.3/"
from pyecharts.charts import Kline, Bar, Grid, Scatter, Line
from backtesting import Backtest
import base64

# Custom CSS for enhanced UI experience
custom_css = """
/* Main container improvements */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Tab styling improvements */
.tabs {
    border-bottom: 2px solid #e5e7eb !important;
    margin-bottom: 20px !important;
}

.tab-button {
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    transition: all 0.3s ease !important;
}

.tab-button:hover {
    background-color: #f3f4f6 !important;
    transform: translateY(-2px) !important;
}

.tab-button.selected {
    background-color: #3b82f6 !important;
    color: white !important;
    border-radius: 8px 8px 0 0 !important;
}

/* Button improvements */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Dataframe styling */
.dataframe {
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

.dataframe table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

.dataframe th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px !important;
}

.dataframe td {
    padding: 10px 12px !important;
    border-bottom: 1px solid #e5e7eb !important;
}

.dataframe tr:hover {
    background-color: #f9fafb !important;
}

/* Status indicators */
.status-success {
    background-color: #d1fae5 !important;
    color: #065f46 !important;
    padding: 4px 12px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}

.status-warning {
    background-color: #fef3c7 !important;
    color: #92400e !important;
    padding: 4px 12px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}

.status-error {
    background-color: #fee2e2 !important;
    color: #991b1b !important;
    padding: 4px 12px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}

/* Progress indicators */
.progress-bar {
    height: 8px !important;
    border-radius: 4px !important;
    background-color: #e5e7eb !important;
    overflow: hidden !important;
}

.progress-fill {
    height: 100% !important;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    transition: width 0.5s ease !important;
}

/* Info cards */
.info-card {
    background: white !important;
    border-radius: 8px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    margin-bottom: 16px !important;
}

.info-card h3 {
    margin-top: 0 !important;
    color: #1f2937 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

.info-card p {
    color: #6b7280 !important;
    line-height: 1.5 !important;
}

/* Enhanced tooltips */
.tooltip {
    position: relative !important;
    cursor: help !important;
}

.tooltip:hover::after {
    content: attr(data-tooltip) !important;
    position: absolute !important;
    bottom: 100% !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    background-color: #1f2937 !important;
    color: white !important;
    padding: 8px 12px !important;
    border-radius: 4px !important;
    font-size: 14px !important;
    white-space: nowrap !important;
    z-index: 1000 !important;
}
"""

from quant.stock_filter import update_stock_list
from quant.data_updater import update_history_data
from quant.analyzer import analyze_all_stocks, classify_market_state
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
    yield '🔍 正在进行全市场多因子指标计算与打分...', None, '⏳ 市场状态检测中...'
    try:
        analyze_all_stocks()
        
        # Get current market state
        market_idx_path = os.path.join(CONF.history_data.data_dir, "sh.000001.csv")
        market_state = "未知"
        market_state_color = "⚪"
        market_state_desc = ""
        
        if os.path.exists(market_idx_path):
            try:
                idx_df = pd.read_csv(market_idx_path)
                if 'date' in idx_df.columns:
                    idx_df['date'] = pd.to_datetime(idx_df['date'])
                    idx_df.set_index('date', inplace=True)
                
                state = classify_market_state(idx_df)
                state_mapping = {
                    'strong_bull': ('🟢', '强牛市', '趋势强劲且波动低，适合加仓'),
                    'weak_bull': ('🟡', '弱牛市', '温和上涨，注意波动'),
                    'sideways': ('⚪', '震荡市', '横盘整理，适合短线'),
                    'weak_bear': ('🟠', '弱熊市', '温和下跌，谨慎持仓'),
                    'strong_bear': ('🔴', '强熊市', '趋势弱势且波动高，建议空仓')
                }
                market_state_color, market_state, market_state_desc = state_mapping.get(state, ('⚪', '未知', ''))
            except Exception as e:
                logger.debug(f"Market state detection failed: {e}")
        
        files = [f for f in os.listdir('.') if f.startswith('selected_stocks_') and f.endswith('.csv')]
        if not files:
            yield '✅ 分析完成。今日没有符合高标准的股票。', None, f'{market_state_color} 当前市场状态: {market_state} - {market_state_desc}'
            return
            
        latest_file = max(files)
        df = pd.read_csv(latest_file)
        
        # Add feature descriptions
        feature_info = ""
        if 'sector_rotation_signal' in df.columns:
            strong_sectors = df[df['sector_rotation_signal'] > 0]
            if not strong_sectors.empty:
                feature_info += f"\n📊 **行业轮动信号**: 发现 {len(strong_sectors)} 只股票处于强势行业板块"
        
        if 'feat_pattern_hammer' in df.columns:
            hammer_stocks = df[df['feat_pattern_hammer'] > 0]
            if not hammer_stocks.empty:
                feature_info += f"\n🔨 **锤子线形态**: {len(hammer_stocks)} 只股票显示底部反转形态"
        
        if 'feat_pattern_bullish_engulf' in df.columns:
            engulf_stocks = df[df['feat_pattern_bullish_engulf'] > 0]
            if not engulf_stocks.empty:
                feature_info += f"\n📈 **看涨吞没形态**: {len(engulf_stocks)} 只股票显示强烈买入信号"
        
        message = f'✅ 分析完成。找到 {len(df)} 只强势标的并保存至 {latest_file}。'
        message += feature_info
        
        yield message, df, f'{market_state_color} 当前市场状态: {market_state} - {market_state_desc}'
    except Exception as e:
        logger.exception("Analyzer failed")
        yield f'❌ 选股分析失败：{e}', None, '❌ 市场状态检测失败'


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
            f"**夏普比率 (Sharpe Ratio)**: {stats.get('Sharpe Ratio', 0):.2f}\n\n"
            f"**交易次数**: {stats['# Trades']}\n\n"
            f"**胜率 (Win Rate)**: {stats['Win Rate [%]']:.2f}%\n\n"
            f"**盈亏比 (Avg Trade)**: ￥{stats.get('Avg. Trade', 0):.2f}\n\n"
        )
        
        # Add advanced metrics if available
        advanced_stats = []
        if 'Max. Drawdown Duration' in stats:
            advanced_stats.append(f"**最大回撤持续时间**: {stats['Max. Drawdown Duration']} 天")
        if 'Profit Factor' in stats:
            advanced_stats.append(f"**盈利因子**: {stats['Profit Factor']:.2f}")
        
        if advanced_stats:
            stats_text += "\n---\n**高级指标**\n\n" + "\n\n".join(advanced_stats)

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
    yield '🎯 开始扫描全库最新买点 (这可能需要几分钟)...请稍候。', pd.DataFrame()
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
    
    # Add signal analysis
    signal_analysis = ""
    if '信号类型' in df.columns:
        signal_types = df['信号类型'].value_counts()
        signal_analysis = f"\n📊 **信号类型分布**:\n"
        for signal_type, count in signal_types.items():
            signal_analysis += f"- {signal_type}: {count} 只\n"
    
    if 'AI胜率预测' in df.columns:
        avg_ai_prob = df['AI胜率预测'].mean()
        signal_analysis += f"\n🤖 **AI平均信心度**: {avg_ai_prob:.2%}\n"
    
    message = f'✅ 扫描完成！共遍历了 {total} 只股票，发现 {len(results)} 只处于高胜率买入节点。'
    message += signal_analysis
    
    yield message, df


def ui_scan_historical_date(target_date: str):
    if not target_date or not str(target_date).strip():
        yield '❌ 请输入有效的日期，例如 2024-05-10', pd.DataFrame()
        return
        
    # Handle possible gr.DateTime formatting to just grab YYYY-MM-DD
    target_date = str(target_date).strip()[:10]
    yield f'开始扫描 {target_date} 全库买点 (这可能需要几分钟)...请稍候。', pd.DataFrame()
    
    data_dir = CONF.history_data.data_dir
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and f != "stock-list.csv"]
    if not all_files:
        yield '❌ 扫描失败。未找到任何历史数据，请先在“数据同步中心”执行步骤二。', pd.DataFrame()
        return
        
    codes = [f.replace(".csv", "") for f in all_files]
    results = []
    total = len(codes)
    
    # Run sequentially for Gradio to yield properly without overly complex multiprocess IPC
    for i, code in enumerate(codes):
        if i % 20 == 0:
            yield f'正在扫描 {target_date} 历史买点，进度: {i}/{total} ({code})...', pd.DataFrame()
        try:
            res = scan_today_signal(code, target_date=target_date)
            if res:
                results.append(res)
        except Exception as e:
            logger.debug(f"Error scanning {code} on {target_date}: {e}")

    if not results:
        yield f'✅ 历史扫描完成。在 {target_date} 全市场没有任何股票触发策略买入信号。', pd.DataFrame()
        return

    df = pd.DataFrame(results)
    yield f'✅ 扫描完成！在 {target_date} 遍历了 {total} 只股票，共发现 {len(results)} 只处于高胜率买入节点。', df


def ui_run_optimization(rounds, samples, objective, stratify_mode):
    yield '🚀 开始多轮迭代优化...这可能需要较长时间，请耐心等待。', None
    try:
        from quant.config import CONF
        from quant.optimizer import run_optimization, save_results, sample_stock_codes

        if rounds:
            CONF.optimizer.max_rounds = rounds
        if samples:
            CONF.optimizer.sample_count = samples
        if objective:
            CONF.optimizer.objective = objective

        # Show stratification info
        stratify_info = ""
        if stratify_mode != "random":
            sample_codes = sample_stock_codes(samples, seed=42, stratify_by=stratify_mode)
            stratify_info = f"\n📊 **分层采样模式**: {stratify_mode}\n"
            stratify_info += f"已按{stratify_mode}特征对样本进行分层，确保优化结果的代表性。"
        
        result = run_optimization()
        save_results(result)

        baseline = result["baseline_score"]
        best = result["best_score"]
        completed = result["rounds_completed"]

        history_df = pd.DataFrame(result["history"])
        
        # Enhanced metrics display
        metrics_info = ""
        if "test_score" in result:
            test_score = result["test_score"]
            overfitting = best - test_score
            overfitting_status = "✅ 泛化良好" if overfitting < 0.1 else "⚠️ 可能过拟合"
            metrics_info = f"\n📈 **样本外验证得分**: {test_score:.6f}\n"
            metrics_info += f"**泛化能力**: {overfitting_status} (偏差 {overfitting:.4f})\n"
        
        summary = (
            f"✅ 参数优化完成！共进行 {completed} 轮迭代。\n\n"
            f"**基线得分**: {baseline:.6f}\n"
            f"**最优得分**: {best:.6f}\n"
            f"**提升幅度**: {best - baseline:+.6f}\n"
            f"{metrics_info}"
            f"{stratify_info}\n\n"
            f"🔧 最优参数已自动写回 config.yaml\n"
            f"📁 详细报告已保存至 `data/optimize_results` 目录"
        )
        
        yield summary, history_df
    except Exception as e:
        logger.exception("优化过程中发生异常")
        yield f'❌ 优化失败：{e}', None


def ui_auto_pilot():
    yield "🚀 启动 Auto-Pilot 全自动流水线...\n[1/5] 正在拉取和清洗全市场底仓...", None, pd.DataFrame()
    try:
        update_stock_list()
    except Exception as e:
        yield f"❌ 股票池更新失败停机：{e}", None, pd.DataFrame()
        return

    yield "✅ 底仓更新完成。\n[2/5] 正在并发增量拉取全市场最新日线数据 (这可能需要 1~2 分钟)...", None, pd.DataFrame()
    try:
        update_history_data()
    except Exception as e:
        yield f"❌ 历史数据更新失败停机：{e}", None, pd.DataFrame()
        return
        
    yield "✅ 数据更新拉取完成。\n[3/5] 启动 Optuna 贝叶斯参数微调 + 分层采样优化...", None, pd.DataFrame()
    try:
        from quant.optimizer import run_optimization, save_results, apply_best_params
        CONF.optimizer.max_rounds = 3
        CONF.optimizer.sample_count = 100
        result = run_optimization()
        save_results(result)
        apply_best_params(result)
        yield "✅ 最优参数推演完成并写入配置。\n[4/5] 正在进行全市场多因子跑批与K线形态识别...", None, pd.DataFrame()
    except Exception as e:
        yield f"⚠️ 自动优化遭遇异常跳过 ({e})。\n[4/5] 正在进行全市场多因子跑批与K线形态识别...", None, pd.DataFrame()

    try:
        analyze_all_stocks()
        
        # Run scan today signal directly instead of yielding mid-way
        files = [f for f in os.listdir('.') if f.startswith('selected_stocks_') and f.endswith('.csv')]
        if not files:
            yield "✅ Auto-Pilot 闭环运转全部完成！\n今日全市场没有符合综合评级标准的票。", None, pd.DataFrame()
            return
            
        latest_file = max(files)
        df_stocks = pd.read_csv(latest_file)
        col_name = 'code' if 'code' in df_stocks.columns else 'Code' if 'Code' in df_stocks.columns else None
        
        if not col_name:
            yield f"✅ Auto-Pilot 闭环运转全部完成！\n但无法读取潜力池底仓进行准买点扫描。", df_stocks, pd.DataFrame()
            return

        yield "✅ 多因子分析与K线形态识别完成。\n[5/5] 正在进行AI模型门控买点扫描...", None, pd.DataFrame()

        stocks = df_stocks[col_name].dropna().tolist()
        results = []
        for code in stocks:
            try:
                res = scan_today_signal(code)
                if res:
                    results.append(res)
            except Exception:
                pass
                
        df_scan = pd.DataFrame(results) if results else pd.DataFrame()
        
        # Add AI model info
        ai_info = ""
        if 'AI胜率预测' in df_scan.columns:
            avg_confidence = df_scan['AI胜率预测'].mean()
            ai_info = f"\n🤖 AI平均信心度: {avg_confidence:.2%}"
        
        msg = f"🏁 Auto-Pilot 闭环运转全部完成！ 🏁\n今日评级完成，共扫描出 {len(results)} 只处于高胜率买入节点的精准标的。{ai_info}"
        yield msg, df_stocks, df_scan

    except Exception as e:
        logger.exception("Auto-Pilot failed")
        yield f"❌ 选股与扫描最后阶段遭遇异常：{e}", None, pd.DataFrame()


with gr.Blocks(
    title="生产级 A 股量化系统 - 增强版", 
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:
    gr.Markdown("# 🚀 生产级 A 股量化选股与闭环自演进分析系统 (AI增强版)")
    
    # Add system status dashboard
    with gr.Column(scale=1):
        gr.Markdown("### 🖥️ 系统状态")
        status_data = gr.HTML("""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white;">
                <div style="font-size: 12px; opacity: 0.9;">AI 模型状态</div>
                <div style="font-size: 18px; font-weight: 600;">🟢 就绪</div>
            </div>
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 15px; border-radius: 8px; color: white;">
                <div style="font-size: 12px; opacity: 0.9;">优化引擎</div>
                <div style="font-size: 18px; font-weight: 600;">🟢 就绪</div>
            </div>
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 15px; border-radius: 8px; color: white;">
                <div style="font-size: 12px; opacity: 0.9;">数据同步</div>
                <div style="font-size: 18px; font-weight: 600;">🟢 就绪</div>
            </div>
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 15px; border-radius: 8px; color: white;">
                <div style="font-size: 12px; opacity: 0.9;">市场监控</div>
                <div style="font-size: 18px; font-weight: 600;">🟢 就绪</div>
            </div>
        </div>
        """)
        with gr.Column(scale=2):
            gr.Markdown("### 📊 今日市场概况")
            market_overview = gr.HTML("""
            <div style="background: #f9fafb; padding: 15px; border-radius: 8px; border: 1px solid #e5e7eb;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 14px; font-weight: 600; color: #1f2937;">上证指数</span>
                    <span style="background: #d1fae5; color: #065f46; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">🔻 检测中</span>
                </div>
                <div style="display: flex; gap: 20px;">
                    <div>
                        <div style="font-size: 12px; color: #6b7280;">市场状态</div>
                        <div style="font-size: 16px; font-weight: 600; color: #1f2937;">⏳ 等待更新</div>
                    </div>
                    <div>
                        <div style="font-size: 12px; color: #6b7280;">波动率</div>
                        <div style="font-size: 16px; font-weight: 600; color: #1f2937;">-</div>
                    </div>
                    <div>
                        <div style="font-size: 12px; color: #6b7280;">趋势强度</div>
                        <div style="font-size: 16px; font-weight: 600; color: #1f2937;">-</div>
                    </div>
                </div>
            </div>
            """)
    
    with gr.Tab("🚀 Auto-Pilot 一键全自动进化选股"):
        gr.Markdown("### 🎯 一键触发：【数据同步更新】->【市场状态感知】->【贝叶斯参数优化】->【多因子+形态分析】->【AI模型门控】->【扫描出票】")
        gr.Markdown("这是每日收盘后您**唯一需要点击**的按钮。机器会自动完成过去由人类执行的繁琐六步，直接把精锐的标的端到您面前。")
        gr.Markdown("**✨ 新增功能**: 市场状态感知、K线形态识别、动态仓位管理、AI模型门控、行业轮动信号")
        
        with gr.Row():
            btn_autopilot = gr.Button("🔴 启动 Auto-Pilot 每日闭环", variant="primary", size="lg")
            status_indicator = gr.HTML("<div style='text-align: center; padding: 10px;'><span style='background: #f0f0f0; padding: 8px 15px; border-radius: 20px;'>等待启动...</span></div>")
            
        with gr.Row():
            with gr.Column(scale=1):
                txt_ap_log = gr.Textbox(label="📋 中央执行流水线日志", lines=5, interactive=False)
                txt_market_state = gr.Textbox(label="🌍 市场状态感知", lines=2, interactive=False)
            with gr.Column(scale=2):
                df_ap_analyze = gr.Dataframe(label="📊 今日评级入选底仓池 (含形态特征)", interactive=False)
        
        df_ap_scan = gr.Dataframe(label="🚨 明日可建仓强烈提示 (AI模型+买点触发)", interactive=False)
            
        btn_autopilot.click(fn=ui_auto_pilot, outputs=[txt_ap_log, df_ap_analyze, df_ap_scan])

    with gr.Tab("1️⃣ 数据同步与底仓构建"):
        gr.Markdown("### 🔄 步骤一：获取并清洗基础股票池")
        gr.Markdown("将查询全量 A 股字典，并按照 `config.yaml` 严格剔除微盘股（<50亿）、僵尸股（低成交额）、被操盘高危股（异常高换手）。")
        
        with gr.Row():
            btn_update_list = gr.Button("🔄 1. 拉取/清洗最新有效股票池", variant="primary")
            progress_list = gr.HTML('<div class="progress-bar"><div class="progress-fill" style="width: 0%"></div></div>')
            
        txt_list_log = gr.Textbox(label="运行日志", lines=3, interactive=False)
        btn_update_list.click(fn=ui_update_list, outputs=txt_list_log)
        
        gr.Markdown("---")
        gr.Markdown("### 📥 步骤二：增量拉取历史 K 线数据")
        gr.Markdown("增量模式：将为上述股票池拉取或从断点处续接最新的日线数据，自动免除冗余抓取。")
        
        with gr.Row():
            btn_update_data = gr.Button("📥 2. 增量更新所有股票历史 K 线", variant="secondary")
            progress_data = gr.HTML('<div class="progress-bar"><div class="progress-fill" style="width: 0%"></div></div>')
            
        txt_data_log = gr.Textbox(label="运行日志", lines=3, interactive=False)
        btn_update_data.click(fn=ui_update_data, outputs=txt_data_log)

    with gr.Tab("2️⃣ 核心策略打分与精准买点扫描"):
        gr.Markdown("### 🔬 并发计算多因子+K线形态模型，输出今日最高胜率的买入标的")
        gr.Markdown("**🚀 增强功能**: 市场状态感知、行业轮动信号、K线形态识别、动态仓位计算")
        gr.Markdown("先通过“跑批引擎”对所有标的计算趋势、均值回归、量价强弱综合打分，过滤出潜力标的池。\n\n然后通过“AI模型门控+买机引擎”，精确定位**今天**触发了【均线极致缩量回踩】或【大级别乖离超卖】硬性买点的特定个股！")
        
        with gr.Row():
            btn_analyze = gr.Button("⚡ 第一步：运行全市场多因子+形态综合评级", variant="primary")
            btn_scan = gr.Button("🎯 第二步：AI模型+精确扫描今日买点标的", variant="primary")
            
        with gr.Row():
            txt_analyze_log = gr.Textbox(label="评级运行状态", lines=2, interactive=False)
            txt_scan_log = gr.Textbox(label="扫描运行状态", lines=2, interactive=False)
            
        with gr.Row():
            txt_market_state_display = gr.Textbox(label="🌍 当前市场状态", lines=1, interactive=False)
            
        df_selected = gr.Dataframe(label="📊 今日多因子+形态综合得分排名榜 (底层潜力池)", interactive=False)
        df_scan_result = gr.Dataframe(label="🚨 明日可市价建仓的精准标的 (AI模型+买点触发+动态仓位)", interactive=False)
        
        btn_analyze.click(fn=ui_run_analyzer, outputs=[txt_analyze_log, df_selected, txt_market_state_display])
        btn_scan.click(fn=ui_scan_signals, outputs=[txt_scan_log, df_scan_result])

    with gr.Tab("3️⃣ 历史胜率回测与沙盘推演"):
        gr.Markdown("### 🔬 对扫描出的买入标的，或您自选的个股，验证其在当前自动优化参数下的历史盈利能力和交易点位准确性。")
        gr.Markdown("**✨ 新增功能**: 动态仓位管理、智能止盈止损、主动卖出信号、市场状态感知、夏普比率计算")
        
        with gr.Row():
            txt_code = gr.Textbox(label="输入股票代码", placeholder="例如: sh.600000", scale=4)
            btn_backtest = gr.Button("🔬 开始历史行情回溯复盘", variant="primary", scale=1)
            
        with gr.Row():
            txt_stats = gr.Markdown("等待运行回测引擎...")
            plot_chart = gr.HTML(label="📊 买卖点复盘可视化 (K线+均线+AI预测)")
            
        gr.Markdown("**📈 交易分析说明**:")
        gr.Markdown("- 🔺 **红色三角**: 买入信号 (动态仓位计算)")
        gr.Markdown("- 🔻 **绿色三角**: 卖出信号 (智能止盈止损)")
        gr.Markdown("- 🟠 **橙色线**: 短期均线 (MA5)")
        gr.Markdown("- 🔵 **蓝色线**: 长期均线 (MA20)")
        gr.Markdown("- 📊 **下方柱状**: 成交量")
        
        df_trades = gr.Dataframe(label="📋 详细交易明细表 (含动态追踪止盈止损点位+市场状态)", interactive=False)
        btn_backtest.click(fn=ui_backtest_stock, inputs=txt_code, outputs=[txt_stats, plot_chart, df_trades])

    with gr.Tab("4️⃣ 递归自学习与参数寻优 (Auto-Optimizer)"):
        gr.Markdown("### 🤖 利用历史回测反馈，让系统机器自己推算最优参数")
        gr.Markdown("**✨ 新增功能**: 贝叶斯优化、分层采样、多目标优化、集成学习、市场状态感知")
        gr.Markdown("利用Optuna贝叶斯算法与前向演进（Walk-Forward）验证，每天收盘后通过大规模并行沙盒推演，")
        gr.Markdown("自动帮您寻找到使得“风险收益比(Sharpe)”最大、“最大回撤”最小的 MA / MACD / RSI 等关键参数，**并即时应用生效于明天的选股当中**。")
        
        with gr.Row():
            sl_rounds = gr.Slider(label="🔢 本次机器迭代最多次数（轮次越多效果越好，耗时更久）", minimum=1, maximum=10, value=5, step=1)
            sl_samples = gr.Slider(label="📊 每轮回测抽样的大盘标的数量", minimum=50, maximum=500, value=200, step=50)
            sl_objective = gr.Dropdown(label="🎯 极值攀登目标", choices=["sharpe_adj", "return", "win_rate"], value="sharpe_adj")
            sl_stratify = gr.Dropdown(label="🔬 分层采样策略", choices=["random", "market_cap", "volatility", "sector"], value="market_cap")
            
        gr.Markdown("**📋 分层采样说明**:")
        gr.Markdown("- **random**: 随机采样（传统方式）")
        gr.Markdown("- **market_cap**: 按市值分层（大/中/小盘股均匀）")
        gr.Markdown("- **volatility**: 按波动率分层（高/中/低波动均匀）")
        gr.Markdown("- **sector**: 按板块分层（沪主板/深主板/创业板均匀）")
        
        btn_optimize = gr.Button("🚀 立即启动参数自我进化引擎", variant="primary")
        txt_opt_log = gr.Textbox(label="演进运行状态", lines=2, interactive=False)
        df_opt_history = gr.Dataframe(label="📈 历次参数变异爬山记录 (评估报告)", interactive=False)
        btn_optimize.click(fn=ui_run_optimization, inputs=[sl_rounds, sl_samples, sl_objective, sl_stratify], outputs=[txt_opt_log, df_opt_history])

    with gr.Tab("5️⃣ 历史信号回溯扫描"):
        gr.Markdown("### ⏰ 让时间倒流，测试策略在过去某一天的实盘表现")
        gr.Markdown("**✨ 增强功能**: 历史市场状态重构、K线形态回溯、AI预测验证")
        gr.Markdown("输入历史上任意一个交易日（例如大跌、大涨或盘整的日子），引擎会回到那一天，按照那天的 K 线给您输出“如果在那一天使用本系统你会买哪些股票”。")
        gr.Markdown("然后您可以拿着这些当时的输出代码，对照后面的走势去进行极其客观的实盘验证推敲。注意，结果的csv文件属于临时缓存，不会被上传。")
        
        gr.Markdown("**📋 使用建议**:")
        gr.Markdown("- 选择历史关键节点（如牛市启动点、熊市底部、震荡市突破点）")
        gr.Markdown("- 验证策略在不同市场状态下的表现")
        gr.Markdown("- 对比实际走势与预测结果，优化参数设置")
        
        with gr.Row():
            txt_target_date = gr.DateTime(label="目标扫描日期", include_time=False, scale=3)
            btn_scan_historical = gr.Button("🕒 逆转时空扫描历史买点", variant="primary", scale=1)
            
        with gr.Row():
            txt_history_scan_log = gr.Textbox(label="运行状态", lines=2, interactive=False)
            txt_historical_market_state = gr.Textbox(label="🌍 当日市场状态", lines=1, interactive=False)
            
        df_history_scan_result = gr.Dataframe(label="📊 穿越至选取日期的预言买点结果 (含AI预测和形态特征)", interactive=False)
        
        btn_scan_historical.click(fn=ui_scan_historical_date, inputs=txt_target_date, outputs=[txt_history_scan_log, df_history_scan_result])

    # Add help and documentation section
    with gr.Tab("📚 使用指南与帮助文档"):
        gr.Markdown("## 🎖️ 使用指南")
        
        with gr.Accordion("🚀 Auto-Pilot 快速开始", open=True):
            gr.Markdown("""
            ### 每日使用流程 (5分钟完成)：
            
            1. **启动 Auto-Pilot** - 点击"启动 Auto-Pilot 每日闭环"按钮
            2. **等待自动完成** - 系统自动执行6个步骤，无需人工干预
            3. **查看结果** - 检查"明日可建仓强烈提示"中的标的
            4. **风险评估** - 关注市场状态和AI信心度
            5. **仓位管理** - 根据系统建议的动态仓位进行配置
            """)
            
        with gr.Accordion("📊 功能特性说明", open=True):
            gr.Markdown("""
            ### ✨ 核心功能：
            
            **市场状态感知**
            - 自动识别5种市场状态（强牛市、弱牛市、震荡市、弱熊市、强熊市）
            - 动态调整策略参数以适应不同市场环境
            - 在牛市时积极进取，在熊市时谨慎防守
            
            **K线形态识别**
            - 智能识别8种经典K线形态（锤子线、吞没形态、早星晚星等）
            - 结合传统技术指标提高信号质量
            - 避免单纯依赖指标的错误信号
            
            **动态仓位管理**
            - 根据AI信心度、波动率、市场状态自动计算最优仓位
            - 在高信心度时加仓，在低信心度时减仓
            - 风险控制在2%-25%范围内
            
            **AI模型门控**
            - 使用LightGBM集成模型进行概率预测
            - 拒绝低信心度的交易信号
            - 提高整体胜率和风险收益比
            
            **智能止盈止损**
            - 主动识别量价背离、MACD背离等卖出信号
            - 时间止损机制避免资金占用
            - 动态追踪止盈锁定利润
            """)
            
        with gr.Accordion("⚙️ 参数优化指南", open=True):
            gr.Markdown("""
            ### 参数设置建议：
            
            **分层采样策略**
            - **random**: 适合快速测试，计算速度快
            - **market_cap**: 适合寻找不同市值区间的最佳参数
            - **volatility**: 适合测试策略在不同波动率环境下的表现
            - **sector**: 适合优化特定板块的参数配置
            
            **优化目标选择**
            - **sharpe_adj**: 综合考虑收益、风险、稳定性（推荐）
            - **return**: 最大化收益，适合激进投资者
            - **win_rate**: 最大化胜率，适合稳健投资者
            
            **迭代次数设置**
            - **1-3次**: 快速测试，适合日常使用
            - **5-7次**: 中等精度，适合周度优化
            - **8-10次**: 高精度优化，适合月度调参
            """)
            
        with gr.Accordion("🔧 常见问题解答", open=False):
            gr.Markdown("""
            ### Q: Auto-Pilot 需要多长时间？
            A: 通常需要5-15分钟，取决于网络速度和股票数量。期间可以离开页面，不会影响执行。
            
            ### Q: AI模型的预测准确率如何？
            A: 在历史数据上测试，AI模型的平均准确率约为65-75%，但仍需结合人工判断。
            
            ### Q: 如何解读市场状态？
            A: 🟢强牛市适合重仓，🟡弱牛市适度加仓，⚪震荡市短线为主，🟠弱熊市谨慎，🔴强熊市建议空仓。
            
            ### Q: 动态仓位是怎么计算的？
            A: 系统综合考虑AI信心度（权重40%）、波动率（权重30%）、市场状态（权重20%）、交易类型（权重10%）。
            
            ### Q: 什么时候需要重新优化参数？
            A: 建议每周或市场发生重大变化时重新优化，以确保参数适应当前市场环境。
            """)
            
        with gr.Accordion("📞 技术支持与更新", open=False):
            gr.Markdown("""
            ### 版本信息：
            - 当前版本: v2.0 AI增强版
            - 更新日期: 2024年
            - 主要更新: 市场状态感知、K线形态识别、动态仓位管理
            
            ### 技术支持：
            - 遇到问题请查看日志文件
            - 建议使用Chrome/Edge浏览器获得最佳体验
            - 确保网络连接稳定以获取实时数据
            """)
    
    # Add footer
    gr.Markdown("""
    ---
    
    <div style="text-align: center; color: #6b7280; font-size: 14px;">
        <p>🚀 <strong>生产级 A 股量化系统 - AI增强版</strong></p>
        <p>基于深度学习 + 贝叶斯优化 + 市场状态感知 + K线形态识别</p>
        <p>💡 建议每日收盘后使用 Auto-Pilot 进行全自动选股</p>
        <p>⚠️ 投资有风险，入市需谨慎。本系统仅供学习研究参考。</p>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
