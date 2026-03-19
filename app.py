"""
量化选股系统 - 可视化界面
========================
核心功能：
1. 今日推荐股票 - 一键扫描
2. 个股分析 - 量价走势 + 买卖信号
3. 回测 - 策略表现验证
4. 数据更新 - 一键更新

启动方式: python main.py ui
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 确保quant模块可导入
sys.path.insert(0, str(Path(__file__).parent))

from quant.app.backtester import run_backtest
from quant.data.data_updater import update_history_data
from quant.data.stock_filter import update_stock_list
from quant.infra.config import CONF
from quant.infra.logger import logger


# =========================
# 配置
# =========================

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TODAY_STR = datetime.now().strftime("%Y-%m-%d")


# =========================
# 工具函数
# =========================

def _calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """计算买卖信号"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    
    if "close" in df.columns:
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        
        # 生成信号
        df["buy_signal"] = (
            (df["ma5"] > df["ma20"]) & 
            (df["ma5"].shift(1) <= df["ma20"].shift(1))
        )
        df["sell_signal"] = (
            (df["ma5"] < df["ma20"]) & 
            (df["ma5"].shift(1) >= df["ma20"].shift(1))
        )
    
    return df


def load_stock_data_cached(code: str, lookback_days: int = 100) -> Optional[pd.DataFrame]:
    """加载股票数据（带缓存）"""
    cache_file = CACHE_DIR / f"{code}_chart.csv"
    
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=1):
            df = pd.read_csv(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            return df
    
    data_path = Path(CONF.history_data.data_dir) / f"{code}.csv"
    if not data_path.exists():
        return None
    
    df = pd.read_csv(data_path)
    if df.empty or "date" not in df.columns:
        return None
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    cutoff = datetime.now() - timedelta(days=lookback_days)
    df = df[df["date"] >= cutoff]
    
    df.to_csv(cache_file, index=False)
    
    return df


# =========================
# 核心功能
# =========================

def scan_today_stocks(date: str = None, min_ai_prob: float = 0.50, max_results: int = 20) -> tuple:
    """扫描今日推荐股票"""
    if date is None:
        date = TODAY_STR
    
    try:
        import subprocess
        
        logger.info(f"开始扫描日期: {date}")
        
        result = subprocess.run(
            [sys.executable, "main.py", "scan-date", "--date", date],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            logger.error(f"扫描失败: {result.stderr}")
        
        hist_file = Path("data") / f"historical_scan_{date.replace('-', '')}.csv"
        
        if not hist_file.exists():
            return pd.DataFrame(), f"未找到 {date} 的扫描结果\n请先运行: python main.py scan-date --date {date}"
        
        result_df = pd.read_csv(hist_file)
        
        if result_df is None or result_df.empty:
            return pd.DataFrame(), f"扫描日期 {date} 未发现符合条件的股票"
        
        if "ai_prob" in result_df.columns:
            result_df = result_df[result_df["ai_prob"] >= min_ai_prob]
        
        result_df = result_df.head(max_results)
        
        display_cols = ["code", "date", "close", "ai_prob", "signal_type", "expected_value_pct"]
        available_cols = [c for c in display_cols if c in result_df.columns]
        display_df = result_df[available_cols].copy()
        display_df.columns = ["代码", "信号日期", "收盘价", "AI胜率", "信号类型", "期望收益(%)"][:len(available_cols)]
        
        if "AI胜率" in display_df.columns:
            display_df["AI胜率"] = (pd.to_numeric(display_df["AI胜率"], errors="coerce") * 100).round(1).astype(str) + "%"
        if "期望收益(%)" in display_df.columns:
            display_df["期望收益(%)"] = pd.to_numeric(display_df["期望收益(%)"], errors="coerce").round(2).astype(str) + "%"
        
        summary = f"""扫描结果
-------------------------
扫描日期: {date}
发现信号: {len(display_df)} 只
AI胜率阈值: {min_ai_prob*100:.0f}%

Top 3 推荐
"""
        
        if len(display_df) >= 3:
            top3 = display_df.head(3)
            for i, (_, row) in enumerate(top3.iterrows()):
                ai_prob_str = row.get('AI胜率', 'N/A')
                summary += f"\n{i+1}. {row['代码']} | AI胜率 {ai_prob_str}"
        
        return display_df, summary
        
    except Exception as e:
        import traceback
        return pd.DataFrame(), f"扫描失败: {str(e)}"


def analyze_stock(code: str, lookback_days: int = 100) -> tuple:
    """分析个股 - 量价走势 + 买卖信号"""
    if not code:
        return "请输入股票代码", None
    
    df = load_stock_data_cached(code, lookback_days)
    if df is None or df.empty:
        return f"未找到股票 {code} 的数据\n请先更新股票数据", None
    
    df = _calculate_signals(df)
    fig = create_stock_chart(df, code)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    latest_date = latest.get('date', 'N/A')
    if hasattr(latest_date, 'strftime'):
        latest_date = latest_date.strftime('%Y-%m-%d')
    
    price_change = ((latest.get('close', 0) / prev.get('close', 1)) - 1) * 100
    
    report = f"""{code} 个股分析报告
-------------------------

最新数据: {latest_date}

价格信息
  收盘价: {latest.get('close', 0):.2f}
  涨跌幅: {price_change:.2f}%

技术指标
  MA5:  {latest.get('ma5', 0):.2f}
  MA20: {latest.get('ma20', 0):.2f}
  MA60: {latest.get('ma60', 0):.2f}
  RSI:  {latest.get('rsi', 0):.1f}

信号状态
"""
    
    if latest.get("buy_signal", False):
        report += "  [买入] MA金叉形成\n"
    else:
        report += "  [无] 买入信号未触发\n"
    
    if latest.get("sell_signal", False):
        report += "  [卖出] MA死叉形成\n"
    else:
        report += "  [无] 卖出信号未触发\n"
    
    rsi = latest.get("rsi", 50)
    if rsi < 30:
        report += f"  RSI超卖: {rsi:.1f} (可能反弹)\n"
    elif rsi > 70:
        report += f"  RSI超买: {rsi:.1f} (注意回调)\n"
    else:
        report += f"  RSI中性: {rsi:.1f}\n"
    
    return report, fig


def create_stock_chart(df: pd.DataFrame, code: str) -> go.Figure:
    """创建股票图表 - K线 + 均线 + 信号"""
    if df is None or df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("价格走势", "成交量", "RSI")
    )
    
    dates = df["date"].tolist()
    opens = df["open"].tolist() if "open" in df.columns else df["close"].tolist()
    highs = df["high"].tolist() if "high" in df.columns else df["close"].tolist()
    lows = df["low"].tolist() if "low" in df.columns else df["close"].tolist()
    closes = df["close"].tolist()
    volumes = df["volume"].tolist() if "volume" in df.columns else [0] * len(df)
    
    # K线
    fig.add_trace(
        go.Candlestick(
            x=dates, open=opens, high=highs, low=lows, close=closes,
            name="K线",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )
    
    # 均线
    if "ma5" in df.columns:
        fig.add_trace(go.Scatter(x=dates, y=df["ma5"].tolist(), name="MA5", line=dict(color="#FF6B6B", width=1)), row=1, col=1)
    if "ma20" in df.columns:
        fig.add_trace(go.Scatter(x=dates, y=df["ma20"].tolist(), name="MA20", line=dict(color="#4ECDC4", width=1.5)), row=1, col=1)
    if "ma60" in df.columns:
        fig.add_trace(go.Scatter(x=dates, y=df["ma60"].tolist(), name="MA60", line=dict(color="#45B7D1", width=1.5)), row=1, col=1)
    
    # 买入信号
    if "buy_signal" in df.columns:
        buy_signals = df[df["buy_signal"] == True]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals["date"].tolist(),
                    y=buy_signals["close"].tolist(),
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=15, color="#26a69a"),
                    name="买入信号"
                ),
                row=1, col=1
            )
    
    # 卖出信号
    if "sell_signal" in df.columns:
        sell_signals = df[df["sell_signal"] == True]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals["date"].tolist(),
                    y=sell_signals["close"].tolist(),
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=15, color="#ef5350"),
                    name="卖出信号"
                ),
                row=1, col=1
            )
    
    # 成交量
    colors = ["#26a69a" if close >= open_ else "#ef5350" for close, open_ in zip(closes, opens)]
    fig.add_trace(
        go.Bar(x=dates, y=volumes, name="成交量", marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # RSI
    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=dates, y=df["rsi"].tolist(), name="RSI", line=dict(color="#9C27B0", width=1.5)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    
    fig.update_layout(
        title=f"{code} 量价走势与信号",
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    
    return fig


def update_all_data() -> str:
    """一键更新所有数据"""
    try:
        update_stock_list()
        update_history_data()
        for f in CACHE_DIR.glob("*.csv"):
            f.unlink()
        return "数据更新完成！股票列表和历史数据已全部更新。"
    except Exception as e:
        return f"更新失败: {str(e)}"


def backtest_stock(code: str, start_date: str = None, end_date: str = None) -> tuple:
    """回测单只股票"""
    if not code:
        return "请输入股票代码", None
    
    try:
        result = run_backtest(code, start_date=start_date, end_date=end_date)
        
        if result is None or result.empty:
            return f"{code} 回测完成，但无交易记录", None
        
        display = result[["date", "close", "signal", "pnl", "pnl_pct"]].copy()
        display.columns = ["日期", "收盘价", "信号", "盈亏", "盈亏率(%)"]
        display["盈亏率(%)"] = (display["盈亏率(%)"] * 100).round(2).astype(str) + "%"
        
        total_trades = len(display)
        wins = len(display[display["盈亏"] > 0])
        losses = len(display[display["盈亏"] <= 0])
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        stats = f"""{code} 回测统计
-------------------------
交易次数: {total_trades}
盈利次数: {wins}
亏损次数: {losses}
胜率: {win_rate:.1f}%

盈亏统计:
  总盈亏: {display['盈亏'].sum():.2f}
  最大单笔盈利: {display['盈亏'].max():.2f}
  最大单笔亏损: {display['盈亏'].min():.2f}
"""
        
        return stats, display
        
    except Exception as e:
        return f"回测失败: {str(e)}", None


# =========================
# Gradio 界面
# =========================

def create_demo():
    """创建Gradio界面"""
    
    with gr.Blocks(title="量化选股系统") as demo:
        gr.Markdown("# 量化选股系统\n### 智能选股 · 量价分析 · 风险控制")
        
        gr.Markdown(f"* 系统就绪 | 数据日期: {TODAY_STR} | 模型已加载")
        
        gr.Markdown("---")
        
        with gr.Tabs():
            # Tab 1: 今日推荐
            with gr.TabItem("今日推荐"):
                gr.Markdown("### 一键扫描今日推荐股票")
                
                with gr.Row():
                    scan_date = gr.Textbox(label="扫描日期", value=TODAY_STR, placeholder="YYYY-MM-DD")
                    min_ai_prob = gr.Slider(label="AI胜率阈值", minimum=0.30, maximum=0.80, value=0.50, step=0.05)
                    max_results = gr.Slider(label="最大结果数", minimum=5, maximum=50, value=20, step=5)
                    scan_btn = gr.Button("开始扫描", variant="primary")
                
                scan_result = gr.DataFrame(label="扫描结果")
                scan_summary = gr.Textbox(label="扫描摘要", lines=8)
                
                scan_btn.click(
                    fn=scan_today_stocks,
                    inputs=[scan_date, min_ai_prob, max_results],
                    outputs=[scan_result, scan_summary]
                )
            
            # Tab 2: 个股分析
            with gr.TabItem("个股分析"):
                gr.Markdown("### 输入股票代码，分析量价走势与买卖信号")
                
                with gr.Row():
                    stock_code = gr.Textbox(label="股票代码", placeholder="sh.600000 或 sz.000001", scale=3)
                    lookback = gr.Number(label="查看天数", value=100, minimum=30, maximum=500, step=10, scale=1)
                    analyze_btn = gr.Button("分析", variant="primary", scale=1)
                
                with gr.Row():
                    analyze_report = gr.Textbox(label="分析报告", lines=12, scale=1)
                    chart = gr.Plot(scale=2)
                
                analyze_btn.click(
                    fn=analyze_stock,
                    inputs=[stock_code, lookback],
                    outputs=[analyze_report, chart]
                )
                
                gr.Examples(
                    examples=[["sh.600000", 100], ["sz.000001", 100], ["sh.603311", 100], ["sz.300661", 100]],
                    inputs=[stock_code, lookback]
                )
            
            # Tab 3: 回测
            with gr.TabItem("回测"):
                gr.Markdown("### 回测指定股票的交易策略表现")
                
                with gr.Row():
                    bt_code = gr.Textbox(label="股票代码", placeholder="sh.600000", scale=2)
                    bt_start = gr.Textbox(label="开始日期", placeholder="YYYY-MM-DD", value="2024-01-01", scale=2)
                    bt_end = gr.Textbox(label="结束日期", placeholder="YYYY-MM-DD", value=TODAY_STR, scale=2)
                    bt_btn = gr.Button("回测", variant="primary", scale=1)
                
                bt_stats = gr.Textbox(label="回测统计", lines=10)
                bt_result = gr.DataFrame(label="交易记录")
                
                bt_btn.click(
                    fn=backtest_stock,
                    inputs=[bt_code, bt_start, bt_end],
                    outputs=[bt_stats, bt_result]
                )
            
            # Tab 4: 数据更新
            with gr.TabItem("数据更新"):
                gr.Markdown("### 一键更新股票列表和历史数据")
                gr.Markdown("**注意**: 更新数据可能需要几分钟时间")
                
                update_btn = gr.Button("开始更新", variant="primary", size="lg")
                update_status = gr.Textbox(label="更新状态", lines=5)
                
                update_btn.click(fn=update_all_data, inputs=[], outputs=[update_status])
            
            # Tab 5: 关于
            with gr.TabItem("关于"):
                gr.Markdown("""
                ### 量化选股系统
                
                **功能**:
                - 今日推荐 - 基于AI模型的智能选股
                - 量价分析 - K线、均线、技术指标
                - 策略回测 - 验证策略有效性
                
                **使用说明**:
                1. 进入「今日推荐」页面，点击扫描获取推荐股票
                2. 进入「个股分析」页面，输入股票代码查看详细分析
                3. 定期点击「数据更新」保持数据最新
                
                **免责声明**: 本系统仅供参考学习，不构成投资建议。股市有风险，投资需谨慎！
                """)
        
        gr.Markdown("---")
        gr.Markdown("*© 2026 量化选股系统 | 仅供学习交流*")
    
    return demo


# 创建demo实例供 main.py 调用
demo = create_demo()


# =========================
# 直接启动
# =========================

if __name__ == "__main__":
    print("=" * 50)
    print("量化选股系统启动中...")
    print("=" * 50)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,
        show_error=True
    )
