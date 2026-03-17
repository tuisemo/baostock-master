"""
快速反向验证测试（简化版）

不依赖 AI 模型，仅使用规则引擎测试选股能力
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.app.backtester import get_market_index, evaluate_buy_signals, _build_column_names
from quant.core.adaptive_strategy import get_dynamic_params_v10
from quant.core.strategy_params import StrategyParams
from quant.features.analyzer import calculate_indicators


def scan_without_ai(code: str, target_date: str, params: StrategyParams) -> dict | None:
    """
    扫描买点信号（不使用 AI 模型）
    
    Args:
        code: 股票代码
        target_date: 目标日期
        params: 策略参数
        
    Returns:
        信号字典或 None
    """
    try:
        # 读取数据
        data_path = Path(CONF.history_data.data_dir) / f"{code}.csv"
        if not data_path.exists():
            return None
        
        df = pd.read_csv(data_path)
        if df.empty or len(df) < 60:
            return None
        
        # 计算指标
        df = calculate_indicators(df, params)
        
        # 添加特征（简化版，不需要 ML 特征）
        if 'vol_slope' not in df.columns:
            df['vol_slope'] = 0.0
        
        # 过滤到目标日期
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.to_datetime(target_date)]
        
        if len(df) < 10:
            return None
        
        # 获取最新数据
        row_1 = df.iloc[-1]
        row_2 = df.iloc[-2]
        row_3 = df.iloc[-3] if len(df) >= 3 else row_2
        
        # 构建列名
        cols = _build_column_names(params)
        
        # 提取需要的数据
        price = row_1.get("close", np.nan)
        if pd.isna(price) or price <= 0:
            return None
        
        # 获取市场状态
        idx_df = get_market_index()
        market_uptrend = True
        market_state = "sideways_low_vol"
        
        if idx_df is not None:
            try:
                current_date_ts = df.index[-1] if 'Date' in df.columns else pd.to_datetime(row_1['date'])
                idx_loc = idx_df.index.get_indexer([current_date_ts], method="pad")[0]
                if idx_loc != -1:
                    market_uptrend = bool(idx_df.iloc[idx_loc].get("market_uptrend", True))
                    market_state = str(idx_df.iloc[idx_loc].get("market_state", "sideways_low_vol"))
            except:
                pass
        
        # 获取动态参数
        dyn_p = get_dynamic_params_v10(params, market_state)
        
        # 评估买入信号
        signal_pullback, signal_rebound, signal_trend_breakout, signal_details = evaluate_buy_signals(
            price=price,
            open_p=float(row_1.get("open", price)),
            low_p=float(row_1.get("low", price)),
            sma_l_1=row_1[cols["sma_l"]],
            sma_l_3=row_3[cols["sma_l"]],
            sma_s_1=row_1[cols["sma_s"]],
            macd_h_1=row_1[cols["macd_h"]],
            macd_h_2=row_2[cols["macd_h"]],
            rsi_1=row_1[cols["rsi"]],
            bb_lower_1=row_1[cols["bb_lower"]],
            vol_1=row_1.get("volume", 0),
            vol_2=row_2.get("volume", 0),
            has_vol_slope="vol_slope" in df.columns,
            vol_slope_1=row_1.get("vol_slope", 0.0),
            has_mom_div="momentum_divergence" in df.columns,
            mom_div_1=row_1.get("momentum_divergence", 0.0),
            market_uptrend=market_uptrend,
            p=dyn_p,
            weekly_data=None,
        )
        
        # 确定信号类型
        signal_type = ""
        if signal_pullback:
            signal_type = "布林带极度下杀反弹 (左侧)"
        elif signal_rebound:
            signal_type = "超卖恐慌底部 (左侧)"
        elif signal_trend_breakout:
            signal_type = "均线放量金叉 (右侧)"
        
        if not signal_type:
            return None
        
        # 计算综合得分
        total_score = signal_details.get('composite_score', 0.0)
        
        return {
            "code": code,
            "date": target_date,
            "close": round(float(price), 2),
            "total_score": round(total_score, 3),
            "signal_type": signal_type,
            "market_state": market_state,
            "signal_details": signal_details,
        }
        
    except Exception as e:
        logger.debug(f"扫描 {code} 失败：{e}")
        return None


def simulate_hold_performance(code: str, buy_date: str, buy_price: float, hold_days: int) -> dict:
    """
    模拟持股 N 天的表现
    
    Args:
        code: 股票代码
        buy_date: 买入日期
        buy_price: 买入价格
        hold_days: 持仓天数
        
    Returns:
        表现字典
    """
    try:
        data_path = Path(CONF.history_data.data_dir) / f"{code}.csv"
        if not data_path.exists():
            return None
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 找到买入日期
        buy_idx = df[df['date'] == pd.to_datetime(buy_date)]
        if buy_idx.empty:
            return None
        
        buy_idx = buy_idx.index[0]
        
        # 计算后续 N 天的价格路径
        end_idx = min(buy_idx + hold_days, len(df) - 1)
        if end_idx <= buy_idx:
            return None
        
        price_path = df.iloc[buy_idx:end_idx + 1]['close'].values
        
        # 计算最大盈利和最大回撤
        cum_max = np.maximum.accumulate(price_path)
        drawdowns = (price_path - cum_max) / cum_max
        max_drawdown_pct = abs(np.min(drawdowns)) * 100
        
        cum_min = np.minimum.accumulate(price_path)
        runups = (price_path - cum_min) / cum_min
        max_profit_pct = np.max(runups) * 100
        
        # 最终收益
        sell_price = price_path[-1]
        pnl_pct = (sell_price - buy_price) / buy_price * 100
        
        sell_date = df.iloc[end_idx]['date'].strftime('%Y-%m-%d')
        
        return {
            "code": code,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "hold_days": end_idx - buy_idx,
            "pnl_pct": pnl_pct,
            "max_profit_pct": max_profit_pct,
            "max_drawdown_pct": max_drawdown_pct,
        }
        
    except Exception as e:
        logger.debug(f"模拟 {code} 失败：{e}")
        return None


def run_quick_validation():
    """运行快速反向验证"""
    
    print("=" * 80)
    print("快速反向验证测试（纯规则引擎）")
    print("=" * 80)
    
    # 测试配置
    test_dates = [
        '2024-01-15',
        '2024-03-15',
        '2024-05-15',
        '2024-07-15',
        '2024-09-15',
        '2024-11-15',
    ]
    
    hold_days = 5
    max_stocks = 20  # 每个日期测试前 N 个信号
    
    # 获取所有股票
    data_path = Path(CONF.history_data.data_dir)
    stock_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in stock_files if f.name not in ["stock-list.csv", "sh.000001.csv"]]
    
    print(f"\n测试配置:")
    print(f"  测试日期：{len(test_dates)} 个")
    print(f"  持仓天数：{hold_days} 天")
    print(f"  每日期最大股票数：{max_stocks}")
    print(f"  可用股票数：{len(stock_files)}")
    print("=" * 80)
    
    # 创建策略参数
    params = StrategyParams.from_app_config(CONF)
    
    all_signals = []
    all_performances = []
    
    for test_date in test_dates:
        print(f"\n测试日期：{test_date}")
        
        # 1. 扫描买点
        signals = []
        for stock_file in tqdm(stock_files[:100], desc="扫描买点"):  # 只测试前 100 只股票加快速度
            code = stock_file.stem
            signal = scan_without_ai(code, test_date, params)
            if signal:
                signals.append(signal)
        
        # 按得分排序
        if signals:
            signals.sort(key=lambda x: x.get('total_score', 0), reverse=True)
            signals = signals[:max_stocks]  # 取前 N 个
        
        print(f"  发现 {len(signals)} 个信号")
        all_signals.extend(signals)
        
        # 2. 模拟持股表现
        for signal in signals:
            perf = simulate_hold_performance(
                signal['code'],
                signal['date'],
                signal['close'],
                hold_days,
            )
            if perf:
                perf['signal_type'] = signal['signal_type']
                perf['market_state'] = signal['market_state']
                perf['signal_score'] = signal['total_score']
                all_performances.append(perf)
        
        print(f"  模拟 {len([p for p in all_performances if p['buy_date'] == test_date])} 笔交易")
    
    # 3. 计算统计
    if not all_performances:
        print("\n没有有效的交易结果")
        return
    
    print("\n" + "=" * 80)
    print("验证结果")
    print("=" * 80)
    
    # 基础统计
    total = len(all_performances)
    wins = [p for p in all_performances if p['pnl_pct'] > 0]
    losses = [p for p in all_performances if p['pnl_pct'] <= 0]
    
    win_rate = len(wins) / total * 100 if total > 0 else 0
    avg_pnl = np.mean([p['pnl_pct'] for p in all_performances])
    median_pnl = np.median([p['pnl_pct'] for p in all_performances])
    max_profit = max([p['pnl_pct'] for p in all_performances])
    max_loss = min([p['pnl_pct'] for p in all_performances])
    
    # 盈亏比
    gross_profit = sum(p['pnl_pct'] for p in wins) if wins else 0
    gross_loss = abs(sum(p['pnl_pct'] for p in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # 路径统计
    avg_max_profit = np.mean([p['max_profit_pct'] for p in all_performances])
    avg_max_drawdown = np.mean([p['max_drawdown_pct'] for p in all_performances])
    
    # 打印结果
    print(f"\n【整体表现】")
    print(f"  总交易数：{total}")
    print(f"  胜率：{win_rate:.2f}% ({len(wins)}赢/{len(losses)}亏)")
    print(f"  平均收益：{avg_pnl:.2f}%")
    print(f"  中位数收益：{median_pnl:.2f}%")
    print(f"  最大盈利：{max_profit:.2f}%")
    print(f"  最大亏损：{max_loss:.2f}%")
    print(f"  平均最大浮盈：{avg_max_profit:.2f}%")
    print(f"  平均最大浮亏：{avg_max_drawdown:.2f}%")
    print(f"  盈亏比：{profit_factor:.2f}")
    
    # 按信号类型分析
    print(f"\n【按信号类型分析】")
    by_signal = {}
    for perf in all_performances:
        sig_type = perf.get('signal_type', 'Unknown')
        if sig_type not in by_signal:
            by_signal[sig_type] = []
        by_signal[sig_type].append(perf)
    
    for sig_type, perfs in by_signal.items():
        pnls = [p['pnl_pct'] for p in perfs]
        wr = len([p for p in perfs if p['pnl_pct'] > 0]) / len(perfs) * 100
        avg = np.mean(pnls)
        print(f"\n  {sig_type}:")
        print(f"    样本数：{len(perfs)}")
        print(f"    胜率：{wr:.1f}%")
        print(f"    平均收益：{avg:.2f}%")
    
    # 按市场状态分析
    print(f"\n【按市场状态分析】")
    by_market = {}
    for perf in all_performances:
        state = perf.get('market_state', 'unknown')
        if state not in by_market:
            by_market[state] = []
        by_market[state].append(perf)
    
    for state, perfs in by_market.items():
        pnls = [p['pnl_pct'] for p in perfs]
        wr = len([p for p in perfs if p['pnl_pct'] > 0]) / len(perfs) * 100
        avg = np.mean(pnls)
        print(f"\n  {state}:")
        print(f"    样本数：{len(perfs)}")
        print(f"    胜率：{wr:.1f}%")
        print(f"    平均收益：{avg:.2f}%")
    
    # 保存结果
    df = pd.DataFrame(all_performances)
    output_csv = f"quick_reverse_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n" + "=" * 80)
    print(f"详细数据已保存至：{output_csv}")
    print("=" * 80)
    
    # 评估选股能力
    print("\n【选股能力评估】")
    if win_rate >= 55 and profit_factor >= 1.3:
        print("  ✓ 选股能力优秀 - 策略具有显著的盈利优势")
    elif win_rate >= 50 and profit_factor >= 1.1:
        print("  △ 选股能力良好 - 策略略优于随机")
    elif win_rate >= 45:
        print("  ⚠ 选股能力一般 - 接近随机水平，需要优化")
    else:
        print("  ✗ 选股能力较差 - 策略可能存在问题，建议大幅优化")
    
    print("=" * 80)
    
    return all_performances


if __name__ == "__main__":
    run_quick_validation()
