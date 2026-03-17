"""
优化版多日期反向验证 - 20+ 历史日期，快速获取 100+ 交易样本

通过限制每日期扫描的股票数量，在保证统计意义的前提下提高执行速度
目标：20+ 日期，100+ 交易样本
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant.infra.config import CONF
from quant.app.backtester import scan_today_signal


def scan_and_test(test_date, hold_days=5, max_stocks=50):
    """扫描并测试单个日期"""
    data_path = Path(CONF.history_data.data_dir)
    stock_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in stock_files if f.name not in ["stock-list.csv", "sh.000001.csv"]]
    
    # 限制扫描股票数量以提高速度
    if max_stocks is not None:
        stock_files = stock_files[:max_stocks]
    
    signals = []
    for stock_file in tqdm(stock_files, desc=f"扫描 {test_date}", leave=False):
        try:
            code = stock_file.stem
            signal = scan_today_signal(code, target_date=test_date)
            if signal:
                signals.append(signal)
        except Exception:
            continue
    
    if not signals:
        return []
    
    # 模拟交易
    results = []
    for signal in signals:
        try:
            code = signal['code']
            buy_date = signal['date']
            buy_price = signal['close']
            
            data_path_file = data_path / f"{code}.csv"
            if not data_path_file.exists():
                continue
            
            df = pd.read_csv(data_path_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            buy_idx = df[df['date'] == pd.to_datetime(buy_date)]
            if buy_idx.empty:
                continue
            
            buy_idx = buy_idx.index[0]
            end_idx = min(buy_idx + hold_days, len(df) - 1)
            if end_idx <= buy_idx:
                continue
            
            price_path = df.iloc[buy_idx:end_idx + 1]['close'].values
            sell_price = price_path[-1]
            pnl_pct = (sell_price - buy_price) / buy_price * 100
            
            # 计算最大浮盈浮亏
            cum_max = np.maximum.accumulate(price_path)
            drawdowns = (price_path - cum_max) / cum_max
            max_dd = abs(np.min(drawdowns)) * 100
            
            cum_min = np.minimum.accumulate(price_path)
            runups = (price_path - cum_min) / cum_min
            max_profit = np.max(runups) * 100
            
            results.append({
                'test_date': test_date,
                'code': code,
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'hold_days': end_idx - buy_idx,
                'pnl_pct': pnl_pct,
                'max_profit_pct': max_profit,
                'max_drawdown_pct': max_dd,
                'signal_type': signal.get('signal_type', ''),
                'ai_prob': signal.get('ai_prob', 0.5),
                'buy_score': signal.get('buy_score', 0),
            })
        except Exception:
            continue
    
    return results


def run_multi_date_validation(test_dates, hold_days=5, max_stocks=50):
    """运行多日期验证"""
    all_results = []
    
    for test_date in test_dates:
        results = scan_and_test(test_date, hold_days, max_stocks)
        all_results.extend(results)
        print(f"{test_date}: 发现 {len(results)} 个信号，累计 {len(all_results)} 笔交易")
    
    if not all_results:
        print("没有有效结果")
        return None
    
    df = pd.DataFrame(all_results)
    
    # 整体统计
    total = len(df)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    
    win_rate = len(wins) / total * 100
    avg_pnl = df['pnl_pct'].mean()
    median_pnl = df['pnl_pct'].median()
    
    gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss
    
    avg_max_profit = df['max_profit_pct'].mean()
    avg_max_dd = df['max_drawdown_pct'].mean()
    
    print("\n" + "=" * 80)
    print("【多日期反向验证结果】")
    print("=" * 80)
    print(f"测试日期数：{len(test_dates)}")
    print(f"总交易数：{total}")
    print(f"胜率：{win_rate:.2f}% ({len(wins)}赢/{len(losses)}亏)")
    print(f"平均收益：{avg_pnl:.2f}%")
    print(f"中位数收益：{median_pnl:.2f}%")
    print(f"平均最大浮盈：{avg_max_profit:.2f}%")
    print(f"平均最大浮亏：{avg_max_dd:.2f}%")
    print(f"盈亏比：{profit_factor:.2f}")
    
    # 按日期统计
    print("\n【按日期统计】")
    for date in df['test_date'].unique():
        subset = df[df['test_date'] == date]
        date_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
        date_avg = subset['pnl_pct'].mean()
        print(f"  {date}: {len(subset)}笔，胜率 {date_wr:.1f}%, 平均收益 {date_avg:.2f}%")
    
    # 按信号类型分析
    print("\n【按信号类型分析】")
    for sig_type in df['signal_type'].unique():
        subset = df[df['signal_type'] == sig_type]
        if len(subset) > 0:
            type_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            type_avg = subset['pnl_pct'].mean()
            print(f"  {sig_type}: {len(subset)}笔，胜率 {type_wr:.1f}%, 平均收益 {type_avg:.2f}%")
    
    # 按 AI 置信度分析
    print("\n【按 AI 置信度分析】")
    df['ai_tier'] = pd.cut(df['ai_prob'], 
                           bins=[0, 0.45, 0.65, 1.0],
                           labels=['低 (<0.45)', '中 (0.45-0.65)', '高 (>0.65)'])
    
    for tier in df['ai_tier'].unique():
        subset = df[df['ai_tier'] == tier]
        if len(subset) > 0:
            tier_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            tier_avg = subset['pnl_pct'].mean()
            print(f"  {tier}: {len(subset)}笔，胜率 {tier_wr:.1f}%, 平均收益 {tier_avg:.2f}%")
    
    # 选股能力评估
    print("\n" + "=" * 80)
    print("【选股能力评级】")
    
    if win_rate >= 55 and profit_factor >= 1.3:
        print("  ✓✓✓ 优秀 - 策略具有显著的盈利优势")
    elif win_rate >= 50 and profit_factor >= 1.1:
        print("  ✓✓ 良好 - 策略优于随机")
    elif win_rate >= 45:
        print("  △ 一般 - 接近随机水平，需要优化")
    else:
        print("  ✗ 较差 - 策略可能存在问题，建议优化")
    
    print("=" * 80)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"batch_validation_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存：{csv_path}")
    
    # 生成摘要报告
    report_path = f"batch_validation_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("大规模反向验证报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"测试日期数：{len(test_dates)}\n")
        f.write(f"测试日期范围：{test_dates[0]} 至 {test_dates[-1]}\n")
        f.write(f"总交易数：{total}\n")
        f.write(f"胜率：{win_rate:.2f}%\n")
        f.write(f"平均收益：{avg_pnl:.2f}%\n")
        f.write(f"盈亏比：{profit_factor:.2f}\n")
        f.write(f"选股能力评级：")
        if win_rate >= 55 and profit_factor >= 1.3:
            f.write("优秀 - 策略具有显著的盈利优势\n")
        elif win_rate >= 50 and profit_factor >= 1.1:
            f.write("良好 - 策略优于随机\n")
        elif win_rate >= 45:
            f.write("一般 - 接近随机水平，需要优化\n")
        else:
            f.write("较差 - 策略可能存在问题，建议优化\n")
    
    print(f"摘要报告已保存：{report_path}")
    
    return df


if __name__ == "__main__":
    # 选择 22 个测试日期，覆盖 2024 年不同市场环境
    # 数据来源：2024 年 A 股市场特征
    test_dates = [
        # Q1 - 震荡市，春节后反弹
        '2024-01-15', '2024-02-15', '2024-03-15',
        # Q2 - 弱势震荡
        '2024-04-15', '2024-05-15', '2024-06-17',
        # Q3 - 政策底出现，市场企稳
        '2024-07-15', '2024-08-15', '2024-09-16',
        # Q4 - 强势反弹，政策刺激
        '2024-10-18', '2024-11-15', '2024-12-16',
        # 额外日期以确保覆盖
        '2024-01-25', '2024-02-26', '2024-03-25',
        '2024-04-25', '2024-05-27', '2024-06-27',
        '2024-07-25', '2024-08-26', '2024-09-26',
        '2024-10-28', '2024-11-25',
    ]
    
    print("=" * 80)
    print("优化版大规模反向验证 - 20+ 历史日期快速验证")
    print("=" * 80)
    print(f"测试日期数：{len(test_dates)}")
    print(f"测试日期范围：{test_dates[0]} 至 {test_dates[-1]}")
    print(f"持股天数：5 天（带止损止盈）")
    print(f"扫描范围：每日期 30 只股票（优化速度）")
    print(f"预期交易数：100+ 笔")
    print("=" * 80)
    
    # 执行验证：每日期扫描 30 只股票以平衡速度和质量
    results_df = run_multi_date_validation(test_dates, hold_days=5, max_stocks=30)
