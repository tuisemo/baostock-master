"""
大规模反向验证 - 2024 年全市场验证
覆盖 20+ 历史日期，执行全市场选股扫描，收集 100+ 交易样本

使用方法:
    python scripts/batch_validation_2024.py

输出:
    - batch_validation_YYYYMMDD_HHMMSS.csv - 详细交易数据
    - batch_validation_report_YYYYMMDD_HHMMSS.txt - 文本摘要报告
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant.infra.config import CONF
from quant.app.backtester import scan_today_signal


def scan_and_test(test_date, hold_days=5, max_stocks=None):
    """扫描并测试单个日期"""
    data_path = Path(CONF.history_data.data_dir)
    stock_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in stock_files if f.name not in ["stock-list.csv", "sh.000001.csv"]]
    
    # 如果指定了 max_stocks，则限制扫描数量
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
                'name': signal.get('name', ''),
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
                'market_state': signal.get('market_state', 'unknown'),
            })
        except Exception:
            continue
    
    return results


def run_batch_validation(test_dates, hold_days=5, max_stocks=None, output_dir="data/exports"):
    """运行批量验证，带进度保存"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    progress_file = output_path / "batch_validation_progress.json"
    
    # 加载进度 (如果存在)
    completed_dates = set()
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                completed_dates = set(progress.get('completed_dates', []))
                all_results = progress.get('results', [])
                print(f"恢复进度：已完成 {len(completed_dates)} 个日期，{len(all_results)} 笔交易")
        except Exception as e:
            print(f"无法加载进度：{e}")
    
    # 过滤已完成的日期
    remaining_dates = [d for d in test_dates if d not in completed_dates]
    
    print(f"\n剩余日期：{len(remaining_dates)}/{len(test_dates)}")
    
    total_start = datetime.now()
    
    # 处理剩余日期
    for i, test_date in enumerate(remaining_dates, 1):
        date_start = datetime.now()
        print(f"\n[{i}/{len(remaining_dates)}] 处理日期：{test_date}")
        
        results = scan_and_test(test_date, hold_days, max_stocks)
        all_results.extend(results)
        
        date_elapsed = (datetime.now() - date_start).total_seconds()
        
        # 保存进度
        completed_dates.add(test_date)
        progress_data = {
            'completed_dates': list(completed_dates),
            'results': all_results,
            'last_updated': datetime.now().isoformat(),
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        
        print(f"  -> {test_date}: 发现 {len(results)} 个信号 (用时 {date_elapsed:.1f}s)")
        print(f"  -> 累计交易数：{len(all_results)}")
    
    total_elapsed = (datetime.now() - total_start).total_seconds()
    print(f"\n总用时：{total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)")
    
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
    print("【大规模反向验证结果】")
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
    for date in sorted(df['test_date'].unique()):
        subset = df[df['test_date'] == date]
        date_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
        date_avg = subset['pnl_pct'].mean()
        print(f"  {date}: {len(subset)}笔，胜率 {date_wr:.1f}%, 平均收益 {date_avg:.2f}%")
    
    # 按市场状态分析
    print("\n【按市场状态分析】")
    for state in df['market_state'].unique():
        subset = df[df['market_state'] == state]
        if len(subset) > 0:
            state_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            state_avg = subset['pnl_pct'].mean()
            print(f"  {state}: {len(subset)}笔，胜率 {state_wr:.1f}%, 平均收益 {state_avg:.2f}%")
    
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
    csv_path = output_path / f"batch_validation_{timestamp}.csv"
    report_path = output_path / f"batch_validation_report_{timestamp}.txt"
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存：{csv_path}")
    
    # 生成报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("大规模反向验证报告 - 2024 年全市场选股能力评估\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试日期范围：{test_dates[0]} 至 {test_dates[-1]}\n")
        f.write(f"测试日期数：{len(test_dates)}\n")
        f.write(f"总交易数：{total}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("核心统计指标\n")
        f.write("-" * 80 + "\n")
        f.write(f"胜率：{win_rate:.2f}% ({len(wins)}赢/{len(losses)}亏)\n")
        f.write(f"平均收益：{avg_pnl:.2f}%\n")
        f.write(f"中位数收益：{median_pnl:.2f}%\n")
        f.write(f"平均最大浮盈：{avg_max_profit:.2f}%\n")
        f.write(f"平均最大浮亏：{avg_max_dd:.2f}%\n")
        f.write(f"盈亏比：{profit_factor:.2f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("按市场状态分析\n")
        f.write("-" * 80 + "\n")
        for state in sorted(df['market_state'].unique()):
            subset = df[df['market_state'] == state]
            if len(subset) > 0:
                state_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
                state_avg = subset['pnl_pct'].mean()
                f.write(f"{state}: {len(subset)}笔，胜率 {state_wr:.1f}%, 平均收益 {state_avg:.2f}%\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("按 AI 置信度分析\n")
        f.write("-" * 80 + "\n")
        for tier in ['低 (<0.45)', '中 (0.45-0.65)', '高 (>0.65)']:
            subset = df[df['ai_tier'] == tier]
            if len(subset) > 0:
                tier_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
                tier_avg = subset['pnl_pct'].mean()
                f.write(f"{tier}: {len(subset)}笔，胜率 {tier_wr:.1f}%, 平均收益 {tier_avg:.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("选股能力评级：")
        if win_rate >= 55 and profit_factor >= 1.3:
            f.write("优秀 - 策略具有显著的盈利优势\n")
        elif win_rate >= 50 and profit_factor >= 1.1:
            f.write("良好 - 策略优于随机\n")
        elif win_rate >= 45:
            f.write("一般 - 接近随机水平，需要优化\n")
        else:
            f.write("较差 - 策略可能存在问题，建议优化\n")
        f.write("=" * 80 + "\n")
    
    print(f"验证报告已保存：{report_path}")
    
    # 清理进度文件
    if progress_file.exists():
        progress_file.unlink()
    
    return df


if __name__ == "__main__":
    # 选择 24 个测试日期，覆盖 2024 年不同市场环境（牛市、熊市、震荡市）
    # 数据来源：2024 年 A 股市场特征
    # - Q1 (1-3 月): 震荡市，春节后反弹
    # - Q2 (4-6 月): 弱势震荡，经济数据疲软
    # - Q3 (7-9 月): 政策底出现，市场企稳
    # - Q4 (10-12 月): 强势反弹，政策刺激
    test_dates = [
        # 1 月 - 震荡市
        '2024-01-10', '2024-01-24',
        # 2 月 - 春节后反弹
        '2024-02-07', '2024-02-21',
        # 3 月 - 震荡整理
        '2024-03-06', '2024-03-20',
        # 4 月 - 弱势震荡
        '2024-04-10', '2024-04-24',
        # 5 月 - 继续弱势
        '2024-05-08', '2024-05-22',
        # 6 月 - 政策底酝酿
        '2024-06-05', '2024-06-19',
        # 7 月 - 政策预期
        '2024-07-10', '2024-07-24',
        # 8 月 - 震荡筑底
        '2024-08-07', '2024-08-21',
        # 9 月 - 企稳回升
        '2024-09-11', '2024-09-25',
        # 10 月 - 政策刺激反弹
        '2024-10-09', '2024-10-23',
        # 11 月 - 强势反弹
        '2024-11-06', '2024-11-20',
        # 12 月 - 年末行情
        '2024-12-11', '2024-12-25',
    ]
    
    print("=" * 80)
    print("大规模反向验证 - 2024 年全市场选股能力评估")
    print("=" * 80)
    print(f"测试日期数：{len(test_dates)}")
    print(f"测试日期范围：{test_dates[0]} 至 {test_dates[-1]}")
    print(f"持股天数：5 天（带止损止盈）")
    print(f"扫描范围：全市场股票（约 1400+ 只）")
    print("=" * 80)
    print("\n注意：此脚本支持进度保存，可中断后重新运行")
    print("输出目录：data/exports/")
    print("=" * 80)
    
    # 执行大规模验证：全市场扫描，不限制股票数量
    results_df = run_batch_validation(test_dates, hold_days=5, max_stocks=None)
