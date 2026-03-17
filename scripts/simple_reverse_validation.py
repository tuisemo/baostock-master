"""
简化的反向验证脚本
目标：选择一个历史日期，执行选股，观察持股 N 天后的盈利情况

使用方法:
    python scripts\simple_reverse_validation.py
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant.infra.config import CONF
from quant.app.backtester import scan_today_signal


def scan_date(target_date, max_stocks=100):
    """扫描历史日期的买点信号"""
    print(f"\n扫描日期：{target_date}")
    
    # 获取所有股票数据
    data_path = Path(CONF.history_data.data_dir)
    stock_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in stock_files if f.name not in ["stock-list.csv", "sh.000001.csv"]]
    
    print(f"扫描 {min(max_stocks, len(stock_files))} 只股票...")
    
    signals = []
    for stock_file in tqdm(stock_files[:max_stocks], desc="扫描"):
        try:
            code = stock_file.stem
            signal = scan_today_signal(code, target_date=target_date)
            if signal:
                signals.append(signal)
        except Exception:
            continue
    
    print(f"发现 {len(signals)} 个信号")
    
    # 按得分排序
    if signals:
        signals.sort(key=lambda x: x.get('buy_score', 0), reverse=True)
    
    return signals


def simulate_hold(signals, hold_days=5):
    """模拟持股 N 天后的表现"""
    print(f"\n模拟持股 {hold_days} 天...")
    
    results = []
    
    for signal in tqdm(signals, desc="模拟交易"):
        try:
            code = signal['code']
            buy_date = signal['date']
            buy_price = signal['close']
            
            # 读取股票数据
            data_path = Path(CONF.history_data.data_dir) / f"{code}.csv"
            if not data_path.exists():
                continue
            
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 找到买入日
            buy_idx = df[df['date'] == pd.to_datetime(buy_date)]
            if buy_idx.empty:
                continue
            
            buy_idx = buy_idx.index[0]
            
            # 计算未来 N 天
            end_idx = min(buy_idx + hold_days, len(df) - 1)
            if end_idx <= buy_idx:
                continue
            
            price_path = df.iloc[buy_idx:end_idx + 1]['close'].values
            
            # 计算收益
            cum_max = np.maximum.accumulate(price_path)
            drawdowns = (price_path - cum_max) / cum_max
            max_dd = abs(np.min(drawdowns)) * 100
            
            cum_min = np.minimum.accumulate(price_path)
            runups = (price_path - cum_min) / cum_min
            max_profit = np.max(runups) * 100
            
            sell_price = price_path[-1]
            pnl_pct = (sell_price - buy_price) / buy_price * 100
            
            results.append({
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
            
        except Exception as e:
            continue
    
    print(f"完成 {len(results)} 笔交易模拟")
    return results


def analyze(results):
    """分析结果"""
    if not results:
        print("没有有效结果")
        return None
    
    df = pd.DataFrame(results)
    
    total = len(df)
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]
    
    win_rate = len(wins) / total * 100
    avg_pnl = df['pnl_pct'].mean()
    median_pnl = df['pnl_pct'].median()
    max_profit = df['pnl_pct'].max()
    max_loss = df['pnl_pct'].min()
    
    gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss
    
    avg_max_profit = df['max_profit_pct'].mean()
    avg_max_dd = df['max_drawdown_pct'].mean()
    
    print("\n" + "=" * 80)
    print("【选股能力评估结果】")
    print("=" * 80)
    print(f"总交易数：{total}")
    print(f"胜率：{win_rate:.2f}% ({len(wins)}赢/{len(losses)}亏)")
    print(f"平均收益：{avg_pnl:.2f}%")
    print(f"中位数收益：{median_pnl:.2f}%")
    print(f"最大盈利：{max_profit:.2f}%")
    print(f"最大亏损：{max_loss:.2f}%")
    print(f"平均最大浮盈：{avg_max_profit:.2f}%")
    print(f"平均最大浮亏：{avg_max_dd:.2f}%")
    print(f"盈亏比：{profit_factor:.2f}")
    
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
    
    # 按信号类型分析
    print("\n【按信号类型分析】")
    for sig_type in df['signal_type'].unique():
        subset = df[df['signal_type'] == sig_type]
        if len(subset) > 0:
            type_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            type_avg = subset['pnl_pct'].mean()
            print(f"  {sig_type}: {len(subset)}笔，胜率 {type_wr:.1f}%, 平均收益 {type_avg:.2f}%")
    
    # 选股能力评估
    print("\n" + "=" * 80)
    print("【选股能力评级】")
    
    if win_rate >= 55 and profit_factor >= 1.3:
        print("  ✓✓✓ 优秀 - 策略具有显著的盈利优势")
        print(f"      高胜率 ({win_rate:.1f}%) + 高盈亏比 ({profit_factor:.2f})")
    elif win_rate >= 50 and profit_factor >= 1.1:
        print("  ✓✓ 良好 - 策略优于随机")
        print(f"      胜率 ({win_rate:.1f}%) + 盈亏比 ({profit_factor:.2f}) 达标")
    elif win_rate >= 45:
        print("  △ 一般 - 接近随机水平，需要优化")
        print(f"      建议：提高 AI 阈值、加强量能确认、优化止盈止损")
    else:
        print("  ✗ 较差 - 策略可能存在问题")
        print(f"      建议：重新训练模型、调整参数、检查数据质量")
    
    print("=" * 80)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"reverse_validation_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存：{csv_path}")
    
    return df


def main():
    """主函数"""
    print("=" * 80)
    print("反向验证 - 选股能力评估")
    print("=" * 80)
    
    # 选择测试日期（可以修改）- 使用已知存在的日期
    test_date = '2024-04-15'  # 周一，工作日
    hold_days = 5  # 持股 5 天
    
    print(f"测试日期：{test_date}")
    print(f"持股天数：{hold_days} 天")
    
    # 1. 扫描买点 - 增加扫描数量
    signals = scan_date(test_date, max_stocks=200)
    
    if not signals:
        print("未发现任何买点信号")
        return
    
    # 2. 模拟交易
    results = simulate_hold(signals, hold_days)
    
    # 3. 分析结果
    analyze(results)


if __name__ == "__main__":
    main()
