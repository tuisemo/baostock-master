"""
小规模批量验证测试 - 验证流程是否正常工作
测试 5 个日期，每个日期扫描前 50 只股票

使用方法:
    python scripts/test_batch_validation.py
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


def test_batch_validation():
    """测试批量验证流程"""
    # 选择 5 个测试日期
    test_dates = [
        '2024-03-15',
        '2024-04-15',
        '2024-05-15',
        '2024-06-17',
        '2024-07-15',
    ]
    
    data_path = Path(CONF.history_data.data_dir)
    stock_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in stock_files if f.name not in ["stock-list.csv", "sh.000001.csv"]]
    
    # 限制测试股票数量
    test_stocks = stock_files[:50]
    
    print("=" * 80)
    print("小规模批量验证测试")
    print("=" * 80)
    print(f"测试日期：{len(test_dates)} 个")
    print(f"测试股票：{len(test_stocks)} 只")
    print("=" * 80)
    
    all_results = []
    
    for i, test_date in enumerate(test_dates, 1):
        print(f"\n[{i}/{len(test_dates)}] 处理日期：{test_date}")
        
        signals = []
        for stock_file in tqdm(test_stocks, desc="扫描", leave=False):
            try:
                code = stock_file.stem
                signal = scan_today_signal(code, target_date=test_date)
                if signal:
                    signals.append(signal)
            except Exception:
                continue
        
        print(f"  发现 {len(signals)} 个信号")
        
        # 模拟交易
        for signal in signals:
            try:
                code = signal['code']
                buy_date = signal['date']
                buy_price = signal['close']
                
                stock_file = data_path / f"{code}.csv"
                if not stock_file.exists():
                    continue
                
                df = pd.read_csv(stock_file)
                if 'date' not in df.columns:
                    continue
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                buy_idx = df[df['date'] == pd.to_datetime(buy_date)]
                if buy_idx.empty:
                    continue
                
                buy_idx = buy_idx.index[0]
                hold_days = 5
                end_idx = min(buy_idx + hold_days, len(df) - 1)
                if end_idx <= buy_idx:
                    continue
                
                price_path = df.iloc[buy_idx:end_idx + 1]['close'].values
                sell_price = price_path[-1]
                pnl_pct = (sell_price - buy_price) / buy_price * 100
                
                all_results.append({
                    'test_date': test_date,
                    'code': code,
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'pnl_pct': pnl_pct,
                    'signal_type': signal.get('signal_type', ''),
                    'ai_prob': signal.get('ai_prob', 0.5),
                })
            except Exception:
                continue
        
        print(f"  累计交易数：{len(all_results)}")
    
    if not all_results:
        print("\n没有有效结果")
        return
    
    df = pd.DataFrame(all_results)
    
    # 统计
    total = len(df)
    wins = df[df['pnl_pct'] > 0]
    win_rate = len(wins) / total * 100
    avg_pnl = df['pnl_pct'].mean()
    
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"总交易数：{total}")
    print(f"胜率：{win_rate:.2f}%")
    print(f"平均收益：{avg_pnl:.2f}%")
    print("=" * 80)
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"test_batch_validation_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存：{csv_path}")


if __name__ == "__main__":
    test_batch_validation()
