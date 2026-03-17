"""
单日反向验证测试

测试单个历史日期的选股表现
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
from quant.features.analyzer import calculate_indicators
from quant.core.strategy_params import StrategyParams

def scan_single_date(target_date='2024-03-15', hold_days=5, max_stocks=50):
    """
    扫描单个日期的买点并模拟后续表现
    """
    print(f"测试日期：{target_date}")
    print(f"持仓天数：{hold_days} 天")
    print(f"最大股票数：{max_stocks}")
    print("=" * 80)
    
    # 获取所有股票
    data_path = Path(CONF.history_data.data_dir)
    stock_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in stock_files if f.name not in ["stock-list.csv", "sh.000001.csv"]]
    
    print(f"扫描 {len(stock_files)} 只股票...")
    
    params = StrategyParams.from_app_config(CONF)
    
    results = []
    
    for stock_file in tqdm(stock_files[:200], desc="扫描"):  # 测试前 200 只
        try:
            code = stock_file.stem
            
            # 读取数据
            df = pd.read_csv(stock_file)
            if df.empty or len(df) < 60:
                continue
            
            # 计算指标
            df = calculate_indicators(df, params)
            
            # 添加 vol_slope 如果不存在
            if 'vol_slope' not in df.columns:
                df['vol_slope'] = 0.0
            
            # 转换日期
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 过滤到目标日期
            target_dt = pd.to_datetime(target_date)
            df_historical = df[df['date'] <= target_dt]
            
            if len(df_historical) < 30:
                continue
            
            # 获取买入日数据
            buy_row = df_historical.iloc[-1]
            buy_price = buy_row['close']
            
            # 获取未来 N 天数据
            future_end_idx = min(len(df_historical) + hold_days - 1, len(df) - 1)
            future_data = df.iloc[len(df_historical)-1:future_end_idx+1]
            
            if len(future_data) < 2:
                continue
            
            # 计算收益
            price_path = future_data['close'].values
            sell_price = price_path[-1]
            pnl_pct = (sell_price - buy_price) / buy_price * 100
            
            # 计算最大浮盈浮亏
            cum_max = np.maximum.accumulate(price_path)
            max_drawdown = abs(np.min((price_path - cum_max) / cum_max)) * 100
            
            cum_min = np.minimum.accumulate(price_path)
            max_profit = np.max((price_path - cum_min) / cum_min) * 100
            
            # 简单的买入信号判断
            # 检查是否满足基本条件（简化版）
            rsi_col = f"RSI_{params.rsi_length}"
            if rsi_col not in buy_row or pd.isna(buy_row[rsi_col]):
                continue
            
            is_signal = False
            signal_type = "其他"
            
            # 左侧信号：RSI 超卖 + 价格接近布林下轨
            bb_lower_col = f"BBL_{params.bbands_length}_{params.bbands_std}"
            if (buy_row[rsi_col] < 30 or buy_row['close'] < buy_row.get(bb_lower_col, buy_row['close']) * 0.98):
                is_signal = True
                signal_type = "左侧超卖"
            
            # 右侧信号：均线金叉 + 放量
            sma_s_col = f"SMA_{params.ma_short}"
            sma_l_col = f"SMA_{params.ma_long}"
            if (sma_s_col in buy_row and sma_l_col in buy_row and 
                buy_row[sma_s_col] > buy_row[sma_l_col] and 
                buy_row['volume'] > buy_row.get('volume', 0) * 1.5):
                is_signal = True
                signal_type = "右侧突破"
            
            if not is_signal:
                continue
            
            results.append({
                'code': code,
                'buy_date': target_date,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'hold_days': len(future_data) - 1,
                'pnl_pct': pnl_pct,
                'max_profit_pct': max_profit,
                'max_drawdown_pct': max_drawdown,
                'signal_type': signal_type,
            })
            
            if len(results) >= max_stocks:
                break
                
        except Exception as e:
            continue
    
    # 分析结果
    if not results:
        print("未发现任何信号")
        return None
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print(f"测试结果 ({len(results)} 笔交易)")
    print("=" * 80)
    
    total = len(df_results)
    wins = df_results[df_results['pnl_pct'] > 0]
    losses = df_results[df_results['pnl_pct'] <= 0]
    
    win_rate = len(wins) / total * 100
    avg_pnl = df_results['pnl_pct'].mean()
    median_pnl = df_results['pnl_pct'].median()
    max_profit = df_results['pnl_pct'].max()
    max_loss = df_results['pnl_pct'].min()
    
    gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss
    
    avg_max_profit = df_results['max_profit_pct'].mean()
    avg_max_drawdown = df_results['max_drawdown_pct'].mean()
    
    print(f"总交易数：{total}")
    print(f"胜率：{win_rate:.2f}% ({len(wins)}赢/{len(losses)}亏)")
    print(f"平均收益：{avg_pnl:.2f}%")
    print(f"中位数收益：{median_pnl:.2f}%")
    print(f"最大盈利：{max_profit:.2f}%")
    print(f"最大亏损：{max_loss:.2f}%")
    print(f"平均最大浮盈：{avg_max_profit:.2f}%")
    print(f"平均最大浮亏：{avg_max_drawdown:.2f}%")
    print(f"盈亏比：{profit_factor:.2f}")
    
    # 按信号类型分析
    print("\n按信号类型:")
    for sig_type in df_results['signal_type'].unique():
        subset = df_results[df_results['signal_type'] == sig_type]
        print(f"  {sig_type}: {len(subset)}笔，胜率 {(len(subset[subset['pnl_pct']>0])/len(subset)*100):.1f}%, "
              f"平均收益 {subset['pnl_pct'].mean():.2f}%")
    
    # 保存结果
    output_file = f"single_date_test_{target_date.replace('-', '')}.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至：{output_file}")
    
    # 评估
    print("\n" + "=" * 80)
    print("选股能力评估:")
    if win_rate >= 55 and profit_factor >= 1.3:
        print("  ✓ 优秀 - 策略具有显著盈利优势")
    elif win_rate >= 50 and profit_factor >= 1.1:
        print("  △ 良好 - 策略略优于随机")
    elif win_rate >= 45:
        print("  ⚠ 一般 - 接近随机水平")
    else:
        print("  ✗ 较差 - 需要优化")
    print("=" * 80)
    
    return df_results

if __name__ == "__main__":
    scan_single_date('2024-03-15', hold_days=5, max_stocks=50)
