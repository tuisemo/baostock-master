"""
完整系统功能验证脚本

在 AI 模型训练完成后，进行全面的系统功能验证：
1. 训练 AI 模型
2. 使用反向验证评估选股能力
3. 生成详细报告

使用方法:
    python scripts/full_system_validation.py
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.core.strategy_params import StrategyParams
from quant.features.analyzer import calculate_indicators
from quant.features.features import extract_features, create_targets
from quant.core.trainer import build_dataset, train_model
from quant.app.backtester import get_market_index, scan_today_signal
from tqdm import tqdm


def step1_train_ai_model():
    """步骤 1: 训练 AI 模型"""
    print("\n" + "=" * 80)
    print("步骤 1: 训练 AI 模型")
    print("=" * 80)
    
    p = StrategyParams.from_app_config(CONF)
    data_dir = CONF.history_data.data_dir
    
    print(f"构建数据集...")
    df = build_dataset(
        data_dir,
        p,
        n_forward_days=p.ai_forward_days,
        target_atr_mult=p.ai_target_atr_mult,
        stop_loss_atr_mult=p.ai_stop_loss_atr_mult,
    )
    
    if df.empty:
        print("❌ 数据集为空，训练中止")
        return False
    
    print(f"✓ 数据集构建完成：{len(df)} 条样本")
    print(f"  正样本：{df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"  负样本：{len(df) - df['target'].sum()} ({(1-df['target'].mean())*100:.1f}%)")
    
    # 检查特征列
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    print(f"  特征数：{len(feat_cols)}")
    
    if len(feat_cols) == 0:
        print("❌ 没有特征列，训练中止")
        return False
    
    print(f"\n开始训练 LightGBM 模型...")
    model_path = "models/alpha_lgbm.txt"
    
    try:
        train_model(df, model_path=model_path)
        print(f"✓ 模型训练完成，保存至：{model_path}")
        
        # 验证模型
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
        print(f"✓ 模型验证成功：{model.num_feature()} 个特征")
        return True
        
    except Exception as e:
        print(f"❌ 模型训练失败：{e}")
        return False


def step2_scan_test_date(target_date='2024-03-15'):
    """步骤 2: 在测试日期扫描买点"""
    print("\n" + "=" * 80)
    print(f"步骤 2: 扫描历史日期 {target_date} 的买点信号")
    print("=" * 80)
    
    # 获取所有股票
    data_path = Path(CONF.history_data.data_dir)
    stock_files = list(data_path.glob("*.csv"))
    stock_files = [f for f in stock_files if f.name not in ["stock-list.csv", "sh.000001.csv"]]
    
    print(f"扫描 {min(100, len(stock_files))} 只股票...")
    
    signals = []
    for stock_file in tqdm(stock_files[:100], desc="扫描买点"):
        try:
            code = stock_file.stem
            signal = scan_today_signal(code, target_date=target_date)
            if signal:
                signals.append(signal)
        except Exception:
            continue
    
    print(f"\n发现 {len(signals)} 个买点信号")
    
    if signals:
        # 按 buy_score 排序
        signals.sort(key=lambda x: x.get('buy_score', 0), reverse=True)
        
        print("\n前 10 个信号:")
        for i, sig in enumerate(signals[:10], 1):
            print(f"  {i}. {sig['code']}: {sig['signal_type']}, "
                  f"score={sig.get('buy_score', 0):.3f}, "
                  f"AI_prob={sig.get('ai_prob', 0):.2%}")
    
    return signals


def step3_simulate_performance(signals, hold_days=5):
    """步骤 3: 模拟持股表现"""
    print("\n" + "=" * 80)
    print(f"步骤 3: 模拟持股 {hold_days} 天的表现")
    print("=" * 80)
    
    if not signals:
        print("没有信号可模拟")
        return []
    
    performances = []
    
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
            
            # 找到买入日期
            buy_idx = df[df['date'] == pd.to_datetime(buy_date)]
            if buy_idx.empty:
                continue
            
            buy_idx = buy_idx.index[0]
            
            # 计算未来 N 天的价格路径
            end_idx = min(buy_idx + hold_days, len(df) - 1)
            if end_idx <= buy_idx:
                continue
            
            price_path = df.iloc[buy_idx:end_idx + 1]['close'].values
            
            # 计算最大浮盈浮亏
            cum_max = np.maximum.accumulate(price_path)
            drawdowns = (price_path - cum_max) / cum_max
            max_drawdown_pct = abs(np.min(drawdowns)) * 100
            
            cum_min = np.minimum.accumulate(price_path)
            runups = (price_path - cum_min) / cum_min
            max_profit_pct = np.max(runups) * 100
            
            # 最终收益
            sell_price = price_path[-1]
            pnl_pct = (sell_price - buy_price) / buy_price * 100
            
            performances.append({
                'code': code,
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'hold_days': end_idx - buy_idx,
                'pnl_pct': pnl_pct,
                'max_profit_pct': max_profit_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'signal_type': signal.get('signal_type', ''),
                'ai_prob': signal.get('ai_prob', 0.5),
                'buy_score': signal.get('buy_score', 0),
            })
            
        except Exception as e:
            continue
    
    print(f"完成 {len(performances)} 笔交易模拟")
    return performances


def step4_analyze_results(performances):
    """步骤 4: 分析结果"""
    print("\n" + "=" * 80)
    print("步骤 4: 分析验证结果")
    print("=" * 80)
    
    if not performances:
        print("没有有效的交易结果")
        return None
    
    df = pd.DataFrame(performances)
    
    # 基础统计
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
    avg_max_drawdown = df['max_drawdown_pct'].mean()
    
    # 打印统计
    print(f"\n【整体表现】")
    print(f"  总交易数：{total}")
    print(f"  胜率：{win_rate:.2f}% ({len(wins)}赢/{len(losses)}亏)")
    print(f"  平均收益：{avg_pnl:.2f}%")
    print(f"  中位数收益：{median_pnl:.2f}%")
    print(f"  最大盈利：{max_profit:.2f}%")
    print(f"  最大亏损：{max_loss:.2f}%")
    print(f"  平均最大浮盈：{avg_max_profit:.2f}%")
    print(f"  平均最大浮亏：{avg_max_drawdown_pct:.2f}%")
    print(f"  盈亏比：{profit_factor:.2f}")
    
    # 按 AI 置信度分析
    print(f"\n【按 AI 置信度分析】")
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
    print(f"\n【按信号类型分析】")
    for sig_type in df['signal_type'].unique():
        subset = df[df['signal_type'] == sig_type]
        if len(subset) > 0:
            type_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            type_avg = subset['pnl_pct'].mean()
            print(f"  {sig_type}: {len(subset)}笔，胜率 {type_wr:.1f}%, 平均收益 {type_avg:.2f}%")
    
    # 评估选股能力
    print("\n" + "=" * 80)
    print("【选股能力评估】")
    
    if win_rate >= 55 and profit_factor >= 1.3:
        print("  ✓ 优秀 - 策略具有显著的盈利优势")
        print(f"    高胜率 ({win_rate:.1f}%) + 高盈亏比 ({profit_factor:.2f})")
    elif win_rate >= 50 and profit_factor >= 1.1:
        print("  △ 良好 - 策略略优于随机")
        print(f"    胜率 ({win_rate:.1f}%) + 盈亏比 ({profit_factor:.2f}) 均达标")
    elif win_rate >= 45:
        print("  ⚠ 一般 - 接近随机水平，需要优化")
        print(f"    建议：提高 AI 阈值、加强量能确认、优化止盈止损")
    else:
        print("  ✗ 较差 - 策略可能存在问题")
        print(f"    建议：重新训练模型、调整参数、检查数据质量")
    
    print("=" * 80)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = f"full_system_validation_{timestamp}.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存至：{output_csv}")
    
    return df


def run_full_validation():
    """运行完整的功能验证流程"""
    
    print("=" * 80)
    print("完整系统功能验证")
    print("目标：评估 AI 模型训练后的选股能力")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 步骤 1: 训练 AI 模型
    if not step1_train_ai_model():
        print("\n❌ AI 模型训练失败，中止验证")
        return None
    
    # 步骤 2: 扫描测试日期
    test_date = '2024-03-15'  # 选择一个历史日期
    signals = step2_scan_test_date(test_date)
    
    # 步骤 3: 模拟表现
    performances = step3_simulate_performance(signals, hold_days=5)
    
    # 步骤 4: 分析结果
    results_df = step4_analyze_results(performances)
    
    # 完成
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n验证完成，耗时：{elapsed/60:.1f} 分钟")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    results = run_full_validation()
    
    if results is not None:
        print("\n✓ 系统功能验证成功完成")
        print("下一步：根据验证结果优化策略参数")
    else:
        print("\n✗ 系统功能验证失败")
        print("请检查日志了解详细原因")
