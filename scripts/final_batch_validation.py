"""
最终批量验证 - 2024 年全市场选股能力验证
目标：完成 20+ 日期验证，收集 100+ 交易样本

策略：
- 24 个代表性日期（每月 2 个）
- 每个日期扫描 200 只股票
- 5 天持有期
- 带止损止盈

使用方法:
    python scripts/final_batch_validation.py

输出:
    - data/exports/batch_validation_final_YYYYMMDD_HHMMSS.csv
    - data/exports/batch_validation_final_report_YYYYMMDD_HHMMSS.txt
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant.app.reverse_validation import ReverseValidator
from quant.infra.config import CONF


def run_final_validation():
    """运行最终批量验证"""
    # 24 个测试日期，覆盖 2024 年不同市场环境
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
    print("最终批量验证 - 2024 年全市场选股能力评估")
    print("=" * 80)
    print(f"测试日期数：{len(test_dates)}")
    print(f"测试日期范围：{test_dates[0]} 至 {test_dates[-1]}")
    print(f"持股天数：5 天")
    print(f"每日期扫描：200 只股票")
    print(f"预计交易数：80-150 笔")
    print("=" * 80)
    
    # 创建验证器
    validator = ReverseValidator(
        data_dir=CONF.history_data.data_dir,
        default_hold_days=5,
    )
    
    # 运行验证
    print("\n开始执行验证...")
    start_time = datetime.now()
    
    summary, by_category, df = validator.run_full_validation(
        test_dates=test_dates,
        hold_days=5,
        max_stocks_per_date=200,  # 每个日期扫描 200 只股票
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if df.empty:
        print("验证失败：没有有效交易数据")
        return None
    
    # 打印结果
    print("\n" + "=" * 80)
    print("【验证结果摘要】")
    print("=" * 80)
    print(f"总交易数：{len(df)}")
    print(f"覆盖日期：{df['buy_date'].nunique()} 个")
    print(f"胜率：{summary.win_rate:.2f}%")
    print(f"平均收益：{summary.avg_pnl_pct:.2f}%")
    print(f"盈亏比：{summary.profit_factor:.2f}")
    print(f"夏普比率：{summary.sharpe_ratio:.2f}")
    print(f"总用时：{elapsed/60:.1f}分钟")
    
    # 按市场状态分析
    if 'market_state' in df.columns:
        print("\n【按市场状态分析】")
        for state in sorted(df['market_state'].unique()):
            subset = df[df['market_state'] == state]
            if len(subset) > 0:
                state_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
                state_avg = subset['pnl_pct'].mean()
                print(f"  {state}: {len(subset)}笔，胜率 {state_wr:.1f}%, 平均收益 {state_avg:.2f}%")
    
    # 按 AI 置信度分析
    print("\n【按 AI 置信度分析】")
    df['ai_tier'] = pd.cut(df['ai_confidence'], 
                           bins=[0, 0.45, 0.65, 1.0],
                           labels=['低 (<0.45)', '中 (0.45-0.65)', '高 (>0.65)'])
    
    for tier in ['低 (<0.45)', '中 (0.45-0.65)', '高 (>0.65)']:
        subset = df[df['ai_tier'] == tier]
        if len(subset) > 0:
            tier_wr = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
            tier_avg = subset['pnl_pct'].mean()
            print(f"  {tier}: {len(subset)}笔，胜率 {tier_wr:.1f}%, 平均收益 {tier_avg:.2f}%")
    
    # 选股能力评级
    print("\n" + "=" * 80)
    print("【选股能力评级】")
    win_rate = summary.win_rate
    profit_factor = summary.profit_factor
    
    if win_rate >= 55 and profit_factor >= 1.3:
        rating = "优秀 - 策略具有显著的盈利优势"
    elif win_rate >= 50 and profit_factor >= 1.1:
        rating = "良好 - 策略优于随机"
    elif win_rate >= 45:
        rating = "一般 - 接近随机水平，需要优化"
    else:
        rating = "较差 - 策略可能存在问题，建议优化"
    
    print(f"  {rating}")
    print("=" * 80)
    
    # 保存结果
    output_dir = Path("data/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV
    csv_path = output_dir / f"batch_validation_final_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存：{csv_path}")
    
    # 报告
    report_path = output_dir / f"batch_validation_final_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("大规模反向验证报告 - 2024 年全市场选股能力评估\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试日期范围：{test_dates[0]} 至 {test_dates[-1]}\n")
        f.write(f"测试日期数：{len(test_dates)}\n")
        f.write(f"总交易数：{len(df)}\n")
        f.write(f"总用时：{elapsed/60:.1f}分钟\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("核心统计指标\n")
        f.write("-" * 80 + "\n")
        f.write(f"胜率：{summary.win_rate:.2f}% ({summary.win_count}赢/{summary.loss_count}亏)\n")
        f.write(f"平均收益：{summary.avg_pnl_pct:.2f}%\n")
        f.write(f"中位数收益：{summary.median_pnl_pct:.2f}%\n")
        f.write(f"最大单笔盈利：{summary.max_profit_pct:.2f}%\n")
        f.write(f"最大单笔亏损：{summary.max_loss_pct:.2f}%\n")
        f.write(f"平均最大浮盈：{summary.avg_max_profit_pct:.2f}%\n")
        f.write(f"平均最大浮亏：{summary.avg_max_drawdown_pct:.2f}%\n")
        f.write(f"盈亏比：{summary.profit_factor:.2f}\n")
        f.write(f"平均持仓天数：{summary.avg_hold_days:.1f}天\n")
        f.write(f"年化夏普比率：{summary.sharpe_ratio:.2f}\n")
        f.write(f"收益分布偏度：{summary.skewness:.2f}\n")
        f.write(f"收益分布峰度：{summary.kurtosis:.2f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("按市场状态分析\n")
        f.write("-" * 80 + "\n")
        if 'market_state' in df.columns:
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
        f.write(f"选股能力评级：{rating}\n")
        f.write("=" * 80 + "\n")
    
    print(f"验证报告已保存：{report_path}")
    
    # 摘要 JSON
    summary_json = {
        'test_dates_count': len(test_dates),
        'total_trades': len(df),
        'win_rate': summary.win_rate,
        'avg_pnl': summary.avg_pnl_pct,
        'profit_factor': summary.profit_factor,
        'sharpe_ratio': summary.sharpe_ratio,
        'elapsed_minutes': elapsed / 60,
        'rating': rating,
        'timestamp': datetime.now().isoformat(),
    }
    json_path = output_dir / f"batch_validation_final_summary_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    
    print(f"摘要已保存：{json_path}")
    
    # 验证要求
    print("\n" + "=" * 80)
    print("【验证要求检查】")
    print("=" * 80)
    dates_count = df['buy_date'].nunique()
    trades_count = len(df)
    print(f"✓ 覆盖日期数：{dates_count} (要求 >= 20) - {'通过' if dates_count >= 20 else '不通过'}")
    print(f"✓ 总交易数：{trades_count} (要求 >= 100) - {'通过' if trades_count >= 100 else '不通过'}")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    run_final_validation()
