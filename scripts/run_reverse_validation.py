"""
反向验证分析脚本

选择一个历史日期范围，执行选股并观察后续表现
评估选股策略的有效性

使用方法:
    python scripts/run_reverse_validation.py
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
import pandas as pd
import numpy as np

from quant.app.reverse_validation import ReverseValidator
from quant.core.strategy_params import StrategyParams
from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.app.backtester import get_market_index


def analyze_date_range():
    """分析历史日期范围的选股表现"""
    
    print("=" * 80)
    print("反向验证分析：选股策略能力评估")
    print("=" * 80)
    
    # 配置分析参数
    # 选择一个历史区间：2024 年不同市场状态的月份
    test_config = {
        # 测试区间 (选择不同市场状态的代表性月份)
        'test_periods': [
            ('2024-01-02', '2024-01-31'),  # 年初震荡
            ('2024-03-01', '2024-03-29'),  # 春季行情
            ('2024-05-06', '2024-05-31'),  # 震荡整理
            ('2024-07-01', '2024-07-31'),  # 夏季调整
            ('2024-09-02', '2024-09-30'),  # 秋季行情
            ('2024-11-01', '2024-11-29'),  # 年底行情
        ],
        # 持仓天数
        'hold_days': 5,
        # 每个日期最多测试的股票数 (前 N 个信号)
        'max_stocks_per_date': 30,
        # 抽样频率 (每隔 N 个交易日测试一次)
        'sample_frequency': 3,
    }
    
    # 获取大盘指数数据来确定交易日
    idx_df = get_market_index()
    if idx_df is None:
        print("错误：无法读取大盘指数数据，请先同步历史数据")
        return
    
    # 生成测试日期列表
    all_test_dates = []
    for start_date, end_date in test_config['test_periods']:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        valid_dates = idx_df.index[(idx_df.index >= start) & (idx_df.index <= end)]
        
        # 抽样：每隔 N 个交易日测试一次
        freq = test_config['sample_frequency']
        sampled_dates = valid_dates[::freq]
        all_test_dates.extend(sampled_dates)
        print(f"期间 {start_date} ~ {end_date}: {len(valid_dates)} 个交易日 -> 抽样 {len(sampled_dates)} 个")
    
    print(f"\n总计：{len(all_test_dates)} 个测试日期")
    print(f"持仓天数：{test_config['hold_days']} 天")
    print(f"每日期最大股票数：{test_config['max_stocks_per_date']}")
    print("=" * 80)
    
    # 创建验证器
    validator = ReverseValidator(
        data_dir=CONF.history_data.data_dir,
        default_hold_days=test_config['hold_days'],
    )
    
    # 运行完整验证
    test_dates_str = [d.strftime('%Y-%m-%d') for d in all_test_dates]
    
    summary, by_category, df = validator.run_full_validation(
        test_dates=test_dates_str,
        hold_days=test_config['hold_days'],
        max_stocks_per_date=test_config['max_stocks_per_date'],
    )
    
    # 导出报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"reverse_validation_{timestamp}.txt"
    validator.export_report(summary, by_category, df, output_path)
    
    # 打印详细分析
    print("\n" + "=" * 80)
    print("核心发现")
    print("=" * 80)
    
    # 1. 整体表现
    print("\n【整体表现】")
    print(f"  总交易数：{summary.total_signals}")
    print(f"  胜率：{summary.win_rate:.2f}% ({summary.win_count}赢/{summary.loss_count}亏)")
    print(f"  平均收益：{summary.avg_pnl_pct:.2f}%")
    print(f"  中位数收益：{summary.median_pnl_pct:.2f}%")
    print(f"  盈亏比：{summary.profit_factor:.2f}")
    print(f"  夏普比率：{summary.sharpe_ratio:.2f}")
    print(f"  平均持仓：{summary.avg_hold_days:.1f}天")
    
    # 2. 收益分布特征
    print("\n【收益分布特征】")
    print(f"  最大盈利：{summary.max_profit_pct:.2f}%")
    print(f"  最大亏损：{summary.max_loss_pct:.2f}%")
    print(f"  平均最大浮盈：{summary.avg_max_profit_pct:.2f}%")
    print(f"  平均最大浮亏：{summary.avg_max_drawdown_pct:.2f}%")
    print(f"  偏度：{summary.skewness:.2f} ({'右偏/正偏' if summary.skewness > 0 else '左偏/负偏'})")
    print(f"  峰度：{summary.kurtosis:.2f} ({'尖峰' if summary.kurtosis > 0 else '平峰'})")
    
    # 3. 关键指标解读
    print("\n【关键指标解读】")
    
    # 胜率评估
    if summary.win_rate >= 55:
        print(f"  ✓ 胜率优秀 ({summary.win_rate:.1f}%)，策略具有显著的选股优势")
    elif summary.win_rate >= 50:
        print(f"  △ 胜率良好 ({summary.win_rate:.1f}%)，策略略优于随机")
    elif summary.win_rate >= 45:
        print(f"  ⚠ 胜率一般 ({summary.win_rate:.1f}%)，接近随机水平")
    else:
        print(f"  ✗ 胜率较差 ({summary.win_rate:.1f}%)，选股逻辑可能存在问题")
    
    # 盈亏比评估
    if summary.profit_factor >= 1.5:
        print(f"  ✓ 盈亏比优秀 ({summary.profit_factor:.2f})，盈利能力强")
    elif summary.profit_factor >= 1.2:
        print(f"  △ 盈亏比良好 ({summary.profit_factor:.2f})，有一定盈利优势")
    elif summary.profit_factor >= 1.0:
        print(f"  ⚠ 盈亏比一般 ({summary.profit_factor:.2f})，勉强覆盖亏损")
    else:
        print(f"  ✗ 盈亏比较差 ({summary.profit_factor:.2f})，亏损超过盈利")
    
    # 夏普比率评估
    if summary.sharpe_ratio >= 1.5:
        print(f"  ✓ 风险调整收益优秀 (Sharpe={summary.sharpe_ratio:.2f})")
    elif summary.sharpe_ratio >= 0.8:
        print(f"  △ 风险调整收益良好 (Sharpe={summary.sharpe_ratio:.2f})")
    elif summary.sharpe_ratio >= 0.3:
        print(f"  ⚠ 风险调整收益一般 (Sharpe={summary.sharpe_ratio:.2f})")
    else:
        print(f"  ✗ 风险调整收益较差 (Sharpe={summary.sharpe_ratio:.2f})")
    
    # 浮盈浮亏分析
    print("\n【浮盈浮亏分析】")
    profit_retention = summary.avg_pnl_pct / summary.avg_max_profit_pct * 100 if summary.avg_max_profit_pct > 0 else 0
    loss_avoidance = summary.avg_pnl_pct / (summary.avg_pnl_pct - summary.avg_max_drawdown_pct) * 100 if (summary.avg_pnl_pct - summary.avg_max_drawdown_pct) != 0 else 0
    
    print(f"  利润留存率：{profit_retention:.1f}% (平均实现收益/平均最大浮盈)")
    print(f"  亏损规避率：{loss_avoidance:.1f}%")
    
    if profit_retention < 50:
        print(f"  ⚠ 利润留存率偏低，建议优化止盈策略")
    elif profit_retention > 80:
        print(f"  ✓ 利润留存率良好")
    
    # 4. 按类别分析
    print("\n【按信号类型分析】")
    for signal_type, stats in by_category.get('by_signal_type', {}).items():
        print(f"\n  {signal_type}:")
        print(f"    样本数：{stats['count']}")
        print(f"    胜率：{stats['win_rate']:.1f}%")
        print(f"    平均收益：{stats['avg_pnl']:.2f}%")
        if stats['count'] >= 10:  # 只有样本足够才评价
            if stats['win_rate'] > 55 and stats['avg_pnl'] > 1:
                print(f"    ✓ 该信号类型表现优秀")
            elif stats['win_rate'] > 50:
                print(f"    △ 该信号类型表现良好")
            else:
                print(f"    ⚠ 该信号类型需要优化")
    
    print("\n【按市场状态分析】")
    for state, stats in by_category.get('by_market_state', {}).items():
        print(f"\n  {state}:")
        print(f"    样本数：{stats['count']}")
        print(f"    胜率：{stats['win_rate']:.1f}%")
        print(f"    平均收益：{stats['avg_pnl']:.2f}%")
    
    print("\n【按 AI 置信度分析】")
    for tier, stats in by_category.get('by_ai_confidence', {}).items():
        print(f"\n  {tier}:")
        print(f"    样本数：{stats['count']}")
        print(f"    胜率：{stats['win_rate']:.1f}%")
        print(f"    平均收益：{stats['avg_pnl']:.2f}%")
    
    print("\n" + "=" * 80)
    print(f"完整报告已保存至：{output_path}")
    print(f"详细数据已保存为 CSV 文件")
    print("=" * 80)
    
    return summary, by_category, df


def identify_optimization_opportunities(summary, by_category):
    """基于验证结果识别优化机会"""
    
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    
    suggestions = []
    
    # 1. 胜率偏低
    if summary.win_rate < 50:
        suggestions.append({
            'issue': '胜率偏低',
            'severity': '高',
            'suggestions': [
                '提高 AI 概率阈值，过滤低质量信号',
                '加强量能确认要求 (vol_up_ratio)',
                '增加多周期一致性确认',
                '考虑提高 min_composite_score 门槛',
            ]
        })
    
    # 2. 盈亏比偏低
    if summary.profit_factor < 1.2:
        suggestions.append({
            'issue': '盈亏比偏低',
            'severity': '高',
            'suggestions': [
                '调整止盈/止损比例 (ai_target_atr_mult / ai_stop_loss_atr_mult)',
                '优化移动止损策略 (trail_atr_mult)',
                '增加早期止盈机制 (breakeven_trigger)',
                '改善持仓周期管理 (max_hold_days)',
            ]
        })
    
    # 3. 最大浮亏过大
    if summary.avg_max_drawdown_pct > 8:
        suggestions.append({
            'issue': '最大浮亏过大',
            'severity': '中',
            'suggestions': [
                '收紧初始止损 (降低 trail_atr_mult)',
                '增加波动率调整的动态止损',
                '提前激活移动止损保护',
                '在市场高波动状态降低仓位',
            ]
        })
    
    # 4. 利润留存率低
    if summary.avg_max_profit_pct > 0:
        profit_retention = summary.avg_pnl_pct / summary.avg_max_profit_pct * 100
        if profit_retention < 40:
            suggestions.append({
                'issue': '利润留存率偏低',
                'severity': '中',
                'suggestions': [
                    '优化止盈策略，避免过早平仓',
                    '使用更宽松的移动止损 (提高 trail_atr_mult)',
                    '区分左侧/右侧交易的止盈策略',
                    '增加趋势延续信号的权重',
                ]
            })
    
    # 5. 按市场状态优化
    by_market = by_category.get('by_market_state', {})
    for state, stats in by_market.items():
        if stats['count'] >= 10 and stats['win_rate'] < 45:
            suggestions.append({
                'issue': f'{state}市场状态下表现不佳',
                'severity': '中',
                'suggestions': [
                    f'调整{state}状态下的 AI 阈值 (market_state_thresholds.{state}.ai_threshold)',
                    f'降低{state}状态下的仓位 (market_state_thresholds.{state}.position_mult)',
                    f'优化{state}状态下的止损策略',
                    f'考虑在{state}状态下暂停交易或大幅降低频率',
                ]
            })
    
    # 6. 按信号类型优化
    by_signal = by_category.get('by_signal_type', {})
    for signal_type, stats in by_signal.items():
        if stats['count'] >= 10 and stats['win_rate'] < 45:
            suggestions.append({
                'issue': f'{signal_type}信号表现不佳',
                'severity': '中',
                'suggestions': [
                    '重新评估该信号类型的权重配置',
                    '增加额外的确认条件',
                    '考虑降低该信号类型的触发频率',
                    '检查是否存在数据泄露或过拟合',
                ]
            })
    
    # 7. 按 AI 置信度优化
    by_ai = by_category.get('by_ai_confidence', {})
    if 'high (>=0.65)' in by_ai:
        high_tier = by_ai['high (>=0.65)']
        if high_tier['win_rate'] < 55:
            suggestions.append({
                'issue': '高置信度信号表现未达预期',
                'severity': '高',
                'suggestions': [
                    '重新训练 AI 模型，检查特征工程',
                    '提高高置信度档位阈值 (CONFIDENCE_HIGH_THRESHOLD)',
                    '增加 AI 门控与其他维度的交叉验证',
                    '检查训练数据与实战数据的分布差异',
                ]
            })
    
    # 打印建议
    if not suggestions:
        print("\n✓ 未发现明显的优化点，策略整体表现健康")
    else:
        for i, sug in enumerate(suggestions, 1):
            print(f"\n[{i}] 问题：{sug['issue']} (严重性：{sug['severity']})")
            print("    建议措施:")
            for j, action in enumerate(sug['suggestions'], 1):
                print(f"      {j}. {action}")
    
    print("\n" + "=" * 80)
    return suggestions


if __name__ == "__main__":
    # 运行分析
    summary, by_category, df = analyze_date_range()
    
    if summary and summary.total_signals > 0:
        # 识别优化机会
        suggestions = identify_optimization_opportunities(summary, by_category)
        
        print("\n" + "=" * 80)
        print("下一步行动")
        print("=" * 80)
        print("1. 查看详细报告文件了解完整统计")
        print("2. 根据优化建议调整策略参数")
        print("3. 使用优化后的参数重新运行反向验证")
        print("4. 确认改进后，可以使用 UI 的闭环寻优功能进一步微调")
        print("=" * 80)
