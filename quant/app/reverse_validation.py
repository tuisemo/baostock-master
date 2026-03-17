"""
反向验证模块：历史选股能力评估

核心逻辑：
1. 选择一个历史日期 T
2. 执行选股策略，得到推荐股票列表
3. 观察这些股票在 T+N 天后的表现
4. 统计胜率、平均收益、最大回撤等指标
5. 评估选股策略的有效性

使用方法：
    python main.py reverse-validate --date 2024-05-10 --hold-days 5
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.app.backtester import scan_today_signal, get_market_index, get_slippage_model
from quant.core.strategy_params import StrategyParams

# 临时禁用 AI 模型（如果模型文件损坏）
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class StockPerformance:
    """股票表现数据"""
    code: str
    name: str
    buy_date: str
    buy_price: float
    sell_date: str
    sell_price: float
    hold_days: int
    pnl_pct: float
    max_profit_pct: float
    max_drawdown_pct: float
    signal_type: str
    signal_score: float
    ai_confidence: float
    market_state: str


@dataclass
class ValidationSummary:
    """验证结果汇总"""
    total_signals: int
    valid_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_pnl_pct: float
    median_pnl_pct: float
    max_profit_pct: float
    max_loss_pct: float
    avg_max_profit_pct: float
    avg_max_drawdown_pct: float
    profit_factor: float
    avg_hold_days: float
    sharpe_ratio: float
    skewness: float
    kurtosis: float


class ReverseValidator:
    """
    反向验证器
    
    通过在历史日期执行选股策略，观察后续表现来评估策略有效性
    """
    
    def __init__(
        self,
        data_dir: str = None,
        default_hold_days: int = 5,
        default_capital: float = 100000.0,
    ):
        self.data_dir = data_dir or CONF.history_data.data_dir
        self.default_hold_days = default_hold_days
        self.default_capital = default_capital
        self.results: List[StockPerformance] = []
        
    def scan_historical_date(
        self,
        target_date: str,
        params: StrategyParams = None,
        max_stocks: int = None,
    ) -> List[Dict]:
        """
        在历史日期执行选股扫描
        
        Args:
            target_date: 目标日期 (YYYY-MM-DD)
            params: 策略参数
            max_stocks: 最多返回的股票数量
            
        Returns:
            推荐股票列表
        """
        logger.info(f"开始扫描历史日期 {target_date} 的买点信号...")
        
        # 获取所有可用的股票数据文件
        data_path = Path(self.data_dir)
        stock_files = list(data_path.glob("*.csv"))
        stock_files = [f for f in stock_files if f.name != "stock-list.csv" and f.name != "sh.000001.csv"]
        
        if not stock_files:
            logger.error(f"在 {self.data_dir} 未找到股票数据文件")
            return []
        
        # 限制扫描数量 (用于快速测试)
        if max_stocks:
            stock_files = stock_files[:max_stocks]
        
        logger.info(f"将扫描 {len(stock_files)} 只股票...")
        
        # 并行扫描所有股票
        results = []
        for stock_file in tqdm(stock_files, desc="扫描买点"):
            try:
                code = stock_file.stem
                signal = scan_today_signal(code, params=params, target_date=target_date)
                if signal:
                    results.append(signal)
            except Exception as e:
                logger.debug(f"扫描 {stock_file.name} 失败：{e}")
        
        # 按 buy_score 排序
        if results:
            results.sort(key=lambda x: x.get('buy_score', 0), reverse=True)
        
        logger.info(f"扫描完成，共发现 {len(results)} 个买点信号")
        return results
    
    def simulate_trades(
        self,
        signals: List[Dict],
        hold_days: int = None,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.10,
    ) -> List[StockPerformance]:
        """
        模拟交易并计算表现
        
        Args:
            signals: 买点信号列表
            hold_days: 持仓天数
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            
        Returns:
            交易表现列表
        """
        hold_days = hold_days or self.default_hold_days
        performances = []
        
        for signal in tqdm(signals, desc="模拟交易"):
            try:
                perf = self._simulate_single_trade(signal, hold_days, stop_loss_pct, take_profit_pct)
                if perf:
                    performances.append(perf)
            except Exception as e:
                logger.debug(f"模拟交易失败 {signal.get('code')}: {e}")
        
        self.results.extend(performances)
        return performances
    
    def _simulate_single_trade(
        self,
        signal: Dict,
        hold_days: int,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> StockPerformance | None:
        """模拟单笔交易"""
        code = signal.get('code')
        buy_date = signal.get('date')
        buy_price = signal.get('close')
        
        if not all([code, buy_date, buy_price]):
            return None
        
        # 读取股票历史数据
        stock_file = Path(self.data_dir) / f"{code}.csv"
        if not stock_file.exists():
            return None
        
        df = pd.read_csv(stock_file)
        if df.empty or 'date' not in df.columns:
            return None
        
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
            return None  # 数据不足
        
        price_path = df.iloc[buy_idx:end_idx + 1]['close'].values
        
        # 计算最大盈利和最大回撤
        cum_max = np.maximum.accumulate(price_path)
        drawdowns = (price_path - cum_max) / cum_max
        max_drawdown_pct = abs(np.min(drawdowns)) * 100
        
        cum_min = np.minimum.accumulate(price_path)
        runups = (price_path - cum_min) / cum_min
        max_profit_pct = np.max(runups) * 100
        
        # 简化：假设持有 N 天后卖出
        sell_price = price_path[-1]
        pnl_pct = (sell_price - buy_price) / buy_price * 100
        
        # 检查止损/止盈
        actual_sell_price = sell_price
        actual_pnl = pnl_pct
        
        for i, price in enumerate(price_path[1:], 1):
            # 止损检查
            if (price - buy_price) / buy_price <= -stop_loss_pct:
                actual_sell_price = price
                actual_pnl = (price - buy_price) / buy_price * 100
                break
            # 止盈检查
            if (price - buy_price) / buy_price >= take_profit_pct:
                actual_sell_price = price
                actual_pnl = (price - buy_price) / buy_price * 100
                break
        
        sell_date = df.iloc[buy_idx + i if 'i' in locals() else end_idx]['date'].strftime('%Y-%m-%d')
        
        return StockPerformance(
            code=code,
            name=signal.get('name', ''),
            buy_date=buy_date,
            buy_price=buy_price,
            sell_date=sell_date,
            sell_price=actual_sell_price,
            hold_days=i if 'i' in locals() else hold_days,
            pnl_pct=actual_pnl,
            max_profit_pct=max_profit_pct,
            max_drawdown_pct=max_drawdown_pct,
            signal_type=signal.get('signal_type', ''),
            signal_score=signal.get('buy_score', 0),
            ai_confidence=signal.get('ai_prob', 0.5),
            market_state=signal.get('market_state', 'unknown'),
        )
    
    def compute_summary(self, performances: List[StockPerformance] = None) -> ValidationSummary:
        """
        计算验证摘要统计
        
        Args:
            performances: 交易表现列表，如果为 None 则使用 self.results
            
        Returns:
            验证摘要
        """
        perfs = performances or self.results
        
        if not perfs:
            return ValidationSummary(
                total_signals=0,
                valid_trades=0,
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                avg_pnl_pct=0.0,
                median_pnl_pct=0.0,
                max_profit_pct=0.0,
                max_loss_pct=0.0,
                avg_max_profit_pct=0.0,
                avg_max_drawdown_pct=0.0,
                profit_factor=0.0,
                avg_hold_days=0.0,
                sharpe_ratio=0.0,
                skewness=0.0,
                kurtosis=0.0,
            )
        
        # 基础统计
        total = len(perfs)
        wins = [p for p in perfs if p.pnl_pct > 0]
        losses = [p for p in perfs if p.pnl_pct <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total * 100 if total > 0 else 0
        
        # PnL 统计
        pnls = [p.pnl_pct for p in perfs]
        avg_pnl = np.mean(pnls)
        median_pnl = np.median(pnls)
        max_profit = max(pnls)
        max_loss = min(pnls)
        
        # 路径统计
        avg_max_profit = np.mean([p.max_profit_pct for p in perfs])
        avg_max_drawdown = np.mean([p.max_drawdown_pct for p in perfs])
        
        # 盈亏比
        gross_profit = sum(p.pnl_pct for p in wins) if wins else 0
        gross_loss = abs(sum(p.pnl_pct for p in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 持仓周期
        avg_hold = np.mean([p.hold_days for p in perfs])
        
        # 夏普比率 (假设无风险利率为 0)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # 年化
        else:
            sharpe = 0.0
        
        # 偏度和峰度
        skewness = pd.Series(pnls).skew()
        kurtosis = pd.Series(pnls).kurtosis()
        
        return ValidationSummary(
            total_signals=total,
            valid_trades=total,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            avg_pnl_pct=avg_pnl,
            median_pnl_pct=median_pnl,
            max_profit_pct=max_profit,
            max_loss_pct=max_loss,
            avg_max_profit_pct=avg_max_profit,
            avg_max_drawdown_pct=avg_max_drawdown,
            profit_factor=profit_factor,
            avg_hold_days=avg_hold,
            sharpe_ratio=sharpe,
            skewness=skewness,
            kurtosis=kurtosis,
        )
    
    def analyze_by_category(self, performances: List[StockPerformance] = None) -> Dict:
        """
        按类别分析表现
        
        Categories:
        - 信号类型 (左侧/右侧)
        - 市场状态
        - AI 置信度档位
        
        Returns:
            分析结果字典
        """
        perfs = performances or self.results
        
        if not perfs:
            return {}
        
        # 按信号类型分析
        by_signal_type = {}
        for signal_type in set(p.signal_type for p in perfs):
            subset = [p for p in perfs if p.signal_type == signal_type]
            summary = self.compute_summary(subset)
            by_signal_type[signal_type] = {
                'count': len(subset),
                'win_rate': summary.win_rate,
                'avg_pnl': summary.avg_pnl_pct,
                'sharpe': summary.sharpe_ratio,
            }
        
        # 按市场状态分析
        by_market_state = {}
        for state in set(p.market_state for p in perfs):
            subset = [p for p in perfs if p.market_state == state]
            summary = self.compute_summary(subset)
            by_market_state[state] = {
                'count': len(subset),
                'win_rate': summary.win_rate,
                'avg_pnl': summary.avg_pnl_pct,
                'sharpe': summary.sharpe_ratio,
            }
        
        # 按 AI 置信度档位分析
        by_ai_tier = {}
        for p in perfs:
            conf = p.ai_confidence
            if conf >= 0.65:
                tier = 'high (>=0.65)'
            elif conf >= 0.45:
                tier = 'medium (0.45-0.65)'
            else:
                tier = 'low (<0.45)'
            
            if tier not in by_ai_tier:
                by_ai_tier[tier] = []
            by_ai_tier[tier].append(p)
        
        ai_analysis = {}
        for tier, subset in by_ai_tier.items():
            summary = self.compute_summary(subset)
            ai_analysis[tier] = {
                'count': len(subset),
                'win_rate': summary.win_rate,
                'avg_pnl': summary.avg_pnl_pct,
            }
        
        return {
            'by_signal_type': by_signal_type,
            'by_market_state': by_market_state,
            'by_ai_confidence': ai_analysis,
        }
    
    def run_full_validation(
        self,
        test_dates: List[str],
        hold_days: int = None,
        max_stocks_per_date: int = 50,
    ) -> Tuple[ValidationSummary, Dict, pd.DataFrame]:
        """
        运行完整的反向验证流程
        
        Args:
            test_dates: 测试日期列表
            hold_days: 持仓天数
            max_stocks_per_date: 每个日期最多测试的股票数
            
        Returns:
            (摘要统计，分类分析，详细数据 DataFrame)
        """
        logger.info(f"开始反向验证：{len(test_dates)} 个日期")
        
        all_performances = []
        
        for i, test_date in enumerate(test_dates, 1):
            logger.info(f"[{i}/{len(test_dates)}] 测试日期：{test_date}")
            
            # 1. 扫描买点
            signals = self.scan_historical_date(
                target_date=test_date,
                max_stocks=max_stocks_per_date,
            )
            
            if not signals:
                logger.warning(f"日期 {test_date} 未发现任何买点信号")
                continue
            
            # 2. 模拟交易
            perfs = self.simulate_trades(signals, hold_days=hold_days)
            all_performances.extend(perfs)
            
            logger.info(f"  -> 发现 {len(signals)} 个信号，模拟 {len(perfs)} 笔交易")
        
        if not all_performances:
            logger.warning("没有有效的交易结果")
            empty_summary = self.compute_summary([])
            return empty_summary, {}, pd.DataFrame()
        
        # 3. 计算统计
        summary = self.compute_summary(all_performances)
        by_category = self.analyze_by_category(all_performances)
        
        # 4. 创建 DataFrame
        df = pd.DataFrame([
            {
                'code': p.code,
                'name': p.name,
                'buy_date': p.buy_date,
                'sell_date': p.sell_date,
                'buy_price': p.buy_price,
                'sell_price': p.sell_price,
                'hold_days': p.hold_days,
                'pnl_pct': p.pnl_pct,
                'max_profit_pct': p.max_profit_pct,
                'max_drawdown_pct': p.max_drawdown_pct,
                'signal_type': p.signal_type,
                'signal_score': p.signal_score,
                'ai_confidence': p.ai_confidence,
                'market_state': p.market_state,
            }
            for p in all_performances
        ])
        
        return summary, by_category, df
    
    def export_report(
        self,
        summary: ValidationSummary,
        by_category: Dict,
        df: pd.DataFrame,
        output_path: str = None,
    ) -> str:
        """导出验证报告"""
        output_path = output_path or f"reverse_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("反向验证报告 - 选股策略能力评估\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试交易日数：{df['buy_date'].nunique() if not df.empty else 0}\n")
            f.write(f"总交易笔数：{summary.total_signals}\n\n")
            
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
            f.write("按信号类型分析\n")
            f.write("-" * 80 + "\n")
            for signal_type, stats in by_category.get('by_signal_type', {}).items():
                f.write(f"\n{signal_type}:\n")
                f.write(f"  交易数：{stats['count']}\n")
                f.write(f"  胜率：{stats['win_rate']:.2f}%\n")
                f.write(f"  平均收益：{stats['avg_pnl']:.2f}%\n")
                f.write(f"  夏普比率：{stats['sharpe']:.2f}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("按市场状态分析\n")
            f.write("-" * 80 + "\n")
            for state, stats in by_category.get('by_market_state', {}).items():
                f.write(f"\n{state}:\n")
                f.write(f"  交易数：{stats['count']}\n")
                f.write(f"  胜率：{stats['win_rate']:.2f}%\n")
                f.write(f"  平均收益：{stats['avg_pnl']:.2f}%\n")
                f.write(f"  夏普比率：{stats['sharpe']:.2f}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("按 AI 置信度分析\n")
            f.write("-" * 80 + "\n")
            for tier, stats in by_category.get('by_ai_confidence', {}).items():
                f.write(f"\n{tier}:\n")
                f.write(f"  交易数：{stats['count']}\n")
                f.write(f"  胜率：{stats['win_rate']:.2f}%\n")
                f.write(f"  平均收益：{stats['avg_pnl']:.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("详细交易数据已保存为 CSV 文件\n")
            f.write("=" * 80 + "\n")
        
        # 保存 CSV
        csv_path = output_path.replace('.txt', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"验证报告已保存至：{output_path}")
        logger.info(f"详细数据已保存至：{csv_path}")
        
        return output_path


def cmd_reverse_validate(args):
    """CLI 命令入口"""
    from quant.core.strategy_params import StrategyParams
    
    # 解析日期范围
    if args.dates:
        test_dates = [d.strip() for d in args.dates.split(',')]
    elif args.start_date and args.end_date:
        start = pd.to_datetime(args.start_date)
        end = pd.to_datetime(args.end_date)
        # 获取区间内的交易日
        idx_df = get_market_index()
        if idx_df is not None:
            valid_dates = idx_df.index[(idx_df.index >= start) & (idx_df.index <= end)]
            test_dates = [d.strftime('%Y-%m-%d') for d in valid_dates]
        else:
            # 简单生成日期范围
            test_dates = pd.date_range(start=start, end=end, freq='B').strftime('%Y-%m-%d').tolist()
    else:
        logger.error("请指定 --dates 或 --start-date 和 --end-date")
        return
    
    logger.info(f"将测试 {len(test_dates)} 个日期")
    
    # 创建验证器
    validator = ReverseValidator(
        data_dir=CONF.history_data.data_dir,
        default_hold_days=args.hold_days,
    )
    
    # 运行验证
    summary, by_category, df = validator.run_full_validation(
        test_dates=test_dates,
        hold_days=args.hold_days,
        max_stocks_per_date=args.max_stocks,
    )
    
    # 导出报告
    if args.output:
        validator.export_report(summary, by_category, df, args.output)
    else:
        validator.export_report(summary, by_category, df)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("反向验证结果摘要")
    print("=" * 80)
    print(f"总交易数：{summary.total_signals}")
    print(f"胜率：{summary.win_rate:.2f}%")
    print(f"平均收益：{summary.avg_pnl_pct:.2f}%")
    print(f"盈亏比：{summary.profit_factor:.2f}")
    print(f"夏普比率：{summary.sharpe_ratio:.2f}")
    print("=" * 80)
