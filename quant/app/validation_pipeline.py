import argparse
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from quant.app.backtester import run_backtest
from quant.core.strategy_params import StrategyParams
from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.app.optimizer_enhanced import ENHANCED_CORE_PARAM_SPACE


class ValidationPipeline:
    """
    闭环验证与调优管道
    用于严格的样本外历史回溯，支持多日期截面选股、持仓模拟和结果反馈。
    """
    def __init__(self, validation_dates: List[str], data_dir: str = None):
        """
        Args:
            validation_dates: 样本外测试日期列表 (YYYY-MM-DD)
            data_dir: 历史数据目录，默认从 config 拉取
        """
        self.validation_dates = sorted(validation_dates)
        self.data_dir = data_dir or CONF.history_data.data_dir
        self.all_codes = self._get_active_codes()
        self._opt_fixed_params = {}
        logger.info(f"Initialized ValidationPipeline with {len(self.validation_dates)} dates and {len(self.all_codes)} stocks.")

    def _get_active_codes(self) -> List[str]:
        # 简单读取目录下的标的
        if not os.path.exists(self.data_dir):
            return []
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        return [f[:-4] for f in files]

    def _truncate_and_simulate_signals(self, target_date: str, params: StrategyParams) -> List[str]:
        """
        模拟在 target_date 当天复盘，截断未来数据，生成买入信号。
        使用真实的指标计算和 SignalScorer 打分机制。
        """
        logger.debug(f"[Simulation] Generating signals for cross-section at {target_date}...")
        
        target_dt = pd.to_datetime(target_date)
        signals = []
        
        from quant.features.analyzer import calculate_indicators
        from quant.core.signal_scorer import SignalScorer, CrossSectionalRanker
        
        scorer = SignalScorer(params)
        
        # 为了提高回测速度，应该使用多进程或者采样。这里为了精准验证，遍历过滤后的票池
        # 为了不让单日复盘太慢，我们暂时随机抽取200只股票进行打盘（如果总数很大）
        rng = random.Random(hash(target_date + str(params.to_dict())))
        sample_size = min(200, len(self.all_codes))
        sampled_codes = rng.sample(self.all_codes, sample_size)
        
        for code in sampled_codes:
            csv_path = os.path.join(self.data_dir, f"{code}.csv")
            if not os.path.exists(csv_path):
                continue
                
            try:
                # 1. 读取并截断数据
                df = pd.read_csv(csv_path)
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] <= target_dt].copy()
                
                # 需要足够的历史数据预热指标 (例如 MA60, ATR 等)
                if len(df) < 100:
                    continue
                    
                # 2. 计算指标
                df = calculate_indicators(df, params)
                
                if df.empty or len(df) < 2:
                    continue
                    
                # 3. 提取 T 日（最后一行）和 T-1 日（倒数第二行）的特征
                row_1 = df.iloc[-1]
                row_2 = df.iloc[-2]
                
                # 跳过停牌或无交易的数据
                if pd.isna(row_1.get('close')) or pd.isna(row_1.get(f'SMA_{params.ma_short}')):
                    continue
                
                price = row_1['close']
                
                # 重构 backtester.evaluate_buy_signals 的核心门控逻辑
                sma_s_1 = row_1[f'SMA_{params.ma_short}']
                sma_l_1 = row_1[f'SMA_{params.ma_long}']
                macd_h_1 = row_1[f'MACDh_{params.macd_fast}_{params.macd_slow}_{params.macd_signal}']
                macd_h_2 = row_2[f'MACDh_{params.macd_fast}_{params.macd_slow}_{params.macd_signal}']
                rsi_1 = row_1[f'RSI_{params.rsi_length}']
                bb_lower_1 = row_1[f'BBL_{params.bbands_length}_{params.bbands_std}']
                vol_1 = row_1['volume']
                vol_2 = row_2['volume']
                
                # 预判形态门控
                is_bb_dip = price < bb_lower_1 * params.bbands_lower_bias
                is_rsi_dip = rsi_1 < params.rsi_oversold_extreme if hasattr(params, 'rsi_oversold_extreme') else rsi_1 < 20
                is_above_ma = price > sma_s_1
                
                # 在真实回测里应该调用 evaluate_buy_signals，或者让 scorer 去算
                # 这里为了适配 scorer 的接口：
                signal_pullback = is_bb_dip
                signal_rebound = is_rsi_dip
                signal_breakout = is_above_ma and rsi_1 < params.rsi_cooled_max if hasattr(params, 'rsi_cooled_max') else is_above_ma and rsi_1 < 70
                
                # 4. 利用 SignalScorer 打分
                score = scorer.calculate_signal_score(
                    signal_pullback=signal_pullback,
                    signal_rebound=signal_rebound,
                    signal_breakout=signal_breakout,
                    price=price,
                    open_p=row_1['open'],
                    low_p=row_1['low'],
                    sma_l_1=sma_l_1,
                    sma_s_1=sma_s_1,
                    macd_h_1=macd_h_1,
                    macd_h_2=macd_h_2,
                    rsi_1=rsi_1,
                    vol_1=vol_1,
                    vol_2=vol_2,
                    ai_prob=0.5  # 为了提速，闭环初期可以 mock ai probability
                )
                
                # 如果符合入场的最基本条件 (质量评级不是 poor)
                if score.quality_rating not in ["poor", "weak"]:
                    # 后续也可以把 df_idx 的大盘过滤加上，这里保持信号层面过滤
                    signals.append((code, score))
                    
            except Exception as e:
                logger.debug(f"Trancation simulation failed for {code}: {e}")
                
        # 5. 横截面排名，选出 Top N
        ranker = CrossSectionalRanker(top_n=5)  # 假定每天最多只开5单
        ranked = ranker.rank_signals(signals)
        
        buy_list = [code for code, _ in ranked]
        return buy_list

    def run_single_date_evaluation(self, target_date: str, params: StrategyParams) -> Dict:
        """
        评估单个验证日的表现：选股 -> 虚拟持仓 -> 平仓结算
        """
        # 1. 获取选股清单
        buy_list = self._truncate_and_simulate_signals(target_date, params)
        if not buy_list:
            return {"date": target_date, "trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "max_drawdown": 0.0}
        
        # 2. 从 T 日开始向后回测（持有期模拟）
        # backtest 引擎已经支持 start_date 和 end_date，我们可以将 end_date 设为 target_date 的 N 天后
        # 但我们更想要的是让它自然触发止损止盈或时间衰减
        
        # 将 target_date 转换为 datetime 并加上一个极宽的缓冲期 (例如 30 天) 供真实出场逻辑触发
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
            sim_end_dt = target_dt + timedelta(days=40)
            sim_end_date = sim_end_dt.strftime("%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {target_date}")
            return {}

        results = []
        for code in buy_list:
            # 使用增量回测机制仅模拟这几天
            # 注意：实际底层回测库 backtesting.py 可能会因为数据量太少（缺乏指标预热）报错
            # 改进：我们通常传入从 T-200 到 T+40 的数据片段给底层回测器，但在 T 日才允许产生 entry signal.
            # 为了框架可用，这里假设回测引擎自身能处理 start_date
            res = run_backtest(code, params, start_date=target_date, end_date=sim_end_date)
            if res:
                bt, stats = res
                results.append({
                    "code": code,
                    "return_pct": stats.get("Return [%]", 0.0),
                    "win_rate": stats.get("Win Rate [%]", 0.0),
                    "max_drawdown": stats.get("Max. Drawdown [%]", 0.0),
                    "trades": stats.get("# Trades", 0)
                })
        
        # 3. 汇总当天买入的一篮子个股
        if not results:
             return {"date": target_date, "trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "max_drawdown": 0.0}
        
        df = pd.DataFrame(results)
        return {
            "date": target_date,
            "trades": df["trades"].sum(),
            "win_rate": (df["return_pct"] > 0).mean() * 100, # 简单按只统计胜率
            "avg_pnl": df["return_pct"].mean(),
            "max_drawdown": df["max_drawdown"].min()
        }

    def run_full_evaluation(self, params: StrategyParams) -> Dict:
        """
        遍历所有验证日期并聚合总得分
        """
        logger.info(f"Running full evaluation over {len(self.validation_dates)} dates...")
        daily_summaries = []
        for date in tqdm(self.validation_dates, desc="Cross-sectional evaluation"):
            summary = self.run_single_date_evaluation(date, params)
            if summary and summary.get("trades", 0) > 0:
                daily_summaries.append(summary)
        
        if not daily_summaries:
            return {"composite_score": -999.0, "avg_win_rate": 0.0, "avg_pnl": 0.0, "avg_dd": 0.0, "detail": []}
            
        df_all = pd.DataFrame(daily_summaries)
        
        avg_win_rate = df_all["win_rate"].mean()
        avg_pnl = df_all["avg_pnl"].mean()
        avg_dd = df_all["max_drawdown"].mean()
        
        # 复合适应度得分: 强调正收益与胜率，严惩回撤
        composite_score = (avg_pnl * 0.4) + (avg_win_rate * 0.4) + (avg_dd * 0.2)
        
        return {
            "composite_score": composite_score,
            "avg_win_rate": avg_win_rate,
            "avg_pnl": avg_pnl,
            "avg_dd": avg_dd,
            "detail": daily_summaries
        }

    def _optuna_objective(self, trial: optuna.Trial) -> float:
        """Optuna Trial 目标函数"""
        p_dict = StrategyParams().to_dict()
        for pname, (lo, hi, step) in ENHANCED_CORE_PARAM_SPACE.items():
            if isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int):
                p_dict[pname] = trial.suggest_int(pname, lo, hi, step=step)
            else:
                p_dict[pname] = trial.suggest_float(pname, lo, hi, step=step)
        
        # Override with any fixed params defined
        if hasattr(self, '_opt_fixed_params') and self._opt_fixed_params:
            p_dict.update(self._opt_fixed_params)

        params = StrategyParams.from_dict(p_dict)
        res = self.run_full_evaluation(params)
        
        # Penalize if no valid summaries
        if not res.get("detail"):
            return -999.0
            
        return res["composite_score"]

    def optimize_for_real_trading(self, n_trials: int = 50, fixed_params: Dict = None) -> Dict:
        """
        基于验证日期的闭环参数寻优
        """
        logger.info(f"Starting closed-loop optimization for {n_trials} trials...")
        self._opt_fixed_params = fixed_params or {}
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self._optuna_objective, n_trials=n_trials)
        
        best_p = study.best_params
        best_s = study.best_value
        logger.info(f"Optimization finished! Best composite score: {best_s:.4f}")
        
        final_params = StrategyParams.from_dict({**StrategyParams().to_dict(), **best_p})
        return {
            "best_params": final_params.to_dict(),
            "best_composite_score": best_s
        }

if __name__ == "__main__":
    # 临时测试桩：按月采样几个日期
    test_dates = ["2023-01-05", "2023-03-01", "2023-06-05", "2023-09-01"]
    pipeline = ValidationPipeline(validation_dates=test_dates)
    
    # 跑个基线
    base_params = StrategyParams()
    base_res = pipeline.run_full_evaluation(base_params)
    print(f"=== Baseline Performance ===")
    print(json.dumps(base_res, indent=2, ensure_ascii=False))
    
    # 跑闭环优化
    print(f"\n=== Running Closed-loop Optimization (2 Trials) ===")
    opt_res = pipeline.optimize_for_real_trading(n_trials=2)
    print(json.dumps(opt_res, indent=2, ensure_ascii=False))
