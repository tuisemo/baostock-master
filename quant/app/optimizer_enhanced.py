"""
增强版参数优化模块
扩展参数空间至 15 维，引入 CMA-ES 优化算法，实现多目标优化
"""
import json
import os
import random
import warnings
from datetime import datetime
from typing import Callable, List, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
import optuna
import multiprocessing

from quant.app.backtester import batch_backtest
from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.core.strategy_params import StrategyParams, CORE_PARAM_SPACE

warnings.filterwarnings("ignore")


# ===== 扩展的参数空间（15 维核心参数）=====
ENHANCED_CORE_PARAM_SPACE: Dict[str, Tuple[float, float, float]] = {
    # === 入场条件（6 维）===
    "vol_up_ratio": (1.1, 1.8, 0.05),
    "rsi_cooled_max": (45, 65, 5),
    "pullback_ma_tolerance": (1.00, 1.05, 0.005),
    "negative_bias_pct": (0.80, 0.98, 0.01),
    "rsi_oversold": (25, 45, 5),
    "bbands_lower_bias": (0.90, 1.10, 0.01),
    
    # === 出场条件（4 维）===
    "trail_atr_mult": (1.0, 2.5, 0.1),
    "take_profit_pct": (0.03, 0.12, 0.01),
    "breakeven_trigger": (0.02, 0.06, 0.005),
    "max_hold_days": (5, 25, 2),
    
    # === 信号权重（3 维）===
    "w_pullback_ma": (0.5, 5.0, 0.5),
    "w_rsi_rebound": (0.5, 5.0, 0.5),
    "w_vol_up": (0.5, 3.0, 0.5),
    
    # === AI 门控（1 维）===
    "ai_prob_threshold": (0.1, 0.5, 0.05),

    # === EV 门控（1 维）===
    # 单位: %。0 表示至少非负期望值，越高越保守、信号更少但更“硬”。
    "min_expected_value_pct": (0.0, 2.0, 0.25),
    
    # === 仓位管理（1 维）===
    "position_size": (0.05, 0.20, 0.01),
}


class EnhancedOptimizer:
    """
    增强版参数优化器
    支持多种优化算法（TPE、CMA-ES、Random）
    支持多目标优化
    """
    
    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float, float]] = None,
        algorithm: str = "tpe",
        n_trials: int = 200,
        n_jobs: int = None,
        multi_objective: bool = False
    ):
        """
        初始化增强版优化器
        
        Args:
            param_space: 参数空间，默认使用 ENHANCED_CORE_PARAM_SPACE
            algorithm: 优化算法，可选 'tpe', 'cmaes', 'random'
            n_trials: 试验次数
            n_jobs: 并行任务数
            multi_objective: 是否启用多目标优化
        """
        self.param_space = param_space or ENHANCED_CORE_PARAM_SPACE
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.n_jobs = n_jobs or (multiprocessing.cpu_count() - 1)
        self.multi_objective = multi_objective
        self.study = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        
        logger.info(f"初始化增强版优化器: 算法={algorithm}, 参数维度={len(self.param_space)}, 试验次数={n_trials}")
    
    def _get_sampler(self):
        """获取采样器"""
        if self.algorithm == "tpe":
            return optuna.samplers.TPESampler(
                n_startup_trials=min(20, self.n_trials // 10),
                multivariate=True,
                seed=42
            )
        elif self.algorithm == "cmaes":
            try:
                return optuna.samplers.CmaEsSampler(
                    n_startup_trials=min(20, self.n_trials // 10),
                    seed=42
                )
            except ImportError:
                logger.warning("CMA-ES 不可用，回退到 TPE")
                return optuna.samplers.TPESampler(seed=42)
        elif self.algorithm == "random":
            return optuna.samplers.RandomSampler(seed=42)
        else:
            raise ValueError(f"未知算法: {self.algorithm}")
    
    def _get_pruner(self):
        """获取剪枝器"""
        return optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=3,
            reduction_factor=3
        )
    
    def _suggest_params(self, trial: optuna.Trial) -> StrategyParams:
        """从试验中建议参数"""
        candidate_dict = StrategyParams().to_dict()
        
        for pname, (lo, hi, step) in self.param_space.items():
            if isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int):
                candidate_dict[pname] = trial.suggest_int(pname, lo, hi, step=step)
            else:
                candidate_dict[pname] = trial.suggest_float(pname, lo, hi, step=step)
        
        return StrategyParams.from_dict(candidate_dict)
    
    def _compute_objective(self, df_results: pd.DataFrame, params: StrategyParams) -> Dict[str, float]:
        """
        计算多目标得分
        
        Returns:
            包含多个目标得分的字典
        """
        if df_results is None or df_results.empty:
            return {
                "sharpe": 0.0,
                "return": 0.0,
                "stability": 0.0,
                "trade_count": 0.0
            }
        
        # 基础指标
        sharpe = df_results["sharpe"].fillna(0.0).mean()
        return_pct = df_results["return_pct"].mean()
        win_rate = (df_results["return_pct"] > 0).sum() / len(df_results)
        max_drawdown = abs(df_results["max_drawdown"].mean())
        num_trades = df_results["num_trades"].mean()
        
        # 稳定性指标（夏普比率的标准差）
        sharpe_std = df_results["sharpe"].std()
        stability = 1.0 / (1.0 + sharpe_std)
        
        # 综合得分
        composite_score = (
            0.4 * sharpe + 
            0.3 * (return_pct / 100.0) + 
            0.2 * win_rate + 
            0.1 * stability * (1.0 - max_drawdown / 100.0)
        )
        
        # L2 正则化惩罚
        penalty = 0.0
        reg_strength = CONF.optimizer.regularization_strength
        if reg_strength > 0:
            p_dict = params.to_dict()
            for name, (lo, hi, _) in self.param_space.items():
                if name in p_dict:
                    center = (lo + hi) / 2
                    scale = (hi - lo) / 2
                    normalized_val = (p_dict[name] - center) / (scale + 1e-8)
                    penalty += (normalized_val ** 2)
            composite_score -= reg_strength * penalty
        
        return {
            "sharpe": sharpe,
            "return": return_pct / 100.0,
            "stability": stability,
            "trade_count": min(1.0, num_trades / 100.0),  # 归一化交易次数
            "composite": composite_score
        }
    
    def _objective_function(self, trial: optuna.Trial, codes: List[str]) -> float:
        """目标函数"""
        # 1. 建议参数
        params = self._suggest_params(trial)
        
        # 2. 执行回测
        df_results = batch_backtest(codes, params)
        
        # 3. 计算得分
        scores = self._compute_objective(df_results, params)
        
        # 4. 记录历史
        self.optimization_history.append({
            "trial": trial.number,
            "params": params.to_dict(),
            "scores": scores
        })
        
        # 5. 报告中间结果（用于 Hyperband Pruning）
        trial.report(scores["composite"], step=1)
        
        # 6. 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return scores["composite"]
    
    def optimize(self, codes: List[str], timeout: int = None) -> Dict:
        """
        执行优化
        
        Args:
            codes: 股票代码列表
            timeout: 超时时间（秒）
        
        Returns:
            优化结果字典
        """
        logger.info(f"开始参数优化: 算法={self.algorithm}, 试验次数={self.n_trials}")
        
        # 创建研究
        directions = ["maximize"]  # 单目标优化
        
        if self.multi_objective:
            # 多目标优化（如果有需求可以扩展）
            directions = ["maximize"] * 4  # sharpe, return, stability, trade_count
        
        self.study = optuna.create_study(
            directions=directions,
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )
        
        # 添加早停回调
        early_stop_callback = EarlyStoppingCallback(
            patience=30,
            min_delta=0.001,
            check_interval=10
        )
        
        # 执行优化
        self.study.optimize(
            lambda t: self._objective_function(t, codes),
            n_trials=self.n_trials,
            n_jobs=1,  # 注意：并行优化可能与某些采样器不兼容
            timeout=timeout,
            callbacks=[early_stop_callback]
        )
        
        # 提取最佳结果
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            logger.error("所有试验均失败或被剪枝")
            return {
                "best_params": StrategyParams().to_dict(),
                "best_score": 0.0,
                "test_score": 0.0,
                "history": []
            }
        
        best_trial = self.study.best_trial
        self.best_params = StrategyParams.from_dict(best_trial.params)
        self.best_score = best_trial.value
        
        logger.info(f"优化完成: 最佳得分={self.best_score:.6f}, 试验次数={len(self.study.trials)}")
        
        # 最终评估
        final_scores = self._evaluate_final(codes, self.best_params)
        
        return {
            "best_params": self.best_params.to_dict(),
            "best_score": self.best_score,
            "test_score": final_scores["composite"],
            "final_scores": final_scores,
            "n_trials": len(self.study.trials),
            "history": self.optimization_history
        }
    
    def _evaluate_final(self, codes: List[str], params: StrategyParams) -> Dict[str, float]:
        """最终评估"""
        df_results = batch_backtest(codes, params)
        return self._compute_objective(df_results, params)
    
    def save_results(self, results: Dict, output_dir: str = None) -> str:
        """保存优化结果"""
        base_dir = output_dir or CONF.optimizer.results_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(base_dir, f"enhanced_{ts}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存最佳参数
        params_path = os.path.join(result_dir, "best_params.yaml")
        sp = StrategyParams.from_dict(results["best_params"])
        sp.to_yaml(params_path)
        
        # 保存完整结果
        report_path = os.path.join(result_dir, "optimization_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"优化结果已保存: {result_dir}")
        return result_dir


class EarlyStoppingCallback:
    """早停回调"""
    
    def __init__(self, patience: int = 30, min_delta: float = 0.001, check_interval: int = 10):
        self.patience = patience
        self.min_delta = min_delta
        self.check_interval = check_interval
        self.best_score = None
        self.best_trial = None
        self.no_improve_count = 0
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        current_score = study.best_value
        
        if self.best_score is None:
            self.best_score = current_score
            self.best_trial = trial.number
            return
        
        is_improved = current_score - self.best_score > self.min_delta
        
        if is_improved:
            self.best_score = current_score
            self.best_trial = trial.number
            self.no_improve_count = 0
            logger.info(f"[EarlyStop] 最佳得分提升: {self.best_score:.6f} (Trial #{trial.number})")
        else:
            self.no_improve_count += 1
        
        if trial.number % self.check_interval == 0:
            logger.info(
                f"[EarlyStop] 进度: Trial #{trial.number} | 最佳得分: {self.best_score:.6f} | "
                f"无提升次数: {self.no_improve_count}/{self.patience}"
            )
        
        if self.no_improve_count >= self.patience:
            logger.info(
                f"[EarlyStop] 早停触发: 连续 {self.patience} 次无显著提升. "
                f"最佳得分: {self.best_score:.6f} (Trial #{self.best_trial})"
            )
            study.stop()


def run_enhanced_optimization(
    algorithm: str = "tpe",
    n_trials: int = 200,
    multi_objective: bool = False
) -> Dict:
    """
    运行增强版参数优化
    
    Args:
        algorithm: 优化算法 ('tpe', 'cmaes', 'random')
        n_trials: 试验次数
        multi_objective: 是否启用多目标优化
    
    Returns:
        优化结果
    """
    logger.info("=== 增强版参数优化引擎启动 ===")
    logger.info(f"参数空间维度: {len(ENHANCED_CORE_PARAM_SPACE)}")
    logger.info(f"优化算法: {algorithm}")
    logger.info(f"试验次数: {n_trials}")
    
    # 采样股票
    codes = sample_stock_codes(CONF.optimizer.sample_count, seed=42)
    
    # 创建优化器
    optimizer = EnhancedOptimizer(
        param_space=ENHANCED_CORE_PARAM_SPACE,
        algorithm=algorithm,
        n_trials=n_trials,
        multi_objective=multi_objective
    )
    
    # 执行优化
    results = optimizer.optimize(codes)
    
    # 保存结果
    optimizer.save_results(results)
    
    # 应用最佳参数（可选）
    # apply_best_params(results)
    
    return results


def sample_stock_codes(n: int, seed: int = None) -> List[str]:
    """采样股票代码"""
    data_dir = CONF.history_data.data_dir
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    codes = [f[:-4] for f in all_files if f.startswith("sh.6") or f.startswith("sz.00") or f.startswith("sz.30")]
    
    rng = random.Random(seed)
    return rng.sample(codes, min(n, len(codes)))
