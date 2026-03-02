from __future__ import annotations

import json
import os
import random
import typing
import warnings
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import yaml
import optuna
import multiprocessing

from quant.backtester import batch_backtest
from quant.config import CONF
from quant.logger import logger
from quant.strategy_params import CORE_PARAM_SPACE, StrategyParams

warnings.filterwarnings("ignore")


class EarlyStoppingCallback:
    """
    Optuna 早停回调
    当连续 N 次试验没有显著提升时，提前终止优化
    """

    def __init__(self, patience: int = 30, min_delta: float = 0.001, check_interval: int = 10):
        self.patience = patience
        self.min_delta = min_delta
        self.check_interval = check_interval
        self.best_score = None
        self.best_trial = None
        self.no_improve_count = 0
        self.is_best_improved = False

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        """
        每次试验后调用

        Args:
            study: Optuna 学习对象
            trial: 当前试验对象
        """
        current_score = study.best_value

        # 首次试验，初始化
        if self.best_score is None:
            self.best_score = current_score
            self.best_trial = trial.number
            return

        # 检查是否有显著提升
        is_improved = current_score - self.best_score > self.min_delta

        if is_improved:
            self.best_score = current_score
            self.best_trial = trial.number
            self.no_improve_count = 0
            self.is_best_improved = True
            logger.info(
                f"[EarlyStop] Best score improved: {self.best_score:.6f} "
                f"(Trial #{trial.number})"
            )
        else:
            self.no_improve_count += 1
            self.is_best_improved = False

        # 每 check_interval 次检查是否需要早停
        if trial.number % self.check_interval == 0:
            logger.info(
                f"[EarlyStop] Progress: Trial #{trial.number} | Best Score: {self.best_score:.6f} | "
                f"No Improvement: {self.no_improve_count}/{self.patience}"
            )

        # 检查是否达到早停条件
        if self.no_improve_count >= self.patience:
            logger.info(
                f"[EarlyStop] Early stopping triggered: No significant improvement for {self.patience} trials. "
                f"Best score: {self.best_score:.6f} (Trial #{self.best_trial})"
            )
            logger.info(f"[EarlyStop] Total trials: {trial.number + 1}")
            study.stop()


def compute_objective(df_results: pd.DataFrame, objective: str, params: StrategyParams | None = None) -> float:
    """
    Compute the objective score for a given set of backtest results.
    Includes regularization to prevent parameters from drifting too far into extreme zones.
    """
    if df_results is None or df_results.empty:
        return 0.0  # Neutral score when AI filters all signals (not catastrophic)

    # Base score selection
    if objective == "sharpe_pure":
        # Pure arithmetic mean of Sharpe Ratios across stocks
        raw_score = float(df_results["sharpe"].fillna(0.0).mean())
    elif objective == "sharpe_adj":
        # Legacy weighted sharpe including stability and drawdown
        sharpe = df_results["sharpe"].fillna(0.0)
        mean_sharpe = sharpe.mean()
        std_sharpe = sharpe.std() if len(sharpe) > 1 else 0.0
        stability_discount = max(0.2, 1.0 - (std_sharpe / (abs(mean_sharpe) + 1e-5)) * 0.2)
        mean_dd = df_results["max_drawdown"].mean()
        count_profitable = (df_results["return_pct"] > 0).sum()
        ratio = count_profitable / len(df_results) if len(df_results) > 0 else 0.0
        raw_score = float(mean_sharpe * stability_discount * (1 - abs(mean_dd) / 100) * np.sqrt(ratio))
    elif objective == "return":
        raw_score = float(df_results["return_pct"].mean())
    else:
        raw_score = -999.0

    # Add L2 Regularization Penalty (Anti-Overfitting)
    # Penalize params that are too far from their space center
    penalty = 0.0
    if params is not None and getattr(CONF.optimizer, "regularization_strength", 0.1) > 0:
        reg_strength = CONF.optimizer.regularization_strength
        p_dict = params.to_dict()
        for name, (lo, hi, _) in CORE_PARAM_SPACE.items():
            if name in p_dict:
                center = (lo + hi) / 2
                scale = (hi - lo) / 2
                normalized_val = (p_dict[name] - center) / (scale + 1e-8)
                penalty += (normalized_val ** 2)
        raw_score -= reg_strength * penalty

    return raw_score


def walk_forward_cv(
    codes: list[str], 
    params: StrategyParams, 
    n_folds: int = 3, 
    train_ratio: float = 0.7,
    objective_name: str = "sharpe_pure"
) -> float:
    """
    Perform multi-fold Walk-Forward Validation.
    Splits the timeline into multiple overlapping or sequential windows.
    Returns the average Test Score across all folds.
    """
    # 1. Get dates from a representative stock
    data_dir = CONF.history_data.data_dir
    sample_file = os.path.join(data_dir, f"{codes[0]}.csv")
    if not os.path.exists(sample_file):
        return -999.0
    
    df_dates = pd.read_csv(sample_file, usecols=["date"])
    dates = pd.to_datetime(df_dates["date"]).sort_values().unique()
    if len(dates) < 100:
        return -999.0

    fold_test_scores = []
    valid_folds = 0
    total_len = len(dates)
    
    # Simple rolling window walk-forward
    # Fold 1: [0% -> 40%] train, [40% -> 60%] test
    # Fold 2: [0% -> 60%] train, [60% -> 80%] test
    # Fold 3: [0% -> 80%] train, [80% -> 100%] test
    for i in range(n_folds):
        pivot_ratio = 0.4 + (i * 0.2) # 0.4, 0.6, 0.8
        test_end_ratio = pivot_ratio + 0.2 # 0.6, 0.8, 1.0
        
        train_end_date = dates[int(total_len * pivot_ratio)].strftime('%Y-%m-%d')
        test_end_date = dates[int(total_len * test_end_ratio) - 1].strftime('%Y-%m-%d')
        
        # We optimize based on TEST score of historical windows to find robust params
        df_fold_test = batch_backtest(
            codes, params, 
            start_date=train_end_date, 
            end_date=test_end_date
        )
        # Pass params=None to avoid per-fold regularization (applied once in objective_function)
        score = compute_objective(df_fold_test, objective_name, params=None)
        
        if score != 0.0:  # Only count folds with actual trades
            fold_test_scores.append(score)
            valid_folds += 1
        
    if valid_folds == 0:
        return 0.0  # No folds had any trades
    return float(np.mean(fold_test_scores))


def objective_function(trial: optuna.Trial, codes: list[str]) -> float:
    opt_cfg = CONF.optimizer
    
    # 1. Suggest parameters ONLY from CORE_PARAM_SPACE to reduce dimensionality
    candidate_dict = StrategyParams().to_dict() # Start with defaults
    for pname, (lo, hi, step) in CORE_PARAM_SPACE.items():
        if isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int):
            candidate_dict[pname] = trial.suggest_int(pname, lo, hi, step=step)
        else:
            candidate_dict[pname] = trial.suggest_float(pname, lo, hi, step=step)

    # 2. Map to StrategyParams object
    nb_params = StrategyParams.from_dict(candidate_dict)

    # 3. Two-Stage Evaluation (Hyperband Pruning)
    # Stage 1: Fast evaluation with 1 fold
    logger.debug(f"[Hyperband] Trial #{trial.number}: Stage 1 - 1 fold quick evaluation")
    score_stage1 = walk_forward_cv(
        codes, nb_params,
        n_folds=1,
        objective_name=opt_cfg.objective
    )

    # Report intermediate result to Hyperband pruner
    trial.report(score_stage1, step=1)

    # Check if trial should be pruned
    if trial.should_prune():
        logger.debug(f"[Hyperband] Trial #{trial.number}: Pruned at stage 1 (score: {score_stage1:.6f})")
        raise optuna.TrialPruned()

    # Stage 2: Full evaluation with 3 folds (only if not pruned)
    n_folds = getattr(opt_cfg, "walk_forward_folds", 3)
    logger.debug(f"[Hyperband] Trial #{trial.number}: Stage 2 - {n_folds} folds full evaluation")
    final_score = walk_forward_cv(
        codes, nb_params,
        n_folds=n_folds,
        objective_name=opt_cfg.objective
    )
    
    # Apply L2 regularization penalty ONCE (not per-fold)
    if getattr(CONF.optimizer, "regularization_strength", 0.1) > 0:
        reg_strength = CONF.optimizer.regularization_strength
        p_dict = nb_params.to_dict()
        penalty = 0.0
        for name, (lo, hi, _) in CORE_PARAM_SPACE.items():
            if name in p_dict:
                center = (lo + hi) / 2
                scale = (hi - lo) / 2
                normalized_val = (p_dict[name] - center) / (scale + 1e-8)
                penalty += (normalized_val ** 2)
        final_score -= reg_strength * penalty

    # Store for later retrieval
    trial.set_user_attr("params_dict", nb_params.to_dict())
    return float(final_score)


def get_all_dates(codes: list[str]) -> pd.DatetimeIndex:
    data_dir = CONF.history_data.data_dir
    sample_file = os.path.join(data_dir, f"{codes[0]}.csv")
    df = pd.read_csv(sample_file)
    return pd.to_datetime(df["date"]).sort_values().unique()


def run_optimization(callback: Callable | None = None) -> dict:
    opt_cfg = CONF.optimizer
    # Increase trials count for 8D space
    n_trials = opt_cfg.max_rounds * 40 
    sample_count = opt_cfg.sample_count
    
    logger.info("=== Optuna 抗过拟合策略引擎启动 (Phase 4) ===")
    logger.info(f"核心维度: {len(CORE_PARAM_SPACE)} | 目标: {opt_cfg.objective} | 试验总数: {n_trials}")

    # 1. Sample stocks
    codes = sample_stock_codes(sample_count, seed=42) # Fixed seed for consistency

    # 1.5 Configure Parallelization
    cpu_cores = multiprocessing.cpu_count()
    n_parallel_jobs = min(cpu_cores - 1, 12)  # 保留一个核心，最多12个并行（保守进阶）
    logger.info(f"🚀 启用并行优化: {n_parallel_jobs} 个并行进程 (CPU核心数: {cpu_cores})")

    # 2. Configuration
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,
            multivariate=True,
            seed=42
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=3,
            reduction_factor=3
        )
    )

    # 2.5 Configure Early Stopping
    early_stop_callback = EarlyStoppingCallback(
        patience=30,
        min_delta=0.001,
        check_interval=10
    )
    logger.info(f"[EarlyStop] patience={early_stop_callback.patience}")

    # 3. Run Optimization (with parallelization)
    study.optimize(
        lambda t: objective_function(t, codes),
        n_trials=n_trials,
        n_jobs=n_parallel_jobs,  # 启用并行化
        catch=(Exception,),
        callbacks=[early_stop_callback]
    )
    
    # 4. Extract best trial safely
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        logger.error("所有试验均失败或被剪枝，无法获取最优参数。")
        return {"best_params": StrategyParams().to_dict(), "best_score": 0.0, "test_score": 0.0,
                "baseline_score": 0.0, "rounds_completed": n_trials, "history": []}

    best_trial = study.best_trial
    logger.info(f"=== 优化结束 | 最佳阶段得分: {best_trial.value:.6f} ===")


    # 4. Final Final Evaluation on OOS (Out of Sample)
    best_params = StrategyParams.from_dict(best_trial.user_attrs["params_dict"])
    dates = get_all_dates(codes)
    oos_start_date = dates[int(len(dates) * opt_cfg.train_ratio)].strftime('%Y-%m-%d')
    
    logger.info(f"正在进行最终 OOS 盲测 (从 {oos_start_date} 开始)...")
    df_oos = batch_backtest(codes, best_params, start_date=oos_start_date)
    oos_score = compute_objective(df_oos, opt_cfg.objective)
    
    logger.info(f"最终样本外盲测得分: {oos_score:.6f}")

    results = {
        "best_params": best_params.to_dict(),
        "best_score": best_trial.value,
        "test_score": oos_score,
        "baseline_score": 0.0,
        "rounds_completed": n_trials,
        "history": []
    }

    save_results(results)
    return results


def sample_stock_codes(n: int, seed: int | None = None) -> list[str]:
    data_dir = CONF.history_data.data_dir
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and f != "stock-list.csv"]
    codes = [f[:-4] for f in all_files if f.startswith("sh.6") or f.startswith("sz.00") or f.startswith("sz.30")]
    
    rng = random.Random(seed)
    return rng.sample(codes, min(n, len(codes)))


def save_results(results: dict, output_dir: str | None = None) -> str:
    base_dir = output_dir or CONF.optimizer.results_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, ts)
    os.makedirs(result_dir, exist_ok=True)

    params_path = os.path.join(result_dir, "best_params.yaml")
    sp = StrategyParams.from_dict(results["best_params"])
    sp.to_yaml(params_path)

    report_path = os.path.join(result_dir, "optimization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"优化结果已保存: {result_dir}")
    return result_dir


def apply_best_params(results: dict) -> None:
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f) or {}

    bp = results["best_params"]
    
    # Update analyzer weights and strategy params
    if "analyzer" not in cfg_data: cfg_data["analyzer"] = {}
    if "strategy" not in cfg_data: cfg_data["strategy"] = {}
    
    # Only update what we optimized (from CORE_PARAM_SPACE keys)
    for key in bp:
        if key in CORE_PARAM_SPACE:
            # Route to either analyzer or strategy based on where it lives in config.yaml
            if key.startswith("w_") or key in ["vol_up_ratio", "rsi_cooled_max", "pullback_ma_tolerance", "trail_atr_mult", "take_profit_pct", "rsi_oversold"]:
                # Most go to strategy in this project's config structure
                cfg_data["strategy"][key] = bp[key]

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"最优参数已写回 {config_path}")


if __name__ == "__main__":
    """Main entry point for running optimization."""
    try:
        run_optimization()
    except KeyboardInterrupt:
        logger.info("\n[STOP] Optimization interrupted by user.")
    except Exception as e:
        logger.error(f"[ERROR] Optimization failed: {e}")
        raise

