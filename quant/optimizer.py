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

from quant.backtester import batch_backtest
from quant.config import CONF
from quant.logger import logger
from quant.strategy_params import PARAM_SPACE, StrategyParams

warnings.filterwarnings("ignore")


def compute_objective(df_results: pd.DataFrame, objective: str) -> float:
    if df_results is None or df_results.empty:
        return -999.0

    if objective == "sharpe_adj":
        sharpe = df_results["sharpe"].fillna(0.0)
        mean_sharpe = sharpe.mean()
        # Stability penalty: penalize high variance in Sharpe across stocks
        std_sharpe = sharpe.std() if len(sharpe) > 1 else 0.0
        stability_discount = max(0.2, 1.0 - (std_sharpe / (abs(mean_sharpe) + 1e-5)) * 0.2)
        
        mean_dd = df_results["max_drawdown"].mean()
        total = len(df_results)
        count_profitable = (df_results["return_pct"] > 0).sum()
        ratio = count_profitable / total if total > 0 else 0.0
        
        return float(mean_sharpe * stability_discount * (1 - abs(mean_dd) / 100) * np.sqrt(ratio))
    elif objective == "return":
        return float(df_results["return_pct"].mean())
    elif objective == "win_rate":
        return float(df_results["win_rate"].mean())
    else:
        return -999.0


def sample_stock_codes(n: int, seed: int | None = None) -> list[str]:
    data_dir = CONF.history_data.data_dir
    all_files = os.listdir(data_dir)
    codes: list[str] = []
    for f in all_files:
        if not f.endswith(".csv"):
            continue
        if f == "stock-list.csv":
            continue
        name = f[:-4]
        if name.startswith("sh.6") or name.startswith("sz.00") or name.startswith("sz.30"):
            codes.append(name)

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    n = min(n, len(codes))
    return rng.sample(codes, n)


def get_train_test_split_dates(codes: list[str], train_ratio: float) -> str | None:
    data_dir = CONF.history_data.data_dir
    if not codes:
        return None
    
    sample_file = os.path.join(data_dir, f"{codes[0]}.csv")
    if not os.path.exists(sample_file):
        return None
        
    df = pd.read_csv(sample_file)
    if "date" not in df.columns:
        return None
        
    dates = pd.to_datetime(df["date"]).sort_values().dropna()
    if len(dates) < 50:
        return None
        
    split_idx = int(len(dates) * train_ratio)
    split_date = dates.iloc[split_idx]
    if isinstance(split_date, pd.Timestamp):
        return split_date.strftime('%Y-%m-%d')
    return str(split_date)


def walk_forward_evaluate(
    params: StrategyParams,
    codes: list[str],
    n_splits: int,
    train_ratio: float,
    objective: str,
    train_end_date: str | None = None,
) -> tuple[float, float]:
    df_train = batch_backtest(codes, params, end_date=train_end_date)
    train_score = compute_objective(df_train, objective)
    
    df_test = batch_backtest(codes, params, start_date=train_end_date)
    test_score = compute_objective(df_test, objective)
    
    return train_score, test_score


import optuna

def objective_function(trial: optuna.Trial, codes_train: list[str], train_end_date: str) -> float:
    opt_cfg = CONF.optimizer
    objective = opt_cfg.objective
    n_splits = opt_cfg.walk_forward_splits
    train_ratio = opt_cfg.train_ratio

    # 1. Suggest parameters from PARAM_SPACE
    candidate: dict[str, float] = {}
    for pname, (lo, hi, step) in PARAM_SPACE.items():
        if isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int):
            candidate[pname] = trial.suggest_int(pname, lo, hi, step=step)
        else:
            candidate[pname] = trial.suggest_float(pname, lo, hi, step=step)

    # 2. Constraints and Normalizations
    w_t = candidate["weight_trend"]
    w_r = candidate["weight_reversion"]
    w_v = candidate["weight_volume"]
    w_sum = w_t + w_r + w_v
    if w_sum > 0:
        candidate["weight_trend"] = round(w_t / w_sum, 4)
        candidate["weight_reversion"] = round(w_r / w_sum, 4)
        candidate["weight_volume"] = round(w_v / w_sum, 4)

    if candidate["macd_fast"] >= candidate["macd_slow"]:
        candidate["macd_fast"] = candidate["macd_slow"] - int(PARAM_SPACE["macd_fast"][2])

    if candidate["ma_short"] >= candidate["ma_long"]:
        candidate["ma_short"] = candidate["ma_long"] - int(PARAM_SPACE["ma_short"][2])

    if candidate["rsi_oversold"] >= candidate["rsi_cooled_max"]:
        candidate["rsi_oversold"] = candidate["rsi_cooled_max"] - PARAM_SPACE["rsi_oversold"][2]

    # Map directly back and evaluate Train-only for objective search
    nb_params = StrategyParams.from_dict(candidate)
    
    df_train = batch_backtest(codes_train, nb_params, end_date=train_end_date)
    train_score = compute_objective(df_train, objective)
    # Store candidate dict to trial so we can reconstruct OOS easily later
    trial.set_user_attr("params_dict", nb_params.to_dict())
    
    return float(train_score)

def run_optimization(callback: Callable | None = None) -> dict:
    opt_cfg = CONF.optimizer
    # max_rounds is repurposed as n_trials for Optuna
    n_trials = opt_cfg.max_rounds * 10
    sample_count = opt_cfg.sample_count
    
    logger.info("=== Optuna 贝叶斯策略引擎启动 (Phase 3) ===")
    logger.info(f"目标函数: {opt_cfg.objective} | 最大试错: {n_trials} | 采样: {sample_count}")

    codes_r0 = sample_stock_codes(sample_count, seed=0)
    train_end_date = get_train_test_split_dates(codes_r0, opt_cfg.train_ratio)
    
    logger.info(f"训练(Train)/测试(OOS) 时间横切点: {train_end_date}")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Run bayesian optimization ONLY on Train sets
    study.optimize(
        lambda t: objective_function(t, codes_r0, train_end_date),
        n_trials=n_trials,
        n_jobs=1,  # Keep single threaded due to Backtesting internal DF overrides
        catch=(Exception,)
    )
    
    best_trial = study.best_trial
    best_train_score = best_trial.value
    best_params_dict = best_trial.user_attrs.get("params_dict", {})
    best_params = StrategyParams.from_dict(best_params_dict)

    logger.info(f"=== Optuna 调参完结 ===")
    logger.info(f"历史最佳样本内得分 (Train Score): {best_train_score:.6f}")
    
    # Now run strict One-Off Test/OOS Evaluation
    df_test = batch_backtest(codes_r0, best_params, start_date=train_end_date)
    best_test_score = compute_objective(df_test, opt_cfg.objective)
    logger.info(f"真实验本外盲测得分 (Test/OOS Score): {best_test_score:.6f}")

    results: dict[str, typing.Any] = {
        "best_params": best_params.to_dict(),
        "best_score": best_train_score,
        "test_score": best_test_score,
        "baseline_score": 0.0,
        "history": [], # Legacy mock to avoid breaking downstream
        "rounds_completed": n_trials,
    }

    save_results(results)
    return results


def save_results(results: dict, output_dir: str | None = None) -> str:
    base_dir = output_dir or CONF.optimizer.results_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, ts)
    os.makedirs(result_dir, exist_ok=True)

    params_path = os.path.join(result_dir, "best_params.yaml")
    sp = StrategyParams.from_dict(results["best_params"])
    sp.to_yaml(params_path)

    report_path = os.path.join(result_dir, "optimization_report.json")
    report = {
        "best_score": results["best_score"],
        "baseline_score": results["baseline_score"],
        "rounds_completed": results["rounds_completed"],
        "history": results["history"],
        "timestamp": ts,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"优化结果已保存: {result_dir}")
    return result_dir


def apply_best_params(results: dict) -> None:
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f) or {}

    bp = results["best_params"]

    cfg_data["analyzer"] = {
        "weights": {
            "trend": bp["weight_trend"],
            "reversion": bp["weight_reversion"],
            "volume": bp["weight_volume"],
        },
        "ma_short": bp["ma_short"],
        "ma_long": bp["ma_long"],
        "macd_fast": bp["macd_fast"],
        "macd_slow": bp["macd_slow"],
        "macd_signal": bp["macd_signal"],
        "rsi_length": bp["rsi_length"],
        "rsi_buy_threshold": bp["rsi_buy_threshold"],
        "rsi_sell_threshold": bp["rsi_sell_threshold"],
        "bbands_length": bp["bbands_length"],
        "bbands_std": bp["bbands_std"],
        "atr_length": bp["atr_length"],
        "atr_multiplier": bp["atr_multiplier"],
    }

    cfg_data["strategy"] = {
        "vol_up_ratio": bp["vol_up_ratio"],
        "rsi_cooled_max": bp["rsi_cooled_max"],
        "pullback_ma_tolerance": bp["pullback_ma_tolerance"],
        "negative_bias_pct": bp["negative_bias_pct"],
        "rsi_oversold": bp["rsi_oversold"],
        "trail_atr_mult": bp["trail_atr_mult"],
        "take_profit_pct": bp["take_profit_pct"],
        "breakeven_trigger": bp["breakeven_trigger"],
        "breakeven_buffer": bp["breakeven_buffer"],
        "w_pullback_ma": bp["w_pullback_ma"],
        "w_macd_cross": bp["w_macd_cross"],
        "w_vol_up": bp["w_vol_up"],
        "w_rsi_rebound": bp["w_rsi_rebound"],
        "w_green_candle": bp["w_green_candle"],
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"最优参数已写回 {config_path}")
