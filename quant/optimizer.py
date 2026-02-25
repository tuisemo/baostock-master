from __future__ import annotations

import json
import os
import random
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
        mean_dd = df_results["max_drawdown"].mean()
        total = len(df_results)
        count_profitable = (df_results["return_pct"] > 0).sum()
        ratio = count_profitable / total if total > 0 else 0.0
        return float(mean_sharpe * (1 - abs(mean_dd) / 100) * np.sqrt(ratio))
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


def walk_forward_evaluate(
    params: StrategyParams,
    codes: list[str],
    n_splits: int,
    train_ratio: float,
    objective: str,
) -> tuple[float, float]:
    df_results = batch_backtest(codes, params)
    score = compute_objective(df_results, objective)
    return score, score


def generate_neighbors(
    base: StrategyParams, param_space: dict, n_neighbors: int = 8
) -> list[StrategyParams]:
    neighbors: list[StrategyParams] = []
    base_dict = base.to_dict()

    param_names = list(param_space.keys())

    for _ in range(n_neighbors):
        candidate = dict(base_dict)
        n_perturb = random.randint(3, 5)
        chosen = random.sample(param_names, min(n_perturb, len(param_names)))

        for pname in chosen:
            lo, hi, step = param_space[pname]
            steps_delta = random.choice([-2, -1, 1, 2])
            new_val = candidate[pname] + steps_delta * step

            if isinstance(base_dict[pname], int):
                new_val = int(round(new_val))
            else:
                new_val = round(new_val, 6)

            new_val = max(lo, min(hi, new_val))
            candidate[pname] = new_val

        # Normalize weights to sum to ~1.0
        w_t = candidate["weight_trend"]
        w_r = candidate["weight_reversion"]
        w_v = candidate["weight_volume"]
        w_sum = w_t + w_r + w_v
        if w_sum > 0:
            candidate["weight_trend"] = round(w_t / w_sum, 4)
            candidate["weight_reversion"] = round(w_r / w_sum, 4)
            candidate["weight_volume"] = round(w_v / w_sum, 4)

        # Ensure macd_fast < macd_slow
        if candidate["macd_fast"] >= candidate["macd_slow"]:
            candidate["macd_fast"] = candidate["macd_slow"] - int(param_space["macd_fast"][2])

        # Ensure ma_short < ma_long
        if candidate["ma_short"] >= candidate["ma_long"]:
            candidate["ma_short"] = candidate["ma_long"] - int(param_space["ma_short"][2])

        # Ensure rsi_oversold < rsi_cooled_max
        if candidate["rsi_oversold"] >= candidate["rsi_cooled_max"]:
            candidate["rsi_oversold"] = candidate["rsi_cooled_max"] - param_space["rsi_oversold"][2]

        # Clamp once more after constraint fixes
        for pname in param_space:
            lo, hi, _ = param_space[pname]
            candidate[pname] = max(lo, min(hi, candidate[pname]))

        neighbors.append(StrategyParams.from_dict(candidate))

    return neighbors


def run_optimization(callback: Callable | None = None) -> dict:
    opt_cfg = CONF.optimizer
    max_rounds = opt_cfg.max_rounds
    sample_count = opt_cfg.sample_count
    objective = opt_cfg.objective
    n_splits = opt_cfg.walk_forward_splits
    train_ratio = opt_cfg.train_ratio

    baseline_params = StrategyParams.from_app_config(CONF)
    history: list[dict] = []

    logger.info("=== 策略优化引擎启动 ===")
    logger.info(f"目标函数: {objective} | 最大轮数: {max_rounds} | 每轮采样: {sample_count}")

    # Round 0: evaluate baseline
    logger.info("[Round 0] 评估基线策略...")
    codes_r0 = sample_stock_codes(sample_count, seed=0)
    baseline_score, _ = walk_forward_evaluate(
        baseline_params, codes_r0, n_splits, train_ratio, objective
    )
    logger.info(f"[Round 0] 基线得分: {baseline_score:.6f}")

    history.append({
        "round": 0,
        "score": baseline_score,
        "params": baseline_params.to_dict(),
        "improved": False,
    })

    current_best_params = baseline_params
    current_best_score = baseline_score
    stale_rounds = 0

    if callback is not None:
        callback(0, max_rounds, current_best_score, current_best_params.to_dict(), history)

    for rnd in range(1, max_rounds + 1):
        logger.info(f"[Round {rnd}/{max_rounds}] 生成邻域参数并评估...")

        codes_rnd = sample_stock_codes(sample_count, seed=rnd * 42)
        neighbors = generate_neighbors(current_best_params, PARAM_SPACE, n_neighbors=8)

        best_neighbor_score = -999.0
        best_neighbor_params: StrategyParams | None = None

        for i, nb_params in enumerate(neighbors):
            logger.info(f"  邻域 {i + 1}/{len(neighbors)} 评估中...")
            nb_score, _ = walk_forward_evaluate(
                nb_params, codes_rnd, n_splits, train_ratio, objective
            )
            logger.info(f"  邻域 {i + 1} 得分: {nb_score:.6f}")

            if nb_score > best_neighbor_score:
                best_neighbor_score = nb_score
                best_neighbor_params = nb_params

        improved = False
        if best_neighbor_params is not None and best_neighbor_score > current_best_score:
            improvement = best_neighbor_score - current_best_score
            current_best_params = best_neighbor_params
            current_best_score = best_neighbor_score
            improved = True
            logger.info(
                f"[Round {rnd}] ✅ 改进! 新最优: {current_best_score:.6f} (+{improvement:.6f})"
            )

            if improvement < opt_cfg.convergence_threshold:
                stale_rounds += 1
            else:
                stale_rounds = 0
        else:
            stale_rounds += 1
            logger.info(
                f"[Round {rnd}] ❌ 未改进, 当前最优保持: {current_best_score:.6f}"
            )

        history.append({
            "round": rnd,
            "score": current_best_score,
            "best_neighbor_score": best_neighbor_score,
            "params": current_best_params.to_dict(),
            "improved": improved,
        })

        if callback is not None:
            callback(rnd, max_rounds, current_best_score, current_best_params.to_dict(), history)

        if stale_rounds >= 2:
            logger.info(f"[Round {rnd}] 连续 {stale_rounds} 轮无显著改进, 提前收敛停止")
            break

    logger.info(f"=== 优化完成 === 最终得分: {current_best_score:.6f}")

    results = {
        "best_params": current_best_params.to_dict(),
        "best_score": current_best_score,
        "baseline_score": baseline_score,
        "history": history,
        "rounds_completed": len(history) - 1,
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
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"最优参数已写回 {config_path}")
