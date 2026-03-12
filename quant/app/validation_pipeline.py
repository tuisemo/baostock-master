import hashlib
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

from quant.app.backtester import scan_today_signal
from quant.app.optimizer_enhanced import ENHANCED_CORE_PARAM_SPACE
from quant.core.strategy_params import StrategyParams
from quant.infra.config import CONF
from quant.infra.logger import logger


def _stable_int_seed(text: str) -> int:
    # Python 内置 hash() 会被随机化，不能用于可复现实验
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


class ValidationPipeline:
    """
    闭环验证与调优管道

    设计原则:
    1) 选股信号必须复用 scan_today_signal，避免“训练/验证/实盘”三套逻辑漂移。
    2) 验证侧只评估“在 target_date 产生的买点”，不允许目标日期之后再入场，避免偷看未来。
    3) 通过简化前瞻模拟(止损/止盈 + 持有期)评估收益与回撤，并供 Optuna 做参数搜索。
    """

    def __init__(
        self,
        validation_dates: List[str],
        data_dir: str | None = None,
        sample_size: int = 200,
        max_trades_per_day: int = 10,
    ):
        self.validation_dates = sorted(validation_dates)
        self.data_dir = data_dir or CONF.history_data.data_dir
        self.sample_size = int(sample_size)
        self.max_trades_per_day = int(max_trades_per_day)

        self.all_codes = self._get_active_codes()
        self._opt_fixed_params: Dict = {}
        self._sample_cache: Dict[str, List[str]] = {}

        logger.info(
            f"Initialized ValidationPipeline with {len(self.validation_dates)} dates and "
            f"{len(self.all_codes)} codes. sample_size={self.sample_size}, "
            f"max_trades_per_day={self.max_trades_per_day}"
        )

    def _get_active_codes(self) -> List[str]:
        if not os.path.exists(self.data_dir):
            return []

        codes: List[str] = []
        for f in os.listdir(self.data_dir):
            if not f.endswith(".csv"):
                continue
            if f in {"stock-list.csv", "sh.000001.csv"}:
                continue
            codes.append(f[:-4])
        return sorted(codes)

    def _sample_codes_for_date(self, target_date: str) -> List[str]:
        cached = self._sample_cache.get(target_date)
        if cached is not None:
            return cached

        if not self.all_codes:
            self._sample_cache[target_date] = []
            return []

        k = min(self.sample_size, len(self.all_codes))
        rng = random.Random(_stable_int_seed(f"sample::{target_date}"))
        sampled = rng.sample(self.all_codes, k)
        self._sample_cache[target_date] = sampled
        return sampled

    def _scan_cross_section(self, target_date: str, params: StrategyParams) -> List[dict]:
        """
        在 target_date 截面上运行真实选股逻辑(scan_today_signal)，返回按 buy_score 排序的信号列表。
        """
        codes = self._sample_codes_for_date(target_date)
        if not codes:
            return []

        signals: List[dict] = []
        for code in codes:
            try:
                sig = scan_today_signal(code, params=params, target_date=target_date)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"[Validation] scan failed: {code} @ {target_date}: {e}")
                continue

        if not signals:
            return []

        signals.sort(key=lambda d: float(d.get("buy_score", d.get("total_score", 0.0))), reverse=True)
        return signals[: self.max_trades_per_day]

    def _simulate_forward_trade(
        self,
        code: str,
        target_date: str,
        entry_atr: float,
        params: StrategyParams,
    ) -> Optional[Dict]:
        """
        对“在 target_date 收盘买入”的交易做简化前瞻模拟:
        - 止损/止盈: ai_stop_loss_atr_mult / ai_target_atr_mult
        - 持有期: min(max_hold_days, ai_forward_days)
        - 同一天同时触及止损/止盈: 悲观假设先触发止损
        """
        csv_path = os.path.join(self.data_dir, f"{code}.csv")
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path)
        if df.empty or "date" not in df.columns:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        target_dt = pd.to_datetime(target_date)
        hist = df[df["date"] <= target_dt]
        if hist.empty:
            return None

        entry_pos = int(hist.index[-1])
        entry_row = df.iloc[entry_pos]

        entry_price = float(entry_row.get("close", np.nan))
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None

        atr = float(entry_atr) if entry_atr is not None else np.nan
        if not np.isfinite(atr) or atr <= 0:
            return None

        hold_days = int(getattr(params, "ai_forward_days", 5))
        max_hold_days = int(getattr(params, "max_hold_days", hold_days))
        hold_days = max(1, min(hold_days, max_hold_days))

        stop_px = entry_price - float(params.ai_stop_loss_atr_mult) * atr
        target_px = entry_price + float(params.ai_target_atr_mult) * atr

        worst_low = entry_price

        exit_price = entry_price
        exit_reason = "timeout"
        exit_dt = entry_row["date"]

        last_pos = min(len(df) - 1, entry_pos + hold_days)

        for pos in range(entry_pos + 1, last_pos + 1):
            r = df.iloc[pos]
            day_low = float(r.get("low", np.nan))
            day_high = float(r.get("high", np.nan))

            if np.isfinite(day_low):
                worst_low = min(worst_low, day_low)

            if np.isfinite(day_low) and day_low <= stop_px:
                exit_price = stop_px
                exit_reason = "stop"
                exit_dt = r["date"]
                break

            if np.isfinite(day_high) and day_high >= target_px:
                exit_price = target_px
                exit_reason = "target"
                exit_dt = r["date"]
                break

        if exit_reason == "timeout":
            r = df.iloc[last_pos]
            exit_dt = r["date"]
            exit_price = float(r.get("close", np.nan))
            if not np.isfinite(exit_price) or exit_price <= 0:
                return None

        gross_ret_pct = (exit_price - entry_price) / entry_price * 100.0

        commission = float(getattr(params, "commission_pct", 0.0))
        slippage = float(getattr(params, "slippage_pct", 0.0))
        cost_pct = 2.0 * (commission + slippage) * 100.0
        net_ret_pct = gross_ret_pct - cost_pct

        max_drawdown_pct = (worst_low - entry_price) / entry_price * 100.0

        return {
            "code": code,
            "entry_date": entry_row["date"].strftime("%Y-%m-%d"),
            "exit_date": exit_dt.strftime("%Y-%m-%d"),
            "exit_reason": exit_reason,
            "return_pct": float(net_ret_pct),
            "max_drawdown": float(max_drawdown_pct),
        }

    def run_single_date_evaluation(self, target_date: str, params: StrategyParams) -> Dict:
        selected_signals = self._scan_cross_section(target_date, params)
        if not selected_signals:
            return {"date": target_date, "trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "max_drawdown": 0.0}

        results: List[Dict] = []
        for sig in selected_signals:
            code = str(sig.get("code"))
            atr = sig.get("atr")
            try:
                out = self._simulate_forward_trade(code, target_date, atr, params)
            except Exception as e:
                logger.debug(f"[Validation] simulate failed: {code} @ {target_date}: {e}")
                out = None

            if out:
                results.append(out)

        if not results:
            return {"date": target_date, "trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "max_drawdown": 0.0}

        df = pd.DataFrame(results)
        return {
            "date": target_date,
            "trades": int(len(df)),
            "win_rate": float((df["return_pct"] > 0).mean() * 100.0),
            "avg_pnl": float(df["return_pct"].mean()),
            "max_drawdown": float(df["max_drawdown"].min()),
        }

    def run_full_evaluation(self, params: StrategyParams) -> Dict:
        logger.info(f"Running full evaluation over {len(self.validation_dates)} dates...")
        daily_summaries: List[Dict] = []

        for date in tqdm(self.validation_dates, desc="Cross-sectional evaluation"):
            summary = self.run_single_date_evaluation(date, params)
            if summary and summary.get("trades", 0) > 0:
                daily_summaries.append(summary)

        if not daily_summaries:
            return {"composite_score": -999.0, "avg_win_rate": 0.0, "avg_pnl": 0.0, "avg_dd": 0.0, "detail": []}

        df_all = pd.DataFrame(daily_summaries)
        avg_win_rate = float(df_all["win_rate"].mean())
        avg_pnl = float(df_all["avg_pnl"].mean())
        avg_dd = float(df_all["max_drawdown"].mean())

        # Composite: reward pnl & win-rate, penalize drawdown magnitude.
        composite_score = (avg_pnl * 0.4) + (avg_win_rate * 0.4) - (abs(avg_dd) * 0.2)

        return {
            "composite_score": float(composite_score),
            "avg_win_rate": avg_win_rate,
            "avg_pnl": avg_pnl,
            "avg_dd": avg_dd,
            "detail": daily_summaries,
        }

    def _optuna_objective(self, trial: optuna.Trial) -> float:
        p_dict = StrategyParams().to_dict()
        for pname, (lo, hi, step) in ENHANCED_CORE_PARAM_SPACE.items():
            if isinstance(lo, int) and isinstance(hi, int) and isinstance(step, int):
                p_dict[pname] = trial.suggest_int(pname, lo, hi, step=step)
            else:
                p_dict[pname] = trial.suggest_float(pname, lo, hi, step=step)

        if self._opt_fixed_params:
            p_dict.update(self._opt_fixed_params)

        params = StrategyParams.from_dict(p_dict)
        res = self.run_full_evaluation(params)
        if not res.get("detail"):
            return -999.0
        return float(res["composite_score"])

    def optimize_for_real_trading(self, n_trials: int = 50, fixed_params: Dict | None = None) -> Dict:
        logger.info(f"Starting closed-loop optimization for {n_trials} trials...")
        self._opt_fixed_params = fixed_params or {}

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self._optuna_objective, n_trials=n_trials)

        best_p = study.best_params
        best_s = float(study.best_value)
        logger.info(f"Optimization finished! Best composite score: {best_s:.4f}")

        final_params = StrategyParams.from_dict({**StrategyParams().to_dict(), **best_p, **(fixed_params or {})})
        return {"best_params": final_params.to_dict(), "best_composite_score": best_s}


if __name__ == "__main__":
    test_dates = ["2023-01-05", "2023-03-01", "2023-06-05", "2023-09-01"]
    pipeline = ValidationPipeline(validation_dates=test_dates)

    base_params = StrategyParams()
    base_res = pipeline.run_full_evaluation(base_params)
    print("=== Baseline Performance ===")
    print(json.dumps(base_res, indent=2, ensure_ascii=False))
