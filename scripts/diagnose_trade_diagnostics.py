from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

# Ensure project root is importable when running as a script from /scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.app.backtester import run_backtest, scan_today_signal
from quant.core.strategy_params import StrategyParams
from quant.infra.config import CONF


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _date_str(x: Any) -> str:
    if hasattr(x, "strftime"):
        return x.strftime("%Y-%m-%d")
    return str(x)[:10]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _summarize_group(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(key)
        .agg(
            trades=("ReturnPct", "size"),
            win_rate=("is_win", "mean"),
            avg_ret=("ReturnPct", "mean"),
            med_ret=("ReturnPct", "median"),
        )
        .sort_values(["trades", "avg_ret"], ascending=[False, False])
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-trade diagnostics for one stock backtest.")
    ap.add_argument("--code", default="sh.600008", help="Stock code like sh.600008")
    ap.add_argument("--export-dir", default="data/exports", help="Directory to write CSV report")
    args = ap.parse_args()

    code = str(args.code).strip()
    export_dir = Path(args.export_dir)
    _ensure_dir(export_dir)

    p = StrategyParams.from_app_config(CONF)

    res = run_backtest(code)
    if not res:
        print(f"[ERROR] run_backtest returned no result for {code}")
        return 2

    _bt, stats = res
    trades = stats.get("_trades")
    if trades is None or trades.empty:
        print(f"[INFO] No trades for {code}")
        return 0

    t = trades.copy()
    # Enrich with scan_today_signal fields at entry date.
    sig_rows: list[dict[str, Any]] = []
    keys = [
        "signal_type",
        "buy_score",
        "total_score",
        "ai_prob",
        "ai_threshold",
        "expected_value_pct",
        "market_state",
        "market_uptrend",
        "atr_pct",
        "volume_ratio",
        "mom_20",
        "ai_model_type",
        "ai_tier",
        "ensemble_disagreement",
    ]

    for _, tr in t.iterrows():
        entry_time = tr.get("EntryTime")
        entry_date = _date_str(entry_time)
        sig = scan_today_signal(code, params=p, target_date=entry_date) or {}
        sig_rows.append({"EntryTime": entry_time, "entry_date": entry_date, **{k: sig.get(k) for k in keys}})

    sig_df = pd.DataFrame(sig_rows)
    d = t.merge(sig_df, on="EntryTime", how="left")

    # Derived metrics (multi-dimension friendly)
    d["is_win"] = d["ReturnPct"] > 0
    d["atr_pct_est"] = d["Entry_atr"] / d["EntryPrice"] * 100
    d["ma_spread_pct"] = (d["Entry_sma_s"] - d["Entry_sma_l"]) / d["Entry_sma_l"] * 100
    d["price_vs_sma_l_pct"] = (d["EntryPrice"] - d["Entry_sma_l"]) / d["Entry_sma_l"] * 100
    d["bb_pos"] = (d["EntryPrice"] - d["Entry_bbl"]) / (d["Entry_bbu"] - d["Entry_bbl"])

    n = len(d)
    win_rate = float(d["is_win"].mean()) if n else 0.0
    print("=== Backtest Diagnostics ===")
    print(f"code: {code}")
    print(f"trades: {n} | win_rate: {win_rate:.2%} | avg_ret%: {d['ReturnPct'].mean():.2f} | med_ret%: {d['ReturnPct'].median():.2f}")
    print("")

    worst = d.sort_values("ReturnPct").head(8)
    best = d.sort_values("ReturnPct", ascending=False).head(8)
    view_cols = [
        "entry_date",
        "ExitTime",
        "Duration",
        "ReturnPct",
        "PnL",
        "signal_type",
        "ai_prob",
        "expected_value_pct",
        "market_state",
        "atr_pct_est",
        "bb_pos",
        "Entry_rsi",
        "Entry_macd_h",
        "Entry_vol_slope",
    ]
    print("Worst trades (by ReturnPct):")
    print(worst[view_cols].to_string(index=False))
    print("")
    print("Best trades (by ReturnPct):")
    print(best[view_cols].to_string(index=False))
    print("")

    print("Win vs Loss means (selected fields):")
    mean_cols = [
        "ai_prob",
        "expected_value_pct",
        "buy_score",
        "atr_pct_est",
        "bb_pos",
        "Entry_rsi",
        "Entry_macd_h",
        "Entry_vol_slope",
        "ma_spread_pct",
        "price_vs_sma_l_pct",
    ]
    means = d.groupby("is_win")[mean_cols].mean(numeric_only=True)
    print(means.to_string())
    print("")

    ms = _summarize_group(d, "market_state")
    if not ms.empty:
        print("By market_state:")
        print(ms.to_string())
        print("")

    st = _summarize_group(d, "signal_type")
    if not st.empty:
        print("By signal_type:")
        print(st.to_string())
        print("")

    # Volume slope segmentation (key finding candidate)
    if "Entry_vol_slope" in d.columns:
        d["vol_slope_pos"] = pd.to_numeric(d["Entry_vol_slope"], errors="coerce") > 0
        vs = (
            d.groupby("vol_slope_pos")
            .agg(
                trades=("ReturnPct", "size"),
                win_rate=("is_win", "mean"),
                avg_ret=("ReturnPct", "mean"),
                med_ret=("ReturnPct", "median"),
            )
            .sort_values(["trades"], ascending=False)
        )
        print("By Entry_vol_slope > 0:")
        print(vs.to_string())
        print("")

        if "signal_type" in d.columns:
            left = d[d["signal_type"].astype(str).str.contains("左侧", na=False)]
            if not left.empty:
                vs_left = (
                    left.groupby("vol_slope_pos")
                    .agg(
                        trades=("ReturnPct", "size"),
                        win_rate=("is_win", "mean"),
                        avg_ret=("ReturnPct", "mean"),
                        med_ret=("ReturnPct", "median"),
                    )
                    .sort_values(["trades"], ascending=False)
                )
                print("Left-side only: by Entry_vol_slope > 0:")
                print(vs_left.to_string())
                print("")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = export_dir / f"diag_{code.replace('.', '_')}_{ts}.csv"
    d.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
