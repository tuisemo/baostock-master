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
import quant.app.backtester as bt_mod
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


def _debug_scan_reason(code: str, target_date: str, p: StrategyParams) -> str:
    """
    Re-run scan logic step-by-step and return the first reason for returning None.
    This keeps the analysis rigorous when scan/backtest disagree.
    """
    # 1) Data preparation (same as scan)
    df = bt_mod._load_and_prepare(code, p)
    if df is None or df.empty:
        return "no_df_or_empty"

    td = None
    try:
        td = pd.to_datetime(str(target_date)[:10])
    except Exception:
        td = None

    if td is not None:
        df = df[df.index <= td]
    if df.empty or len(df) < 3:
        return "not_enough_rows_after_date_filter"

    cols = bt_mod._build_column_names(p)
    required = [cols["sma_s"], cols["sma_l"], cols["macd_h"], cols["rsi"], cols["bb_lower"], cols["atr"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"missing_required_cols: {missing}"

    row_1 = df.iloc[-1]
    row_2 = df.iloc[-2]
    row_3 = df.iloc[-3] if len(df) >= 3 else row_2

    price = row_1.get("Close", row_1.get("close", np.nan))
    open_p = row_1.get("Open", row_1.get("open", np.nan))
    low_p = row_1.get("Low", row_1.get("low", np.nan))
    if not np.isfinite(float(price)) or not np.isfinite(float(open_p)) or not np.isfinite(float(low_p)):
        return "bad_price_open_low"

    sma_l_1 = row_1[cols["sma_l"]]
    sma_l_3 = row_3[cols["sma_l"]]
    sma_s_1 = row_1[cols["sma_s"]]
    macd_h_1 = row_1[cols["macd_h"]]
    macd_h_2 = row_2[cols["macd_h"]]
    rsi_val = row_1[cols["rsi"]]
    vol_1 = row_1.get("Volume", row_1.get("volume", 0))
    vol_2 = row_2.get("Volume", row_2.get("volume", 0))

    has_vol_slope = "vol_slope" in df.columns
    vol_slope_1 = row_1.get("vol_slope", 0.0) if has_vol_slope else 0.0
    has_mom_div = "momentum_divergence" in df.columns
    mom_div_1 = row_1.get("momentum_divergence", 0.0) if has_mom_div else 0.0

    # Market regime
    current_date_ts = df.index[-1]
    market_uptrend = True
    market_state = "sideways_low_vol"
    idx_df = bt_mod.get_market_index()
    if idx_df is not None:
        try:
            idx_loc = idx_df.index.get_indexer([current_date_ts], method="pad")[0]
            if idx_loc != -1 and idx_loc < len(idx_df):
                market_uptrend = bool(idx_df.iloc[idx_loc].get("market_uptrend", True))
                market_state = str(idx_df.iloc[idx_loc].get("market_state", market_state))
        except Exception:
            pass

    dyn_p = p
    try:
        if market_state:
            dyn_p = bt_mod.get_dynamic_params(p, market_state)
    except Exception:
        dyn_p = p

    weekly_data = None
    try:
        weekly_data = bt_mod.get_weekly_confirmation(code, current_date_ts.strftime("%Y-%m-%d"), bt_mod.CONF.history_data.data_dir)
    except Exception:
        weekly_data = None

    signal_pullback, signal_rebound, signal_trend_breakout, signal_details = bt_mod.evaluate_buy_signals(
        price=float(price),
        open_p=float(open_p),
        low_p=float(low_p),
        sma_l_1=float(sma_l_1),
        sma_l_3=float(sma_l_3) if sma_l_3 is not None else None,
        sma_s_1=float(sma_s_1),
        macd_h_1=float(macd_h_1),
        macd_h_2=float(macd_h_2),
        rsi_1=float(rsi_val),
        bb_lower_1=float(row_1[cols["bb_lower"]]),
        vol_1=float(vol_1),
        vol_2=float(vol_2),
        has_vol_slope=has_vol_slope,
        vol_slope_1=float(vol_slope_1) if np.isfinite(float(vol_slope_1)) else 0.0,
        has_mom_div=has_mom_div,
        mom_div_1=float(mom_div_1) if np.isfinite(float(mom_div_1)) else 0.0,
        market_uptrend=bool(market_uptrend),
        p=dyn_p,
        weekly_data=weekly_data,
    )

    signal_type = ""
    if signal_pullback:
        signal_type = "布林带极度下杀反弹 (左侧)"
    elif signal_rebound:
        signal_type = "超卖恐慌底部 (左侧)"
    elif signal_trend_breakout:
        signal_type = "均线放量金叉 (右侧)"
    if not signal_type:
        cs = signal_details.get("composite_score")
        qs = signal_details.get("quality_gate_passed")
        ms = signal_details.get("min_composite_score")
        tf = signal_details.get("timeframe_alignment")
        vr = (float(vol_1) / float(vol_2)) if float(vol_2) > 0 else None
        return (
            f"no_rule_signal: state={market_state} composite={cs} min={ms} quality_ok={qs} "
            f"tf={tf} vol_slope={float(vol_slope_1):+.3f} vol_ratio={vr}"
        )

    # AI gate
    ai_model = bt_mod._get_ai_model()
    ensemble_model = bt_mod._get_ensemble_model()
    use_ensemble = ensemble_model is not None
    model_present = use_ensemble or (ai_model is not None)
    ai_confidence = 0.5
    ensemble_disagreement = None

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if model_present:
        if not feat_cols:
            return "model_present_but_no_feat_cols"
        feat_row = df.iloc[-1][feat_cols]
        if feat_row.isna().any():
            return "feat_row_has_nan"
        if use_ensemble:
            X_pred = feat_row.values.reshape(1, -1)
            ai_confidence, ensemble_disagreement = bt_mod.get_ensemble_prediction_and_disagreement(
                pd.DataFrame(X_pred, columns=feat_cols)
            )
        elif ai_model is not None:
            ai_confidence = float(ai_model.predict(feat_row.values.reshape(1, -1))[0])

    try:
        ai_thresh = bt_mod.get_dynamic_ai_threshold(
            market_state=market_state,
            base_threshold=float(getattr(dyn_p, "ai_prob_threshold", 0.5)),
            volatility_regime="normal",
        )
    except Exception:
        ai_thresh = float(getattr(dyn_p, "ai_prob_threshold", 0.5))

    if float(ai_confidence) < float(ai_thresh):
        return f"ai_below_threshold: ai={ai_confidence:.4f} thresh={ai_thresh:.4f} state={market_state}"

    confidence_factor, tier = bt_mod.get_tiered_confidence_factor(
        ai_confidence=ai_confidence,
        ensemble_disagreement=ensemble_disagreement,
        use_ensemble=use_ensemble,
    )
    if tier == "block":
        return f"tier_block: ai={ai_confidence:.4f} disagreement={ensemble_disagreement}"

    atr_raw = row_1[cols["atr"]]
    if pd.isna(atr_raw) or not np.isfinite(float(atr_raw)) or float(atr_raw) <= 0:
        return "bad_atr"

    atr_val = float(atr_raw)
    price_f = float(price)
    target_r = (float(p.ai_target_atr_mult) * atr_val) / price_f
    stop_r = (float(p.ai_stop_loss_atr_mult) * atr_val) / price_f
    cost_r = 2.0 * (float(p.commission_pct) + float(p.slippage_pct))
    ev_pct = (ai_confidence * target_r - (1.0 - ai_confidence) * stop_r - cost_r) * 100.0
    if ev_pct < float(getattr(p, "min_expected_value_pct", 0.0)):
        return f"ev_below_threshold: ev={ev_pct:.3f} min={getattr(p, 'min_expected_value_pct', 0.0)}"

    return "ok_but_scan_returned_none"


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
        sig = scan_today_signal(code, params=p, target_date=entry_date)
        reason = None
        if not sig:
            reason = _debug_scan_reason(code, entry_date, p)
            sig = {}
        sig_rows.append(
            {
                "EntryTime": entry_time,
                "entry_date": entry_date,
                "scan_reason": reason,
                **{k: sig.get(k) for k in keys},
            }
        )

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
        "scan_reason",
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
