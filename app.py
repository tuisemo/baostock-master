from __future__ import annotations

import base64
import concurrent.futures
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

# Charts (optional, used in Backtest panel)
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Kline, Line, Scatter
from pyecharts.globals import CurrentConfig

CurrentConfig.ONLINE_HOST = "https://cdn.staticfile.net/echarts/5.4.3/"

from quant.app.backtester import run_backtest, scan_today_signal
from quant.data.data_updater import update_history_data
from quant.data.stock_filter import update_stock_list
from quant.infra.config import CONF
from quant.infra.logger import logger


# =========================
# UI Theme (CSS)
# =========================

CUSTOM_CSS = """
:root {
  --bg0: #0b1020;
  --bg1: #0f172a;
  --card: rgba(255, 255, 255, 0.06);
  --card2: rgba(255, 255, 255, 0.10);
  --ink: rgba(255, 255, 255, 0.92);
  --muted: rgba(255, 255, 255, 0.70);
  --faint: rgba(255, 255, 255, 0.55);
  --line: rgba(255, 255, 255, 0.12);
  --accent: #06b6d4;     /* cyan */
  --accent2: #f59e0b;    /* amber */
  --good: #22c55e;
  --warn: #f97316;
  --bad: #ef4444;
  --radius: 14px;
  --shadow2: 0 8px 24px rgba(0,0,0,0.25);
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: "Segoe UI Variable", "Segoe UI", "Noto Sans SC", "PingFang SC", "Hiragino Sans GB", Arial, sans-serif;
}

.gradio-container {
  max-width: 1400px !important;
  margin: 0 auto !important;
  font-family: var(--sans) !important;
  color: var(--ink) !important;
  background:
    radial-gradient(1200px 700px at 10% 10%, rgba(6, 182, 212, 0.16), transparent 55%),
    radial-gradient(900px 600px at 90% 20%, rgba(245, 158, 11, 0.10), transparent 55%),
    radial-gradient(900px 700px at 40% 95%, rgba(99, 102, 241, 0.12), transparent 60%),
    linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
}

.block, .gr-box, .gr-panel, .gr-group {
  border-radius: var(--radius) !important;
  border: 1px solid var(--line) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow2) !important;
}

.gr-button-primary {
  background: linear-gradient(135deg, var(--accent), #3b82f6) !important;
  border: none !important;
  font-weight: 700 !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}
.gr-button-primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 14px 30px rgba(6, 182, 212, 0.25) !important;
}

.dataframe {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
.dataframe th {
  background: rgba(255, 255, 255, 0.10) !important;
  color: var(--ink) !important;
  font-weight: 700 !important;
}
.dataframe tr:hover td {
  background: rgba(6, 182, 212, 0.06) !important;
}

.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.07);
  font-size: 12px;
  color: var(--muted);
}
.pill.good { border-color: rgba(34,197,94,0.35); color: rgba(34,197,94,0.95); }
.pill.warn { border-color: rgba(249,115,22,0.35); color: rgba(249,115,22,0.95); }
.pill.bad  { border-color: rgba(239,68,68,0.35);  color: rgba(239,68,68,0.95); }

@keyframes fadein {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
.fadein { animation: fadein .45s ease both; }
"""


# =========================
# Helpers
# =========================

MARKET_STATES = [
    "strong_bull",
    "bull_momentum",
    "bull_volume",
    "weak_bull",
    "sideways_low_vol",
    "sideways_high_vol",
    "weak_bear",
    "bear_momentum",
    "bear_panic",
    "strong_bear",
]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _read_stock_list() -> tuple[list[str], dict[str, str]]:
    path = Path(CONF.history_data.data_dir) / "stock-list.csv"
    if not path.exists():
        return [], {}
    df = pd.read_csv(path)
    if "code" not in df.columns:
        return [], {}
    codes = df["code"].dropna().astype(str).tolist()
    name_map: dict[str, str] = {}
    if "code_name" in df.columns:
        # strict=False for older pandas compatibility
        name_map = dict(zip(df["code"].astype(str).tolist(), df["code_name"].fillna("").astype(str).tolist(), strict=False))
    return codes, name_map


def _parse_codes_text(text: str) -> list[str]:
    codes: list[str] = []
    for raw in str(text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        codes.append(s)
    # de-dup while keeping order
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _format_scan_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make scan results user-friendly for UI tables (Chinese columns + stable ordering)."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    df2 = df.copy()

    for c in (
        "close",
        "total_score",
        "buy_score",
        "ai_prob",
        "ai_threshold",
        "ensemble_disagreement",
        "atr",
        "atr_pct",
        "expected_value_pct",
        "volume_ratio",
        "mom_20",
    ):
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    if "buy_score" in df2.columns:
        df2 = df2.sort_values("buy_score", ascending=False, na_position="last")
    elif "total_score" in df2.columns:
        df2 = df2.sort_values("total_score", ascending=False, na_position="last")

    rename_map = {
        "code": "代码",
        "name": "名称",
        "date": "日期",
        "close": "收盘价",
        "signal_type": "信号类型",
        "total_score": "规则得分",
        "buy_score": "买点得分",
        "expected_value_pct": "EV(%)",
        "ai_prob": "AI胜率",
        "ai_threshold": "AI阈值",
        "ai_tier": "AI档位",
        "ensemble_disagreement": "集成分歧",
        "market_state": "市场状态",
        "market_uptrend": "大盘上行",
        "atr_pct": "ATR(%)",
        "atr": "ATR",
        "volume_ratio": "量比",
        "mom_20": "20日动量",
        "ai_model_type": "模型类型",
    }
    df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})

    preferred_order = [
        "代码",
        "名称",
        "日期",
        "收盘价",
        "信号类型",
        "买点得分",
        "EV(%)",
        "AI胜率",
        "AI阈值",
        "AI档位",
        "集成分歧",
        "市场状态",
        "大盘上行",
        "ATR(%)",
        "规则得分",
        "量比",
        "20日动量",
        "模型类型",
    ]
    keep = [c for c in preferred_order if c in df2.columns]
    rest = [c for c in df2.columns if c not in keep]
    return df2[keep + rest]


def _summarize_scan_results(df: pd.DataFrame, total_universe: int) -> str:
    if df is None or df.empty:
        return f"- 扫描股票数: {total_universe}\n- 命中买点数: 0\n"

    lines: list[str] = [
        f"- 扫描股票数: {total_universe}",
        f"- 命中买点数: {len(df)}",
    ]

    if "ai_prob" in df.columns:
        v = pd.to_numeric(df["ai_prob"], errors="coerce").mean()
        if pd.notna(v):
            lines.append(f"- AI胜率均值: {float(v):.2%}")

    if "expected_value_pct" in df.columns:
        ev_mean = pd.to_numeric(df["expected_value_pct"], errors="coerce").mean()
        ev_med = pd.to_numeric(df["expected_value_pct"], errors="coerce").median()
        if pd.notna(ev_mean) and pd.notna(ev_med):
            lines.append(f"- EV(%) 均值/中位数: {float(ev_mean):.2f} / {float(ev_med):.2f}")

    if "market_state" in df.columns:
        try:
            ms = str(df["market_state"].mode().iloc[0])
            lines.append(f"- 主要市场状态: {ms}")
        except Exception:
            pass

    return "\n".join(lines)


def _apply_scan_filters(
    df: pd.DataFrame,
    min_buy_score: float,
    min_expected_value_pct: float,
    min_ai_prob: float,
    market_state_allowlist: list[str],
    max_atr_pct: float,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    out = df.copy()

    if "buy_score" in out.columns:
        out = out[pd.to_numeric(out["buy_score"], errors="coerce") >= float(min_buy_score)]
    if "expected_value_pct" in out.columns:
        out = out[pd.to_numeric(out["expected_value_pct"], errors="coerce") >= float(min_expected_value_pct)]
    if "ai_prob" in out.columns:
        out = out[pd.to_numeric(out["ai_prob"], errors="coerce") >= float(min_ai_prob)]

    if market_state_allowlist and "market_state" in out.columns:
        out = out[out["market_state"].astype(str).isin(set(market_state_allowlist))]

    if max_atr_pct is not None and float(max_atr_pct) > 0 and "atr_pct" in out.columns:
        out = out[pd.to_numeric(out["atr_pct"], errors="coerce") <= float(max_atr_pct)]

    return out.reset_index(drop=True)


def _export_scan_csv(df_internal: pd.DataFrame, tag: str) -> str | None:
    if df_internal is None or df_internal.empty:
        return None
    out_dir = Path("data") / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scan_candidates_{tag}_{_now_tag()}.csv"
    df_internal.to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)


def _status_html() -> str:
    data_dir = Path(CONF.history_data.data_dir)
    stock_list_path = data_dir / "stock-list.csv"
    idx_path = data_dir / "sh.000001.csv"

    codes, _name_map = _read_stock_list()

    def pill(text: str, cls: str) -> str:
        return f"<span class='pill {cls}'>{text}</span>"

    # Stock list
    if stock_list_path.exists():
        stock_list_state = pill("stock-list.csv 已就绪", "good")
        stock_list_meta = f"{len(codes)} 只 | mtime: {datetime.fromtimestamp(stock_list_path.stat().st_mtime):%Y-%m-%d %H:%M}"
    else:
        stock_list_state = pill("缺少 stock-list.csv", "warn")
        stock_list_meta = "请先在 Setup 中更新股票池"

    # Index coverage
    idx_state = pill("指数数据未知", "warn")
    idx_meta = "请先同步历史数据"
    if idx_path.exists():
        try:
            idx_df = pd.read_csv(idx_path)
            if "date" in idx_df.columns:
                dates = pd.to_datetime(idx_df["date"], errors="coerce").dropna()
            elif "Date" in idx_df.columns:
                dates = pd.to_datetime(idx_df["Date"], errors="coerce").dropna()
            else:
                dates = pd.Series([], dtype="datetime64[ns]")
            if len(dates) > 0:
                idx_state = pill("指数数据已就绪", "good")
                idx_meta = f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d} | rows: {len(dates)}"
        except Exception:
            pass

    # Model status
    model_dir = Path("models")
    model_file = model_dir / "alpha_lgbm.txt"
    if model_file.exists():
        model_state = pill("AI 模型: alpha_lgbm.txt", "good")
        model_meta = f"mtime: {datetime.fromtimestamp(model_file.stat().st_mtime):%Y-%m-%d %H:%M} | size: {model_file.stat().st_size/1024/1024:.1f} MB"
    else:
        model_state = pill("AI 模型缺失", "warn")
        model_meta = "可先训练/放置模型文件，再进行 AI 门控扫描"

    # Strategy params snapshot
    try:
        from quant.core.strategy_params import StrategyParams

        p = StrategyParams.from_app_config(CONF)
        params_meta = (
            f"ai_prob_threshold={p.ai_prob_threshold:.2f} | min_EV(%)={p.min_expected_value_pct:.2f} | "
            f"hold_window=min({p.max_hold_days},{p.ai_forward_days})"
        )
        params_state = pill("策略参数已加载", "good")
    except Exception as e:
        params_state = pill("策略参数加载失败", "bad")
        params_meta = str(e)

    return f"""
    <div class="fadein" style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
      <div style="padding:14px; border-radius: var(--radius); border: 1px solid var(--line); background: var(--card2);">
        <div style="font-size:12px; color: var(--faint); margin-bottom:8px;">股票池</div>
        <div style="font-size:14px; font-weight:700; margin-bottom:6px;">{stock_list_state}</div>
        <div style="font-size:12px; color: var(--muted);">{stock_list_meta}</div>
      </div>
      <div style="padding:14px; border-radius: var(--radius); border: 1px solid var(--line); background: var(--card2);">
        <div style="font-size:12px; color: var(--faint); margin-bottom:8px;">数据覆盖</div>
        <div style="font-size:14px; font-weight:700; margin-bottom:6px;">{idx_state}</div>
        <div style="font-size:12px; color: var(--muted);">{idx_meta}</div>
      </div>
      <div style="padding:14px; border-radius: var(--radius); border: 1px solid var(--line); background: var(--card2);">
        <div style="font-size:12px; color: var(--faint); margin-bottom:8px;">模型状态</div>
        <div style="font-size:14px; font-weight:700; margin-bottom:6px;">{model_state}</div>
        <div style="font-size:12px; color: var(--muted);">{model_meta}</div>
      </div>
      <div style="padding:14px; border-radius: var(--radius); border: 1px solid var(--line); background: var(--card2);">
        <div style="font-size:12px; color: var(--faint); margin-bottom:8px;">策略口径</div>
        <div style="font-size:14px; font-weight:700; margin-bottom:6px;">{params_state}</div>
        <div style="font-size:12px; color: var(--muted); font-family: var(--mono);">{params_meta}</div>
      </div>
    </div>
    """


# =========================
# Actions (Setup)
# =========================


def ui_refresh_status() -> str:
    return _status_html()


def ui_update_stock_pool():
    yield "正在更新股票池 (生成 data/stock-list.csv)...", _status_html()
    try:
        update_stock_list()
    except Exception as e:
        yield f"更新失败: {e}", _status_html()
        return
    yield "股票池更新完成。", _status_html()


def ui_update_kline_data():
    yield "正在增量更新历史 K 线数据 (写入 data/*.csv)...", _status_html()
    try:
        update_history_data()
    except Exception as e:
        yield f"更新失败: {e}", _status_html()
        return
    yield "历史数据更新完成。", _status_html()


# =========================
# Actions (Scan)
# =========================


def ui_run_scan(
    universe_source: str,
    custom_codes_text: str,
    target_date: str,
    max_codes: int,
    workers: int,
    min_buy_score: float,
    min_expected_value_pct: float,
    min_ai_prob: float,
    market_state_allowlist: list[str],
    max_atr_pct: float,
):
    # Prime UI
    yield "准备扫描...", pd.DataFrame(), None, pd.DataFrame()

    # Resolve universe
    name_map: dict[str, str] = {}
    if universe_source == "stock-list.csv":
        codes, name_map = _read_stock_list()
        if not codes:
            yield "未找到可用股票池，请先在 Setup 中更新股票池。", pd.DataFrame(), None, pd.DataFrame()
            return
        universe_note = "stock-list.csv"
    else:
        codes = _parse_codes_text(custom_codes_text)
        universe_note = "custom codes"

    if max_codes is not None and int(max_codes) > 0:
        codes = codes[: int(max_codes)]

    if not codes:
        yield "股票列表为空。", pd.DataFrame(), None, pd.DataFrame()
        return

    date_s = str(target_date or "").strip()[:10]
    date_arg = date_s if date_s else None

    # Params snapshot
    try:
        from quant.core.strategy_params import StrategyParams

        p = StrategyParams.from_app_config(CONF)
    except Exception as e:
        yield f"策略参数加载失败: {e}", pd.DataFrame(), None, pd.DataFrame()
        return

    total = len(codes)
    results: list[dict[str, Any]] = []
    errors = 0

    progress_every = max(20, total // 50)
    workers_i = max(1, int(workers or 1))

    def scan_one(code: str) -> dict[str, Any] | None:
        try:
            sig = scan_today_signal(code, params=p, target_date=date_arg)
        except Exception:
            return None
        if not sig:
            return None
        if name_map:
            sig = dict(sig)
            sig["name"] = name_map.get(code, "")
        return sig

    started = datetime.now()
    yield (
        f"开始扫描 ({universe_note}) | 代码数={total} | target_date={date_arg or 'latest'} | workers={workers_i} ...",
        pd.DataFrame(),
        None,
        pd.DataFrame(),
    )

    if workers_i <= 1:
        for i, code in enumerate(codes, start=1):
            sig = scan_one(code)
            if sig is not None:
                results.append(sig)

            if i % progress_every == 0 or i == total:
                df_internal = pd.DataFrame(results)
                df_internal = _apply_scan_filters(
                    df_internal,
                    min_buy_score=min_buy_score,
                    min_expected_value_pct=min_expected_value_pct,
                    min_ai_prob=min_ai_prob,
                    market_state_allowlist=market_state_allowlist,
                    max_atr_pct=max_atr_pct,
                )
                msg = f"进度: {i}/{total} | 命中(过滤后): {len(df_internal)}"
                yield msg, _format_scan_results_df(df_internal), None, df_internal
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers_i) as ex:
            futs = [ex.submit(scan_one, c) for c in codes]
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                done += 1
                try:
                    sig = fut.result()
                except Exception:
                    errors += 1
                    sig = None
                if sig is not None:
                    results.append(sig)

                if done % progress_every == 0 or done == total:
                    df_internal = pd.DataFrame(results)
                    df_internal = _apply_scan_filters(
                        df_internal,
                        min_buy_score=min_buy_score,
                        min_expected_value_pct=min_expected_value_pct,
                        min_ai_prob=min_ai_prob,
                        market_state_allowlist=market_state_allowlist,
                        max_atr_pct=max_atr_pct,
                    )
                    msg = f"进度: {done}/{total} | 命中(过滤后): {len(df_internal)} | errors: {errors}"
                    yield msg, _format_scan_results_df(df_internal), None, df_internal

    df_internal = pd.DataFrame(results)
    df_filtered = _apply_scan_filters(
        df_internal,
        min_buy_score=min_buy_score,
        min_expected_value_pct=min_expected_value_pct,
        min_ai_prob=min_ai_prob,
        market_state_allowlist=market_state_allowlist,
        max_atr_pct=max_atr_pct,
    )

    tag = f"{universe_source}_{date_arg or 'latest'}".replace(" ", "_").replace("/", "_")
    export_path = _export_scan_csv(df_filtered, tag=tag)

    elapsed = (datetime.now() - started).total_seconds()
    summary = _summarize_scan_results(df_filtered, total_universe=total)
    msg = (
        f"扫描完成: {elapsed:.1f}s | errors: {errors}\n\n"
        f"{summary}\n\n"
        f"导出文件: {export_path or 'N/A'}"
    )

    yield msg, _format_scan_results_df(df_filtered), export_path, df_filtered


def ui_export_selected(df_state: pd.DataFrame) -> str | None:
    if df_state is None or df_state.empty:
        return None
    return _export_scan_csv(df_state, tag="manual_export")


def ui_on_scan_select(df_state: pd.DataFrame, capital: float, evt: gr.SelectData):
    if df_state is None or df_state.empty:
        return "", "", "未选择任何行。", None

    row = None
    try:
        if isinstance(evt.index, (tuple, list)) and len(evt.index) >= 1:
            row = int(evt.index[0])
        else:
            row = int(evt.index)
    except Exception:
        row = None

    if row is None or row < 0 or row >= len(df_state):
        return "", "", "选择行无效。", None

    code = str(df_state.iloc[row].get("code", "")).strip()
    date_s = str(df_state.iloc[row].get("date", "")).strip()[:10]
    plan_md, sig = ui_trade_plan(code=code, target_date=date_s, capital=capital)
    return code, date_s, plan_md, sig


# =========================
# Actions (Plan + Backtest)
# =========================


def ui_trade_plan(code: str, target_date: str | None, capital: float) -> tuple[str, dict[str, Any] | None]:
    code = str(code or "").strip()
    if not code:
        return "请输入股票代码，例如 `sh.600000`。", None

    date_s = str(target_date or "").strip()[:10]
    date_arg = date_s if date_s else None

    try:
        from quant.core.adaptive_strategy import get_dynamic_params
        from quant.core.strategy_params import StrategyParams
        from quant.app.backtester import get_tiered_confidence_factor
    except Exception as e:
        return f"依赖加载失败: {e}", None

    p = StrategyParams.from_app_config(CONF)
    sig = scan_today_signal(code, params=p, target_date=date_arg)
    if not sig:
        return "未触发买点信号 (或数据不足/指标无法计算)。", None

    close = _safe_float(sig.get("close"))
    atr = _safe_float(sig.get("atr"))
    if not close or not atr or close <= 0 or atr <= 0:
        return "信号数据不完整 (缺少 close/atr)。", sig

    stop_px = close - float(p.ai_stop_loss_atr_mult) * atr
    target_px = close + float(p.ai_target_atr_mult) * atr
    stop_pct = (stop_px / close - 1.0) * 100.0
    target_pct = (target_px / close - 1.0) * 100.0

    market_state = str(sig.get("market_state", "") or "")
    try:
        dyn_p = get_dynamic_params(p, market_state) if market_state else p
    except Exception:
        dyn_p = p

    ai_prob = float(sig.get("ai_prob", 0.5))
    ai_thresh = sig.get("ai_threshold", None)
    tier = str(sig.get("ai_tier", "") or "")
    disagreement = sig.get("ensemble_disagreement", None)
    use_ensemble = str(sig.get("ai_model_type", "")).lower() == "ensemble"

    dis_v = _safe_float(disagreement)
    try:
        confidence_factor, _ = get_tiered_confidence_factor(
            ai_confidence=ai_prob,
            ensemble_disagreement=dis_v,
            use_ensemble=use_ensemble,
        )
    except Exception:
        confidence_factor = 1.0

    suggested_pos = float(getattr(dyn_p, "position_size", 0.1)) * float(confidence_factor)

    try:
        cap = float(capital) if capital else 100000.0
    except Exception:
        cap = 100000.0

    risk_budget = cap * float(getattr(dyn_p, "atr_risk_per_trade", 0.02))
    risk_per_share = max(1e-8, close - stop_px)
    shares = int((risk_budget / risk_per_share) // 100) * 100
    risk_based_pos = (shares * close / cap) if shares > 0 else None

    final_pos = suggested_pos
    if risk_based_pos is not None and risk_based_pos > 0:
        final_pos = min(final_pos, float(risk_based_pos))

    hold_days = min(int(getattr(p, "max_hold_days", 5)), int(getattr(p, "ai_forward_days", 5)))

    ev_pct = sig.get("expected_value_pct", None)
    buy_score = sig.get("buy_score", None)
    total_score = sig.get("total_score", None)
    sig_type = str(sig.get("signal_type", "") or "")

    ai_thresh_s = f"{float(ai_thresh):.2f}" if ai_thresh is not None else "N/A"
    ev_s = f"{float(ev_pct):+.2f}%" if ev_pct is not None else "N/A"
    buy_score_s = f"{float(buy_score):.3f}" if buy_score is not None else "N/A"
    rules_s = f"{float(total_score):.3f}" if total_score is not None else "N/A"

    lines = [
        "### 交易计划 (Buy Plan)",
        f"- 标的: `{code}`  日期: `{sig.get('date','') or (date_s or 'latest')}`",
        f"- 信号: {sig_type}  市场状态: `{market_state or 'N/A'}`",
        f"- 买点得分: `{buy_score_s}`  规则得分: `{rules_s}`  EV: `{ev_s}`",
        "",
        "#### 价格区间 (以收盘价为参考)",
        f"- 参考买入价: `{close:.2f}`",
        f"- 止损价(ATR): `{stop_px:.2f}` ({stop_pct:.2f}%)",
        f"- 止盈价(ATR): `{target_px:.2f}` (+{target_pct:.2f}%)",
        f"- 建议观察/持有窗口: `{hold_days}` 天 (与 AI 标签周期对齐)",
        "",
        "#### AI 门控",
        f"- AI胜率: `{ai_prob:.2%}`  阈值: `{ai_thresh_s}`  档位: `{tier or 'N/A'}`  分歧: `{disagreement if disagreement is not None else 'N/A'}`",
        "",
        "#### 仓位建议 (参考)",
        f"- 策略仓位(含置信度因子): `{suggested_pos:.2%}`",
    ]

    if risk_based_pos is not None and risk_based_pos > 0:
        lines.append(f"- 风险预算仓位(按 atr_risk_per_trade): `{risk_based_pos:.2%}`  (约 `{shares}` 股)")
        lines.append(f"- 建议执行仓位: `{final_pos:.2%}`  (约 `{cap*final_pos:,.0f}` 元)")
    else:
        lines.append(f"- 建议执行仓位: `{final_pos:.2%}`  (约 `{cap*final_pos:,.0f}` 元)")

    # Lightweight risk hints
    hints: list[str] = []
    atr_pct = _safe_float(sig.get("atr_pct"))
    if atr_pct is not None and atr_pct > 8:
        hints.append(f"- 波动提示: ATR%={atr_pct:.2f} 偏高，注意止损滑点与仓位。")
    if market_state in ("strong_bear", "bear_panic"):
        hints.append("- 市场提示: 当前市场状态偏熊，建议降低仓位或提高筛选门槛。")
    if hints:
        lines.extend(["", "#### 风险提示"] + hints)

    return "\n".join(lines), sig


def ui_backtest(code: str):
    code = str(code or "").strip()
    if not code:
        return "请输入股票代码，例如 sh.600000", None, None

    file_path = Path(CONF.history_data.data_dir) / f"{code}.csv"
    if not file_path.exists():
        return f"本地未找到 {code} 的数据，请先更新历史数据。", None, None

    try:
        result = run_backtest(code)
    except Exception as e:
        return f"回测失败: {e}", None, None

    if not result:
        return "回测失败，可能是数据量过少或无法计算指标。", None, None

    bt, stats = result
    stats_text = (
        f"**回测标的**: {code}\n\n"
        f"**起止时间**: {stats['Start'].strftime('%Y-%m-%d')} -> {stats['End'].strftime('%Y-%m-%d')}\n\n"
        f"**初始资金**: ￥100,000.00\n\n"
        f"**最终资金**: ￥{stats['Equity Final [$]']:,.2f}\n\n"
        f"**收益率 (Return)**: {stats['Return [%]']:.2f}%\n\n"
        f"**最大回撤 (Max Drawdown)**: {stats['Max. Drawdown [%]']:.2f}%\n\n"
        f"**夏普比率 (Sharpe Ratio)**: {stats.get('Sharpe Ratio', 0):.2f}\n\n"
        f"**交易次数**: {stats['# Trades']}\n\n"
        f"**胜率 (Win Rate)**: {stats['Win Rate [%]']:.2f}%\n\n"
    )

    iframe_html = None
    try:
        df = bt._data
        df.reset_index(inplace=True)
        dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        kline_data = df[["Open", "Close", "Low", "High"]].values.tolist()

        kline = (
            Kline()
            .add_xaxis(dates)
            .add_yaxis(
                "K线",
                kline_data,
                itemstyle_opts=opts.ItemStyleOpts(color="#ef4444", color0="#22c55e"),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True),
                title_opts=opts.TitleOpts(title=f"{code} 策略回测"),
                datazoom_opts=[
                    opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=[0, 1], range_start=80, range_end=100),
                    opts.DataZoomOpts(is_show=True, xaxis_index=[0, 1], type_="slider", pos_top="95%", range_start=80, range_end=100),
                ],
                legend_opts=opts.LegendOpts(pos_top="5%", pos_left="center"),
            )
        )

        ma_s_col = f"SMA_{CONF.analyzer.ma_short}"
        ma_l_col = f"SMA_{CONF.analyzer.ma_long}"
        line = Line().add_xaxis(dates)
        if ma_s_col in df.columns:
            line.add_yaxis(
                f"MA{CONF.analyzer.ma_short}",
                df[ma_s_col].tolist(),
                is_symbol_show=False,
                color="#f59e0b",
                label_opts=opts.LabelOpts(is_show=False),
            )
        if ma_l_col in df.columns:
            line.add_yaxis(
                f"MA{CONF.analyzer.ma_long}",
                df[ma_l_col].tolist(),
                is_symbol_show=False,
                color="#06b6d4",
                label_opts=opts.LabelOpts(is_show=False),
            )
        kline.overlap(line)

        trades_df = stats.get("_trades", pd.DataFrame())
        if trades_df is not None and not trades_df.empty:
            buy_y = [None] * len(dates)
            sell_y = [None] * len(dates)
            for _, row in trades_df.iterrows():
                entry_t = row["EntryTime"]
                exit_t = row["ExitTime"]
                entry_idx = df[df["Date"] == entry_t].index[0] if isinstance(entry_t, pd.Timestamp) else int(entry_t)
                exit_idx = df[df["Date"] == exit_t].index[0] if isinstance(exit_t, pd.Timestamp) else int(exit_t)
                buy_y[entry_idx] = row["EntryPrice"]
                sell_y[exit_idx] = row["ExitPrice"]

            if any(y is not None for y in buy_y):
                buy_scatter = (
                    Scatter()
                    .add_xaxis(dates)
                    .add_yaxis(
                        "买入",
                        buy_y,
                        symbol="triangle",
                        symbol_size=14,
                        itemstyle_opts=opts.ItemStyleOpts(color="#ef4444"),
                        label_opts=opts.LabelOpts(is_show=False),
                    )
                )
                kline.overlap(buy_scatter)

            if any(y is not None for y in sell_y):
                sell_scatter = (
                    Scatter()
                    .add_xaxis(dates)
                    .add_yaxis(
                        "卖出",
                        sell_y,
                        symbol="triangle-down",
                        symbol_size=14,
                        itemstyle_opts=opts.ItemStyleOpts(color="#22c55e"),
                        label_opts=opts.LabelOpts(is_show=False),
                    )
                )
                kline.overlap(sell_scatter)

        volumes = df["Volume"].tolist() if "Volume" in df.columns else []
        bar = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis("成交量", volumes, label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category", grid_index=1, axislabel_opts=opts.LabelOpts(is_show=False)),
                yaxis_opts=opts.AxisOpts(is_scale=True, grid_index=1, axislabel_opts=opts.LabelOpts(is_show=False)),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        grid = (
            Grid(init_opts=opts.InitOpts(width="100%", height="800px"))
            .add(kline, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", height="65%"))
            .add(bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", pos_top="80%", height="15%"))
        )

        html_file = "temp_backtest_chart.html"
        grid.render(html_file)
        raw_html = Path(html_file).read_text(encoding="utf-8")
        b64_html = base64.b64encode(raw_html.encode("utf-8")).decode("utf-8")
        iframe_html = f'<iframe src="data:text/html;base64,{b64_html}" width="100%" height="850px" frameborder="0"></iframe>'
    except Exception as e:
        logger.debug(f"Render chart failed: {e}")

    return stats_text, iframe_html, stats.get("_trades", pd.DataFrame())


# =========================
# Actions (Verify)
# =========================


def _parse_date_lines(text: str) -> list[str]:
    dates: list[str] = []
    for raw in str(text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        dates.append(s[:10])

    seen: set[str] = set()
    out: list[str] = []
    for d in dates:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return sorted(out)


def ui_generate_validation_dates(start_date: str, end_date: str, n_dates: int):
    start = str(start_date or "").strip()[:10]
    end = str(end_date or "").strip()[:10]
    if not start or not end:
        return "请先选择开始/结束日期。", ""

    from quant.app.backtester import get_market_index

    idx_df = get_market_index()
    if idx_df is None or idx_df.empty:
        return "无法读取大盘指数数据 (sh.000001.csv)，请先同步数据。", ""

    dates = idx_df.index
    valid = dates[(dates >= start) & (dates <= end)]
    if len(valid) == 0:
        return f"在区间 {start} ~ {end} 没有可用交易日。", ""

    n = int(n_dates) if n_dates else 20
    n = max(1, min(n, len(valid)))
    indices = np.linspace(0, len(valid) - 1, n, dtype=int)
    sample_dates = [valid[i].strftime("%Y-%m-%d") for i in indices]
    return f"已生成 {len(sample_dates)} 个验证日期。", "\n".join(sample_dates)


def ui_run_validation(date_list_text: str, sample_size: int, max_trades_per_day: int, max_hold_days: int):
    dates = _parse_date_lines(date_list_text)
    if not dates:
        return "请先生成或输入验证日期列表 (每行一个 YYYY-MM-DD)。", pd.DataFrame()

    from quant.app.validation_pipeline import ValidationPipeline
    from quant.core.strategy_params import StrategyParams

    p = StrategyParams.from_app_config(CONF)
    try:
        if max_hold_days is not None:
            p.max_hold_days = int(max_hold_days)
    except Exception:
        pass

    pipe = ValidationPipeline(
        validation_dates=dates,
        sample_size=int(sample_size),
        max_trades_per_day=int(max_trades_per_day),
    )
    res = pipe.run_full_evaluation(p)
    detail_df = pd.DataFrame(res.get("detail", []))

    summary = (
        f"策略体检完成\n\n"
        f"日期数: {len(dates)} | sample_size: {int(sample_size)} | topN: {int(max_trades_per_day)} | max_hold_days: {p.max_hold_days}\n\n"
        f"复合得分: {res.get('composite_score', -999.0):.4f}\n"
        f"平均胜率: {res.get('avg_win_rate', 0.0):.2f}%\n"
        f"平均PnL: {res.get('avg_pnl', 0.0):.2f}%\n"
        f"平均回撤: {res.get('avg_dd', 0.0):.2f}%\n"
    )
    return summary, detail_df


def ui_optimize_validation(date_list_text: str, sample_size: int, max_trades_per_day: int, max_hold_days: int, n_trials: int):
    yield "开始闭环寻优 (基于买点信号的样本外体检得分)...", pd.DataFrame()

    dates = _parse_date_lines(date_list_text)
    if not dates:
        yield "请先生成或输入验证日期列表 (每行一个 YYYY-MM-DD)。", pd.DataFrame()
        return

    from quant.app.validation_pipeline import ValidationPipeline

    fixed: dict[str, Any] = {}
    try:
        if max_hold_days is not None:
            fixed["max_hold_days"] = int(max_hold_days)
    except Exception:
        fixed = {}

    pipe = ValidationPipeline(
        validation_dates=dates,
        sample_size=int(sample_size),
        max_trades_per_day=int(max_trades_per_day),
    )

    try:
        opt_res = pipe.optimize_for_real_trading(n_trials=int(n_trials), fixed_params=fixed or None)
    except Exception as e:
        yield f"闭环寻优失败: {e}", pd.DataFrame()
        return

    best_score = float(opt_res.get("best_composite_score", -999.0))
    best_params = opt_res.get("best_params", {}) or {}
    best_df = pd.DataFrame([{"param": k, "value": v} for k, v in sorted(best_params.items())])

    msg = (
        f"闭环寻优完成\n\n"
        f"Best composite_score: {best_score:.4f}\n"
        f"Trials: {int(n_trials)} | fixed_params: {fixed if fixed else 'None'}\n\n"
        f"提示: 建议用更大样本的体检验证确认后，再更新 config.yaml。"
    )
    yield msg, best_df


# =========================
# App Layout
# =========================


with gr.Blocks(
    title="买点选股平台 (规则 + AI + EV + 验证闭环)",
) as demo:
    gr.Markdown(
        """
        # 买点选股平台

        用 **规则 + AI 门控 + 期望值 EV + 风险约束** 精准筛选处于买点的股票，并通过 **体检验证** 提升稳健性。
        """
    )

    with gr.Row():
        status_html = gr.HTML(_status_html())
        btn_refresh_status = gr.Button("刷新状态", variant="secondary")

    btn_refresh_status.click(fn=ui_refresh_status, inputs=[], outputs=[status_html])

    with gr.Tabs():
        with gr.Tab("1) Setup"):
            gr.Markdown("## 数据与股票池")
            setup_msg = gr.Markdown()
            with gr.Row():
                btn_update_pool = gr.Button("更新股票池 (stock-list.csv)", variant="primary")
                btn_update_data = gr.Button("增量更新历史数据", variant="primary")

            btn_update_pool.click(fn=ui_update_stock_pool, inputs=[], outputs=[setup_msg, status_html])
            btn_update_data.click(fn=ui_update_kline_data, inputs=[], outputs=[setup_msg, status_html])

            gr.Markdown(
                """
                提示:
                - 首次使用请先执行: 更新股票池 -> 更新历史数据
                - 选股默认使用 `data/stock-list.csv` 作为扫描范围
                """
            )

        with gr.Tab("2) Scan"):
            gr.Markdown("## 买点扫描与筛选")

            scan_state = gr.State(pd.DataFrame())
            scan_msg = gr.Markdown()

            with gr.Row():
                with gr.Column(scale=1):
                    universe_source = gr.Radio(
                        ["stock-list.csv", "custom"],
                        value="stock-list.csv",
                        label="扫描范围",
                    )
                    custom_codes = gr.Textbox(
                        label="自定义代码 (每行一个, 仅在 custom 生效)",
                        placeholder="sh.600000\nsz.000001",
                        lines=6,
                    )
                    target_date = gr.Textbox(
                        label="历史扫描日期 (YYYY-MM-DD，可空表示 latest)",
                        placeholder="2024-05-10",
                    )
                    max_codes = gr.Number(label="最多扫描代码数 (0=全部)", value=0, precision=0)
                    workers = gr.Slider(1, 32, value=8, step=1, label="并发 workers")

                with gr.Column(scale=1):
                    gr.Markdown("### 过滤器 (MVP 刚需)")
                    min_buy_score = gr.Number(label="min_buy_score", value=0.0)
                    min_expected_value_pct = gr.Number(label="min_expected_value_pct (EV%)", value=0.0)
                    min_ai_prob = gr.Slider(0, 1, value=0.5, step=0.01, label="min_ai_prob")
                    market_state_allowlist = gr.CheckboxGroup(
                        choices=MARKET_STATES,
                        value=[],
                        label="允许的 market_state (空=不过滤)",
                    )
                    max_atr_pct = gr.Number(label="max_atr_pct (0=不过滤)", value=0.0)

                    with gr.Row():
                        btn_scan = gr.Button("开始扫描", variant="primary")
                        btn_export = gr.Button("导出当前表格", variant="secondary")

            scan_table = gr.Dataframe(
                label="买点候选列表 (点击行将自动生成交易计划)",
                value=pd.DataFrame(),
                interactive=False,
                wrap=True,
                max_height=520,
            )
            export_file = gr.File(label="导出文件", visible=True)

            btn_scan.click(
                fn=ui_run_scan,
                inputs=[
                    universe_source,
                    custom_codes,
                    target_date,
                    max_codes,
                    workers,
                    min_buy_score,
                    min_expected_value_pct,
                    min_ai_prob,
                    market_state_allowlist,
                    max_atr_pct,
                ],
                outputs=[scan_msg, scan_table, export_file, scan_state],
            )
            btn_export.click(fn=ui_export_selected, inputs=[scan_state], outputs=[export_file])

        with gr.Tab("3) Plan"):
            gr.Markdown("## 标的详情与交易计划")

            with gr.Row():
                plan_code = gr.Textbox(label="code", placeholder="sh.600000")
                plan_date = gr.Textbox(label="date (可空)", placeholder="2024-05-10")
                plan_capital = gr.Number(label="capital", value=100000.0)

            with gr.Row():
                btn_plan = gr.Button("生成交易计划", variant="primary")
                btn_bt = gr.Button("回测该标的", variant="secondary")

            plan_md = gr.Markdown()
            plan_sig = gr.JSON(label="信号原始输出 (scan_today_signal)")

            bt_md = gr.Markdown()
            bt_chart = gr.HTML(label="回测图表")
            bt_trades = gr.Dataframe(label="交易明细", value=pd.DataFrame(), interactive=False)

            btn_plan.click(fn=ui_trade_plan, inputs=[plan_code, plan_date, plan_capital], outputs=[plan_md, plan_sig])
            btn_bt.click(fn=ui_backtest, inputs=[plan_code], outputs=[bt_md, bt_chart, bt_trades])

        with gr.Tab("4) Verify"):
            gr.Markdown("## 策略体检与闭环寻优")

            with gr.Row():
                v_start = gr.Textbox(label="start_date", placeholder="2022-01-01")
                v_end = gr.Textbox(label="end_date", placeholder="2024-01-01")
                v_n = gr.Number(label="n_dates", value=20, precision=0)
            with gr.Row():
                btn_gen_dates = gr.Button("生成验证日期", variant="secondary")
                gen_msg = gr.Markdown()
            v_dates = gr.Textbox(label="验证日期列表 (每行一个 YYYY-MM-DD)", lines=8)

            btn_gen_dates.click(fn=ui_generate_validation_dates, inputs=[v_start, v_end, v_n], outputs=[gen_msg, v_dates])

            with gr.Row():
                v_sample = gr.Number(label="sample_size", value=200, precision=0)
                v_topn = gr.Number(label="max_trades_per_day (topN)", value=10, precision=0)
                v_hold = gr.Number(label="max_hold_days", value=5, precision=0)
            with gr.Row():
                btn_validate = gr.Button("运行策略体检", variant="primary")
                btn_opt = gr.Button("闭环寻优 (Optuna)", variant="secondary")
                v_trials = gr.Number(label="n_trials", value=30, precision=0)

            v_summary = gr.Markdown()
            v_detail = gr.Dataframe(label="体检明细", value=pd.DataFrame(), interactive=False)
            opt_best = gr.Dataframe(label="最优参数", value=pd.DataFrame(), interactive=False)

            btn_validate.click(fn=ui_run_validation, inputs=[v_dates, v_sample, v_topn, v_hold], outputs=[v_summary, v_detail])
            btn_opt.click(fn=ui_optimize_validation, inputs=[v_dates, v_sample, v_topn, v_hold, v_trials], outputs=[v_summary, opt_best])

    # Cross-tab linkage: select a scan row -> fill Plan + generate plan
    scan_table.select(
        fn=ui_on_scan_select,
        inputs=[scan_state, plan_capital],
        outputs=[plan_code, plan_date, plan_md, plan_sig],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
