from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from quant.config import CONF
from quant.logger import logger

try:
    from quant.strategy_params import StrategyParams
except ImportError:
    StrategyParams = None


def _resolve(params: StrategyParams | None, name: str, conf_path: str | None = None):
    """Resolve a parameter from StrategyParams or CONF."""
    if params is not None:
        return getattr(params, name)
    if conf_path is not None:
        obj = CONF
        for part in conf_path.split("."):
            obj = getattr(obj, part)
        return obj
    obj = CONF.analyzer
    return getattr(obj, name)


def calculate_indicators(df: pd.DataFrame, params: StrategyParams | None = None) -> pd.DataFrame:
    """Calculate multi-factor technical indicators."""
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    ma_short = _resolve(params, "ma_short")
    ma_long = _resolve(params, "ma_long")
    macd_fast = _resolve(params, "macd_fast")
    macd_slow = _resolve(params, "macd_slow")
    macd_signal = _resolve(params, "macd_signal")
    rsi_length = _resolve(params, "rsi_length")
    bbands_length = _resolve(params, "bbands_length")
    bbands_std = _resolve(params, "bbands_std")
    atr_length = _resolve(params, "atr_length")

    if len(df) < max(ma_long, bbands_length, atr_length, macd_slow):
        return df

    try:
        df.ta.sma(length=ma_short, append=True)
        df.ta.sma(length=ma_long, append=True)
        df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
        df.ta.rsi(length=rsi_length, append=True)
        
        # Manually compute BBands to avoid pandas-ta std parsing bugs
        sma_bb = df['close'].rolling(window=bbands_length).mean()
        std_bb = df['close'].rolling(window=bbands_length).std()
        df[f'BBL_{bbands_length}_{bbands_std}'] = sma_bb - bbands_std * std_bb
        df[f'BBM_{bbands_length}_{bbands_std}'] = sma_bb
        df[f'BBU_{bbands_length}_{bbands_std}'] = sma_bb + bbands_std * std_bb
        
        df.ta.obv(append=True)
        df.ta.atr(length=atr_length, append=True)

        _add_volume_slope(df)
        _add_momentum_leads(df)

    except Exception as e:
        logger.warning(f"指标计算异常: {e}")

    return df


def _add_volume_slope(df: pd.DataFrame, window: int = 5) -> None:
    """Add normalized linear-regression slope of volume over `window` bars."""
    vol = df["volume"].values.astype(float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    slopes = np.full(len(vol), np.nan)
    for i in range(window - 1, len(vol)):
        y = vol[i - window + 1 : i + 1]
        y_mean = y.mean()
        if y_mean == 0:
            slopes[i] = 0.0
            continue
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        slopes[i] = slope / y_mean

    df["vol_slope"] = slopes


def _add_momentum_leads(df: pd.DataFrame) -> None:
    """Add short-term momentum lead indicators and divergence."""
    close = df["close"]
    df["momentum_3"] = close.pct_change(3)
    df["momentum_5"] = close.pct_change(5)
    df["momentum_10"] = close.pct_change(10)
    df["momentum_divergence"] = df["momentum_3"] - df["momentum_10"]


def score_stock(df: pd.DataFrame, params: StrategyParams | None = None) -> dict:
    """Score a stock on trend / reversion / volume factors."""
    ma_short = _resolve(params, "ma_short")
    ma_long = _resolve(params, "ma_long")
    macd_fast = _resolve(params, "macd_fast")
    macd_slow = _resolve(params, "macd_slow")
    macd_signal = _resolve(params, "macd_signal")
    rsi_length = _resolve(params, "rsi_length")
    rsi_buy_threshold = _resolve(params, "rsi_buy_threshold")
    bbands_length = _resolve(params, "bbands_length")
    bbands_std = _resolve(params, "bbands_std")
    atr_length = _resolve(params, "atr_length")
    atr_multiplier = _resolve(params, "atr_multiplier")

    if params is not None:
        w_trend = params.weight_trend
        w_reversion = params.weight_reversion
        w_volume = params.weight_volume
    else:
        w_trend = CONF.analyzer.weights.trend
        w_reversion = CONF.analyzer.weights.reversion
        w_volume = CONF.analyzer.weights.volume

    if df.empty or len(df) < ma_long:
        return {"total_score": 0.0}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # 20-day momentum for ranking if available
    mom_20 = 0.0
    if len(df) >= 20:
        mom_20 = (latest["close"] - df.iloc[-20]["close"]) / df.iloc[-20]["close"]

    sma_s = f"SMA_{ma_short}"
    sma_l = f"SMA_{ma_long}"
    macd_h = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
    rsi_col = f"RSI_{rsi_length}"
    bb_lower = f"BBL_{bbands_length}_{bbands_std}_{bbands_std}"
    obv_col = "OBV"
    atr_col = f"ATRr_{atr_length}"

    req_cols = [sma_s, sma_l, macd_h, rsi_col, bb_lower, obv_col, atr_col]
    if not all(col in df.columns for col in req_cols):
        return {"total_score": 0.0}

    try:
        trend_score = 0.0
        if latest[sma_s] > latest[sma_l]:
            trend_score += 0.5
        if latest[macd_h] > 0 and prev[macd_h] <= 0:
            trend_score += 0.5
        elif latest[macd_h] > 0:
            trend_score += 0.3

        reversion_score = 0.0
        if latest[rsi_col] <= rsi_buy_threshold:
            reversion_score += 0.5
        if latest["low"] <= latest[bb_lower] or prev["low"] <= prev[bb_lower]:
            if latest["close"] > latest[bb_lower]:
                reversion_score += 0.5

        volume_score = 0.0
        if latest[obv_col] > prev[obv_col]:
            volume_score += 0.5
        if latest["close"] > prev["close"] and latest["volume"] > prev["volume"] * 1.5:
            volume_score += 0.5

        total_score = (
            trend_score * w_trend
            + reversion_score * w_reversion
            + volume_score * w_volume
        )

        stop_loss = latest["close"] - (atr_multiplier * latest[atr_col])

        return {
            "total_score": round(total_score, 3),
            "trend": trend_score,
            "reversion": reversion_score,
            "volume": volume_score,
            "close": latest["close"],
            "stop_loss": round(stop_loss, 2),
            "rsi": round(latest[rsi_col], 2),
            "mom_20": round(mom_20, 4),
            "date": (
                latest["date"].strftime("%Y-%m-%d")
                if pd.api.types.is_datetime64_any_dtype(latest["date"])
                else latest["date"]
            ),
        }
    except Exception:
        return {"total_score": 0.0}


def _analyze_single_file(file_path: str) -> dict | None:
    """Helper function to process a single stock file for parallel execution."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None

        code = os.path.basename(file_path).replace(".csv", "")
        df = calculate_indicators(df)
        score_data = score_stock(df)

        if score_data.get("total_score", 0) >= 0.15:
            res = {"code": code}
            res.update(score_data)
            return res
    except Exception as e:
        logger.debug(f"处理文件异常 {file_path}: {e}")
    return None


def analyze_all_stocks() -> None:
    """Batch analysis entry point using CONF-based defaults."""
    data_files = glob(os.path.join(CONF.history_data.data_dir, "*.csv"))
    data_files = [f for f in data_files if "stock-list.csv" not in f]

    if not data_files:
        logger.error(f"在 {CONF.history_data.data_dir} 未找到相关的历史数据。请先执行 update-data。")
        return

    logger.info(f"开始对 {len(data_files)} 支股票执行多因子综合分析...")

    results = []

    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_single_file, f): f for f in data_files}
        for future in tqdm(as_completed(futures), total=len(data_files), desc="量化分析进度"):
            res = future.result()
            if res is not None:
                results.append(res)

    logger.info(f"分析完成。共有 {len(results)} 支股票满足评分阈值。")

    if results:
        res_df = pd.DataFrame(results)
        
        # Add True Cross-Sectional Ranking for today's snapshot
        if 'rsi' in res_df.columns:
            res_df['rsi_rank'] = res_df['rsi'].rank(pct=True)
        if 'mom_20' in res_df.columns:
            res_df['mom_rank'] = res_df['mom_20'].rank(pct=True)
            
        res_df = res_df.sort_values("total_score", ascending=False)

        date_str = datetime.now().strftime("%Y%m%d")
        out_filename = f"selected_stocks_{date_str}.csv"
        res_df.to_csv(out_filename, index=False, encoding="utf-8-sig")
        logger.info(f"高分优选股列表已保存至: {out_filename}")
    else:
        logger.info("今日没有匹配到强势标的，建议空仓观望。")
