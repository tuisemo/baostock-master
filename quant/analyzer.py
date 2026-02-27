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
    obj = getattr(CONF, 'analyzer', None)
    if obj is None:
        raise AttributeError(f"Configuration 'analyzer' not found in CONF")
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
    if df.empty or len(df) < window:
        df["vol_slope"] = 0.0
        return

    if "volume" not in df.columns:
        df["vol_slope"] = 0.0
        return

    vol = df["volume"].values.astype(float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    if x_var == 0:
        df["vol_slope"] = 0.0
        return

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
    if df.empty or "close" not in df.columns:
        df["momentum_3"] = 0.0
        df["momentum_5"] = 0.0
        df["momentum_10"] = 0.0
        df["momentum_divergence"] = 0.0
        return

    close = df["close"]
    df["momentum_3"] = close.pct_change(3).fillna(0)
    df["momentum_5"] = close.pct_change(5).fillna(0)
    df["momentum_10"] = close.pct_change(10).fillna(0)
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
    try:
        if len(df) >= 20:
            mom_20 = (latest["close"] - df.iloc[-20]["close"]) / df.iloc[-20]["close"]
    except (IndexError, KeyError, ZeroDivisionError):
        mom_20 = 0.0

    sma_s = f"SMA_{ma_short}"
    sma_l = f"SMA_{ma_long}"
    macd_h = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
    rsi_col = f"RSI_{rsi_length}"
    bb_lower = f"BBL_{bbands_length}_{bbands_std}"
    obv_col = "OBV"
    atr_col = f"ATRr_{atr_length}"

    req_cols = [sma_s, sma_l, macd_h, rsi_col, bb_lower, obv_col, atr_col, "low", "close", "volume"]
    if not all(col in df.columns for col in req_cols):
        return {"total_score": 0.0}

    try:
        trend_score = 0.0
        if pd.notna(latest[sma_s]) and pd.notna(latest[sma_l]):
            if latest[sma_s] > latest[sma_l]:
                trend_score += 0.5
        if pd.notna(latest[macd_h]) and pd.notna(prev[macd_h]):
            if latest[macd_h] > 0 and prev[macd_h] <= 0:
                trend_score += 0.5
            elif latest[macd_h] > 0:
                trend_score += 0.3

        reversion_score = 0.0
        if pd.notna(latest[rsi_col]):
            if latest[rsi_col] <= rsi_buy_threshold:
                reversion_score += 0.5
        if (pd.notna(latest["low"]) and pd.notna(latest[bb_lower]) and
            pd.notna(prev["low"]) and pd.notna(prev[bb_lower])):
            if latest["low"] <= latest[bb_lower] or prev["low"] <= prev[bb_lower]:
                if pd.notna(latest["close"]) and pd.notna(latest[bb_lower]):
                    if latest["close"] > latest[bb_lower]:
                        reversion_score += 0.5

        volume_score = 0.0
        if pd.notna(latest[obv_col]) and pd.notna(prev[obv_col]):
            if latest[obv_col] > prev[obv_col]:
                volume_score += 0.5
        if (pd.notna(latest["close"]) and pd.notna(prev["close"]) and
            pd.notna(latest["volume"]) and pd.notna(prev["volume"])):
            if latest["close"] > prev["close"] and latest["volume"] > prev["volume"] * 1.5:
                volume_score += 0.5

        total_score = (
            trend_score * w_trend
            + reversion_score * w_reversion
            + volume_score * w_volume
        )

        if pd.notna(latest["close"]) and pd.notna(latest[atr_col]):
            stop_loss = latest["close"] - (atr_multiplier * latest[atr_col])
        else:
            stop_loss = latest["close"] if pd.notna(latest["close"]) else 0.0

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


def classify_market_state(index_df: pd.DataFrame, lookback_days: int = 60) -> str:
    """
    Classify market state into 5 categories:
    - strong_bull: Strong uptrend with low volatility
    - weak_bull: Mild uptrend
    - sideways: Range-bound with low volatility
    - weak_bear: Mild downtrend
    - strong_bear: Strong downtrend with high volatility
    """
    if index_df is None or len(index_df) < lookback_days:
        return "sideways"

    # Use latest data
    recent = index_df.tail(lookback_days).copy()

    # Validate required column
    if 'close' not in recent.columns:
        return "sideways"

    # Calculate moving averages
    ma20 = recent['close'].rolling(window=20).mean()
    ma60 = recent['close'].rolling(window=60).mean()

    # Check if MA60 has valid value
    if pd.isna(ma60.iloc[-1]) or pd.isna(ma20.iloc[-1]):
        return "sideways"

    # Trend strength (distance between MA20 and MA60)
    try:
        trend_strength = (ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]
    except ZeroDivisionError:
        return "sideways"

    # Volatility (20-day return standard deviation)
    returns = recent['close'].pct_change().dropna()
    if len(returns) < 20:
        return "sideways"
    volatility = returns.tail(20).std()

    # Rate of change (20-day return)
    if len(recent) < 20:
        return "sideways"
    try:
        roc_20 = (recent['close'].iloc[-1] - recent['close'].iloc[-20]) / recent['close'].iloc[-20]
    except ZeroDivisionError:
        return "sideways"

    # Classification logic
    if trend_strength > 0.02 and volatility < 0.02 and roc_20 > 0.05:
        return "strong_bull"
    elif trend_strength > 0.005:
        return "weak_bull"
    elif -0.005 <= trend_strength <= 0.005 and volatility < 0.025:
        return "sideways"
    elif trend_strength < -0.005 and volatility < 0.03:
        return "weak_bear"
    elif trend_strength < -0.02 or (trend_strength < -0.01 and volatility > 0.03):
        return "strong_bear"
    else:
        return "sideways"


def get_dynamic_params(base_params: StrategyParams, market_state: str) -> StrategyParams:
    """
    Dynamically adjust strategy parameters based on market state.
    """
    from dataclasses import replace
    try:
        params = replace(base_params)
    except Exception:
        # Fallback: try to convert to dict and recreate
        try:
            params = StrategyParams(**base_params.__dict__)
        except Exception as e:
            logger.error(f"Failed to create params copy: {e}")
            raise

    # Market state adjustments
    state_adjustments = {
        'strong_bull': {
            'position_size': 1.3,
            'ai_prob_threshold': -0.05,
            'max_hold_days': 5,
            'trail_atr_mult': 2.2,
            'take_profit_pct': 0.001,  # Increase by 1%
        },
        'weak_bull': {
            'position_size': 1.1,
            'ai_prob_threshold': -0.02,
            'max_hold_days': 2,
            'trail_atr_mult': 2.0,
        },
        'sideways': {
            'position_size': 1.0,
            'ai_prob_threshold': 0.0,
            'max_hold_days': 0,
        },
        'weak_bear': {
            'position_size': 0.8,
            'ai_prob_threshold': 0.08,
            'max_hold_days': -2,
            'trail_atr_mult': 1.6,
        },
        'strong_bear': {
            'position_size': 0.6,
            'ai_prob_threshold': 0.12,
            'max_hold_days': -5,
            'trail_atr_mult': 1.4,
            'take_profit_pct': -0.002,  # Decrease by 0.2%
        },
    }

    # Apply adjustments
    adjustments = state_adjustments.get(market_state, {})
    for key, delta in adjustments.items():
        current_value = getattr(params, key, None)
        if current_value is not None:
            if isinstance(current_value, float):
                # Add delta (or multiply for position_size)
                if key == 'position_size':
                    setattr(params, key, min(0.25, max(0.02, current_value * delta)))
                elif key in ['ai_prob_threshold', 'take_profit_pct']:
                    setattr(params, key, current_value + delta)
                else:
                    setattr(params, key, current_value + delta)
            elif isinstance(current_value, int):
                setattr(params, key, current_value + int(delta))

    return params


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
            
        # Add sector rotation signals
        sector_signals = calculate_sector_rotation_signals(res_df)
        res_df = pd.merge(res_df, sector_signals, on='code', how='left')
            
        res_df = res_df.sort_values("total_score", ascending=False)

        date_str = datetime.now().strftime("%Y%m%d")
        out_filename = f"selected_stocks_{date_str}.csv"
        res_df.to_csv(out_filename, index=False, encoding="utf-8-sig")
        logger.info(f"高分优选股列表已保存至: {out_filename}")
    else:
        logger.info("今日没有匹配到强势标的，建议空仓观望。")

def calculate_sector_rotation_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sector rotation signals based on stock codes.
    Group stocks by exchange and market cap, then calculate relative strength.
    """
    if df.empty or 'code' not in df.columns:
        return pd.DataFrame()

    sector_data = []

    # Group by sector (using stock code prefix)
    def get_sector(code: str) -> str:
        if code.startswith("sh.6"):
            return "shanghai_main"
        elif code.startswith("sz.00"):
            return "shenzhen_main"
        elif code.startswith("sz.30"):
            return "shenzhen_chi_next"
        else:
            return "other"

    # Add sector to dataframe temporarily
    df = df.copy()
    df['sector'] = df['code'].apply(get_sector)

    # Calculate sector performance metrics
    for sector in df['sector'].unique():
        sector_stocks = df[df['sector'] == sector]

        if len(sector_stocks) > 0:
            # Average metrics for the sector
            avg_mom = sector_stocks['mom_20'].mean() if 'mom_20' in sector_stocks.columns else 0.0
            avg_rsi = sector_stocks['rsi'].mean() if 'rsi' in sector_stocks.columns else 50.0
            avg_score = sector_stocks['total_score'].mean() if 'total_score' in sector_stocks.columns else 0.0

            # Store sector data (keep total_score for relative strength calculation)
            for _, stock in sector_stocks.iterrows():
                sector_data.append({
                    'code': stock.get('code', ''),
                    'total_score': stock.get('total_score', 0),
                    'sector': sector,
                    'sector_avg_mom': avg_mom,
                    'sector_avg_rsi': avg_rsi,
                    'sector_avg_score': avg_score,
                })

    if not sector_data:
        return pd.DataFrame()

    # Create sector signals dataframe
    sector_df = pd.DataFrame(sector_data)

    # Calculate relative strength within sector
    sector_df['stock_sector_relative_strength'] = (
        sector_df['total_score'] - sector_df['sector_avg_score']
    )

    # Drop total_score from sector signals to avoid conflict with original data
    sector_df = sector_df.drop(columns=['total_score'], errors='ignore')

    # Identify top performing sectors
    if not sector_df.empty and 'sector_avg_score' in sector_df.columns:
        sector_performance = sector_df.groupby('sector')['sector_avg_score'].mean().sort_values(ascending=False)
        top_sectors = sector_performance.head(2).index.tolist()
        bottom_sectors = sector_performance.tail(2).index.tolist()

        sector_df['sector_rotation_signal'] = sector_df['sector'].apply(
            lambda x: 1 if x in top_sectors else (-1 if x in bottom_sectors else 0)
        )

    return sector_df.drop(columns=['sector'], errors='ignore')
