import pandas as pd
import numpy as np

import os
from quant.config import CONF
from quant.logger import logger
from quant.numba_accelerator import (
    compute_vol_slope_numba,
    create_targets_numba,
    create_multi_class_targets_numba,
    get_numba_status
)

# Cache for the market index dataframe to avoid reading it on every stock
_MARKET_INDEX_CACHE = None

# Numba 状态
_NUMBA_STATUS = get_numba_status()

def _get_market_features() -> pd.DataFrame | None:
    global _MARKET_INDEX_CACHE
    if _MARKET_INDEX_CACHE is not None:
        return _MARKET_INDEX_CACHE
        
    idx_path = os.path.join(CONF.history_data.data_dir, "sh.000001.csv")
    if not os.path.exists(idx_path):
        return None
        
    try:
        idx_df = pd.read_csv(idx_path)
        if "date" in idx_df.columns:
            idx_df["date"] = pd.to_datetime(idx_df["date"])
            idx_df.set_index("date", inplace=True)
            idx_df.sort_index(inplace=True)
        elif "Date" in idx_df.columns:
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            idx_df.set_index("Date", inplace=True)
            idx_df.sort_index(inplace=True)
            
        close_col = 'close' if 'close' in idx_df.columns else ('Close' if 'Close' in idx_df.columns else None)
        if not close_col:
            return None
            
        features = pd.DataFrame(index=idx_df.index)
        close_s = idx_df[close_col]
        features['feat_market_pct_1'] = close_s.pct_change(1)
        features['feat_market_pct_5'] = close_s.pct_change(5)
        features['feat_market_pct_10'] = close_s.pct_change(10)
        features['feat_market_pct_20'] = close_s.pct_change(20)
        
        # Market Regime (Distance to 20 MA)
        ma_20 = close_s.rolling(window=20).mean()
        features['feat_market_bias_20'] = (close_s - ma_20) / (ma_20 + 1e-8)
        
        _MARKET_INDEX_CACHE = features
        return _MARKET_INDEX_CACHE
    except Exception:
        return None

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于已有的指标列，构造用于机器学习的高级时序和截面特征。
    输入 df 已包含 `calculate_indicators` 计算出的基础指标。
    """
    df = df.copy()
    
    # 获取列名，支持大小写兼容（为了适应不同阶段的数据列名）
    close_col = 'close' if 'close' in df.columns else 'Close'
    open_col = 'open' if 'open' in df.columns else 'Open'
    low_col = 'low' if 'low' in df.columns else 'Low'
    high_col = 'high' if 'high' in df.columns else 'High'
    vol_col = 'volume' if 'volume' in df.columns else 'Volume'
    
    close_series = df[close_col]
    
    # 1. 价格动量特征 (Price Momentum)
    # n日收益率
    df['feat_pct_chg_1'] = close_series.pct_change(1)
    df['feat_pct_chg_3'] = close_series.pct_change(3)
    df['feat_pct_chg_5'] = close_series.pct_change(5)
    df['feat_pct_chg_10'] = close_series.pct_change(10)
    df['feat_pct_chg_20'] = close_series.pct_change(20)
    
    # 当前价位于近 20 日高低点的分位值 (Price Percentile)
    rolling_high_20 = close_series.rolling(window=20).max()
    rolling_low_20 = close_series.rolling(window=20).min()
    df['feat_price_pctl_20'] = (close_series - rolling_low_20) / (rolling_high_20 - rolling_low_20 + 1e-8)
    
    # 2. 波动率与形态特征 (Volatility & Candlestick)
    df['feat_body_ratio'] = (close_series - df[open_col]) / close_series
    df['feat_lower_shadow_ratio'] = (df[[open_col, close_col]].min(axis=1) - df[low_col]) / close_series
    df['feat_upper_shadow_ratio'] = (df[high_col] - df[[open_col, close_col]].max(axis=1)) / close_series
    
    # 3. 真实波动率 ATR (Fixed 14)
    if low_col in df.columns and high_col in df.columns:
        tr1 = df[high_col] - df[low_col]
        tr2 = (df[high_col] - close_series.shift()).abs()
        tr3 = (df[low_col] - close_series.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()
        df['feat_atr_ratio'] = atr_14 / close_series
        df['feat_atr_mom_5'] = atr_14.pct_change(5)
    
    # 4. 相对强弱 RSI (Fixed 14)
    delta = close_series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.ewm(com=13, adjust=False).mean()
    roll_down = down.abs().ewm(com=13, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-8)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))
    df['feat_rsi_val'] = rsi_14 / 100.0
    df['feat_rsi_diff'] = df['feat_rsi_val'].diff(1)
        
    # 5. 趋势与乖离率 Bias (Fixed SMA 5, 20)
    sma_5 = close_series.rolling(window=5).mean()
    sma_20 = close_series.rolling(window=20).mean()
    df['feat_bias_s'] = (close_series - sma_5) / (sma_5 + 1e-8)
    df['feat_bias_l'] = (close_series - sma_20) / (sma_20 + 1e-8)
        
    # 6. MACD 衍生 (Fixed 12, 26, 9)
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macdh = macd - macd_signal
    df['feat_macd_norm'] = macdh / close_series
    df['feat_macd_diff'] = df['feat_macd_norm'].diff(1)
        
    # 7. 布林带衍生 (Fixed 20, 2)
    bb_std_20 = close_series.rolling(window=20).std()
    bbl = sma_20 - 2.0 * bb_std_20
    bbu = sma_20 + 2.0 * bb_std_20
    df['feat_bb_pos'] = (close_series - bbl) / (close_series + 1e-8)
    df['feat_bb_width'] = (bbu - bbl) / (close_series + 1e-8)

    # 8. 量能特征 (Volume/Liquidity)
    df['feat_vol_pct_chg_1'] = df[vol_col].pct_change(1)
    df['feat_vol_pct_chg_5'] = df[vol_col].pct_change(5)
    vol_ma_5 = df[vol_col].rolling(window=5).mean()
    df['feat_vol_ratio_5'] = df[vol_col] / (vol_ma_5 + 1e-8)
    
    # 9. Volume Slope (Fixed 5) - 使用 Numba 加速
    vol_arr = df[vol_col].values.astype(float)
    if _NUMBA_STATUS['available']:
        # 使用 Numba 加速版本
        slope_arr = compute_vol_slope_numba(vol_arr, window=5)
        logger.debug("使用 Numba 加速计算 Volume Slope")
    else:
        # 回退到原始版本
        n_days = len(vol_arr)
        slope_arr = np.full(n_days, np.nan)
        x = np.arange(5, dtype=float)
        x_mean = 2.0
        x_var = 10.0
        for i in range(4, n_days):
            y = vol_arr[i-4:i+1]
            y_mean = np.mean(y)
            if y_mean > 0:
                cov = np.sum((x - x_mean) * (y - y_mean)) / 5.0
                slope = cov / x_var
                slope_arr[i] = slope / y_mean
        logger.debug("使用原始 Python 版本计算 Volume Slope")
    df['feat_vol_slope'] = slope_arr
    
    # 10. Realized Volatility (20-day return std)
    daily_ret = close_series.pct_change(1)
    df['feat_realized_vol_20'] = daily_ret.rolling(window=20).std()
    # Volatility regime change (5-day pct change of realized vol)
    df['feat_vol_regime_chg'] = df['feat_realized_vol_20'].pct_change(5)
        
    if 'turn' in df.columns:
        # turn 已经是百分比起步
        df['feat_turnover'] = df['turn'] / 100.0

    # 6. 大盘宏观特征 (Macro Regime)
    market_feats = _get_market_features()
    if market_feats is not None:
        # Assuming df has datetime index or a recognizable Date column
        has_joined = False
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.join(market_feats, how='left')
            has_joined = True
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.merge(market_feats, left_on='date', right_index=True, how='left')
            has_joined = True
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.merge(market_feats, left_on='Date', right_index=True, how='left')
            has_joined = True
            
        if has_joined:
            # fillna with 0 for macro features (e.g. early dates or suspensions)
            m_cols = [c for c in df.columns if c.startswith('feat_market_')]
            df[m_cols] = df[m_cols].fillna(0.0)

    # 提取有用的特征列
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    
    # 10. 截面/相对分位特征 (Self-Relative Rank / Percentiles)
    # Since we process stocks individually, true cross-sectional rank across all stocks is not feasible here.
    # Instead, we calculate the rolling percentile of RSI and Momentum relative to the stock's own recent 252-day history.
    # This gives the AI a sense of "is this unusually high/low for *this* stock?".
    if 'feat_rsi_val' in df.columns:
        df['feat_rsi_rank_252'] = df['feat_rsi_val'].rolling(window=252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
    if 'feat_pct_chg_20' in df.columns:
        df['feat_mom_rank_252'] = df['feat_pct_chg_20'].rolling(window=252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
    # Fill NAs for these new rolling ranks with 0.5 (median)
    rank_cols = ['feat_rsi_rank_252', 'feat_mom_rank_252']
    for c in rank_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.5)
    
    # Add candlestick pattern features
    df = add_candlestick_patterns(df)
    
    return df

def create_targets(df: pd.DataFrame, n_forward_days: int = 5, target_atr_mult: float = 2.0, stop_loss_atr_mult: float = 1.5) -> pd.DataFrame:
    """
    构建预测目标 (Target Y) - 路径依赖版本：
    如果买单在当天收盘买入，未来 n_forward_days 内：
    - 如果先跌破 (当天收盘价 - stop_loss_atr_mult * 当天ATR)，标记为失败 (0)
    - 如果在没有跌破止损线的前提下，最高价触及了 (当天收盘价 + target_atr_mult * 当天ATR)，标记为成功 (1)
    - 其他情况 (既没止赢也没止损)，标记为失败 (0) 以鼓励高效资金周转。
    """
    df = df.copy()

    close_col = 'close' if 'close' in df.columns else 'Close'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'
    atr_cols = [c for c in df.columns if c.startswith('ATRr_')]

    if not atr_cols:
        logger.warning("No ATR column found, falling back to fixed 5% target.")
        # 降级回老逻辑 (fallback to fixed 5% if ATR is missing)
        return _fallback_create_targets(df, n_forward_days, 0.05, close_col, high_col)

    atr_col = atr_cols[0]

    close_series = df[close_col].values
    high_series = df[high_col].values
    low_series = df[low_col].values
    atr_series = df[atr_col].values

    # 使用 Numba 加速版本（如果可用）
    if _NUMBA_STATUS['available']:
        labels = create_targets_numba(
            close_series, high_series, low_series, atr_series,
            n_forward_days, target_atr_mult, stop_loss_atr_mult
        )
        logger.debug("使用 Numba 加速计算 Binary Targets")
    else:
        # 回退到原始版本
        n = len(df)
        labels = np.full(n, np.nan)

        # 向量化处理难以直接实现严密的基于时间的路径依赖（谁先发生），
        # 由于 n_forward_days 很小 (比如5)，可以使用一个微循环来对每个样本盘点未来几天轨迹
        for i in range(n - n_forward_days):
            entry_price = close_series[i]
            curr_atr = atr_series[i]

            # 动态止损与止盈位 (Dynamic Target & Stop based on Volatility)
            initial_sl = entry_price - stop_loss_atr_mult * curr_atr
            target_price = entry_price + target_atr_mult * curr_atr

            is_success = 0

            # 逐日审视未来 N 天
            for lag in range(1, n_forward_days + 1):
                future_idx = int(i + lag)
                day_high = float(high_series[future_idx])
                day_low = float(low_series[future_idx])

                # 悲观假定：如果同一天既碰到了止损又碰到了止盈，我们悲观认为是先触发止损 (洗盘扫损)
                if day_low <= initial_sl:
                    is_success = 0
                    break # 被扫损出局，直接结束对这笔交易的未来观望

                if day_high >= target_price:
                    is_success = 1
                    break # 触及止盈，漂亮赢下一单

            labels[i] = is_success
        logger.debug("使用原始 Python 版本计算 Binary Targets")

    df['label_max_ret_5d'] = labels
    return df

def _fallback_create_targets(df, n_forward_days, target_pct, close_col, high_col):
    close_series = df[close_col]
    high_series = df[high_col]
    future_highest = high_series.shift(-1).rolling(window=n_forward_days, min_periods=1).max()
    future_max_return = (future_highest - close_series) / close_series
    df['label_max_ret_5d'] = (future_max_return >= target_pct).astype(int)
    df.loc[df.index[-n_forward_days:], 'label_max_ret_5d'] = np.nan
    return df

def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick pattern recognition features.
    """
    df = df.copy()
    
    # Get required columns
    close_col = 'close' if 'close' in df.columns else 'Close'
    open_col = 'open' if 'open' in df.columns else 'Open'
    low_col = 'low' if 'low' in df.columns else 'Low'
    high_col = 'high' if 'high' in df.columns else 'High'
    
    close = df[close_col]
    open_p = df[open_col]
    low = df[low_col]
    high = df[high_col]
    
    # Calculate body and shadows
    body = abs(close - open_p)
    lower_shadow = df[[open_col, close_col]].min(axis=1) - low
    upper_shadow = high - df[[open_col, close_col]].max(axis=1)
    body_range = high - low + 1e-8
    
    # 1. Hammer Pattern (potential reversal at bottom)
    # Long lower shadow, small body at top, little upper shadow
    hammer = (
        (lower_shadow >= 2 * body) &
        (body <= body_range * 0.3) &
        (upper_shadow <= body_range * 0.1)
    )
    df['feat_pattern_hammer'] = hammer.astype(float)
    
    # 2. Shooting Star (potential reversal at top)
    # Long upper shadow, small body at bottom, little lower shadow
    shooting_star = (
        (upper_shadow >= 2 * body) &
        (body <= body_range * 0.3) &
        (lower_shadow <= body_range * 0.1)
    )
    df['feat_pattern_shooting_star'] = shooting_star.astype(float)
    
    # 3. Bullish Engulfing (strong reversal signal)
    # Red candle followed by green candle that completely engulfs it
    prev_close = close.shift(1)
    prev_open = open_p.shift(1)
    
    # Previous candle is bearish (close < open)
    prev_bearish = prev_close < prev_open
    # Current candle is bullish (close > open)
    current_bullish = close > open_p
    # Current candle engulfs previous
    engulfing = (
        (open_p <= prev_close) &  # Current open <= previous close
        (close >= prev_open)      # Current close >= previous open
    )
    
    bullish_engulfing = prev_bearish & current_bullish & engulfing
    df['feat_pattern_bullish_engulf'] = bullish_engulfing.astype(float)
    
    # 4. Bearish Engulfing
    prev_bullish = prev_close > prev_open
    current_bearish = close < open_p
    bearish_engulfing = (
        prev_bullish &
        current_bearish &
        (open_p >= prev_close) &
        (close <= prev_open)
    )
    df['feat_pattern_bearish_engulf'] = bearish_engulfing.astype(float)
    
    # 5. Doji (indecision, potential reversal)
    # Very small body
    doji = body <= body_range * 0.1
    df['feat_pattern_doji'] = doji.astype(float)
    
    # 6. Morning Star (3-candle bullish reversal pattern)
    # First: large bearish candle
    # Second: small body (can be red or green)
    # Third: large bullish candle that closes above first candle's midpoint
    if len(df) >= 3:
        first_bearish = close.shift(2) < open_p.shift(2)
        second_small = body.shift(1) <= body_range.shift(1) * 0.3
        third_bullish = close > open_p
        third_recovery = close > (open_p.shift(2) + close.shift(2)) / 2
        
        morning_star = first_bearish & second_small & third_bullish & third_recovery
        df['feat_pattern_morning_star'] = morning_star.astype(float).fillna(0)
    else:
        df['feat_pattern_morning_star'] = 0.0
    
    # 7. Evening Star (3-candle bearish reversal pattern)
    if len(df) >= 3:
        first_bullish = close.shift(2) > open_p.shift(2)
        second_small = body.shift(1) <= body_range.shift(1) * 0.3
        third_bearish = close < open_p
        third_decline = close < (open_p.shift(2) + close.shift(2)) / 2
        
        evening_star = first_bullish & second_small & third_bearish & third_decline
        df['feat_pattern_evening_star'] = evening_star.astype(float).fillna(0)
    else:
        df['feat_pattern_evening_star'] = 0.0
    
    # 8. Three White Soldiers (strong bullish continuation)
    if len(df) >= 3:
        candle1 = close.shift(2) > open_p.shift(2)
        candle2 = close.shift(1) > open_p.shift(1)
        candle3 = close > open_p
        all_bullish = candle1 & candle2 & candle3
        
        # Each candle closes higher than previous
        higher_highs = (close.shift(1) > close.shift(2)) & (close > close.shift(1))
        
        # Small upper shadows
        small_shadows = (
            (upper_shadow.shift(2) <= body_range.shift(2) * 0.3) &
            (upper_shadow.shift(1) <= body_range.shift(1) * 0.3) &
            (upper_shadow <= body_range * 0.3)
        )
        
        three_white = all_bullish & higher_highs & small_shadows
        df['feat_pattern_three_white'] = three_white.astype(float).fillna(0)
    else:
        df['feat_pattern_three_white'] = 0.0
    
    return df

def create_multi_class_targets(df: pd.DataFrame, n_forward_days: int = 5, target_atr_mult: float = 2.0, stop_loss_atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Enhanced multi-class target variable for better prediction granularity.

    Labels:
    0: Loss (hit stop loss)
    1: Small profit (0-3% return)
    2: Medium profit (3-8% return)
    3: Large profit (>8% return)

    This helps the model distinguish between different quality trades.
    """
    df = df.copy()

    close_col = 'close' if 'close' in df.columns else 'Close'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'
    atr_cols = [c for c in df.columns if c.startswith('ATRr_')]

    if not atr_cols:
        # Fallback to simple multi-class if ATR is missing
        logger.warning("No ATR column found, using fixed thresholds for multi-class targets.")
        return _fallback_multi_class_targets(df, n_forward_days, close_col, high_col, low_col)

    atr_col = atr_cols[0]

    close_series = df[close_col].values
    high_series = df[high_col].values
    low_series = df[low_col].values
    atr_series = df[atr_col].values

    # 使用 Numba 加速版本（如果可用）
    if _NUMBA_STATUS['available']:
        labels = create_multi_class_targets_numba(
            close_series, high_series, low_series, atr_series,
            n_forward_days, target_atr_mult, stop_loss_atr_mult
        )
        logger.debug("使用 Numba 加速计算 Multi-class Targets")
    else:
        # 回退到原始版本
        n = len(df)
        labels = np.full(n, np.nan)

        # Define return thresholds
        small_profit_threshold = 0.03  # 3%
        medium_profit_threshold = 0.08  # 8%

        for i in range(n - n_forward_days):
            entry_price = close_series[i]
            curr_atr = atr_series[i]

            # Dynamic stop loss
            stop_loss = entry_price - stop_loss_atr_mult * curr_atr

            # Track maximum gain and whether stop loss was hit
            max_gain = 0.0
            hit_stop = False

            for lag in range(1, n_forward_days + 1):
                future_idx = int(i + lag)
                day_high = float(high_series[future_idx])
                day_low = float(low_series[future_idx])

                # Check stop loss first (pessimistic assumption)
                if day_low <= stop_loss:
                    hit_stop = True
                    break

                # Calculate gain
                gain = (day_high - entry_price) / entry_price
                max_gain = max(max_gain, gain)

                # Early exit if we hit target (2 ATR)
                if gain >= target_atr_mult * curr_atr / entry_price:
                    break

            # Assign label based on outcome
            if hit_stop:
                labels[i] = 0  # Loss
            elif max_gain < small_profit_threshold:
                labels[i] = 1  # Small profit
            elif max_gain < medium_profit_threshold:
                labels[i] = 2  # Medium profit
            else:
                labels[i] = 3  # Large profit
        logger.debug("使用原始 Python 版本计算 Multi-class Targets")

    df['label_multi_class'] = labels
    return df

def _fallback_multi_class_targets(df, n_forward_days, close_col, high_col, low_col):
    """Fallback multi-class targets using fixed thresholds."""
    close_series = df[close_col]
    high_series = df[high_col]
    low_series = df[low_col]
    
    # Calculate future highs and lows
    future_high = high_series.shift(-1).rolling(window=n_forward_days, min_periods=1).max()
    future_low = low_series.shift(-1).rolling(window=n_forward_days, min_periods=1).min()
    
    # Calculate gains and losses
    future_max_gain = (future_high - close_series) / close_series
    future_max_loss = (future_low - close_series) / close_series
    
    labels = np.full(len(df), np.nan)
    
    for i in range(len(df) - n_forward_days):
        if future_max_loss.iloc[i] < -0.02:  # Hit 2% stop loss
            labels[i] = 0
        elif future_max_gain.iloc[i] < 0.03:
            labels[i] = 1
        elif future_max_gain.iloc[i] < 0.08:
            labels[i] = 2
        else:
            labels[i] = 3
    
    df['label_multi_class'] = labels
    return df
