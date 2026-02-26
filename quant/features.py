import pandas as pd
import numpy as np

import os
from quant.config import CONF

# Cache for the market index dataframe to avoid reading it on every stock
_MARKET_INDEX_CACHE = None

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
    
    # 9. Volume Slope (Fixed 5)
    vol_arr = df[vol_col].values.astype(float)
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
    df['feat_vol_slope'] = slope_arr
        
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
    
    return df

def create_targets(df: pd.DataFrame, n_forward_days: int = 5, target_pct: float = 0.05, stop_loss_atr_mult: float = 1.5) -> pd.DataFrame:
    """
    构建预测目标 (Target Y) - 路径依赖版本：
    如果买单在当天收盘买入，未来 n_forward_days 内：
    - 如果先跌破 (当天收盘价 - stop_loss_atr_mult * 当天ATR)，标记为失败 (0)
    - 如果在没有跌破止损线的前提下，最高价触及了 (当天收盘价 * (1 + target_pct))，标记为成功 (1)
    - 其他情况 (既没止赢也没止损)，标记为失败 (0) 以鼓励高效资金周转。
    """
    df = df.copy()
    
    close_col = 'close' if 'close' in df.columns else 'Close'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'
    atr_cols = [c for c in df.columns if c.startswith('ATRr_')]
    
    if not atr_cols:
        # 降级回老逻辑
        return _fallback_create_targets(df, n_forward_days, target_pct, close_col, high_col)
        
    atr_col = atr_cols[0]
    
    close_series = df[close_col].values
    high_series = df[high_col].values
    low_series = df[low_col].values
    atr_series = df[atr_col].values
    
    n = len(df)
    labels = np.full(n, np.nan)
    
    # 向量化处理难以直接实现严密的基于时间的路径依赖（谁先发生），
    # 由于 n_forward_days 很小 (比如5)，可以使用一个微循环来对每个样本盘点未来几天轨迹
    for i in range(n - n_forward_days):
        entry_price = close_series[i]
        initial_sl = entry_price - stop_loss_atr_mult * atr_series[i]
        target_price = entry_price * (1.0 + target_pct)
        
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
