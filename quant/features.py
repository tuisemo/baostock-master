import pandas as pd
import numpy as np

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
    
    # 判断列中是否已有 ATR
    atr_cols = [c for c in df.columns if c.startswith('ATRr_')]
    if atr_cols:
        atr_col = atr_cols[0]
        df['feat_atr_ratio'] = df[atr_col] / close_series
        # ATR 动量（波动率扩张/收缩率）
        df['feat_atr_mom_5'] = df[atr_col].pct_change(5)
    
    # 3. 相对强弱 (RSI 衍生)
    rsi_cols = [c for c in df.columns if c.startswith('RSI_')]
    if rsi_cols:
        rsi_col = rsi_cols[0]
        df['feat_rsi_val'] = df[rsi_col] / 100.0  # 缩放到 0-1
        # RSI 下降速度 (一阶差分)
        df['feat_rsi_diff'] = df[rsi_col].diff(1) / 100.0
        
    # 4. 趋势与乖离率 (Bias, MACD 衍生)
    ma_s_cols = [c for c in df.columns if c.startswith('SMA_5') or c.startswith('SMA_10')]
    if ma_s_cols:
        df['feat_bias_s'] = (close_series - df[ma_s_cols[0]]) / df[ma_s_cols[0]]
        
    ma_l_cols = [c for c in df.columns if c.startswith('SMA_20') or c.startswith('SMA_30')]
    if ma_l_cols:
        df['feat_bias_l'] = (close_series - df[ma_l_cols[0]]) / df[ma_l_cols[0]]
        
    macd_cols = [c for c in df.columns if c.startswith('MACDh_')]
    if macd_cols:
        macd_col = macd_cols[0]
        # 对均线归一化以防止绝对值带来的跨股票差异
        df['feat_macd_norm'] = df[macd_col] / close_series
        df['feat_macd_diff'] = df[macd_col].diff(1) / close_series
        
    # 布林带衍生
    bb_l_cols = [c for c in df.columns if c.startswith('BBL_')]
    bb_u_cols = [c for c in df.columns if c.startswith('BBU_')]
    if bb_l_cols and bb_u_cols:
        bbl = df[bb_l_cols[0]]
        bbu = df[bb_u_cols[0]]
        # 价格相对于布林下轨的位置 (< 0 代表跌穿下轨)
        df['feat_bb_pos'] = (close_series - bbl) / close_series
        # 布林带宽度
        df['feat_bb_width'] = (bbu - bbl) / close_series

    # 5. 量能特征 (Volume/Liquidity)
    df['feat_vol_pct_chg_1'] = df[vol_col].pct_change(1)
    df['feat_vol_pct_chg_5'] = df[vol_col].pct_change(5)
    vol_ma_5 = df[vol_col].rolling(window=5).mean()
    df['feat_vol_ratio_5'] = df[vol_col] / (vol_ma_5 + 1e-8)
    
    if 'vol_slope' in df.columns:
        df['feat_vol_slope'] = df['vol_slope']
        
    if 'turn' in df.columns:
        # turn 已经是百分比起步
        df['feat_turnover'] = df['turn'] / 100.0

    # 前向填充可能存在的少量 NaN（由指标产生的），然后扔掉前面的极早期数据
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    
    return df

def create_targets(df: pd.DataFrame, n_forward_days: int = 5, target_pct: float = 0.05) -> pd.DataFrame:
    """
    构建预测目标 (Target Y)，定义为：如果买单在当天收盘买入，
    未来 n_forward_days 内（从第 1 天到 N 天），最高价能否到达 target_pct 的收益？
    如果达到，设为 1；否则设为 0。
    """
    df = df.copy()
    
    close_col = 'close' if 'close' in df.columns else 'Close'
    high_col = 'high' if 'high' in df.columns else 'High'
    open_col = 'open' if 'open' in df.columns else 'Open' # 保留开盘价作为更合理的介入价模拟也可以，这里默认由于按收盘跑买入信号，买点为次日开盘或其他，此代码为了简单起见，按当日收盘算起点。

    close_series = df[close_col]
    high_series = df[high_col]
    
    # 1. 未来 N 天的最高价
    # 不包括今天，而是从明天开始的 N 天
    future_highest = high_series.shift(-1).rolling(window=n_forward_days, min_periods=1).max()
    
    # 2. 最高可能收益率
    future_max_return = (future_highest - close_series) / close_series
    
    # 3. 二分类目标
    df['label_max_ret_5d'] = (future_max_return >= target_pct).astype(int)
    
    # 4. 把由于 shift 导致最后几天的目标是 NaN 的处理一下（实际会变成负几率或失效）
    df.loc[df.index[-n_forward_days:], 'label_max_ret_5d'] = np.nan
    
    return df
