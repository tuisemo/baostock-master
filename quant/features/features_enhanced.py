"""
增强版特征工程模块
在原有 features.py 基础上新增 10+ 维高级特征，提升模型表达能力
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 尝试导入 Numba 加速器
try:
    from quant.infra.numba_accelerator import (
        compute_enhanced_features_numba,
        get_numba_status
    )
    _NUMBA_AVAILABLE = get_numba_status()['available']
except ImportError:
    _NUMBA_AVAILABLE = False


def extract_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取增强版特征，在原有特征基础上新增高级特征

    新增特征类别：
    1. 价格动量特征（3 维）
    2. 波动率特征（3 维）
    3. 量价关系特征（3 维）
    4. 跨截面特征（2 维）
    """
    df = df.copy()
    
    # 获取列名，支持大小写兼容
    close_col = 'close' if 'close' in df.columns else 'Close'
    open_col = 'open' if 'open' in df.columns else 'Open'
    low_col = 'low' if 'low' in df.columns else 'Low'
    high_col = 'high' if 'high' in df.columns else 'High'
    vol_col = 'volume' if 'volume' in df.columns else 'Volume'
    
    close_series = df[close_col]
    n = len(df)
    
    # ========== 1. 价格动量特征 (3 维) ==========
    
    # 动量加速度 (Momentum Acceleration): 动量的一阶导数
    # 捕捉价格变化趋势的加速或减速
    momentum_5 = close_series.pct_change(5)
    momentum_10 = close_series.pct_change(10)
    df['feat_mom_acceleration'] = momentum_5 - momentum_5.shift(1)
    
    # 动量持续性 (Momentum Persistence): 近期动量的一致性
    # 计算过去 5 天动量的标准差，标准差越小表示动量越持续
    df['feat_mom_persistence'] = momentum_5.rolling(window=5).std()
    
    # 相对动量强度 (Relative Momentum Strength): 相对于过去 20 天的动量强度
    # 动量占过去 20 天最高动量的比例
    rolling_max_mom = momentum_10.rolling(window=20).max()
    rolling_min_mom = momentum_10.rolling(window=20).min()
    df['feat_mom_relative_strength'] = (momentum_10 - rolling_min_mom) / (rolling_max_mom - rolling_min_mom + 1e-8)
    
    # ========== 2. 波动率特征 (3 维) ==========
    
    # 历史波动率 (Historical Volatility): 20 日收益率标准差
    daily_ret = close_series.pct_change(1)
    df['feat_volatility_20'] = daily_ret.rolling(window=20).std()
    
    # 波动率分解 (Volatility Decomposition): 短期波动率相对于长期波动率的比值
    vol_short = daily_ret.rolling(window=5).std()
    vol_long = daily_ret.rolling(window=20).std()
    df['feat_volatility_ratio'] = vol_short / (vol_long + 1e-8)
    
    # 波动率变化率 (Volatility Change): 波动率的变化趋势
    df['feat_volatility_change'] = df['feat_volatility_20'].pct_change(5)
    
    # ========== 3. 量价关系特征 (3 维) ==========
    
    # 价量背离 (Price-Volume Divergence): 价格与成交量的背离程度
    # 计算价格变化率和成交量变化率的相关性
    price_change = close_series.pct_change(1)
    vol_change = df[vol_col].pct_change(1)
    df['feat_price_volume_divergence'] = (price_change * vol_change).rolling(window=5).mean()
    
    # 量价协同 (Price-Volume Coordination): 价格上涨时的成交量放大程度
    # 价格上涨且成交量放大的天数比例
    price_up = price_change > 0
    vol_up = vol_change > 0
    df['feat_price_volume_coordination'] = (price_up & vol_up).rolling(window=5).sum() / 5.0
    
    # 放量突破 (Volume Breakout): 成交量突破近期均值的程度
    vol_ma = df[vol_col].rolling(window=20).mean()
    df['feat_volume_breakout'] = df[vol_col] / vol_ma
    
    # ========== 4. 跨截面特征（2 维）==========
    
    # 市场相对强度 (Market Relative Strength): 相对于大盘指数的强弱
    # 这个特征在 merge_market_features 中计算
    
    # 行业相对强度 (Sector Relative Strength): 相对于行业平均的强弱
    # 这个特征在 merge_sector_features 中计算
    
    # 填充 NaN 值
    new_features = [
        'feat_mom_acceleration', 'feat_mom_persistence', 'feat_mom_relative_strength',
        'feat_volatility_20', 'feat_volatility_ratio', 'feat_volatility_change',
        'feat_price_volume_divergence', 'feat_price_volume_coordination', 'feat_volume_breakout'
    ]
    
    for feat in new_features:
        if feat in df.columns:
            df[feat] = df[feat].fillna(0.0)
    
    # Numba 加速版本（如果可用）
    if _NUMBA_AVAILABLE:
        try:
            df = _apply_numba_acceleration(df)
        except Exception:
            pass  # 如果 Numba 失败，使用原始版本
    
    return df


def _apply_numba_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 Numba 加速部分特征计算

    Args:
        df: 原始 DataFrame

    Returns:
        添加了 Numba 加速特征的 DataFrame
    """
    try:
        from quant.infra.numba_accelerator import compute_enhanced_features_numba
        
        close_col = 'close' if 'close' in df.columns else 'Close'
        vol_col = 'volume' if 'volume' in df.columns else 'Volume'
        
        close_arr = df[close_col].values.astype(float)
        vol_arr = df[vol_col].values.astype(float)
        
        # 使用 Numba 加速计算
        mom_acc, mom_persist, vol_change = compute_enhanced_features_numba(close_arr, vol_arr)
        
        # 更新 DataFrame
        df['feat_mom_acceleration'] = mom_acc
        df['feat_mom_persistence'] = mom_persist
        df['feat_volatility_change'] = vol_change
        
    except Exception as e:
        import warnings
        warnings.warn(f"Numba acceleration failed: {e}, using fallback implementation")
    
    return df


def merge_market_features(df: pd.DataFrame, market_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    合并市场特征，添加跨截面特征

    Args:
        df: 股票数据 DataFrame
        market_df: 市场指数 DataFrame（如果为 None，尝试加载）

    Returns:
        合并了市场特征的 DataFrame
    """
    if market_df is None:
        return df
    
    # 尝试获取大盘指数数据
    try:
        # 确保 DataFrame 有日期索引
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
        
        # 计算市场相对强度
        if 'close' in market_df.columns:
            market_close = market_df['close']
            if close_col := 'close' if 'close' in df.columns else 'Close':
                stock_return = df[close_col].pct_change(20)
                market_return = market_close.pct_change(20)
                # 重新对齐索引
                market_return_aligned = market_return.reindex(df.index, method='pad')
                df['feat_market_relative_strength'] = stock_return - market_return_aligned
        else:
            df['feat_market_relative_strength'] = 0.0
            
    except Exception:
        df['feat_market_relative_strength'] = 0.0
    
    # 填充 NaN
    df['feat_market_relative_strength'] = df['feat_market_relative_strength'].fillna(0.0)
    
    return df


def merge_sector_features(df: pd.DataFrame, sector_avg_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    合并行业特征，添加跨截面特征

    Args:
        df: 股票数据 DataFrame
        sector_avg_df: 行业平均数据 DataFrame

    Returns:
        合并了行业特征的 DataFrame
    """
    if sector_avg_df is None:
        df['feat_sector_relative_strength'] = 0.0
        return df
    
    try:
        close_col = 'close' if 'close' in df.columns else 'Close'
        stock_return = df[close_col].pct_change(20)
        
        # 假设 sector_avg_df 包含 'sector_avg_return' 列
        if 'sector_avg_return' in sector_avg_df.columns:
            sector_return_aligned = sector_avg_df['sector_avg_return'].reindex(df.index, method='pad')
            df['feat_sector_relative_strength'] = stock_return - sector_return_aligned
        else:
            df['feat_sector_relative_strength'] = 0.0
            
    except Exception:
        df['feat_sector_relative_strength'] = 0.0
    
    # 填充 NaN
    df['feat_sector_relative_strength'] = df['feat_sector_relative_strength'].fillna(0.0)
    
    return df


def extract_all_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    sector_avg_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    提取所有特征（原始特征 + 增强特征）

    Args:
        df: 原始数据 DataFrame
        market_df: 市场指数 DataFrame（可选）
        sector_avg_df: 行业平均数据 DataFrame（可选）

    Returns:
        包含所有特征的 DataFrame
    """
    # 1. 先调用原始的特征提取函数
    try:
        from quant.features.features import extract_features
        df = extract_features(df)
    except Exception as e:
        import warnings
        warnings.warn(f"Original feature extraction failed: {e}")
    
    # 2. 添加增强特征
    df = extract_enhanced_features(df)
    
    # 3. 合并市场特征
    if market_df is not None:
        df = merge_market_features(df, market_df)
    
    # 4. 合并行业特征
    if sector_avg_df is not None:
        df = merge_sector_features(df, sector_avg_df)
    
    return df
