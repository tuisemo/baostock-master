"""
统一的市场状态分类模块

本模块提供市场状态分类的统一接口，整合了以下功能：
- classify_market_state: 市场状态分类函数
- get_market_state_thresholds: 获取市场状态阈值
- get_market_index: 大盘指数数据获取（带缓存）
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.core.cache import GLOBAL_CACHE


# ===== 市场状态阈值配置 =====
MARKET_STATE_THRESHOLDS = {
    "strong_bull": {
        "ai_threshold": 0.30,
        "trend_strength_min": 0.02,
        "volatility_max": 0.015,
        "roc_min": 0.05,
        "risk_level": 2,
    },
    "bull_momentum": {
        "ai_threshold": 0.32,
        "trend_strength_min": 0.02,
        "volatility_max": 0.015,
        "roc_min": 0.05,
        "momentum_acceleration_min": 0.001,
        "volume_ratio_min": 1.5,
        "risk_level": 1,
    },
    "bull_volume": {
        "ai_threshold": 0.31,
        "trend_strength_min": 0.02,
        "volatility_max": 0.015,
        "roc_min": 0.05,
        "volume_ratio_min": 1.5,
        "risk_level": 3,
    },
    "weak_bull": {
        "ai_threshold": 0.35,
        "trend_strength_min": 0.02,
        "risk_level": 4,
    },
    "sideways_low_vol": {
        "ai_threshold": 0.35,
        "volatility_max": 0.025,
        "risk_level": 5,
    },
    "sideways_high_vol": {
        "ai_threshold": 0.40,
        "volatility_min": 0.025,
        "risk_level": 6,
    },
    "weak_bear": {
        "ai_threshold": 0.40,
        "trend_strength_max": -0.02,
        "risk_level": 7,
    },
    "strong_bear": {
        "ai_threshold": 0.45,
        "trend_strength_max": -0.02,
        "volatility_min": 0.025,
        "risk_level": 8,
    },
    "bear_momentum": {
        "ai_threshold": 0.45,
        "trend_strength_max": -0.02,
        "momentum_acceleration_max": -0.001,
        "risk_level": 9,
    },
    "bear_panic": {
        "ai_threshold": 0.50,
        "trend_strength_max": -0.02,
        "volatility_min": 0.03,
        "volume_ratio_min": 1.5,
        "risk_level": 10,
    },
}

# 旧版阈值（用于向后兼容）
DEFAULT_THRESHOLDS = {
    "trend_strength_bull": 0.02,
    "trend_strength_bear": -0.02,
    "volatility_high": 0.025,
    "volatility_low": 0.015,
    "roc_strong": 0.05,
    "volume_ratio_high": 1.5,
}


def get_market_state_thresholds(state: Optional[str] = None) -> Dict[str, float]:
    """
    获取市场状态阈值

    Args:
        state: 市场状态字符串，如果为None则返回所有状态的阈值汇总

    Returns:
        如果state指定，返回该状态的阈值字典；
        如果state为None，返回默认阈值字典
    """
    if state is None:
        return DEFAULT_THRESHOLDS.copy()

    return MARKET_STATE_THRESHOLDS.get(state, DEFAULT_THRESHOLDS.copy())


def get_market_state_risk_level(market_state: str) -> int:
    """
    获取市场状态风险等级

    Args:
        market_state: 市场状态字符串

    Returns:
        风险等级 (1-10, 1为最低风险，10为最高风险)
    """
    thresholds = get_market_state_thresholds(market_state)
    return int(thresholds.get("risk_level", 5))


def calculate_adaptive_thresholds(
    index_df: pd.DataFrame, lookback_days: int = 252
) -> Dict[str, float]:
    """
    计算自适应阈值（基于历史数据分位数）

    Args:
        index_df: 指数数据DataFrame
        lookback_days: 回看天数

    Returns:
        自适应阈值字典
    """
    if index_df is None or len(index_df) < lookback_days:
        return DEFAULT_THRESHOLDS.copy()

    recent = index_df.tail(lookback_days).copy()
    if "close" not in recent.columns:
        return DEFAULT_THRESHOLDS.copy()

    ma20 = recent["close"].rolling(window=20).mean()
    ma60 = recent["close"].rolling(window=60).mean()
    trend_strength = (ma20 - ma60) / ma60
    returns = recent["close"].pct_change()
    rolling_vol = returns.rolling(window=20).std()
    roc_20 = recent["close"].pct_change(20)
    vol_ma20 = (
        recent["volume"].rolling(window=20).mean()
        if "volume" in recent.columns
        else None
    )
    vol_ratio = (
        recent["volume"] / vol_ma20 if vol_ma20 is not None else pd.Series([1.0] * len(recent))
    )

    thresholds = {
        "trend_strength_bull": trend_strength.dropna().quantile(0.70),
        "trend_strength_bear": trend_strength.dropna().quantile(0.30),
        "volatility_high": rolling_vol.dropna().quantile(0.70),
        "volatility_low": rolling_vol.dropna().quantile(0.30),
        "roc_strong": roc_20.dropna().quantile(0.70),
        "volume_ratio_high": vol_ratio.dropna().quantile(0.70),
    }

    # 限制阈值范围
    thresholds["trend_strength_bull"] = max(0.01, min(0.05, thresholds["trend_strength_bull"]))
    thresholds["trend_strength_bear"] = max(-0.05, min(-0.01, thresholds["trend_strength_bear"]))
    thresholds["volatility_high"] = max(0.015, min(0.05, thresholds["volatility_high"]))
    thresholds["volatility_low"] = max(0.005, min(0.025, thresholds["volatility_low"]))
    thresholds["roc_strong"] = max(0.03, min(0.10, thresholds["roc_strong"]))
    thresholds["volume_ratio_high"] = max(1.2, min(2.0, thresholds["volume_ratio_high"]))

    return thresholds


def default_thresholds() -> Dict[str, float]:
    """返回默认阈值（向后兼容）"""
    return DEFAULT_THRESHOLDS.copy()


def classify_market_state(
    index_df: pd.DataFrame, lookback_days: int = 60, use_adaptive_thresholds: bool = True
) -> str:
    """
    将市场状态分类为10种类型之一

    市场状态分类：
    - strong_bull: 强势牛市（强趋势+低波动+高动量）
    - bull_momentum: 牛市动量（强势牛市+成交量放大+动量加速）
    - bull_volume: 牛市放量（强势牛市+成交量放大）
    - weak_bull: 弱牛市（有趋势但不符合强势条件）
    - sideways_low_vol: 盘整低波动（无明显趋势+低波动）
    - sideways_high_vol: 盘整高波动（无明显趋势+高波动）
    - weak_bear: 弱熊市（有下降趋势但不符合强势条件）
    - strong_bear: 强势熊市（强下降趋势+高波动）
    - bear_momentum: 熊市动量（强势熊市+动量加速）
    - bear_panic: 恐慌下跌（强下降趋势+高波动+高成交量）

    Args:
        index_df: 包含OHLCV数据的DataFrame
        lookback_days: 回看天数
        use_adaptive_thresholds: 是否使用自适应阈值

    Returns:
        市场状态字符串
    """
    if index_df is None or len(index_df) < lookback_days:
        return "sideways_low_vol"

    recent = index_df.tail(lookback_days).copy()
    if "close" not in recent.columns:
        return "sideways_low_vol"

    ma20 = recent["close"].rolling(window=20).mean()
    ma60 = recent["close"].rolling(window=60).mean()

    if pd.isna(ma60.iloc[-1]) or pd.isna(ma20.iloc[-1]):
        return "sideways_low_vol"

    try:
        trend_strength = (ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]
    except ZeroDivisionError:
        return "sideways_low_vol"

    returns = recent["close"].pct_change().dropna()
    if len(returns) < 20:
        return "sideways_low_vol"

    volatility = returns.tail(20).std()
    if len(recent) < 20:
        return "sideways_low_vol"

    try:
        roc_20 = (recent["close"].iloc[-1] - recent["close"].iloc[-20]) / recent["close"].iloc[-20]
    except ZeroDivisionError:
        return "sideways_low_vol"

    # 动量加速
    if len(returns) >= 15:
        mom_5_short = recent["close"].pct_change(5).tail(10).mean()
        mom_5_long = recent["close"].pct_change(5).head(10).mean()
        mom_acceleration = mom_5_short - mom_5_long
    else:
        mom_acceleration = 0.0

    # 成交量比率
    volume_ratio = 1.0
    if "volume" in recent.columns:
        vol_ma20 = recent["volume"].rolling(window=20).mean()
        if not pd.isna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] > 0:
            volume_ratio = recent["volume"].iloc[-1] / vol_ma20.iloc[-1]

    thresholds = (
        calculate_adaptive_thresholds(index_df, 252)
        if use_adaptive_thresholds
        else default_thresholds()
    )

    # 牛市判断
    if trend_strength > thresholds["trend_strength_bull"]:
        if (
            trend_strength > thresholds["trend_strength_bull"]
            and volatility < thresholds["volatility_low"]
            and roc_20 > thresholds["roc_strong"]
        ):
            if mom_acceleration > 0.001 and volume_ratio > thresholds["volume_ratio_high"]:
                return "bull_momentum"
            elif volume_ratio > thresholds["volume_ratio_high"]:
                return "bull_volume"
            else:
                return "strong_bull"
        else:
            return "weak_bull"

    # 熊市判断
    elif trend_strength < thresholds["trend_strength_bear"]:
        if trend_strength < thresholds["trend_strength_bear"] or (
            trend_strength < -0.01 and volatility > thresholds["volatility_high"]
        ):
            if volume_ratio > thresholds["volume_ratio_high"] and volatility > 0.03:
                return "bear_panic"
            elif mom_acceleration < -0.001:
                return "bear_momentum"
            else:
                return "strong_bear"
        else:
            return "weak_bear"

    # 盘整判断
    else:
        if volatility > thresholds["volatility_high"]:
            return "sideways_high_vol"
        else:
            return "sideways_low_vol"


def get_market_state_dimensions(
    index_df: pd.DataFrame, lookback_days: int = 60
) -> Dict[str, float]:
    """
    获取市场状态维度指标

    Args:
        index_df: 指数数据DataFrame
        lookback_days: 回看天数

    Returns:
        包含趋势强度、波动率、动量、成交量比率的字典
    """
    if index_df is None or len(index_df) < lookback_days:
        return {}

    recent = index_df.tail(lookback_days).copy()
    if "close" not in recent.columns:
        return {}

    ma20 = recent["close"].rolling(window=20).mean()
    ma60 = recent["close"].rolling(window=60).mean()
    trend_strength = (
        (ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]
        if not pd.isna(ma60.iloc[-1])
        else 0.0
    )

    returns = recent["close"].pct_change().dropna()
    volatility = returns.tail(20).std() if len(returns) >= 20 else 0.0

    roc_20 = recent["close"].pct_change(20).iloc[-1] if len(recent) >= 20 else 0.0

    volume_ratio = 1.0
    if "volume" in recent.columns:
        vol_ma20 = recent["volume"].rolling(window=20).mean()
        if not pd.isna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] > 0:
            volume_ratio = recent["volume"].iloc[-1] / vol_ma20.iloc[-1]

    return {
        "trend_strength": trend_strength,
        "volatility": volatility,
        "momentum_20d": roc_20,
        "volume_ratio": volume_ratio,
    }


def get_market_index() -> Optional[pd.DataFrame]:
    """
    获取大盘指数数据（带缓存）

    加载并预处理上证指数数据，计算技术指标和市场状态。
    使用全局缓存避免重复加载。

    Returns:
        包含市场数据的DataFrame，失败时返回None
    """
    if GLOBAL_CACHE.market_index is not None:
        return GLOBAL_CACHE.market_index

    file_path = os.path.join(CONF.history_data.data_dir, "sh.000001.csv")
    if not os.path.exists(file_path):
        logger.debug(f"大盘指数文件不存在: {file_path}")
        return None

    df_idx = pd.read_csv(file_path)
    if df_idx.empty or "date" not in df_idx.columns:
        return None

    df_idx.rename(columns={"date": "Date"}, inplace=True)
    df_idx["Date"] = pd.to_datetime(df_idx["Date"])
    df_idx.set_index("Date", inplace=True)
    df_idx.sort_index(inplace=True)

    close_col = "close" if "close" in df_idx.columns else "Close"
    close_s = pd.to_numeric(df_idx[close_col], errors="coerce").astype(float)

    # 基础趋势指标
    ma20 = close_s.rolling(window=20).mean()
    ma60 = close_s.rolling(window=60).mean()
    df_idx["MA20"] = ma20
    df_idx["MA60"] = ma60
    df_idx["market_uptrend"] = close_s > ma20

    # 使用默认阈值预计算市场状态
    try:
        thr = default_thresholds()
    except Exception:
        thr = DEFAULT_THRESHOLDS

    trend_strength = (ma20 - ma60) / ma60.replace(0, np.nan)
    returns = close_s.pct_change()
    volatility = returns.rolling(window=20).std()
    roc_20 = close_s.pct_change(20)

    # 动量加速代理
    mom5 = close_s.pct_change(5)
    mom5_short = mom5.rolling(window=10).mean()
    mom5_long = mom5.shift(50).rolling(window=10).mean()
    mom_acc = mom5_short - mom5_long

    # 成交量比率
    vol_col = (
        "volume"
        if "volume" in df_idx.columns
        else ("Volume" if "Volume" in df_idx.columns else None)
    )
    if vol_col is not None:
        vol_s = pd.to_numeric(df_idx[vol_col], errors="coerce").astype(float)
        vol_ma20 = vol_s.rolling(window=20).mean()
        volume_ratio = vol_s / vol_ma20.replace(0, np.nan)
    else:
        volume_ratio = pd.Series(1.0, index=df_idx.index, dtype=float)

    # 预计算市场状态
    state = pd.Series("sideways_low_vol", index=df_idx.index, dtype=object)

    bull = trend_strength > float(thr.get("trend_strength_bull", 0.02))
    bear = trend_strength < float(thr.get("trend_strength_bear", -0.02))
    sideways = ~(bull | bear)

    # 盘整状态
    sideways_high_vol = sideways & (volatility > float(thr.get("volatility_high", 0.025)))
    state[sideways_high_vol] = "sideways_high_vol"
    state[sideways & ~sideways_high_vol] = "sideways_low_vol"

    # 牛市状态
    strong_bull_cond = (
        bull
        & (volatility < float(thr.get("volatility_low", 0.015)))
        & (roc_20 > float(thr.get("roc_strong", 0.05)))
    )
    bull_momentum = (
        strong_bull_cond
        & (mom_acc > 0.001)
        & (volume_ratio > float(thr.get("volume_ratio_high", 1.5)))
    )
    bull_volume = (
        strong_bull_cond & ~bull_momentum & (volume_ratio > float(thr.get("volume_ratio_high", 1.5)))
    )
    strong_bull = strong_bull_cond & ~bull_momentum & ~bull_volume
    weak_bull = bull & ~strong_bull_cond

    state[weak_bull] = "weak_bull"
    state[strong_bull] = "strong_bull"
    state[bull_volume] = "bull_volume"
    state[bull_momentum] = "bull_momentum"

    # 熊市状态
    strong_bear_cond = bear & (
        (trend_strength < float(thr.get("trend_strength_bear", -0.02)))
        | ((trend_strength < -0.01) & (volatility > float(thr.get("volatility_high", 0.025))))
    )
    bear_panic = (
        strong_bear_cond
        & (volume_ratio > float(thr.get("volume_ratio_high", 1.5)))
        & (volatility > 0.03)
    )
    bear_momentum = strong_bear_cond & ~bear_panic & (mom_acc < -0.001)
    strong_bear = strong_bear_cond & ~bear_panic & ~bear_momentum
    weak_bear = bear & ~strong_bear_cond

    state[weak_bear] = "weak_bear"
    state[strong_bear] = "strong_bear"
    state[bear_momentum] = "bear_momentum"
    state[bear_panic] = "bear_panic"

    df_idx["market_state"] = state
    df_idx["market_volatility"] = volatility

    GLOBAL_CACHE.market_index = df_idx
    return GLOBAL_CACHE.market_index


def clear_market_index_cache() -> None:
    """清除大盘指数缓存（主要用于测试）"""
    GLOBAL_CACHE.clear_market_index()


# 向后兼容：导出classify_market_state_enhanced作为classify_market_state
classify_market_state_enhanced = classify_market_state
