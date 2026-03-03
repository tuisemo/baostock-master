content = '''"""
Enhanced Market State Classifier v2
10-class market state with multi-dimensional analysis and adaptive thresholds
"""
import pandas as pd
import numpy as np
from typing import Dict
from quant.logger import logger

def calculate_adaptive_thresholds(index_df: pd.DataFrame, lookback_days: int = 252) -> Dict[str, float]:
    if index_df is None or len(index_df) < lookback_days:
        return {'trend_strength_bull': 0.02, 'trend_strength_bear': -0.02, 'volatility_high': 0.025, 'volatility_low': 0.015, 'roc_strong': 0.05, 'volume_ratio_high': 1.5}
    recent = index_df.tail(lookback_days).copy()
    if 'close' not in recent.columns:
        return {'trend_strength_bull': 0.02, 'trend_strength_bear': -0.02, 'volatility_high': 0.025, 'volatility_low': 0.015, 'roc_strong': 0.05, 'volume_ratio_high': 1.5}
    ma20 = recent['close'].rolling(window=20).mean()
    ma60 = recent['close'].rolling(window=60).mean()
    trend_strength = (ma20 - ma60) / ma60
    returns = recent['close'].pct_change()
    rolling_vol = returns.rolling(window=20).std()
    roc_20 = recent['close'].pct_change(20)
    vol_ma20 = recent['volume'].rolling(window=20).mean() if 'volume' in recent.columns else None
    vol_ratio = recent['volume'] / vol_ma20 if vol_ma20 is not None else pd.Series([1.0] * len(recent))
    thresholds = {'trend_strength_bull': trend_strength.dropna().quantile(0.70), 'trend_strength_bear': trend_strength.dropna().quantile(0.30), 'volatility_high': rolling_vol.dropna().quantile(0.70), 'volatility_low': rolling_vol.dropna().quantile(0.30), 'roc_strong': roc_20.dropna().quantile(0.70), 'volume_ratio_high': vol_ratio.dropna().quantile(0.70)}
    thresholds['trend_strength_bull'] = max(0.01, min(0.05, thresholds['trend_strength_bull']))
    thresholds['trend_strength_bear'] = max(-0.05, min(-0.01, thresholds['trend_strength_bear']))
    thresholds['volatility_high'] = max(0.015, min(0.05, thresholds['volatility_high']))
    thresholds['volatility_low'] = max(0.005, min(0.025, thresholds['volatility_low']))
    thresholds['roc_strong'] = max(0.03, min(0.10, thresholds['roc_strong']))
    thresholds['volume_ratio_high'] = max(1.2, min(2.0, thresholds['volume_ratio_high']))
    return thresholds

def default_thresholds() -> Dict[str, float]:
    return {'trend_strength_bull': 0.02, 'trend_strength_bear': -0.02, 'volatility_high': 0.025, 'volatility_low': 0.015, 'roc_strong': 0.05, 'volume_ratio_high': 1.5}

def classify_market_state_enhanced(index_df: pd.DataFrame, lookback_days: int = 60, use_adaptive_thresholds: bool = True) -> str:
    if index_df is None or len(index_df) < lookback_days: return "sideways_low_vol"
    recent = index_df.tail(lookback_days).copy()
    if 'close' not in recent.columns: return "sideways_low_vol"
    ma20 = recent['close'].rolling(window=20).mean()
    ma60 = recent['close'].rolling(window=60).mean()
    if pd.isna(ma60.iloc[-1]) or pd.isna(ma20.iloc[-1]): return "sideways_low_vol"
    try: trend_strength = (ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1]
    except ZeroDivisionError: return "sideways_low_vol"
    returns = recent['close'].pct_change().dropna()
    if len(returns) < 20: return "sideways_low_vol"
    volatility = returns.tail(20).std()
    if len(recent) < 20: return "sideways_low_vol"
    try: roc_20 = (recent['close'].iloc[-1] - recent['close'].iloc[-20]) / recent['close'].iloc[-20]
    except ZeroDivisionError: return "sideways_low_vol"
    if len(returns) >= 15:
        mom_5_short = recent['close'].pct_change(5).tail(10).mean()
        mom_5_long = recent['close'].pct_change(5).head(10).mean()
        mom_acceleration = mom_5_short - mom_5_long
    else: mom_acceleration = 0.0
    volume_ratio = 1.0
    if 'volume' in recent.columns:
        vol_ma20 = recent['volume'].rolling(window=20).mean()
        if not pd.isna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] > 0: volume_ratio = recent['volume'].iloc[-1] / vol_ma20.iloc[-1]
    thresholds = calculate_adaptive_thresholds(index_df, 252) if use_adaptive_thresholds else default_thresholds()
    if trend_strength > thresholds['trend_strength_bull']:
        if trend_strength > thresholds['trend_strength_bull'] and volatility < thresholds['volatility_low'] and roc_20 > thresholds['roc_strong']:
            if mom_acceleration > 0.001 and volume_ratio > thresholds['volume_ratio_high']: return "bull_momentum"
            elif volume_ratio > thresholds['volume_ratio_high']: return "bull_volume"
            else: return "strong_bull"
        else: return "weak_bull"
    elif trend_strength < thresholds['trend_strength_bear']:
        if trend_strength < thresholds['trend_strength_bear'] or (trend_strength < -0.01 and volatility > thresholds['volatility_high']):
            if volume_ratio > thresholds['volume_ratio_high'] and volatility > 0.03: return "bear_panic"
            elif mom_acceleration < -0.001: return "bear_momentum"
            else: return "strong_bear"
        else: return "weak_bear"
    else:
        if volatility > thresholds['volatility_high']: return "sideways_high_vol"
        else: return "sideways_low_vol"

def get_market_state_risk_level(market_state: str) -> int:
    risk_levels = {'bull_momentum': 1, 'strong_bull': 2, 'bull_volume': 3, 'weak_bull': 4, 'sideways_low_vol': 5, 'sideways_high_vol': 6, 'weak_bear': 7, 'strong_bear': 8, 'bear_momentum': 9, 'bear_panic': 10}
    return risk_levels.get(market_state, 5)

def get_market_state_dimensions(index_df: pd.DataFrame, lookback_days: int = 60) -> Dict[str, float]:
    if index_df is None or len(index_df) < lookback_days: return {}
    recent = index_df.tail(lookback_days).copy()
    if 'close' not in recent.columns: return {}
    ma20 = recent['close'].rolling(window=20).mean()
    ma60 = recent['close'].rolling(window=60).mean()
    trend_strength = (ma20.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1] if not pd.isna(ma60.iloc[-1]) else 0.0
    returns = recent['close'].pct_change().dropna()
    volatility = returns.tail(20).std() if len(returns) >= 20 else 0.0
    roc_20 = recent['close'].pct_change(20).iloc[-1] if len(recent) >= 20 else 0.0
    volume_ratio = 1.0
    if 'volume' in recent.columns:
        vol_ma20 = recent['volume'].rolling(window=20).mean()
        if not pd.isna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] > 0: volume_ratio = recent['volume'].iloc[-1] / vol_ma20.iloc[-1]
    return {'trend_strength': trend_strength, 'volatility': volatility, 'momentum_20d': roc_20, 'volume_ratio': volume_ratio}

classify_market_state = classify_market_state_enhanced
'''
with open('quant/market_classifier.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Market classifier v2 written successfully')
