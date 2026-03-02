"""
增强版市场状态分类模块
将市场状态从 5 类扩展至 7 类，提升状态识别精度
"""
import pandas as pd
import numpy as np
from typing import Tuple

from quant.logger import logger


def classify_market_state_enhanced(
    index_df: pd.DataFrame,
    lookback_days: int = 60
) -> str:
    """
    增强版市场状态分类（7 类）

    分类逻辑：
    1. strong_bull: 强牛市 - MA20 > MA60，趋势强度 > 2%，低波动率 < 2%，20日涨幅 > 5%
    2. bull_momentum: 牛市动量 - 强牛市 + 动量加速（价格上涨速度加快）
    3. weak_bull: 弱牛市 - MA20 > MA60，趋势强度 > 0.5%
    4. sideways: 横盘 - 趋势强度在 ±0.5% 之间，低波动率
    5. weak_bear: 弱熊市 - MA20 < MA60，趋势强度 < -0.5%
    6. bear_momentum: 熊市动量 - 弱熊市 + 动量加速（价格下跌速度加快）
    7. strong_bear: 强熊市 - MA20 < MA60，趋势强度 < -2%，高波动率

    Args:
        index_df: 市场指数 DataFrame（必须包含 'close' 列）
        lookback_days: 回看天数

    Returns:
        市场状态字符串
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

    # Momentum acceleration (change in 5-day momentum over 10 days)
    if len(returns) >= 15:
        mom_5_short = recent['close'].pct_change(5).tail(10).mean()
        mom_5_long = recent['close'].pct_change(5).head(10).mean()
        mom_acceleration = mom_5_short - mom_5_long
    else:
        mom_acceleration = 0.0

    # Classification logic (7 classes)
    
    # Bull market conditions
    if trend_strength > 0.005:
        if trend_strength > 0.02 and volatility < 0.02 and roc_20 > 0.05:
            # Check for momentum acceleration
            if mom_acceleration > 0.001:
                return "bull_momentum"
            else:
                return "strong_bull"
        else:
            return "weak_bull"
    
    # Bear market conditions
    elif trend_strength < -0.005:
        if trend_strength < -0.02 or (trend_strength < -0.01 and volatility > 0.03):
            # Check for momentum acceleration (negative acceleration means faster decline)
            if mom_acceleration < -0.001:
                return "bear_momentum"
            else:
                return "strong_bear"
        else:
            return "weak_bear"
    
    # Sideways conditions
    else:
        if volatility < 0.025:
            return "sideways"
        else:
            # High volatility sideways
            return "sideways"


def get_market_state_transitions(
    index_df: pd.DataFrame,
    lookback_days: int = 60
) -> pd.DataFrame:
    """
    计算市场状态的历史变化序列

    Args:
        index_df: 市场指数 DataFrame
        lookback_days: 回看天数

    Returns:
        包含市场状态历史的 DataFrame
    """
    if index_df is None or len(index_df) < lookback_days:
        return pd.DataFrame()

    # 创建滑动窗口计算状态
    states = []
    dates = []

    for i in range(lookback_days, len(index_df) + 1):
        window_df = index_df.iloc[max(0, i - lookback_days):i]
        state = classify_market_state_enhanced(window_df, lookback_days=lookback_days)
        states.append(state)
        if 'date' in index_df.columns:
            dates.append(index_df.iloc[i-1]['date'])
        else:
            dates.append(index_df.index[i-1])

    result_df = pd.DataFrame({
        'date': dates,
        'market_state': states
    })

    # 检测状态转换
    result_df['state_changed'] = result_df['market_state'].ne(
        result_df['market_state'].shift()
    )

    return result_df


def analyze_market_regime(
    index_df: pd.DataFrame,
    lookback_days: int = 60
) -> dict:
    """
    分析当前市场状态的历史特征

    Args:
        index_df: 市场指数 DataFrame
        lookback_days: 回看天数

    Returns:
        市场状态特征字典
    """
    state_history = get_market_state_transitions(index_df, lookback_days)
    
    if state_history.empty:
        return {}

    current_state = state_history['market_state'].iloc[-1]
    state_counts = state_history['market_state'].value_counts()
    transition_count = state_history['state_changed'].sum()

    # 计算状态持续时间
    state_durations = {}
    last_change_idx = 0
    current_duration = 0

    for i, (state, changed) in enumerate(zip(state_history['market_state'], state_history['state_changed'])):
        if changed:
            # 记录上一个状态的持续时间
            if last_change_idx > 0:
                prev_state = state_history['market_state'].iloc[last_change_idx]
                duration = i - last_change_idx
                state_durations[prev_state] = state_durations.get(prev_state, [])
                state_durations[prev_state].append(duration)
            last_change_idx = i
        current_duration = len(state_history) - i

    # 计算平均持续时间
    avg_durations = {
        state: np.mean(durations) if durations else 0
        for state, durations in state_durations.items()
    }

    return {
        'current_state': current_state,
        'state_counts': state_counts.to_dict(),
        'total_transitions': transition_count,
        'avg_state_durations': avg_durations,
        'current_duration_days': current_duration
    }


# 兼容旧版函数
classify_market_state = classify_market_state_enhanced


if __name__ == "__main__":
    # 测试代码
    import os
    
    # 加载上证指数数据
    data_dir = "data"
    idx_path = os.path.join(data_dir, "sh.000001.csv")
    
    if os.path.exists(idx_path):
        idx_df = pd.read_csv(idx_path)
        
        # 分析市场状态
        current_state = classify_market_state_enhanced(idx_df)
        print(f"当前市场状态: {current_state}")
        
        # 分析市场状态特征
        regime = analyze_market_regime(idx_df)
        print(f"\n市场状态分析:")
        print(f"  当前状态: {regime.get('current_state')}")
        print(f"  状态分布: {regime.get('state_counts')}")
        print(f"  状态转换次数: {regime.get('total_transitions')}")
        print(f"  当前状态持续时间: {regime.get('current_duration_days')} 天")
