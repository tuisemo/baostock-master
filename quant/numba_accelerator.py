"""
Numba 加速的计算函数
用于优化性能瓶颈，同时保持向后兼容
"""
import numpy as np
from typing import Tuple

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 创建装饰器的 fallback 版本
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args, **kwargs):
        return range(*args)


def get_numba_status() -> dict:
    """获取 Numba 状态信息"""
    return {
        'available': NUMBA_AVAILABLE,
        'jit_enabled': NUMBA_AVAILABLE
    }


@jit(nopython=True, parallel=False)
def compute_vol_slope_numba(vol_arr: np.ndarray, window: int = 5) -> np.ndarray:
    """
    使用 numba 加速的成交量斜率计算

    Args:
        vol_arr: 成交量数组
        window: 窗口大小（固定为 5）

    Returns:
        斜率数组
    """
    n = len(vol_arr)
    slopes = np.full(n, np.nan)

    if n < window:
        return slopes

    # 注意：为了保持与原始 Python 版本的兼容性，使用相同的硬编码值
    # 原始版本使用 x = [0, 1, 2, 3, 4]，均值是 2.0
    # x_var 硬编码为 10.0（虽然数学上应该是 2.0）
    x = np.empty(window, dtype=np.float64)
    x_mean = 2.0
    x_var = 10.0  # 硬编码，与原始版本保持一致

    for j in range(window):
        x[j] = j - x_mean

    for i in range(window - 1, n):
        # 计算窗口均值
        y_mean = 0.0
        for j in range(window):
            y_mean += vol_arr[i - window + 1 + j]
        y_mean /= window

        if y_mean > 0:
            # 计算协方差
            covariance = 0.0
            for j in range(window):
                y_j = vol_arr[i - window + 1 + j] - y_mean
                covariance += x[j] * y_j

            # 样本协方差（除以 window）
            covariance /= window

            # 斜率 = Cov(x, y) / Var(x)
            slope = covariance / x_var

            # 归一化到 y 的均值
            slopes[i] = slope / y_mean

    return slopes


@jit(nopython=True, parallel=True)
def create_targets_numba(
    close_series: np.ndarray,
    high_series: np.ndarray,
    low_series: np.ndarray,
    atr_series: np.ndarray,
    n_forward_days: int,
    target_atr_mult: float,
    stop_loss_atr_mult: float
) -> np.ndarray:
    """
    使用 numba 加速的目标变量生成

    Args:
        close_series: 收盘价数组
        high_series: 最高价数组
        low_series: 最低价数组
        atr_series: ATR 数组
        n_forward_days: 向前看的天数
        target_atr_mult: 目标ATR倍数
        stop_loss_atr_mult: 止损ATR倍数

    Returns:
        标签数组 (0=失败, 1=成功)
    """
    n = len(close_series)
    labels = np.full(n, np.nan)

    if n < n_forward_days:
        return labels

    for i in prange(n - n_forward_days):
        entry_price = close_series[i]
        curr_atr = atr_series[i]

        initial_sl = entry_price - stop_loss_atr_mult * curr_atr
        target_price = entry_price + target_atr_mult * curr_atr

        is_success = 0

        for lag in range(1, n_forward_days + 1):
            future_idx = i + lag
            day_high = high_series[future_idx]
            day_low = low_series[future_idx]

            # 悲观假定：如果同一天既碰到了止损又碰到了止盈，我们悲观认为是先触发止损
            if day_low <= initial_sl:
                is_success = 0
                break

            if day_high >= target_price:
                is_success = 1
                break

        labels[i] = is_success

    return labels


@jit(nopython=True, parallel=True)
def create_multi_class_targets_numba(
    close_series: np.ndarray,
    high_series: np.ndarray,
    low_series: np.ndarray,
    atr_series: np.ndarray,
    n_forward_days: int,
    target_atr_mult: float,
    stop_loss_atr_mult: float
) -> np.ndarray:
    """
    使用 numba 加速的多分类目标变量生成

    Labels:
    0: Loss (hit stop loss)
    1: Small profit (0-3% return)
    2: Medium profit (3-8% return)
    3: Large profit (>8% return)

    Args:
        close_series: 收盘价数组
        high_series: 最高价数组
        low_series: 最低价数组
        atr_series: ATR 数组
        n_forward_days: 向前看的天数
        target_atr_mult: 目标ATR倍数
        stop_loss_atr_mult: 止损ATR倍数

    Returns:
        标签数组 (0-3)
    """
    n = len(close_series)
    labels = np.full(n, np.nan)

    if n < n_forward_days:
        return labels

    small_profit_threshold = 0.03
    medium_profit_threshold = 0.08

    for i in prange(n - n_forward_days):
        entry_price = close_series[i]
        curr_atr = atr_series[i]

        stop_loss = entry_price - stop_loss_atr_mult * curr_atr

        max_gain = 0.0
        hit_stop = False

        for lag in range(1, n_forward_days + 1):
            future_idx = i + lag
            day_high = high_series[future_idx]
            day_low = low_series[future_idx]

            if day_low <= stop_loss:
                hit_stop = True
                break

            gain = (day_high - entry_price) / entry_price
            max_gain = max(max_gain, gain)

            if gain >= target_atr_mult * curr_atr / entry_price:
                break

        if hit_stop:
            labels[i] = 0
        elif max_gain < small_profit_threshold:
            labels[i] = 1
        elif max_gain < medium_profit_threshold:
            labels[i] = 2
        else:
            labels[i] = 3

    return labels


@jit(nopython=True)
def compute_rolling_stats_numba(
    series: np.ndarray,
    window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 rolling mean, std, skewness

    Args:
        series: 输入序列
        window: 窗口大小

    Returns:
        (mean, std, skew) 三个数组
    """
    n = len(series)
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    skews = np.full(n, np.nan)

    if n < window:
        return means, stds, skews

    for i in range(window - 1, n):
        # 计算窗口数据
        window_data = series[i - window + 1:i + 1]

        # Mean
        mean_val = 0.0
        for j in range(window):
            mean_val += window_data[j]
        mean_val /= window
        means[i] = mean_val

        # Std
        var_val = 0.0
        for j in range(window):
            diff = window_data[j] - mean_val
            var_val += diff * diff
        var_val /= window
        stds[i] = np.sqrt(var_val)

        # Skewness
        std_val = stds[i]
        if std_val > 0:
            skew_val = 0.0
            for j in range(window):
                diff = window_data[j] - mean_val
                skew_val += (diff / std_val) ** 3
            skew_val /= window
            skews[i] = skew_val

    return means, stds, skews
