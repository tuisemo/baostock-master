"""
Numba 加速的计算函数
用于优化性能瓶颈，同时保持向后兼容
"""
import numpy as np
from typing import Tuple, Optional
import warnings

try:
    from numba import jit, prange, njit
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 创建装饰器的 fallback 版本
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args, **kwargs):
        return range(*args)
    
    NumbaList = list


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


@jit(nopython=True, parallel=True)
def compute_enhanced_features_numba(
    close_arr: np.ndarray,
    vol_arr: np.ndarray,
    window_short: int = 5,
    window_long: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 numba 加速计算增强版特征

    Args:
        close_arr: 收盘价数组
        vol_arr: 成交量数组
        window_short: 短期窗口
        window_long: 长期窗口

    Returns:
        (mom_acceleration, mom_persistence, volatility_change) 三个数组
    """
    n = len(close_arr)
    mom_acc = np.full(n, np.nan)
    mom_persist = np.full(n, np.nan)
    vol_change = np.full(n, np.nan)

    if n < window_long:
        return mom_acc, mom_persist, vol_change

    # 计算价格动量
    mom_5 = np.full(n, np.nan)
    for i in range(1, n):
        mom_5[i] = (close_arr[i] - close_arr[i - window_short]) / close_arr[i - window_short]

    # 动量加速度（动量的一阶导数）
    for i in range(2, n):
        if not np.isnan(mom_5[i - 1]) and not np.isnan(mom_5[i]):
            mom_acc[i] = mom_5[i] - mom_5[i - 1]

    # 动量持续性（动量的标准差）
    for i in range(window_short - 1, n):
        valid_count = 0
        sum_mom = 0.0
        sum_sq = 0.0
        
        for j in range(i - window_short + 1, i + 1):
            if not np.isnan(mom_5[j]):
                valid_count += 1
                sum_mom += mom_5[j]
        
        if valid_count >= 3:
            mean_mom = sum_mom / valid_count
            for j in range(i - window_short + 1, i + 1):
                if not np.isnan(mom_5[j]):
                    sum_sq += (mom_5[j] - mean_mom) ** 2
            mom_persist[i] = np.sqrt(sum_sq / valid_count)

    # 波动率变化率
    vol_20 = np.full(n, np.nan)
    for i in range(window_long - 1, n):
        sum_ret = 0.0
        sum_sq = 0.0
        valid_count = 0
        
        for j in range(i - window_long + 1, i + 1):
            if j > 0:
                ret = (close_arr[j] - close_arr[j - 1]) / close_arr[j - 1]
                sum_ret += ret
                sum_sq += ret * ret
                valid_count += 1
        
        if valid_count > 0:
            mean_ret = sum_ret / valid_count
            vol_20[i] = np.sqrt((sum_sq / valid_count) - (mean_ret ** 2))

    # 波动率变化率（5 天波动率的变化）
    for i in range(window_long + 4, n):
        if not np.isnan(vol_20[i - 5]) and not np.isnan(vol_20[i]):
            vol_change[i] = (vol_20[i] - vol_20[i - 5]) / (abs(vol_20[i - 5]) + 1e-8)

    return mom_acc, mom_persist, vol_change


@jit(nopython=True, parallel=True)
def compute_cross_sectional_features_numba(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
    sector_returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 numba 加速计算跨截面特征

    Args:
        stock_returns: 个股收益率数组
        market_returns: 市场收益率数组
        sector_returns: 行业收益率数组

    Returns:
        (market_relative_strength, sector_relative_strength) 两个数组
    """
    n = len(stock_returns)
    market_rel = np.full(n, 0.0)
    sector_rel = np.full(n, 0.0)

    for i in range(n):
        if not np.isnan(stock_returns[i]):
            if i < len(market_returns) and not np.isnan(market_returns[i]):
                market_rel[i] = stock_returns[i] - market_returns[i]
            
            if i < len(sector_returns) and not np.isnan(sector_returns[i]):
                sector_rel[i] = stock_returns[i] - sector_returns[i]

    return market_rel, sector_rel


# =============================================================================
# Phase 9: Enhanced Numba Acceleration for Technical Indicators
# =============================================================================

@njit
def compute_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    使用 Numba 加速的 RSI 计算
    
    Args:
        prices: 价格数组
        period: RSI 周期
        
    Returns:
        RSI 值数组 (0-100)
    """
    n = len(prices)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    # 计算价格变化
    deltas = np.diff(prices)
    
    # 初始化增益和损失
    avg_gain = 0.0
    avg_loss = 0.0
    
    # 第一个窗口
    for i in range(period):
        if deltas[i] > 0:
            avg_gain += deltas[i]
        else:
            avg_loss += abs(deltas[i])
    
    avg_gain /= period
    avg_loss /= period
    
    # 第一个 RSI 值
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    else:
        rsi[period] = 100.0
    
    # 后续值使用平滑
    for i in range(period + 1, n):
        gain = deltas[i - 1] if deltas[i - 1] > 0 else 0.0
        loss = abs(deltas[i - 1]) if deltas[i - 1] < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi


@njit
def compute_macd_numba(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Numba 加速的 MACD 计算
    
    Args:
        prices: 价格数组
        fast_period: 快速 EMA 周期
        slow_period: 慢速 EMA 周期
        signal_period: 信号线周期
        
    Returns:
        (macd, signal, histogram) 三个数组
    """
    n = len(prices)
    macd = np.full(n, np.nan)
    signal = np.full(n, np.nan)
    histogram = np.full(n, np.nan)
    
    if n < slow_period:
        return macd, signal, histogram
    
    # 计算 EMA
    ema_fast = np.full(n, np.nan)
    ema_slow = np.full(n, np.nan)
    
    # 初始化 EMA
    ema_fast[slow_period - 1] = np.mean(prices[slow_period - fast_period:slow_period])
    ema_slow[slow_period - 1] = np.mean(prices[:slow_period])
    
    k_fast = 2.0 / (fast_period + 1)
    k_slow = 2.0 / (slow_period + 1)
    
    for i in range(slow_period, n):
        ema_fast[i] = prices[i] * k_fast + ema_fast[i - 1] * (1 - k_fast)
        ema_slow[i] = prices[i] * k_slow + ema_slow[i - 1] * (1 - k_slow)
    
    # 计算 MACD
    for i in range(slow_period, n):
        macd[i] = ema_fast[i] - ema_slow[i]
    
    # 计算信号线 (EMA of MACD)
    valid_start = slow_period + signal_period - 1
    signal[valid_start] = np.mean(macd[slow_period:valid_start + 1])
    
    k_signal = 2.0 / (signal_period + 1)
    
    for i in range(valid_start + 1, n):
        signal[i] = macd[i] * k_signal + signal[i - 1] * (1 - k_signal)
    
    # 计算 histogram
    for i in range(valid_start, n):
        histogram[i] = macd[i] - signal[i]
    
    return macd, signal, histogram


@njit
def compute_bollinger_bands_numba(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Numba 加速的布林带计算
    
    Args:
        prices: 价格数组
        period: 移动平均周期
        std_dev: 标准差倍数
        
    Returns:
        (upper, middle, lower) 三个数组
    """
    n = len(prices)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, middle, lower
    
    for i in range(period - 1, n):
        # 计算窗口均值和标准差
        window_sum = 0.0
        for j in range(i - period + 1, i + 1):
            window_sum += prices[j]
        
        mean_val = window_sum / period
        middle[i] = mean_val
        
        # 计算标准差
        variance = 0.0
        for j in range(i - period + 1, i + 1):
            diff = prices[j] - mean_val
            variance += diff * diff
        
        std_val = np.sqrt(variance / period)
        
        upper[i] = mean_val + std_dev * std_val
        lower[i] = mean_val - std_dev * std_val
    
    return upper, middle, lower


@njit
def compute_atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """
    使用 Numba 加速的 ATR (Average True Range) 计算
    
    Args:
        high: 最高价数组
        low: 最低价数组
        close: 收盘价数组
        period: ATR 周期
        
    Returns:
        ATR 值数组
    """
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period:
        return atr
    
    # 计算 True Range
    tr_values = np.zeros(n)
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr_values[i] = max(tr1, max(tr2, tr3))
    
    # 第一个 ATR 使用简单平均
    atr[period] = np.mean(tr_values[1:period + 1])
    
    # 后续使用 Wilder's smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr_values[i]) / period
    
    return atr


@njit
def compute_features_numba(
    close: np.ndarray,
    volume: np.ndarray,
    high: np.ndarray,
    low: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    批量计算多种特征（用于特征提取加速）
    
    Args:
        close: 收盘价数组
        volume: 成交量数组
        high: 最高价数组
        low: 最低价数组
        
    Returns:
        (returns_1d, returns_5d, returns_20d, vol_ma_ratio, price_range_ratio)
    """
    n = len(close)
    
    returns_1d = np.full(n, np.nan)
    returns_5d = np.full(n, np.nan)
    returns_20d = np.full(n, np.nan)
    vol_ma_ratio = np.full(n, np.nan)
    price_range_ratio = np.full(n, np.nan)
    
    if n < 20:
        return returns_1d, returns_5d, returns_20d, vol_ma_ratio, price_range_ratio
    
    # 计算收益率
    for i in range(1, n):
        returns_1d[i] = (close[i] - close[i - 1]) / close[i - 1]
    
    for i in range(5, n):
        returns_5d[i] = (close[i] - close[i - 5]) / close[i - 5]
    
    for i in range(20, n):
        returns_20d[i] = (close[i] - close[i - 20]) / close[i - 20]
    
    # 计算成交量比率
    for i in range(20, n):
        vol_ma = 0.0
        for j in range(i - 19, i + 1):
            vol_ma += volume[j]
        vol_ma /= 20.0
        vol_ma_ratio[i] = volume[i] / (vol_ma + 1e-8)
    
    # 计算价格波动率
    for i in range(n):
        price_range = high[i] - low[i]
        avg_price = (high[i] + low[i]) / 2.0
        price_range_ratio[i] = price_range / (avg_price + 1e-8)
    
    return returns_1d, returns_5d, returns_20d, vol_ma_ratio, price_range_ratio


@njit(parallel=True)
def batch_process_stocks_numba(
    stock_data_list: list,
    n_forward_days: int = 5,
    target_atr_mult: float = 2.0,
    stop_loss_atr_mult: float = 1.5
) -> list:
    """
    并行处理多只股票的目标变量生成
    
    Args:
        stock_data_list: 股票数据列表，每项为 (close, high, low, atr) 元组
        n_forward_days: 向前看的天数
        target_atr_mult: 目标 ATR 倍数
        stop_loss_atr_mult: 止损 ATR 倍数
        
    Returns:
        标签数组列表
    """
    n_stocks = len(stock_data_list)
    results = []
    
    for idx in prange(n_stocks):
        close_series, high_series, low_series, atr_series = stock_data_list[idx]
        labels = create_targets_numba(
            close_series, high_series, low_series, atr_series,
            n_forward_days, target_atr_mult, stop_loss_atr_mult
        )
        results.append(labels)
    
    return results


@njit
def compute_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    使用 Numba 加速的 EMA 计算
    
    Args:
        prices: 价格数组
        period: EMA 周期
        
    Returns:
        EMA 值数组
    """
    n = len(prices)
    ema = np.full(n, np.nan)
    
    if n < period:
        return ema
    
    # 初始化 EMA
    ema[period - 1] = np.mean(prices[:period])
    
    k = 2.0 / (period + 1)
    
    for i in range(period, n):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    
    return ema


@njit(parallel=True)
def compute_sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    使用 Numba 加速的 SMA（简单移动平均）计算
    
    Args:
        prices: 价格数组
        period: SMA 周期
        
    Returns:
        SMA 值数组
    """
    n = len(prices)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    # 第一个 SMA
    sma[period - 1] = np.mean(prices[:period])
    
    # 使用滑动窗口优化
    for i in prange(period, n):
        sma[i] = sma[i - 1] + (prices[i] - prices[i - period]) / period
    
    return sma


@njit
def compute_std_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    使用 Numba 加速的标准差计算
    
    Args:
        prices: 价格数组
        period: 标准差周期
        
    Returns:
        标准差值数组
    """
    n = len(prices)
    std_arr = np.full(n, np.nan)
    
    if n < period:
        return std_arr
    
    for i in range(period - 1, n):
        window_sum = 0.0
        window_sq_sum = 0.0
        
        for j in range(i - period + 1, i + 1):
            window_sum += prices[j]
            window_sq_sum += prices[j] * prices[j]
        
        mean_val = window_sum / period
        # 方差 = E[X^2] - E[X]^2
        variance = (window_sq_sum / period) - (mean_val * mean_val)
        std_arr[i] = np.sqrt(max(0.0, variance))  # 防止数值误差导致负值
    
    return std_arr


def get_performance_report() -> dict:
    """
    获取 Numba 加速器性能报告
    
    Returns:
        包含 Numba 状态和可用功能的字典
    """
    return {
        'numba_available': NUMBA_AVAILABLE,
        'accelerated_functions': [
            'compute_vol_slope_numba',
            'create_targets_numba',
            'create_multi_class_targets_numba',
            'compute_rolling_stats_numba',
            'compute_enhanced_features_numba',
            'compute_cross_sectional_features_numba',
            'compute_rsi_numba',
            'compute_macd_numba',
            'compute_bollinger_bands_numba',
            'compute_atr_numba',
            'compute_features_numba',
            'compute_ema_numba',
            'compute_sma_numba',
            'compute_std_numba',
            'batch_process_stocks_numba',
        ]
    }
