"""
自适应策略参数调整模块
实现基于市场状态（7类）的自适应参数调整系统
"""
from dataclasses import replace
from typing import Dict

from quant.logger import logger
from quant.strategy_params import StrategyParams


def get_dynamic_params_v10(
    base_params: StrategyParams,
    market_state: str
) -> StrategyParams:
    """
    增强版自适应参数调整系统（支持 10 类市场状态）

    参数调整策略：
    1. bull_momentum: 牛市动量 - 最激进策略，最高仓位，快速止盈
    2. strong_bull: 强牛市 - 激进策略，高仓位
    3. bull_volume: 牛市放量 - 积极策略，高仓位
    4. weak_bull: 弱牛市 - 稳健策略，中高仓位
    5. sideways_low_vol: 低波震荡 - 保守策略，标准仓位
    6. sideways_high_vol: 高波震荡 - 谨慎策略，降低仓位
    7. weak_bear: 弱熊市 - 防御策略，中低仓位
    8. bear_momentum: 熊市动量 - 退守策略，低仓位
    9. bear_panic: 熊市恐慌 - 极端防御，极低仓位
    10. strong_bear: 强熊市 - 极端防御，最低仓位

    Args:
        base_params: 基础策略参数
        market_state: 市场状态（10 类）

    Returns:
        调整后的策略参数
    """
    try:
        params = replace(base_params)
    except Exception:
        # Fallback: try to convert to dict and recreate
        try:
            params = StrategyParams(**base_params.__dict__)
        except Exception as e:
            logger.error(f"Failed to create params copy: {e}")
            raise

    # 市场状态参数调整表（10 类）
    STATE_ADJUSTMENTS_10: Dict[str, Dict[str, float]] = {
        # === 牛市策略（激进）===
        'bull_momentum': {
            'position_size': 0.20,      # 最高仓位
            'ai_prob_threshold': 0.15,   # 最宽松 AI 过滤
            'max_hold_days': 5,         # 短期快速交易
            'trail_atr_mult': 2.5,      # 宽松移动止损
            'take_profit_pct': 0.06,     # 快速止盈
            'w_pullback_ma': 1.0,       # 最低左侧权重
            'w_vol_up': 3.0,           # 最高右侧权重
            'vol_up_ratio': 1.3,        # 降低放量门槛
        },
        'strong_bull': {
            'position_size': 0.18,      # 高仓位
            'ai_prob_threshold': 0.20,   # 宽松 AI 过滤
            'max_hold_days': 8,         # 中等持仓周期
            'trail_atr_mult': 2.3,      # 宽松移动止损
            'take_profit_pct': 0.08,     # 适度止盈
            'w_pullback_ma': 1.2,       # 降低左侧权重
            'w_vol_up': 2.5,           # 提高右侧权重
        },
        'bull_volume': {
            'position_size': 0.16,      # 中高仓位
            'ai_prob_threshold': 0.25,   # 适度 AI 过滤
            'max_hold_days': 8,         # 中等持仓周期
            'trail_atr_mult': 2.2,      # 适中的移动止损
            'take_profit_pct': 0.08,     # 适度止盈
            'w_pullback_ma': 1.5,       # 平衡的权重
            'w_vol_up': 2.5,           # 高权重（放量特征）
        },
        'weak_bull': {
            'position_size': 0.12,      # 中高仓位
            'ai_prob_threshold': 0.30,   # 适度 AI 过滤
            'max_hold_days': 12,        # 中等持仓周期
            'trail_atr_mult': 2.0,      # 适中的移动止损
            'take_profit_pct': 0.08,     # 适度止盈
            'w_pullback_ma': 2.0,       # 平衡的权重
            'w_vol_up': 2.0,           # 平衡的权重
        },

        # === 横盘策略（保守）===
        'sideways_low_vol': {
            'position_size': 0.08,      # 标准仓位
            'ai_prob_threshold': 0.35,   # 严格 AI 过滤
            'max_hold_days': 15,        # 较长持仓周期
            'trail_atr_mult': 1.8,      # 紧密移动止损
            'take_profit_pct': 0.06,     # 快速止盈
            'w_pullback_ma': 2.5,       # 提高左侧权重
            'w_vol_up': 1.5,           # 降低右侧权重
        },
        'sideways_high_vol': {
            'position_size': 0.06,      # 降低仓位
            'ai_prob_threshold': 0.40,   # 更严格 AI 过滤
            'max_hold_days': 10,        # 短期持仓
            'trail_atr_mult': 1.6,      # 紧密移动止损
            'take_profit_pct': 0.05,     # 快速止盈
            'w_pullback_ma': 3.0,       # 提高左侧权重
            'w_vol_up': 1.2,           # 降低右侧权重（高波动不加仓）
        },

        # === 熊市策略（防御）===
        'weak_bear': {
            'position_size': 0.05,      # 中低仓位
            'ai_prob_threshold': 0.45,   # 严格 AI 过滤
            'max_hold_days': 10,        # 短期持仓
            'trail_atr_mult': 1.6,      # 紧密移动止损
            'take_profit_pct': 0.05,     # 快速止盈
            'w_pullback_ma': 3.0,       # 提高左侧权重
            'w_vol_up': 1.0,           # 降低右侧权重
        },
        'bear_momentum': {
            'position_size': 0.03,      # 低仓位
            'ai_prob_threshold': 0.55,   # 极严格 AI 过滤
            'max_hold_days': 5,         # 最短期持仓
            'trail_atr_mult': 1.4,      # 极紧密移动止损
            'take_profit_pct': 0.04,     # 快速止盈
            'w_pullback_ma': 4.0,       # 最高左侧权重
            'w_vol_up': 0.5,           # 最低右侧权重
        },
        'bear_panic': {
            'position_size': 0.02,      # 极低仓位
            'ai_prob_threshold': 0.60,   # 极严格 AI 过滤
            'max_hold_days': 3,         # 最短期持仓
            'trail_atr_mult': 1.2,      # 极紧密移动止损
            'take_profit_pct': 0.03,     # 快速止盈
            'w_pullback_ma': 5.0,       # 最高左侧权重
            'w_vol_up': 0.3,           # 最低右侧权重
            'rsi_oversold': 25,        # 极严格超卖条件
        },
        'strong_bear': {
            'position_size': 0.01,      # 接近空仓
            'ai_prob_threshold': 0.65,   # 极严格 AI 过滤
            'max_hold_days': 2,         # 最短期持仓
            'trail_atr_mult': 1.0,      # 极紧密移动止损
            'take_profit_pct': 0.02,     # 快速止盈
            'w_pullback_ma': 5.0,       # 最高左侧权重
            'w_vol_up': 0.3,           # 最低右侧权重
            'rsi_oversold': 25,        # 极严格超卖条件
        },
    }

    # 获取当前市场状态的调整
    adjustments = STATE_ADJUSTMENTS_10.get(
        market_state,
        STATE_ADJUSTMENTS_10['sideways_low_vol']  # 默认使用低波震荡
    )

    # 应用调整（相对于基准值的增量调整）
    for key, value in adjustments.items():
        current_value = getattr(params, key, None)
        if current_value is not None:
            if isinstance(current_value, float):
                # 对于连续参数，直接应用调整值
                setattr(params, key, value)
            elif isinstance(current_value, int):
                # 对于整数参数，应用四舍五入
                setattr(params, key, int(round(value)))

    logger.debug(f"市场状态: {market_state}, 应用调整: {adjustments}")

    return params


def get_param_transition_matrix() -> Dict[str, Dict[str, float]]:
    """
    获取参数转移矩阵
    定义不同市场状态之间的参数平滑过渡规则

    Returns:
        参数转移矩阵
    """
    # 简化的转移矩阵（实际应用中可以基于历史数据学习）
    return {
        'strong_bull': {
            'bull_momentum': 0.9,  # 强牛市 -> 牛市动量（90% 概率）
            'weak_bull': 0.1,
        },
        'bull_momentum': {
            'strong_bull': 0.7,
            'weak_bull': 0.3,
        },
        'weak_bull': {
            'strong_bull': 0.2,
            'sideways': 0.5,
            'bull_momentum': 0.3,
        },
        'sideways': {
            'weak_bull': 0.3,
            'weak_bear': 0.3,
            'strong_bull': 0.2,
            'strong_bear': 0.2,
        },
        'weak_bear': {
            'sideways': 0.5,
            'strong_bear': 0.2,
            'bear_momentum': 0.3,
        },
        'bear_momentum': {
            'weak_bear': 0.3,
            'strong_bear': 0.7,
        },
        'strong_bear': {
            'bear_momentum': 0.9,
            'weak_bear': 0.1,
        },
    }


def smooth_param_transition(
    current_params: StrategyParams,
    target_params: StrategyParams,
    smoothing_factor: float = 0.3
) -> StrategyParams:
    """
    参数平滑过渡
    在市场状态转换时，平滑过渡参数值，避免剧烈变化

    Args:
        current_params: 当前参数
        target_params: 目标参数
        smoothing_factor: 平滑因子（0-1），越小表示过渡越慢

    Returns:
        平滑过渡后的参数
    """
    try:
        params = replace(current_params)
    except Exception:
        params = StrategyParams(**current_params.__dict__)

    # 获取所有参数名称
    param_names = [f.name for f in params.__dataclass_fields__]

    # 对每个参数进行平滑过渡
    for name in param_names:
        current_value = getattr(current_params, name)
        target_value = getattr(target_params, name)

        if isinstance(current_value, (int, float)):
            # 计算平滑后的值
            smoothed_value = current_value + smoothing_factor * (target_value - current_value)
            
            # 根据类型进行转换
            if isinstance(current_value, int):
                setattr(params, name, int(round(smoothed_value)))
            else:
                setattr(params, name, smoothed_value)

    return params


# 新增：Kelly 公式仓位计算
def calculate_kelly_position(
    win_rate: float,
    avg_profit: float,
    avg_loss: float,
    risk_fraction: float = 0.5  # Kelly 分数（通常使用半 Kelly）
) -> float:
    """
    计算 Kelly 最优仓位

    Args:
        win_rate: 胜率 (0-1)
        avg_profit: 平均盈利 (正值)
        avg_loss: 平均亏损 (负值)
        risk_fraction: Kelly 分数 (0-1)，通常使用 0.5 降低风险

    Returns:
        最优仓位比例 (0-1)
    """
    if avg_loss >= 0 or avg_profit <= 0:
        return 0.05  # 默认最小仓位

    # 盈亏比
    b = abs(avg_profit / avg_loss)

    # Kelly 公式
    if b == 0:
        return 0.05

    kelly_f = (win_rate * b - (1 - win_rate)) / b

    # 应用分数降低风险
    kelly_f *= risk_fraction

    # 限制范围
    kelly_f = max(0.01, min(0.30, kelly_f))

    return kelly_f


# 新增：动态仓位管理器
class DynamicPositionManager:
    """
    动态仓位管理器

    功能：
    1. Kelly 公式动态仓位计算
    2. 基于市场状态的风险调整
    3. 基于连败的调整
    """
    
    def __init__(
        self,
        initial_equity: float = 100000.0,
        risk_fraction: float = 0.5,
        max_consecutive_losses: int = 5,
    ):
        self.equity = initial_equity
        self.risk_fraction = risk_fraction
        self.max_consecutive_losses = max_consecutive_losses

        self.trade_history = []
        self.consecutive_losses = 0

    def calculate_position_size(
        self,
        base_position_size: float,
        market_state: str,
        win_rate: float = 0.5,
        avg_profit: float = 0.05,
        avg_loss: float = -0.03,
    ) -> float:
        """
        动态计算仓位大小

        Args:
            base_position_size: 基础仓位大小（来自策略参数）
            market_state: 市场状态
            win_rate: 历史胜率
            avg_profit: 平均盈利
            avg_loss: 平均亏损

        Returns:
            调整后的仓位大小 (0-1)
        """
        # 1. Kelly 公式仓位
        kelly_position = calculate_kelly_position(
            win_rate, avg_profit, avg_loss, self.risk_fraction
        )

        # 2. 基于市场状态的风险调整
        from quant.market_classifier import get_market_state_risk_level
        risk_level = get_market_state_risk_level(market_state)
        risk_factor = max(0.3, 1.0 - (risk_level - 1) * 0.1)

        # 3. 基于连败的调整
        if self.consecutive_losses > 0:
            loss_factor = max(0.3, 1.0 - self.consecutive_losses * 0.2)
        else:
            loss_factor = 1.0

        # 4. 综合调整
        adjusted_size = kelly_position * risk_factor * loss_factor

        # 5. 与基础仓位结合
        final_size = (adjusted_size + base_position_size) / 2.0

        # 6. 限制范围
        final_size = max(0.01, min(0.30, final_size))

        return final_size

    def record_trade(self, profit_pct: float):
        """
        记录交易结果

        Args:
            profit_pct: 盈亏百分比
        """
        self.trade_history.append(profit_pct)

        if profit_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # 更新权益
        self.equity *= (1 + profit_pct / 100.0)


# 更新：参数转移矩阵扩展为 10 类
def get_param_transition_matrix() -> Dict[str, Dict[str, float]]:
    """
    获取参数转移矩阵
    定义不同市场状态之间的参数平滑过渡规则

    Returns:
        参数转移矩阵
    """
    # 简化的转移矩阵（实际应用中可以基于历史数据学习）
    return {
        'bull_momentum': {
            'strong_bull': 0.8,
            'bull_volume': 0.2,
        },
        'strong_bull': {
            'bull_momentum': 0.7,
            'bull_volume': 0.3,
        },
        'bull_volume': {
            'strong_bull': 0.6,
            'weak_bull': 0.4,
        },
        'weak_bull': {
            'strong_bull': 0.2,
            'sideways_low_vol': 0.5,
            'sideways_high_vol': 0.3,
        },
        'sideways_low_vol': {
            'weak_bull': 0.3,
            'weak_bear': 0.3,
            'sideways_high_vol': 0.4,
        },
        'sideways_high_vol': {
            'weak_bull': 0.2,
            'weak_bear': 0.4,
            'strong_bear': 0.4,
        },
        'weak_bear': {
            'sideways_low_vol': 0.4,
            'bear_momentum': 0.4,
            'strong_bear': 0.2,
        },
        'bear_momentum': {
            'weak_bear': 0.3,
            'strong_bear': 0.5,
            'bear_panic': 0.2,
        },
        'bear_panic': {
            'bear_momentum': 0.6,
            'strong_bear': 0.4,
        },
        'strong_bear': {
            'bear_momentum': 0.8,
            'bear_panic': 0.2,
        },
    }


# 兼容旧版函数
get_dynamic_params = get_dynamic_params_v10
get_dynamic_params_enhanced = get_dynamic_params_v10


if __name__ == "__main__":
    # 测试代码
    from quant.strategy_params import StrategyParams
    
    base_params = StrategyParams()
    
    # 测试不同市场状态的参数调整
    for state in ['bull_momentum', 'strong_bull', 'bull_volume', 'weak_bull',
                   'sideways_low_vol', 'sideways_high_vol',
                   'weak_bear', 'bear_momentum', 'bear_panic', 'strong_bear']:
        adjusted_params = get_dynamic_params_v10(base_params, state)
        
        print(f"\n市场状态: {state}")
        print(f"  仓位大小: {base_params.position_size:.3f} -> {adjusted_params.position_size:.3f}")
        print(f"  AI 概率阈值: {base_params.ai_prob_threshold:.3f} -> {adjusted_params.ai_prob_threshold:.3f}")
        print(f"  最大持仓天数: {base_params.max_hold_days} -> {adjusted_params.max_hold_days}")
