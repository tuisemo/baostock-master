"""
集中管理所有魔法数字（常量）

本模块包含系统中使用的所有硬编码阈值和常量，包括：
- 置信度阈值 (ConfidenceThresholds)
- 评分阈值 (ScoreThresholds)
- 仓位限制 (PositionLimits)
- 质量评级 (QualityRatings)
- 其他交易相关常量

所有常量应从本模块导入，避免在代码中硬编码。
"""
from dataclasses import dataclass
from typing import Dict


# =============================================================================
# 置信度阈值 (Confidence Thresholds)
# 用于AI模型预测结果的置信度分级
# =============================================================================

@dataclass
class ConfidenceThresholds:
    """AI预测置信度阈值"""
    HIGH: float = 0.65   # 高置信度阈值
    MEDIUM: float = 0.45  # 中置信度阈值
    LOW: float = 0.30     # 低置信度阈值


# 单例实例
CONFIDENCE = ConfidenceThresholds()


# =============================================================================
# 评分阈值 (Score Thresholds)
# 用于信号评分的质量评级
# =============================================================================

@dataclass
class ScoreThresholds:
    """信号评分阈值"""
    STRONG: float = 15.0   # 强烈信号 (>= 15)
    GOOD: float = 10.0     # 良好信号 (>= 10)
    NEUTRAL: float = 5.0   # 中性信号 (>= 5)
    WEAK: float = 2.0      # 弱势信号 (>= 2)


# 单例实例
SCORE = ScoreThresholds()


# =============================================================================
# 仓位限制 (Position Limits)
# 仓位大小的最大/最小限制
# =============================================================================

@dataclass
class PositionLimits:
    """仓位限制"""
    MIN_SIZE: float = 0.01    # 最小仓位 1%
    MAX_SIZE: float = 0.30   # 最大仓位 30%


# 单例实例
POSITION = PositionLimits()


# =============================================================================
# 质量评级乘数 (Quality Rating Multipliers)
# 根据信号质量评级调整仓位和止盈止损
# =============================================================================

QUALITY_MULTIPLIERS: Dict[str, float] = {
    'strong': 1.5,   # 强信号乘数
    'good': 1.2,     # 良好信号乘数
    'neutral': 1.0,  # 中性信号乘数
    'weak': 0.7,     # 弱信号乘数
    'poor': 0.4,     # 差信号乘数
}


# =============================================================================
# TP/SL 质量评级乘数 (Quality Rating Multipliers for Take-Profit/Stop-Loss)
# 用于止盈止损计算的信号质量调整
# =============================================================================

QUALITY_MULTIPLIERS_TP_SL: Dict[str, float] = {
    'strong': 1.5,   # 强信号乘数
    'good': 1.2,     # 良好信号乘数
    'neutral': 1.0,  # 中性信号乘数
    'weak': 0.8,     # 弱信号乘数 (TP/SL 用较高值)
    'poor': 0.6,     # 差信号乘数 (TP/SL 用较高值)
}


# =============================================================================
# 交易类型乘数 (Trade Type Multipliers)
# 左侧交易 vs 右侧交易的止盈止损倍数
# =============================================================================

@dataclass
class TradeTypeMultipliers:
    """交易类型乘数"""
    # 止盈止损倍数
    LEFT_TP_MULT: float = 2.0   # 左侧交易止盈倍数
    LEFT_SL_MULT: float = 1.5  # 左侧交易止损倍数
    RIGHT_TP_MULT: float = 3.0  # 右侧交易止盈倍数
    RIGHT_SL_MULT: float = 2.5  # 右侧交易止损倍数


# 单例实例
TRADE_TYPE = TradeTypeMultipliers()


# =============================================================================
# 回撤控制阈值 (Drawdown Control Thresholds)
# 用于风险管理的回撤控制
# =============================================================================

DRAWDOWN_LEVELS: Dict[float, float] = {
    0.05: 1.0,   # 0-5% 回撤: 满仓
    0.10: 0.6,   # 5-10% 回撤: 60% 仓位
    0.15: 0.3,   # 10-15% 回撤: 30% 仓位
    1.00: 0.0,   # >15% 回撤: 空仓
}


# =============================================================================
# VaR (Value at Risk) 配置
# 风险价值配置
# =============================================================================

VAR_CONFIG: Dict[str, float] = {
    'confidence_level': 0.95,   # 95% VaR
    'max_var_95': 0.05,         # 最大 5% VaR
    'history_window': 252,       # 1年数据窗口
}


# =============================================================================
# Kelly 公式限制
# 仓位计算的 Kelly 限制
# =============================================================================

KELLY_LIMITS: Dict[str, float] = {
    'min': 0.01,   # 最小 Kelly 仓位 1%
    'max': 0.30,   # 最大 Kelly 仓位 30%
}


# =============================================================================
# 相关性风险配置
# 组合相关性风险管理
# =============================================================================

CORRELATION_CONFIG: Dict[str, float] = {
    'max_correlation': 0.7,     # 最大相关性
    'lookback_window': 60,       # 相关性计算窗口
    'min_periods': 30,           # 最小数据周期
}


# =============================================================================
# 滑点模型配置 (Slippage Model)
# 基于市值等级的基准滑点
# =============================================================================

SLIPPAGE_BASE: Dict[str, float] = {
    'large': 0.0005,   # 大市值: 0.05%
    'mid': 0.001,      # 中市值: 0.1%
    'small': 0.002,    # 小市值: 0.2%
    'micro': 0.005,    # 微市值: 0.5%
}


SLIPPAGE_CONFIG: Dict[str, float] = {
    'volume_impact_factor': 0.0001,   # 成交量影响因子
    'max_volume_impact': 0.002,       # 最大成交量影响 0.2%
}


# 市值阈值 (单位: 亿CNY)
MARKET_CAP_THRESHOLDS: Dict[str, float] = {
    'large': 100,   # > 100B
    'mid': 30,      # 30B - 100B
    'small': 5,     # 5B - 30B
    'micro': 0      # < 5B
}


# =============================================================================
# 组合风险限制
# =============================================================================

PORTFOLIO_LIMITS: Dict[str, float] = {
    'max_sector_weight': 0.30,           # 单行业最大权重 30%
    'max_single_position': 0.25,          # 单只股票最大权重 25%
    'target_portfolio_volatility': 0.15,   # 目标组合波动率 15%
    'max_positions': 10,                  # 最大持仓数量
}


# =============================================================================
# 波动率相关常量
# =============================================================================

VOLATILITY_CONFIG: Dict[str, float] = {
    'normal_volatility': 0.02,     # 正常波动率 2%
    'high_volatility': 0.025,      # 高波动率阈值
    'low_volatility': 0.015,       # 低波动率阈值
}


# =============================================================================
# 市场状态分类阈值 (Market State Thresholds)
# 用于 classify_market_state 函数
# =============================================================================

MARKET_STATE_THRESHOLDS_LEGACY: Dict[str, float] = {
    'trend_strength_bull': 0.02,      # 牛市趋势强度
    'trend_strength_bear': -0.02,      # 熊市趋势强度
    'volatility_high': 0.025,         # 高波动率
    'volatility_low': 0.015,          # 低波动率
    'roc_strong': 0.05,               # 强劲ROC
    'volume_ratio_high': 1.5,          # 高成交量比率
}


# =============================================================================
# 风险评级阈值
# 风险等级 1-10, 1为最低风险, 10为最高风险
# =============================================================================

RISK_LEVELS: Dict[str, int] = {
    'strong_bull': 2,
    'bull_momentum': 1,
    'bull_volume': 3,
    'weak_bull': 4,
    'sideways_low_vol': 5,
    'sideways_high_vol': 6,
    'weak_bear': 7,
    'strong_bear': 8,
    'bear_momentum': 9,
    'bear_panic': 10,
}


# =============================================================================
# Monte Carlo 模拟配置
# =============================================================================

MONTE_CARLO_CONFIG: Dict[str, float] = {
    'n_paths': 1000,              # 模拟路径数
    'block_size': 20,             # Bootstrap 块大小
    'confidence_levels': [0.95, 0.99],  # 置信水平
}


# =============================================================================
# 兼容性别名 (保留给向后兼容)
# 这些常量已迁移到本模块，其他模块应逐步更新使用新名称
# =============================================================================

# 置信度阈值别名 (backtester.py 使用)
CONFIDENCE_HIGH_THRESHOLD = CONFIDENCE.HIGH       # 0.65
CONFIDENCE_MEDIUM_THRESHOLD = CONFIDENCE.MEDIUM   # 0.45
CONFIDENCE_LOW_THRESHOLD = CONFIDENCE.LOW          # 0.30

# 评分阈值别名 (signal_scorer.py 使用)
SCORE_THRESHOLD_STRONG = SCORE.STRONG    # 15.0
SCORE_THRESHOLD_GOOD = SCORE.GOOD        # 10.0
SCORE_THRESHOLD_NEUTRAL = SCORE.NEUTRAL  # 5.0
SCORE_THRESHOLD_WEAK = SCORE.WEAK        # 2.0
