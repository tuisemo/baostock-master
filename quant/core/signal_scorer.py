"""
信号评分和仓位分配优化模块
优化信号评分系统，实现基于信号质量的动态仓位分配
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from quant.infra.logger import logger
from quant.core.strategy_params import StrategyParams


@dataclass
class SignalScore:
    """信号评分数据类"""
    total_score: float = 0.0
    pullback_score: float = 0.0
    rebound_score: float = 0.0
    breakout_score: float = 0.0
    volume_score: float = 0.0
    macd_score: float = 0.0
    rsi_score: float = 0.0
    ai_score: float = 0.0
    quality_rating: str = "neutral"  # strong, good, neutral, weak, poor
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'total_score': self.total_score,
            'pullback_score': self.pullback_score,
            'rebound_score': self.rebound_score,
            'breakout_score': self.breakout_score,
            'volume_score': self.volume_score,
            'macd_score': self.macd_score,
            'rsi_score': self.rsi_score,
            'ai_score': self.ai_score,
            'quality_rating': self.quality_rating,
        }


class SignalScorer:
    """
    信号评分器
    
    功能：
    1. 多维度信号评分
    2. 信号质量评级
    3. 动态仓位分配
    4. 止盈止损动态调整
    """
    
    def __init__(self, params: StrategyParams):
        self.params = params
        self.score_weights = self._initialize_weights()
    
    def _initialize_weights(self) -> Dict[str, float]:
        """
        初始化评分权重
        
        Returns:
            权重字典
        """
        return {
            'pullback_ma': self.params.w_pullback_ma,
            'macd_cross': self.params.w_macd_cross,
            'vol_up': self.params.w_vol_up,
            'rsi_rebound': self.params.w_rsi_rebound,
            'green_candle': self.params.w_green_candle,
        }
    
    def calculate_signal_score(
        self,
        signal_pullback: bool,
        signal_rebound: bool,
        signal_breakout: bool,
        price: float,
        open_p: float,
        low_p: float,
        sma_l_1: float,
        sma_s_1: float,
        macd_h_1: float,
        macd_h_2: float,
        rsi_1: float,
        vol_1: float,
        vol_2: float,
        ai_prob: float = 0.5,
    ) -> SignalScore:
        """
        计算信号评分（多维度）
        
        Args:
            signal_pullback: 回调信号
            signal_rebound: 反弹信号
            signal_breakout: 突破信号
            price: 当前价格
            open_p: 开盘价
            low_p: 最低价
            sma_l_1: 长期均线
            sma_s_1: 短期均线
            macd_h_1: MACD 直方图
            macd_h_2: 前一日 MACD 直方图
            rsi_1: RSI
            vol_1: 成交量
            vol_2: 前一日成交量
            ai_prob: AI 预测概率
        
        Returns:
            信号评分对象
        """
        score = SignalScore()
        
        # 1. 计算各维度评分
        
        # 回调评分
        if signal_pullback:
            # 检查回调深度
            pullback_depth = (sma_l_1 - price) / sma_l_1
            if pullback_depth > 0.02:
                score.pullback_score = self.score_weights['pullback_ma'] * 1.5
            elif pullback_depth > 0.01:
                score.pullback_score = self.score_weights['pullback_ma'] * 1.2
            else:
                score.pullback_score = self.score_weights['pullback_ma']
        
        # 反弹评分
        if signal_rebound:
            # 检查 RSI 超卖程度
            if rsi_1 < 25:
                score.rebound_score = self.score_weights['rsi_rebound'] * 1.5
            elif rsi_1 < 35:
                score.rebound_score = self.score_weights['rsi_rebound'] * 1.2
            else:
                score.rebound_score = self.score_weights['rsi_rebound']
        
        # 突破评分
        if signal_breakout:
            # 检查突破强度
            if price > sma_s_1 * 1.02:
                score.breakout_score = self.score_weights['pullback_ma'] * 1.5
            elif price > sma_s_1 * 1.01:
                score.breakout_score = self.score_weights['pullback_ma'] * 1.2
            else:
                score.breakout_score = self.score_weights['pullback_ma']
        
        # 成交量评分
        vol_ratio = vol_1 / vol_2 if vol_2 > 0 else 1.0
        if vol_ratio > self.params.vol_up_ratio:
            score.volume_score = self.score_weights['vol_up'] * 1.5
        elif vol_ratio > 1.3:
            score.volume_score = self.score_weights['vol_up'] * 1.2
        elif vol_ratio > 1.1:
            score.volume_score = self.score_weights['vol_up']
        else:
            score.volume_score = self.score_weights['vol_up'] * 0.5
        
        # MACD 评分
        macd_golden_cross = macd_h_2 <= 0 and macd_h_1 > 0
        if macd_golden_cross:
            score.macd_score = self.score_weights['macd_cross'] * 1.5
        elif macd_h_1 > macd_h_2:
            score.macd_score = self.score_weights['macd_cross']
        else:
            score.macd_score = self.score_weights['macd_cross'] * 0.5
        
        # RSI 评分
        if signal_rebound and rsi_1 < self.params.rsi_oversold:
            score.rsi_score = self.score_weights['rsi_rebound'] * 1.5
        elif signal_breakout and 40 < rsi_1 < 70:
            score.rsi_score = self.score_weights['rsi_rebound'] * 1.2
        else:
            score.rsi_score = self.score_weights['rsi_rebound'] * 0.5
        
        # 阳线评分
        is_green_candle = price > open_p
        if is_green_candle:
            lower_shadow = min(open_p, price) - low_p
            body = abs(price - open_p)
            if lower_shadow > body * 1.5 and lower_shadow > 0:
                # 长下影线（锤子线）
                score.rsi_score += self.score_weights['green_candle'] * 0.5
            else:
                score.rsi_score += self.score_weights['green_candle']
        
        # AI 评分
        score.ai_score = (ai_prob - 0.5) * 10  # 将概率转换为评分
        
        # 2. 计算总评分
        score.total_score = (
            score.pullback_score + 
            score.rebound_score + 
            score.breakout_score + 
            score.volume_score + 
            score.macd_score + 
            score.rsi_score + 
            score.ai_score
        )
        
        # 3. 信号质量评级
        if score.total_score >= 15:
            score.quality_rating = "strong"
        elif score.total_score >= 10:
            score.quality_rating = "good"
        elif score.total_score >= 5:
            score.quality_rating = "neutral"
        elif score.total_score >= 2:
            score.quality_rating = "weak"
        else:
            score.quality_rating = "poor"
        
        return score
    
    def calculate_dynamic_position_size(
        self,
        signal_score: SignalScore,
        base_position_size: float,
        equity: float,
        volatility: float = None,
    ) -> float:
        """
        动态仓位分配
        
        根据信号质量、权益、波动率等因素动态调整仓位
        
        Args:
            signal_score: 信号评分
            base_position_size: 基础仓位大小
            equity: 当前权益
            volatility: 市场波动率（可选）
        
        Returns:
            调整后的仓位大小（0-1）
        """
        # 1. 基于信号质量的调整
        quality_multipliers = {
            'strong': 1.5,
            'good': 1.2,
            'neutral': 1.0,
            'weak': 0.7,
            'poor': 0.4,
        }
        quality_factor = quality_multipliers.get(signal_score.quality_rating, 1.0)
        
        # 2. 基于总评分的连续调整
        score_factor = 1.0
        if signal_score.total_score > 0:
            score_factor = min(1.5, max(0.5, signal_score.total_score / 10.0))
        
        # 3. 基于波动率的调整
        volatility_factor = 1.0
        if volatility is not None and volatility > 0:
            normal_volatility = 0.02  # 假设正常波动率为 2%
            volatility_factor = max(0.5, normal_volatility / max(volatility, 0.01))
        
        # 4. 综合调整
        adjusted_size = base_position_size * quality_factor * score_factor * volatility_factor
        
        # 5. 限制仓位范围
        min_size = 0.01  # 最小 1% 仓位
        max_size = 0.30  # 最大 30% 仓位
        
        adjusted_size = max(min_size, min(max_size, adjusted_size))
        
        logger.debug(
            f"动态仓位分配: 质量={signal_score.quality_rating}({quality_factor:.2f}), "
            f"评分={signal_score.total_score:.2f}({score_factor:.2f}), "
            f"波动率={volatility:.4f}({volatility_factor:.2f}), "
            f"基础={base_position_size:.3f} -> 调整后={adjusted_size:.3f}"
        )
        
        return adjusted_size
    
    def calculate_dynamic_tp_sl(
        self,
        signal_score: SignalScore,
        price: float,
        atr: float,
        trade_type: str = "left",
    ) -> Tuple[float, float]:
        """
        动态止盈止损
        
        根据信号质量动态调整止盈止损
        
        Args:
            signal_score: 信号评分
            price: 当前价格
            atr: ATR
            trade_type: 交易类型（'left' 或 'right'）
        
        Returns:
            (止盈价, 止损价)
        """
        # 基于信号质量的调整系数
        quality_multipliers = {
            'strong': 1.5,
            'good': 1.2,
            'neutral': 1.0,
            'weak': 0.8,
            'poor': 0.6,
        }
        quality_factor = quality_multipliers.get(signal_score.quality_rating, 1.0)
        
        # 基础止盈止损参数
        base_tp_pct = self.params.take_profit_pct
        base_trail_atr_mult = self.params.trail_atr_mult
        
        # 根据交易类型调整
        if trade_type == "right":
            # 右侧交易：更宽松的止盈止损
            base_tp_mult = 3.0  # 止盈为 3 倍 ATR
            base_sl_mult = 2.5  # 止损为 2.5 倍 ATR
        else:
            # 左侧交易：更严格的止盈止损
            base_tp_mult = 2.0
            base_sl_mult = 1.5
        
        # 动态调整
        tp_mult = base_tp_mult * quality_factor
        sl_mult = base_sl_mult * quality_factor
        
        # 计算止盈止损价格
        take_profit_price = price + tp_mult * atr
        stop_loss_price = price - sl_mult * atr
        
        logger.debug(
            f"动态止盈止损: 质量={signal_score.quality_rating}({quality_factor:.2f}), "
            f"类型={trade_type}, TP={tp_mult:.2f}ATR, SL={sl_mult:.2f}ATR"
        )
        
        return take_profit_price, stop_loss_price


class CrossSectionalRanker:
    """
    横截面排名器
    
    功能：
    1. 对多个信号进行横截面排名
    2. 基于排名调整仓位
    3. 实现资金分散管理
    """
    
    def __init__(self, top_n: int = 10):
        self.top_n = top_n
    
    def rank_signals(
        self,
        signals: List[Tuple[str, SignalScore]],
    ) -> List[Tuple[str, float]]:
        """
        对信号进行横截面排名
        
        Args:
            signals: 信号列表，格式为 [(code, SignalScore), ...]
        
        Returns:
            排名后的信号列表，格式为 [(code, rank), ...]
        """
        # 计算归一化评分
        max_score = max((s.total_score for _, s in signals), default=1.0)
        
        # 计算排名权重
        ranked_signals = []
        for code, score in signals:
            normalized_score = score.total_score / max_score if max_score > 0 else 0
            rank_weight = normalized_score
            ranked_signals.append((code, rank_weight))
        
        # 按权重排序
        ranked_signals.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_signals
    
    def allocate_capital(
        self,
        ranked_signals: List[Tuple[str, float]],
        total_capital: float,
        max_position_size: float = 0.10,
    ) -> Dict[str, float]:
        """
        基于排名分配资金
        
        Args:
            ranked_signals: 排名后的信号列表
            total_capital: 总资金
            max_position_size: 最大单个仓位比例
        
        Returns:
            资金分配字典，格式为 {code: position_size}
        """
        # 只分配给前 N 个信号
        top_signals = ranked_signals[:self.top_n]
        
        # 计算权重总和
        total_weight = sum(weight for _, weight in top_signals)
        
        # 分配资金
        capital_allocation = {}
        for code, weight in top_signals:
            # 基于权重分配
            if total_weight > 0:
                allocation_ratio = weight / total_weight
            else:
                allocation_ratio = 1.0 / len(top_signals)
            
            # 限制最大单个仓位
            allocation_ratio = min(allocation_ratio, max_position_size)
            
            # 计算仓位大小（相对于总资金）
            position_size = allocation_ratio * max_position_size * len(top_signals)
            position_size = min(position_size, 0.3)  # 最大 30% 仓位
            
            capital_allocation[code] = position_size
        
        return capital_allocation


# 便捷函数
def calculate_signal_score(
    signal_pullback: bool,
    signal_rebound: bool,
    signal_breakout: bool,
    price: float,
    open_p: float,
    low_p: float,
    sma_l_1: float,
    sma_s_1: float,
    macd_h_1: float,
    macd_h_2: float,
    rsi_1: float,
    vol_1: float,
    vol_2: float,
    ai_prob: float = 0.5,
    params: StrategyParams = None,
) -> SignalScore:
    """
    便捷函数：计算信号评分
    """
    if params is None:
        from quant.core.strategy_params import StrategyParams
        params = StrategyParams()
    
    scorer = SignalScorer(params)
    return scorer.calculate_signal_score(
        signal_pullback, signal_rebound, signal_breakout,
        price, open_p, low_p,
        sma_l_1, sma_s_1,
        macd_h_1, macd_h_2,
        rsi_1, vol_1, vol_2,
        ai_prob,
    )


if __name__ == "__main__":
    # 测试代码
    from quant.core.strategy_params import StrategyParams
    
    params = StrategyParams()
    scorer = SignalScorer(params)
    
    # 测试信号评分
    signal_score = scorer.calculate_signal_score(
        signal_pullback=True,
        signal_rebound=False,
        signal_breakout=False,
        price=10.0,
        open_p=9.9,
        low_p=9.8,
        sma_l_1=10.2,
        sma_s_1=10.1,
        macd_h_1=0.1,
        macd_h_2=-0.05,
        rsi_1=30.0,
        vol_1=1000000,
        vol_2=600000,
        ai_prob=0.7,
    )
    
    print(f"信号评分: {signal_score.total_score:.2f}")
    print(f"质量评级: {signal_score.quality_rating}")
    print(f"各维度评分: {signal_score.to_dict()}")
    
    # 测试动态仓位分配
    position_size = scorer.calculate_dynamic_position_size(
        signal_score, base_position_size=0.1, equity=100000, volatility=0.02
    )
    print(f"动态仓位: {position_size:.3f}")
    
    # 测试动态止盈止损
    tp, sl = scorer.calculate_dynamic_tp_sl(
        signal_score, price=10.0, atr=0.5, trade_type="left"
    )
    print(f"动态止盈止损: TP={tp:.2f}, SL={sl:.2f}")
