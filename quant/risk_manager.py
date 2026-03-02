"""
风险控制模块
实现最大回撤限制、仓位管理和连败保护机制
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from quant.logger import logger
from quant.strategy_params import StrategyParams


@dataclass
class RiskMetrics:
    """风险指标数据类"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    win_rate: float = 0.5
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'volatility': self.volatility,
        }


class RiskController:
    """
    风险控制器
    
    功能：
    1. 最大回撤限制
    2. 动态仓位管理
    3. 连败保护机制
    4. 资金曲线监控
    """
    
    def __init__(
        self,
        max_drawdown_pct: float = 0.15,  # 最大回撤限制 15%
        max_consecutive_losses: int = 5,  # 最大连续亏损次数
        min_win_rate: float = 0.4,  # 最小胜率
        initial_equity: float = 100000.0,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.min_win_rate = min_win_rate
        self.initial_equity = initial_equity
        
        # 风险指标
        self.risk_metrics = RiskMetrics()
        self.risk_metrics.peak_equity = initial_equity
        
        # 交易历史
        self.trade_history = []
        
    def update_equity(self, current_equity: float):
        """
        更新权益并计算风险指标
        
        Args:
            current_equity: 当前权益
        """
        # 更新权益曲线
        self.risk_metrics.equity_curve.append(current_equity)
        
        # 更新峰值
        if current_equity > self.risk_metrics.peak_equity:
            self.risk_metrics.peak_equity = current_equity
        
        # 计算当前回撤
        if self.risk_metrics.peak_equity > 0:
            self.risk_metrics.current_drawdown = (
                self.risk_metrics.peak_equity - current_equity
            ) / self.risk_metrics.peak_equity
        
        # 更新最大回撤
        if self.risk_metrics.current_drawdown > self.risk_metrics.max_drawdown:
            self.risk_metrics.max_drawdown = self.risk_metrics.current_drawdown
        
        # 计算波动率
        if len(self.risk_metrics.equity_curve) > 20:
            returns = pd.Series(self.risk_metrics.equity_curve[-20:]).pct_change().dropna()
            self.risk_metrics.volatility = returns.std()
        
        # 计算夏普比率
        if len(self.trade_history) >= 10:
            trades_df = pd.DataFrame(self.trade_history)
            returns = trades_df['return_pct'] / 100.0
            if len(returns) > 0 and returns.std() > 0:
                self.risk_metrics.sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        
        # 计算胜率
        if len(self.trade_history) > 0:
            wins = sum(1 for t in self.trade_history if t['return_pct'] > 0)
            self.risk_metrics.win_rate = wins / len(self.trade_history)
    
    def record_trade(self, trade_result: Dict):
        """
        记录交易结果
        
        Args:
            trade_result: 交易结果字典，包含 'return_pct', 'hold_days' 等
        """
        self.trade_history.append(trade_result)
        
        # 更新连败计数
        if trade_result['return_pct'] < 0:
            self.risk_metrics.consecutive_losses += 1
            # 更新最大连败
            if self.risk_metrics.consecutive_losses > self.risk_metrics.max_consecutive_losses:
                self.risk_metrics.max_consecutive_losses = self.risk_metrics.consecutive_losses
        else:
            self.risk_metrics.consecutive_losses = 0
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """
        检查风险限制
        
        Returns:
            风险限制状态字典
        """
        return {
            'max_drawdown_exceeded': self.risk_metrics.current_drawdown > self.max_drawdown_pct,
            'max_consecutive_losses_exceeded': (
                self.risk_metrics.consecutive_losses >= self.max_consecutive_losses
            ),
            'min_win_rate_breached': self.risk_metrics.win_rate < self.min_win_rate,
            'trading_allowed': self.is_trading_allowed(),
        }
    
    def is_trading_allowed(self) -> bool:
        """
        检查是否允许继续交易
        
        Returns:
            是否允许交易
        """
        limits = self.check_risk_limits()
        
        # 如果任何一个风险限制被触发，暂停交易
        if limits['max_drawdown_exceeded']:
            logger.warning(f"最大回撤限制被触发: {self.risk_metrics.current_drawdown:.2%} > {self.max_drawdown_pct:.2%}")
            return False
        
        if limits['max_consecutive_losses_exceeded']:
            logger.warning(f"最大连败限制被触发: {self.risk_metrics.consecutive_losses} >= {self.max_consecutive_losses}")
            return False
        
        if limits['min_win_rate_breached'] and len(self.trade_history) >= 20:
            logger.warning(f"最小胜率限制被触发: {self.risk_metrics.win_rate:.2%} < {self.min_win_rate:.2%}")
            return False
        
        return True
    
    def calculate_position_size(
        self,
        base_params: StrategyParams,
        current_equity: float,
        volatility: float = None,
    ) -> float:
        """
        动态仓位管理
        
        根据以下因素调整仓位：
        1. 基础仓位（来自策略参数）
        2. 当前回撤（回撤越大，仓位越小）
        3. 连败次数（连败越多，仓位越小）
        4. 市场波动率（波动率越大，仓位越小）
        
        Args:
            base_params: 基础策略参数
            current_equity: 当前权益
            volatility: 市场波动率（可选）
        
        Returns:
            调整后的仓位大小（0-1）
        """
        base_size = base_params.position_size
        
        # 1. 回撤调整
        drawdown_factor = 1.0
        if self.risk_metrics.current_drawdown > 0:
            # 回撤越大，仓位越小
            drawdown_factor = max(0.3, 1.0 - (self.risk_metrics.current_drawdown / self.max_drawdown_pct))
        
        # 2. 连败调整
        loss_factor = 1.0
        if self.risk_metrics.consecutive_losses > 0:
            # 连败越多，仓位越小
            loss_factor = max(0.3, 1.0 - (self.risk_metrics.consecutive_losses / self.max_consecutive_losses))
        
        # 3. 胜率调整
        win_rate_factor = 1.0
        if len(self.trade_history) >= 10:
            # 胜率越低，仓位越小
            win_rate_factor = max(0.5, self.risk_metrics.win_rate / self.min_win_rate)
        
        # 4. 波动率调整
        volatility_factor = 1.0
        if volatility is not None and volatility > 0:
            # 波动率越大，仓位越小
            normal_volatility = 0.02  # 假设正常波动率为 2%
            volatility_factor = max(0.5, normal_volatility / max(volatility, 0.01))
        
        # 综合调整因子
        adjustment_factor = (
            drawdown_factor * 
            loss_factor * 
            win_rate_factor * 
            volatility_factor
        )
        
        # 计算最终仓位
        adjusted_size = base_size * adjustment_factor
        
        # 限制仓位范围
        min_size = 0.01  # 最小 1% 仓位
        max_size = 0.25  # 最大 25% 仓位
        
        adjusted_size = max(min_size, min(max_size, adjusted_size))
        
        logger.debug(
            f"仓位调整: 基础={base_size:.3f}, 回撤因子={drawdown_factor:.3f}, "
            f"连败因子={loss_factor:.3f}, 胜率因子={win_rate_factor:.3f}, "
            f"波动率因子={volatility_factor:.3f}, 最终={adjusted_size:.3f}"
        )
        
        return adjusted_size
    
    def get_risk_report(self) -> Dict:
        """
        生成风险报告
        
        Returns:
            风险报告字典
        """
        risk_limits = self.check_risk_limits()
        
        return {
            'risk_metrics': self.risk_metrics.to_dict(),
            'risk_limits': risk_limits,
            'trading_allowed': risk_limits['trading_allowed'],
            'total_trades': len(self.trade_history),
            'risk_score': self._calculate_risk_score(),
        }
    
    def _calculate_risk_score(self) -> float:
        """
        计算风险评分（0-100）
        
        评分越高，风险越大
        """
        score = 0.0
        
        # 回撤评分（0-40 分）
        drawdown_score = min(40, (self.risk_metrics.current_drawdown / self.max_drawdown_pct) * 40)
        score += drawdown_score
        
        # 连败评分（0-30 分）
        loss_score = min(30, (self.risk_metrics.consecutive_losses / self.max_consecutive_losses) * 30)
        score += loss_score
        
        # 胜率评分（0-30 分）
        if len(self.trade_history) >= 10:
            win_rate_score = max(0, (1.0 - (self.risk_metrics.win_rate / self.min_win_rate)) * 30)
            score += win_rate_score
        
        return round(score, 2)


class CapitalCurveMonitor:
    """
    资金曲线监控器
    
    功能：
    1. 实时监控资金曲线
    2. 检测异常波动
    3. 生成预警信号
    """
    
    def __init__(
        self,
        alert_threshold_pct: float = 0.05,  # 预警阈值 5%
        volatility_window: int = 20,
    ):
        self.alert_threshold_pct = alert_threshold_pct
        self.volatility_window = volatility_window
        
        self.equity_history = []
        self.alerts = []
    
    def update(self, equity: float, timestamp: str = None):
        """
        更新资金曲线
        
        Args:
            equity: 当前权益
            timestamp: 时间戳
        """
        self.equity_history.append({
            'equity': equity,
            'timestamp': timestamp or pd.Timestamp.now(),
        })
        
        # 检测异常波动
        if len(self.equity_history) >= self.volatility_window:
            self._check_anomalies(equity)
    
    def _check_anomalies(self, current_equity: float):
        """
        检测异常波动
        
        Args:
            current_equity: 当前权益
        """
        if len(self.equity_history) < self.volatility_window:
            return
        
        # 计算历史均值和标准差
        recent_equities = [
            e['equity'] 
            for e in self.equity_history[-self.volatility_window:]
        ]
        mean_equity = np.mean(recent_equities)
        std_equity = np.std(recent_equities)
        
        # 检测异常下跌
        if current_equity < mean_equity - self.alert_threshold_pct * mean_equity:
            alert = {
                'type': 'capital_drawdown',
                'severity': 'high',
                'message': f"资金异常下跌: {current_equity:.2f} < {mean_equity - self.alert_threshold_pct * mean_equity:.2f}",
                'equity': current_equity,
                'expected': mean_equity,
                'timestamp': pd.Timestamp.now(),
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
    
    def get_alerts(self, severity: str = None) -> List[Dict]:
        """
        获取预警信息
        
        Args:
            severity: 严重程度（'high', 'medium', 'low'），None 表示全部
        
        Returns:
            预警列表
        """
        if severity is None:
            return self.alerts
        else:
            return [a for a in self.alerts if a['severity'] == severity]


# 全局风险控制器实例
_global_risk_controller: Optional[RiskController] = None
_global_capital_monitor: Optional[CapitalCurveMonitor] = None


def get_risk_controller(
    max_drawdown_pct: float = 0.15,
    max_consecutive_losses: int = 5,
    min_win_rate: float = 0.4,
) -> RiskController:
    """获取全局风险控制器实例"""
    global _global_risk_controller
    
    if _global_risk_controller is None:
        _global_risk_controller = RiskController(
            max_drawdown_pct=max_drawdown_pct,
            max_consecutive_losses=max_consecutive_losses,
            min_win_rate=min_win_rate,
        )
    
    return _global_risk_controller


def get_capital_monitor(
    alert_threshold_pct: float = 0.05,
) -> CapitalCurveMonitor:
    """获取全局资金曲线监控器实例"""
    global _global_capital_monitor
    
    if _global_capital_monitor is None:
        _global_capital_monitor = CapitalCurveMonitor(
            alert_threshold_pct=alert_threshold_pct,
        )
    
    return _global_capital_monitor


if __name__ == "__main__":
    # 测试代码
    risk_controller = get_risk_controller()
    
    # 模拟交易
    equity = 100000.0
    for i in range(30):
        # 模拟权益波动
        equity *= (1 + np.random.normal(0, 0.02))
        
        # 更新风险指标
        risk_controller.update_equity(equity)
        
        # 记录交易
        trade_result = {
            'return_pct': np.random.normal(0, 5),
            'hold_days': np.random.randint(1, 10),
        }
        risk_controller.record_trade(trade_result)
        
        # 检查风险限制
        risk_limits = risk_controller.check_risk_limits()
        print(f"第 {i+1} 天: 权益={equity:.2f}, "
              f"回撤={risk_controller.risk_metrics.current_drawdown:.2%}, "
              f"交易允许={risk_limits['trading_allowed']}")
        
        # 获取仓位调整
        from quant.strategy_params import StrategyParams
        params = StrategyParams()
        adjusted_size = risk_controller.calculate_position_size(params, equity)
        print(f"  调整后仓位: {adjusted_size:.3f}")
    
    # 生成风险报告
    risk_report = risk_controller.get_risk_report()
    print(f"\n风险评分: {risk_report['risk_score']}")
    print(f"风险指标: {risk_report['risk_metrics']}")
