"""
Portfolio Optimizer (Milestone 5.1)
实现组合优化器，包括 Kelly 仓位和风险平价分配
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from quant.infra.logger import logger
from quant.core.strategy_params import StrategyParams


def calculate_kelly_position(
    win_rate: float,
    avg_profit: float,
    avg_loss: float,
    risk_fraction: float = 0.5
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


def calculate_risk_parity_weights(
    assets: List[str],
    volatilities: Dict[str, float],
    correlations: Dict[str, Dict[str, float]] = None
) -> Dict[str, float]:
    """
    计算风险平价权重

    Args:
        assets: 资产列表
        volatilities: 各资产波动率
        correlations: 相关系数矩阵 {asset_i: {asset_j: corr}}

    Returns:
        各资产的权重字典 {asset: weight}
    """
    n = len(assets)
    if n == 0:
        return {}

    # 初始权重：与波动率成反比
    weights = {asset: 1.0 / max(volatilities.get(asset, 0.01), 0.01) for asset in assets}

    # 归一化
    total = sum(weights.values())
    if total > 0:
        weights = {asset: w / total for asset, w in weights.items()}

    # 如果有相关系数矩阵，进行二次优化
    if correlations is not None:
        # 迭代优化（简化版）
        for _ in range(10):
            # 计算组合波动率
            portfolio_vol = 0.0
            for i, asset_i in enumerate(assets):
                for j, asset_j in enumerate(assets):
                    corr = correlations.get(asset_i, {}).get(asset_j, 0.0)
                    portfolio_vol += (
                        weights[asset_i] * weights[asset_j] *
                        volatilities.get(asset_i, 0.02) * volatilities.get(asset_j, 0.02) * corr
                    )

            portfolio_vol = np.sqrt(max(portfolio_vol, 0.0))

            # 调整权重使风险贡献相等
            risk_contributions = {}
            for asset in assets:
                risk_contrib = weights[asset] * volatilities.get(asset, 0.02)
                if correlations:
                    # 考虑相关性的风险贡献
                    corr_sum = sum(
                        correlations.get(asset, {}).get(other, 0.0)
                        for other in assets
                    )
                    risk_contrib *= (1 + corr_sum / (2 * n))
                risk_contributions[asset] = risk_contrib

            # 重新分配权重
            total_risk = sum(risk_contributions.values())
            if total_risk > 0:
                weights = {asset: risk_contrib / total_risk for asset, risk_contrib in risk_contributions.items()}

    # 最终归一化
    total = sum(weights.values())
    if total > 0:
        weights = {asset: w / total for asset, w in weights.items()}

    return weights


class PortfolioOptimizer:
    """
    组合优化器 (Milestone 5.1 & 5.2 Enhanced)

    功能：
    1. 基于信号的组合优化
    2. Kelly 仓位计算
    3. 风险平价分配
    4. 行业集中度限制
    5. 相关性风控 (增强)
    6. 动态相关性监控 (新增)
    7. 相关性矩阵质量评估 (新增)
    """

    def __init__(
        self,
        max_positions: int = 10,
        max_single_weight: float = 0.15,
        risk_free_rate: float = 0.03,
        correlation_lookback: int = 60,  # 相关性计算回看窗口
        correlation_min_periods: int = 30,  # 最小数据量
        dynamic_correlation_update: bool = True,  # 启用动态更新
    ):
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        self.risk_free_rate = risk_free_rate
        self.correlation_lookback = correlation_lookback
        self.correlation_min_periods = correlation_min_periods
        self.dynamic_correlation_update = dynamic_correlation_update

        # 动态相关性监控状态
        self.correlation_history = []  # 历史相关性矩阵
        self.correlation_regime = "normal"  # 当前相关性状态
        self.avg_correlation = 0.0  # 平均相关性
        self.correlation_stability = 1.0  # 相关性稳定性指标

    def optimize_portfolio(
        self,
        signals: List[Dict],
        current_positions: Dict[str, float] = None,
        sector_mapping: Dict[str, str] = None,
        correlation_matrix: pd.DataFrame = None,
    ) -> Dict[str, float]:
        """
        优化投资组合

        Args:
            signals: 信号列表，每个信号包含 code, score, expected_return, volatility, sector
            current_positions: 当前持仓 {code: weight}
            sector_mapping: 行业映射 {code: sector}
            correlation_matrix: 相关系数矩阵

        Returns:
            最优权重分配 {code: weight}
        """
        if not signals:
            return {}

        # 1. 选择 top N 个信号
        sorted_signals = sorted(signals, key=lambda x: x.get('total_score', 0), reverse=True)
        selected_signals = sorted_signals[:self.max_positions]

        # 2. 初始权重分配（基于 Kelly）
        weights = {}
        for signal in selected_signals:
            code = signal['code']
            expected_return = signal.get('expected_return', 0.05)
            volatility = signal.get('volatility', 0.02)

            # 简化假设：胜率与信号评分正相关
            win_rate = min(0.6, 0.3 + signal['total_score'] * 0.03)
            avg_profit = abs(expected_return) * 0.8
            avg_loss = -abs(expected_return) * 0.4

            kelly_weight = calculate_kelly_position(
                win_rate, avg_profit, avg_loss
            )

            # 限制单个仓位权重
            kelly_weight = min(kelly_weight, self.max_single_weight)
            weights[code] = kelly_weight

        # 3. 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {code: w / total for code, w in weights.items()}

        # 4. 应用行业集中度限制
        if sector_mapping:
            weights = self._apply_sector_constraints(weights, selected_signals, sector_mapping)

        # 5. 应用相关性限制
        if correlation_matrix is not None:
            weights = self._apply_correlation_constraints(weights, selected_signals, correlation_matrix)

        return weights

    def _apply_sector_constraints(
        self,
        weights: Dict[str, float],
        signals: List[Dict],
        sector_mapping: Dict[str, str],
        max_sector_weight: float = 0.30,
    ) -> Dict[str, float]:
        """
        应用行业集中度限制（单行业 < 30%）

        Args:
            weights: 当前权重
            signals: 信号列表
            sector_mapping: 行业映射
            max_sector_weight: 最大行业权重

        Returns:
            调整后的权重
        """
        # 聚合行业权重
        sector_weights = {}
        for code, weight in weights.items():
            sector = sector_mapping.get(code, "未知")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

        # 检查并调整
        for sector, sector_weight in sector_weights.items():
            if sector_weight > max_sector_weight:
                # 按比例缩减该行业的所有股票
                reduction_factor = max_sector_weight / sector_weight
                for code in weights.keys():
                    code_sector = sector_mapping.get(code, "未知")
                    if code_sector == sector:
                        weights[code] *= reduction_factor

        # 重新归一化
        total = sum(weights.values())
        if total > 0:
            weights = {code: w / total for code, w in weights.items()}

        return weights

    def calculate_correlation_matrix(
        self,
        returns_data: pd.DataFrame,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        计算改进的相关性矩阵

        Args:
            returns_data: 收益率数据
            method: 计算方法 (pearson, spearman, kendall)

        Returns:
            相关性矩阵
        """
        if returns_data is None or len(returns_data) < self.correlation_min_periods:
            logger.warning(f"Insufficient data for correlation calculation")
            return None

        # 使用回看窗口
        if len(returns_data) > self.correlation_lookback:
            returns_data = returns_data.iloc[-self.correlation_lookback:]

        # 计算相关性
        if method == "spearman":
            corr_matrix = returns_data.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = returns_data.corr(method='kendall')
        else:
            corr_matrix = returns_data.corr(method='pearson')

        # 处理缺失值
        corr_matrix = corr_matrix.fillna(0)

        # 确保对角线为1
        for col in corr_matrix.columns:
            corr_matrix.loc[col, col] = 1.0

        # 更新动态监控状态
        if self.dynamic_correlation_update:
            self._update_correlation_monitor(corr_matrix)

        return corr_matrix

    def _update_correlation_monitor(self, corr_matrix: pd.DataFrame):
        """更新相关性监控状态"""
        # 计算平均相关性（排除对角线）
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        correlations = corr_matrix.values[mask]
        self.avg_correlation = np.mean(np.abs(correlations))

        # 记录历史
        self.correlation_history.append({
            'timestamp': pd.Timestamp.now(),
            'avg_correlation': self.avg_correlation,
            'matrix': corr_matrix.copy()
        })

        # 保持历史在合理范围
        if len(self.correlation_history) > 20:
            self.correlation_history = self.correlation_history[-20:]

        # 计算稳定性
        if len(self.correlation_history) >= 5:
            recent_avgs = [h['avg_correlation'] for h in self.correlation_history[-5:]]
            self.correlation_stability = 1.0 - np.std(recent_avgs)

        # 判断相关性状态
        if self.avg_correlation > 0.7:
            self.correlation_regime = "high"
        elif self.avg_correlation > 0.5:
            self.correlation_regime = "elevated"
        else:
            self.correlation_regime = "normal"

    def get_dynamic_correlation_threshold(self) -> float:
        """获取动态相关性阈值"""
        # 在高相关性环境下降低阈值，增加分散要求
        if self.correlation_regime == "high":
            return 0.5  # 更严格的阈值
        elif self.correlation_regime == "elevated":
            return 0.6
        return 0.7  # 正常环境

    def _apply_correlation_constraints(
        self,
        weights: Dict[str, float],
        signals: List[Dict],
        correlation_matrix: pd.DataFrame,
        max_correlation: float = None,
    ) -> Dict[str, float]:
        """
        应用增强的相关性限制（避免高相关股票集中）

        Args:
            weights: 当前权重
            signals: 信号列表
            correlation_matrix: 相关系数矩阵
            max_correlation: 最大允许相关性 (None for dynamic)

        Returns:
            调整后的权重
        """
        if max_correlation is None:
            max_correlation = self.get_dynamic_correlation_threshold()

        codes = list(weights.keys())

        # 找到高相关的股票对
        high_corr_pairs = []
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                if code1 in correlation_matrix.columns and code2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[code1, code2]
                    if abs(corr) > max_correlation:
                        high_corr_pairs.append((code1, code2, corr))

        # 按相关性排序，优先处理最高相关的
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        # 相关性风险分级处理
        if high_corr_pairs:
            # 找到所有涉及高相关股票对的股票
            high_corr_codes = set()
            extreme_corr_codes = set()  # 极端高相关 (>0.9)

            for code1, code2, corr in high_corr_pairs:
                high_corr_codes.add(code1)
                high_corr_codes.add(code2)
                if abs(corr) > 0.9:
                    extreme_corr_codes.add(code1)
                    extreme_corr_codes.add(code2)

            # 动态调整因子：根据相关性环境调整
            base_reduction = 0.8
            if self.correlation_regime == "high":
                base_reduction = 0.6  # 高相关环境下更激进地降低

            # 极端高相关股票：大幅削减或剔除
            for code in extreme_corr_codes:
                if code in weights:
                    weights[code] *= 0.3  # 削减70%

            # 一般高相关股票：适度降低
            for code in high_corr_codes - extreme_corr_codes:
                if code in weights:
                    weights[code] *= base_reduction

            # 重新归一化
            total = sum(weights.values())
            if total > 0:
                weights = {code: w / total for code, w in weights.items()}

            logger.debug(f"Correlation adjustment: {len(high_corr_pairs)} pairs, "
                        f"regime={self.correlation_regime}, "
                        f"threshold={max_correlation}")

        return weights

    def get_correlation_risk_report(self) -> Dict:
        """生成相关性风险报告"""
        return {
            'avg_correlation': self.avg_correlation,
            'correlation_regime': self.correlation_regime,
            'correlation_stability': self.correlation_stability,
            'dynamic_threshold': self.get_dynamic_correlation_threshold(),
            'history_length': len(self.correlation_history),
        }

    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        rebalance_threshold: float = 0.05,
    ) -> Dict[str, float]:
        """
        组合再平衡

        Args:
            target_weights: 目标权重
            current_weights: 当前权重
            rebalance_threshold: 再平衡阈值（5%）

        Returns:
            调整后的权重
        """
        # 合并所有股票
        all_codes = set(target_weights.keys()) | set(current_weights.keys())

        adjusted_weights = {}

        for code in all_codes:
            target = target_weights.get(code, 0.0)
            current = current_weights.get(code, 0.0)

            # 检查是否需要调整
            if abs(target - current) > rebalance_threshold:
                adjusted_weights[code] = target
            else:
                # 保持在当前权重
                adjusted_weights[code] = current

        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0 and total != 1.0:
            adjusted_weights = {code: w / total for code, w in adjusted_weights.items()}

        return adjusted_weights

    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns_data: pd.DataFrame,
        risk_free_rate: float = None,
    ) -> Dict:
        """
        计算组合指标

        Args:
            weights: 组合权重
            returns_data: 收益率数据（DataFrame，每列是一只股票的收益率）
            risk_free_rate: 无风险利率

        Returns:
            组合指标字典
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # 提取有效股票
        valid_codes = [code for code in weights.keys() if code in returns_data.columns]

        if not valid_codes:
            return {
                'return': 0.0,
                'volatility': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
            }

        # 提取权重和收益率
        weight_array = np.array([weights[code] for code in valid_codes])
        returns_array = returns_data[valid_codes].values

        # 计算组合收益率
        portfolio_returns = (returns_array * weight_array).sum(axis=1)

        # 计算指标
        portfolio_return = portfolio_returns.mean() * 252  # 年化收益率
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # 年化波动率
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0

        # 计算最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
        }

    def optimize_mean_variance(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: pd.DataFrame,
        target_volatility: Optional[float] = None,
        target_return: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        均值-方差优化（经典 Markowitz）

        Args:
            expected_returns: 预期收益率 {code: return}
            cov_matrix: 协方差矩阵
            target_volatility: 目标波动率（可选）
            target_return: 目标收益率（可选）

        Returns:
            最优权重
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("scipy 未安装，无法进行均值-方差优化")
            return {}

        codes = list(expected_returns.keys())

        # 目标函数：最小化组合方差
        def objective(weights):
            return weights @ cov_matrix.values @ weights

        # 约束条件
        constraints = []

        # 权重和为 1
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        # 如果指定了目标收益率
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: w @ np.array([expected_returns[code] for code in codes]) - target_return
            })

        # 权重边界（允许卖空或不允许）
        bounds = [(0, 1) for _ in codes]  # 不允许卖空

        # 初始权重（等权重）
        initial_weights = np.array([1.0 / len(codes) for _ in codes])

        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = {code: result.x[i] for i, code in enumerate(codes)}
            logger.info(f"均值-方差优化成功，目标函数值: {result.fun:.6f}")
            return weights
        else:
            logger.error(f"均值-方差优化失败: {result.message}")
            return {}


# 全局实例
_global_portfolio_optimizer: Optional[PortfolioOptimizer] = None


def get_portfolio_optimizer(
    max_positions: int = 10,
    max_single_weight: float = 0.15,
) -> PortfolioOptimizer:
    """获取全局组合优化器实例"""
    global _global_portfolio_optimizer

    if _global_portfolio_optimizer is None:
        _global_portfolio_optimizer = PortfolioOptimizer(
            max_positions=max_positions,
            max_single_weight=max_single_weight,
        )

    return _global_portfolio_optimizer


if __name__ == "__main__":
    # 测试代码
    optimizer = get_portfolio_optimizer()

    # 测试 Kelly 公式
    kelly = calculate_kelly_position(0.5, 0.05, -0.03)
    print(f"Kelly 仓位: {kelly:.3f}")

    # 测试风险平价
    assets = ['stock1', 'stock2', 'stock3']
    volatilities = {'stock1': 0.02, 'stock2': 0.03, 'stock3': 0.015}
    weights_rp = calculate_risk_parity_weights(assets, volatilities)
    print(f"风险平价权重: {weights_rp}")

    # 测试组合优化
    signals = [
        {'code': 'sh.600000', 'total_score': 0.8, 'expected_return': 0.05, 'volatility': 0.02, 'sector': '金融'},
        {'code': 'sh.600519', 'total_score': 0.7, 'expected_return': 0.06, 'volatility': 0.025, 'sector': '消费'},
        {'code': 'sz.000001', 'total_score': 0.6, 'expected_return': 0.04, 'volatility': 0.022, 'sector': '金融'},
    ]
    sector_mapping = {
        'sh.600000': '金融',
        'sh.600519': '消费',
        'sz.000001': '金融',
    }

    weights = optimizer.optimize_portfolio(signals, sector_mapping=sector_mapping)
    print(f"优化后权重: {weights}")
