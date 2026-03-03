"""
Stress Testing Framework (Milestone 5.2)
实现压力测试框架，模拟极端市场行情
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from quant.logger import logger


@dataclass
class StressScenario:
    """压力测试场景数据类"""
    scenario_id: str  # 场景ID
    name: str  # 场景名称
    description: str  # 场景描述
    market_drop: float = 0.30  # 市场下跌 30%
    volatility_spike: float = 2.0  # 波动率翻倍
    correlation_increase: float = 0.3  # 相关性增加 0.3
    duration_days: int = 60  # 持续 60 天
    liquidity_crunch: bool = False  # 流动性危机
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'scenario_id': self.scenario_id,
            'name': self.name,
            'description': self.description,
            'market_drop': self.market_drop,
            'volatility_spike': self.volatility_spike,
            'correlation_increase': self.correlation_increase,
            'duration_days': self.duration_days,
            'liquidity_crunch': self.liquidity_crunch,
        }


class StressTester:
    """
    压力测试框架

    功能：
    1. 模拟多种极端市场场景
    2. 计算压力下的组合表现
    3. 生成压力测试报告
    4. 风险等级评估
    """
    
    def __init__(
        self,
        scenarios: Dict[str, StressScenario] = None
    ):
        self.scenarios = scenarios or self._get_default_scenarios()
    
    def _get_default_scenarios(self) -> Dict[str, StressScenario]:
        """
        获取默认压力测试场景
        """
        return {
            '2015_stock_crash': StressScenario(
                scenario_id='2015_stock_crash',
                name='2015年股灾',
                description='6-8月暴跌30%+，高波动，流动性危机',
                market_drop=0.30,       # 市场下跌 30%
                volatility_spike=2.0,   # 波动率翻倍
                correlation_increase=0.3, # 相关性增加 0.3
                duration_days=60,       # 持续 60 天
                liquidity_crunch=True,   # 流动性危机
            ),
            '2020_covid_crash': StressScenario(
                scenario_id='2020_covid_crash',
                name='2020年疫情',
                description='2-3月暴跌20%+，快速反弹，政策救市',
                market_drop=0.20,       # 市场下跌 20%
                volatility_spike=1.5,   # 波动率增加 50%
                correlation_increase=0.2, # 相关性增加 0.2
                duration_days=30,       # 持续 30 天
                liquidity_crunch=False,
            ),
            '2022_april_crash': StressScenario(
                scenario_id='2022_april_crash',
                name='2022年4月暴跌',
                description='单日暴跌10%+，情绪恐慌',
                market_drop=0.10,       # 市场下跌 10%
                volatility_spike=2.5,   # 波动率增加 150%
                correlation_increase=0.5, # 相关性增加 0.5
                duration_days=10,       # 持续 10 天
                liquidity_crunch=False,
            ),
            'black_monday': StressScenario(
                scenario_id='black_monday',
                name='黑色星期一',
                description='全市场暴跌，流动性枯竭',
                market_drop=0.40,       # 市场下跌 40%
                volatility_spike=3.0,   # 波动率增加 200%
                correlation_increase=0.8, # 相关性增加 0.8
                duration_days=5,        # 持续 5 天
                liquidity_crunch=True,   # 流动性危机
            ),
        }
    
    def add_scenario(self, scenario: StressScenario):
        """添加压力测试场景"""
        self.scenarios[scenario.scenario_id] = scenario
        logger.info(f"添加压力测试场景: {scenario.name}")
    
    def remove_scenario(self, scenario_id: str):
        """移除压力测试场景"""
        if scenario_id in self.scenarios:
            del self.scenarios[scenario_id]
            logger.info(f"移除压力测试场景: {scenario_id}")
    
    def list_scenarios(self) -> List[StressScenario]:
        """列出现有场景"""
        return list(self.scenarios.values())
    
    def run_stress_test(
        self,
        portfolio: Dict[str, float],
        securities_info: Dict[str, Dict],
        scenario_name: str = None
    ) -> Dict:
        """
        运行压力测试

        Args:
            portfolio: 组合持仓 {code: weight}
            securities_info: 证券信息 {code: {'sector': str, 'volatility': float, 'beta': float, 'correlations': Dict}}
            scenario_name: 场景名称（None 表示运行所有场景）

        Returns:
            压力测试结果 {scenario_id: result}
        """
        if scenario_name:
            scenarios = {k: v for k, v in self.scenarios.items() if k == scenario_name or v.name == scenario_name}
        else:
            scenarios = self.scenarios
        
        results = {}
        
        for scenario_key, scenario in scenarios.items():
            try:
                result = self._run_single_scenario(portfolio, securities_info, scenario)
                results[scenario_key] = result
            except Exception as e:
                logger.error(f"压力测试失败: {scenario_key}, 错误: {e}")
                results[scenario_key] = {
                    'scenario_name': scenario.name,
                    'error': str(e),
                    'survived': False,
                }
        
        return results
    
    def _run_single_scenario(
        self,
        portfolio: Dict[str, float],
        securities_info: Dict[str, Dict],
        scenario: StressScenario
    ) -> Dict:
        """
        运行单个压力测试场景
        """
        # 计算组合初始价值
        initial_value = 1.0  # 标准化为 1
        
        # 应用场景冲击
        drop_factor = 1.0 - scenario.market_drop
        vol_factor = scenario.volatility_spike
        corr_factor = scenario.correlation_increase
        
        # 计算压力下的组合价值
        stressed_value = 0.0
        stressed_positions = {}
        
        for code, weight in portfolio.items():
            info = securities_info.get(code, {})
            beta = info.get('beta', 1.0)  # Beta 系数
            sector = info.get('sector', '未知')
            
            # 个性化冲击（考虑 Beta）
            personal_drop = scenario.market_drop * beta
            personal_drop_factor = 1.0 - personal_drop
            
            # 如果有流动性危机，额外扣减
            if scenario.liquidity_crunch:
                # 假设小盘股流动性更差，下跌更多
                liquidity_penalty = 0.05 if beta > 1.2 else 0.0
                personal_drop_factor -= liquidity_penalty
            
            stressed_value += weight * personal_drop_factor
            stressed_positions[code] = {
                'initial_weight': weight,
                'final_weight': weight * personal_drop_factor,
                'loss_pct': personal_drop * 100,
            }
        
        # 归一化组合价值（考虑总权重）
        total_weight = sum(portfolio.values())
        if total_weight > 0:
            stressed_value = stressed_value / total_weight
        
        # 计算组合波动率（在压力场景下）
        portfolio_vol = self._calculate_stressed_volatility(
            portfolio, securities_info, vol_factor, corr_factor
        )
        
        # 计算最大回撤（在压力场景下）
        # 简化假设：最大回撤 = 市场下跌 + 波动率影响
        max_drawdown = scenario.market_drop + portfolio_vol * np.sqrt(scenario.duration_days)
        
        # 计算收益/风险比
        expected_return = 0.15  # 假设年化收益 15%
        stress_return = expected_return - scenario.market_drop
        sharpe = stress_return / portfolio_vol if portfolio_vol > 0 else -999.0
        
        # 风险等级评估
        risk_level = self._assess_risk_level(max_drawdown, portfolio_vol, sharpe)
        
        # 计算各行业暴露
        sector_exposure = {}
        for code, weight in portfolio.items():
            info = securities_info.get(code, {})
            sector = info.get('sector', '未知')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        # 检查是否通过测试
        passed = (
            max_drawdown < 0.25 and  # 最大回撤 < 25%
            sharpe > 0.0 and         # 夏普比率 > 0
            stressed_value > 0.50     # 组合价值保留 > 50%
        )
        
        return {
            'scenario_id': scenario.scenario_id,
            'scenario_name': scenario.name,
            'description': scenario.description,
            'stressed_value': stressed_value,
            'loss': initial_value - stressed_value,
            'loss_pct': (1.0 - stressed_value) * 100,
            'portfolio_volatility': portfolio_vol,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'risk_level': risk_level,
            'survived': passed,
            'stressed_positions': stressed_positions,
            'sector_exposure': sector_exposure,
        }
    
    def _calculate_stressed_volatility(
        self,
        portfolio: Dict[str, float],
        securities_info: Dict[str, Dict],
        vol_factor: float,
        corr_factor: float,
    ) -> float:
        """
        计算压力下的组合波动率
        """
        codes = list(portfolio.keys())
        n = len(codes)
        
        if n == 0:
            return 0.0
        
        # 构建协方差矩阵
        cov_matrix = np.zeros((n, n))
        
        for i, code_i in enumerate(codes):
            info_i = securities_info.get(code_i, {})
            vol_i = info_i.get('volatility', 0.02) * vol_factor
            
            # 单资产方差
            cov_matrix[i, i] = vol_i ** 2
            
            # 协方差
            for j, code_j in enumerate(codes):
                if i < j:
                    info_j = securities_info.get(code_j, {})
                    vol_j = info_j.get('volatility', 0.02) * vol_factor
                    
                    # 获取基础相关性
                    base_corr = info_i.get('correlations', {}).get(code_j, 0.0)
                    
                    # 应用压力相关性增加
                    stressed_corr = min(1.0, max(-1.0, base_corr + corr_factor))
                    
                    cov_matrix[i, j] = vol_i * vol_j * stressed_corr
                    cov_matrix[j, i] = cov_matrix[i, j]
        
        # 计算组合波动率
        weights = np.array([portfolio.get(code, 0) for code in codes])
        
        # 归一化权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(max(0.0, portfolio_var))
        
        return portfolio_vol
    
    def _assess_risk_level(
        self,
        max_drawdown: float,
        portfolio_volatility: float,
        sharpe: float,
    ) -> str:
        """
        评估风险等级
        """
        if max_drawdown < 0.10 and sharpe > 1.5:
            return '低风险'
        elif max_drawdown < 0.15 and sharpe > 1.0:
            return '中等风险'
        elif max_drawdown < 0.20 and sharpe > 0.5:
            return '中高风险'
        elif max_drawdown < 0.25:
            return '高风险'
        else:
            return '极高风险'
    
    def generate_stress_report(self, results: Dict) -> str:
        """
        生成压力测试报告

        Args:
            results: 压力测试结果字典

        Returns:
            报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("压力测试报告".center(80))
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for scenario_key, result in results.items():
            if 'error' in result:
                report.append(f"【{result['scenario_name']}】")
                report.append(f"  状态: ❌ 测试失败")
                report.append(f"  错误: {result['error']}")
                report.append("")
                continue
            
            report.append(f"【{result['scenario_name']}】")
            report.append(f"  描述: {result['description']}")
            report.append(f"  压力后价值: {result['stressed_value']:.4f}")
            report.append(f"  亏损比例: {result['loss_pct']:.2f}%")
            report.append(f"  组合波动率: {result['portfolio_volatility']:.4f}")
            report.append(f"  最大回撤: {result['max_drawdown']:.2%}")
            report.append(f"  夏普比率: {result['sharpe']:.4f}")
            report.append(f"  风险等级: {result['risk_level']}")
            report.append(f"  测试结果: {'✅ 通过' if result['survived'] else '❌ 失败'}")
            
            # 显示各行业暴露
            if 'sector_exposure' in result and result['sector_exposure']:
                report.append(f"  行业暴露:")
                for sector, weight in sorted(result['sector_exposure'].items(), key=lambda x: -x[1])[:5]:
                    report.append(f"    {sector}: {weight:.2%}")
            
            report.append("")
        
        # 总体评估
        passed_scenarios = sum(1 for r in results.values() if r.get('survived', False))
        total_scenarios = len(results)
        pass_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        report.append("=" * 80)
        report.append("总体评估".center(80))
        report.append("=" * 80)
        report.append(f"  通过场景: {passed_scenarios}/{total_scenarios}")
        report.append(f"  通过率: {pass_rate:.1%}")
        report.append(f"  综合风险等级: {self._get_overall_risk_level(results)}")
        report.append("")
        
        # 建议
        if pass_rate < 0.5:
            report.append("⚠️  建议: 组合风险过高，建议降低仓位或调整持仓结构")
        elif pass_rate < 0.75:
            report.append("⚠️  建议: 组合存在一定风险，建议优化行业配置或降低集中度")
        else:
            report.append("✅ 建议: 组合风险可控，可以继续持有")
        report.append("")
        
        return "\n".join(report)
    
    def _get_overall_risk_level(self, results: Dict) -> str:
        """
        获取综合风险等级
        """
        failed_count = sum(1 for r in results.values() if not r.get('survived', False))
        
        if failed_count == 0:
            return '稳健型'
        elif failed_count == 1:
            return '平衡型'
        elif failed_count == 2:
            return '进取型'
        else:
            return '激进型'
    
    def save_report(self, results: Dict, output_path: str):
        """
        保存压力测试报告到文件

        Args:
            results: 压力测试结果
            output_path: 输出文件路径
        """
        report = self.generate_stress_report(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"压力测试报告已保存到: {output_path}")
    
    def export_results_json(self, results: Dict, output_path: str):
        """
        导出压力测试结果为 JSON

        Args:
            results: 压力测试结果
            output_path: 输出文件路径
        """
        import json
        
        # 转换结果为可序列化的格式
        export_data = {}
        for key, result in results.items():
            export_data[key] = {
                'scenario_id': result.get('scenario_id'),
                'scenario_name': result.get('scenario_name'),
                'description': result.get('description'),
                'stressed_value': float(result.get('stressed_value', 0)),
                'loss_pct': float(result.get('loss_pct', 0)),
                'portfolio_volatility': float(result.get('portfolio_volatility', 0)),
                'max_drawdown': float(result.get('max_drawdown', 0)),
                'sharpe': float(result.get('sharpe', 0)),
                'risk_level': result.get('risk_level'),
                'survived': bool(result.get('survived', False)),
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"压力测试结果已导出到: {output_path}")


# 全局实例
_global_stress_tester: Optional[StressTester] = None


def get_stress_tester() -> StressTester:
    """获取全局压力测试器实例"""
    global _global_stress_tester
    
    if _global_stress_tester is None:
        _global_stress_tester = StressTester()
    
    return _global_stress_tester


if __name__ == "__main__":
    # 测试代码
    tester = get_stress_tester()
    
    # 定义测试组合
    portfolio = {
        'sh.600000': 0.15,  # 浦发银行
        'sh.600519': 0.12,  # 贵州茅台
        'sz.000001': 0.10,  # 平安银行
        'sz.000858': 0.08,  # 五粮液
        'sh.601318': 0.08,  # 中国平安
    }
    
    # 获取证券信息
    securities_info = {
        'sh.600000': {'sector': '金融', 'volatility': 0.025, 'beta': 0.9, 'correlations': {}},
        'sh.600519': {'sector': '消费', 'volatility': 0.022, 'beta': 1.1, 'correlations': {}},
        'sz.000001': {'sector': '金融', 'volatility': 0.026, 'beta': 0.95, 'correlations': {}},
        'sz.000858': {'sector': '消费', 'volatility': 0.023, 'beta': 1.15, 'correlations': {}},
        'sh.601318': {'sector': '金融', 'volatility': 0.024, 'beta': 0.85, 'correlations': {}},
    }
    
    # 运行压力测试
    results = tester.run_stress_test(portfolio, securities_info)
    
    # 生成报告
    report = tester.generate_stress_report(results)
    print(report)
    
    # 保存报告
    tester.save_report(results, 'stress_test_report.txt')
    
    # 导出 JSON
    tester.export_results_json(results, 'stress_test_results.json')
