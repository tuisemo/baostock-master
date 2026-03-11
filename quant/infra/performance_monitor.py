"""
性能监控与优化验证模块
监控系统性能指标，验证优化效果，生成详细报告
"""
import os
import json
import time
import psutil
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from quant.infra.logger import logger


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, output_dir: str = "data/performance_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {}
        self.start_time = None
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        logger.info("开始性能监控...")
    
    def stop_monitoring(self):
        """停止监控"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.info(f"性能监控结束，总耗时: {elapsed:.2f} 秒")
    
    def record_metric(self, name: str, value: float, unit: str = ""):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_system_metrics(self) -> Dict[str, float]:
        """获取系统性能指标"""
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
        }
        return metrics
    
    def save_metrics(self, filename: str = None):
        """保存指标到文件"""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"指标已保存: {filepath}")


class OptimizationValidator:
    """优化验证器"""
    
    def __init__(self, output_dir: str = "data/performance_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def compare_results(
        self,
        baseline_results: Dict,
        optimized_results: Dict,
        save_report: bool = True
    ) -> Dict:
        """
        对比基线结果和优化结果
        
        Args:
            baseline_results: 基线结果
            optimized_results: 优化结果
            save_report: 是否保存报告
        
        Returns:
            对比报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline": {},
            "optimized": {},
            "improvements": {},
            "summary": {}
        }
        
        # 提取关键指标
        baseline_score = baseline_results.get("best_score", 0.0)
        optimized_score = optimized_results.get("best_score", 0.0)
        baseline_test = baseline_results.get("test_score", 0.0)
        optimized_test = optimized_results.get("test_score", 0.0)
        
        report["baseline"]["best_score"] = baseline_score
        report["baseline"]["test_score"] = baseline_test
        report["optimized"]["best_score"] = optimized_score
        report["optimized"]["test_score"] = optimized_test
        
        # 计算改进
        score_improvement = (optimized_score - baseline_score) / (abs(baseline_score) + 1e-8) * 100
        test_improvement = (optimized_test - baseline_test) / (abs(baseline_test) + 1e-8) * 100
        
        report["improvements"]["best_score_pct"] = score_improvement
        report["improvements"]["test_score_pct"] = test_improvement
        report["improvements"]["best_score_abs"] = optimized_score - baseline_score
        report["improvements"]["test_score_abs"] = optimized_test - baseline_test
        
        # 总结
        report["summary"]["successful"] = score_improvement > 0
        report["summary"]["significant"] = abs(score_improvement) > 5.0
        report["summary"]["recommendation"] = (
            "建议使用优化后的参数" if score_improvement > 0 else "建议保持原有参数"
        )
        
        logger.info(f"优化验证完成:")
        logger.info(f"  基线得分: {baseline_score:.6f} (测试: {baseline_test:.6f})")
        logger.info(f"  优化得分: {optimized_score:.6f} (测试: {optimized_test:.6f})")
        logger.info(f"  改进幅度: {score_improvement:+.2f}% (测试: {test_improvement:+.2f}%)")
        logger.info(f"  建议: {report['summary']['recommendation']}")
        
        if save_report:
            self._save_comparison_report(report)
        
        return report
    
    def validate_model_performance(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        验证模型性能
        
        Args:
            model_name: 模型名称
            y_true: 真实标签
            y_pred_proba: 预测概率
            threshold: 分类阈值
        
        Returns:
            性能指标
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix, classification_report
        )
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_pred_proba),
            "threshold": threshold
        }
        
        # 添加分类报告
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        # 添加混淆矩阵
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        logger.info(f"模型 {model_name} 性能验证:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def _save_comparison_report(self, report: Dict):
        """保存对比报告"""
        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"对比报告已保存: {filepath}")
    
    def generate_summary_report(
        self,
        optimization_results: List[Dict],
        model_performance: Dict,
        system_metrics: Dict,
        output_file: str = None
    ) -> str:
        """
        生成优化总结报告
        
        Args:
            optimization_results: 优化结果列表
            model_performance: 模型性能指标
            system_metrics: 系统性能指标
            output_file: 输出文件路径
        
        Returns:
            Markdown 格式的报告
        """
        if output_file is None:
            output_file = os.path.join(
                self.output_dir, 
                f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
        
        # 生成 Markdown 报告
        md_lines = [
            "# 量化交易系统优化总结报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 优化目标",
            "",
            "- 提升策略模型的预测准确性和稳定性",
            "- 优化参数搜索空间和算法",
            "- 增强特征工程的表达能力",
            "- 实现多模型集成提升鲁棒性",
            "",
            "## 2. 优化结果",
            ""
        ]
        
        # 添加优化结果
        for i, result in enumerate(optimization_results, 1):
            md_lines.append(f"### 优化方案 {i}")
            md_lines.append("")
            md_lines.append(f"- **最佳得分**: {result.get('best_score', 0.0):.6f}")
            md_lines.append(f"- **测试得分**: {result.get('test_score', 0.0):.6f}")
            md_lines.append(f"- **试验次数**: {result.get('n_trials', 0)}")
            md_lines.append("")
            
            # 添加参数信息
            best_params = result.get('best_params', {})
            if best_params:
                md_lines.append("**最优参数**:")
                md_lines.append("")
                md_lines.append("```yaml")
                for key, value in best_params.items():
                    md_lines.append(f"{key}: {value}")
                md_lines.append("```")
                md_lines.append("")
        
        # 添加模型性能
        md_lines.extend([
            "## 3. 模型性能",
            ""
        ])
        
        if model_performance:
            for model_name, metrics in model_performance.items():
                md_lines.append(f"### {model_name}")
                md_lines.append("")
                md_lines.append(f"- **AUC**: {metrics.get('auc', 0.0):.4f}")
                md_lines.append(f"- **Accuracy**: {metrics.get('accuracy', 0.0):.4f}")
                md_lines.append(f"- **Precision**: {metrics.get('precision', 0.0):.4f}")
                md_lines.append(f"- **Recall**: {metrics.get('recall', 0.0):.4f}")
                md_lines.append(f"- **F1 Score**: {metrics.get('f1_score', 0.0):.4f}")
                md_lines.append("")
        
        # 添加系统性能
        md_lines.extend([
            "## 4. 系统性能",
            ""
        ])
        
        if system_metrics:
            md_lines.append(f"- **CPU 使用率**: {system_metrics.get('cpu_percent', 0.0):.1f}%")
            md_lines.append(f"- **内存使用率**: {system_metrics.get('memory_percent', 0.0):.1f}%")
            md_lines.append(f"- **磁盘使用率**: {system_metrics.get('disk_usage_percent', 0.0):.1f}%")
            md_lines.append("")
        
        # 添加改进建议
        md_lines.extend([
            "## 5. 改进建议",
            "",
            "基于本次优化结果，建议：",
            "",
            "1. **特征工程**: 继续探索更多高级特征，如行业特征、宏观经济指标",
            "2. **模型优化**: 尝试深度学习模型，如 LSTM、Transformer",
            "3. **参数优化**: 考虑使用更先进的优化算法，如 Bayesian Optimization",
            "4. **风险管理**: 增加更多风险控制机制，如最大回撤限制、仓位管理",
            "5. **实时监控**: 建立实时的性能监控系统，及时发现和解决问题",
            "",
            "---",
            "",
            "*本报告由自动化优化系统生成*"
        ])
        
        # 保存报告
        report_content = "\n".join(md_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"总结报告已保存: {output_file}")
        
        return report_content


def run_performance_validation():
    """运行性能验证流程"""
    logger.info("=== 性能验证流程启动 ===")
    
    # 初始化监控器
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # 记录系统指标
    system_metrics = monitor.get_system_metrics()
    logger.info(f"系统指标: CPU={system_metrics['cpu_percent']:.1f}%, "
                f"Memory={system_metrics['memory_percent']:.1f}%")
    
    # 停止监控
    monitor.stop_monitoring()
    
    return system_metrics
