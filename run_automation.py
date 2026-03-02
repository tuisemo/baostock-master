"""
量化交易系统自动化运行主脚本
执行：数据更新、特征提取、模型训练、回测验证、参数优化等
"""
import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from quant.config import CONF
from quant.logger import logger
from quant.performance_monitor import PerformanceMonitor, OptimizationValidator
from quant.data_updater import update_history_data
from quant.stock_filter import update_stock_list
from quant.features import extract_features
from quant.features_enhanced import extract_all_features
from quant.trainer import load_processed_dataset
from quant.ensemble_trainer import MultiModelEnsemble
from quant.backtester_optimized import batch_backtest_optimized
from quant.optimizer_enhanced import run_enhanced_optimization
from quant.strategy_params import StrategyParams


class AutomationRunner:
    """自动化运行器"""
    
    def __init__(
        self,
        skip_data_update: bool = False,
        skip_training: bool = False,
        skip_backtest: bool = False,
        skip_optimization: bool = False,
        sample_count: int = 100,
        optimization_trials: int = 50,
    ):
        """
        初始化自动化运行器
        
        Args:
            skip_data_update: 是否跳过数据更新
            skip_training: 是否跳过模型训练
            skip_backtest: 是否跳过回测
            skip_optimization: 是否跳过参数优化
            sample_count: 样本股票数量
            optimization_trials: 优化试验次数
        """
        self.skip_data_update = skip_data_update
        self.skip_training = skip_training
        self.skip_backtest = skip_backtest
        self.skip_optimization = skip_optimization
        self.sample_count = sample_count
        self.optimization_trials = optimization_trials
        
        # 初始化性能监控器
        self.monitor = PerformanceMonitor()
        self.monitor.start_monitoring()
        
        # 结果存储
        self.results = {}
        
        logger.info("=" * 80)
        logger.info("量化交易系统自动化运行启动")
        logger.info(f"配置: 数据更新={not self.skip_data_update}, "
                   f"模型训练={not self.skip_training}, "
                   f"回测={not self.skip_backtest}, "
                   f"参数优化={not self.skip_optimization}")
        logger.info(f"样本数量: {self.sample_count}, 优化试验次数: {self.optimization_trials}")
        logger.info("=" * 80)
    
    def run(self):
        """执行完整的自动化运行流程"""
        start_time = time.time()
        
        try:
            # Milestone 1: 数据更新和准备
            self.milestone_1_data_update()
            
            # Milestone 2: 特征工程和特征提取
            self.milestone_2_feature_extraction()
            
            # Milestone 3: 数据集构建和预处理
            self.milestone_3_dataset_preparation()
            
            # Milestone 4: 模型训练（多模型集成）
            if not self.skip_training:
                self.milestone_4_model_training()
            else:
                logger.info("跳过模型训练")
            
            # Milestone 5: 回测验证（优化版）
            if not self.skip_backtest:
                self.milestone_5_backtest()
            else:
                logger.info("跳过回测验证")
            
            # Milestone 6-7: 参数优化
            if not self.skip_optimization:
                self.milestone_6_7_optimization()
            else:
                logger.info("跳过参数优化")
            
            # Milestone 8: 性能评估和报告生成
            self.milestone_8_performance_evaluation()
            
            # Milestone 9: 任务验收和总结
            self.milestone_9_summary()
            
        except Exception as e:
            logger.error(f"自动化运行失败: {e}", exc_info=True)
            raise
        
        finally:
            # 停止性能监控
            self.monitor.stop_monitoring()
            elapsed = time.time() - start_time
            logger.info(f"自动化运行完成，总耗时: {elapsed/60:.2f} 分钟")
    
    def milestone_1_data_update(self):
        """Milestone 1: 数据更新和准备"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 1: 数据更新和准备")
        logger.info("=" * 80)
        
        if self.skip_data_update:
            logger.info("跳过数据更新")
            return
        
        # 拉取最新数据
        logger.info("开始拉取最新历史数据...")
        update_history_data()
        
        # 统计数据
        data_dir = CONF.history_data.data_dir
        data_files = list(Path(data_dir).glob("*.csv"))
        
        self.results['milestone_1'] = {
            'data_files_count': len(data_files),
            'data_dir': str(data_dir),
        }
        
        logger.info(f"数据更新完成: {len(data_files)} 个数据文件")
    
    def milestone_2_feature_extraction(self):
        """Milestone 2: 特征工程和特征提取"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 2: 特征工程和特征提取")
        logger.info("=" * 80)
        
        # 采样股票
        data_dir = CONF.history_data.data_dir
        data_files = list(Path(data_dir).glob("*.csv"))
        sampled_files = np.random.choice(data_files, 
                                         min(self.sample_count, len(data_files)), 
                                         replace=False)
        
        logger.info(f"采样 {len(sampled_files)} 只股票进行特征提取...")
        
        # 特征提取统计
        feature_counts = []
        
        for data_file in sampled_files:
            try:
                # 读取数据
                df = pd.read_csv(data_file)
                
                # 计算技术指标
                params = StrategyParams()
                from quant.analyzer import calculate_indicators
                df = calculate_indicators(df, params)
                
                # 提取所有特征（基础 + 增强）
                df = extract_all_features(df)
                
                # 统计特征数量
                feature_cols = [c for c in df.columns if c.startswith('feat_')]
                feature_counts.append(len(feature_cols))
                
            except Exception as e:
                logger.debug(f"特征提取失败 {data_file}: {e}")
        
        avg_features = np.mean(feature_counts) if feature_counts else 0
        
        self.results['milestone_2'] = {
            'sampled_stocks': len(sampled_files),
            'avg_features_per_stock': avg_features,
        }
        
        logger.info(f"特征提取完成: 平均 {avg_features:.1f} 维特征/股票")
    
    def milestone_3_dataset_preparation(self):
        """Milestone 3: 数据集构建和预处理"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 3: 数据集构建和预处理")
        logger.info("=" * 80)
        
        # 加载处理后的数据集
        X_train, y_train, X_eval, y_eval = load_processed_dataset()
        
        dataset_info = {
            'train_samples': len(X_train),
            'eval_samples': len(X_eval),
            'feature_count': X_train.shape[1],
            'positive_ratio': y_train.sum() / len(y_train),
        }
        
        self.results['milestone_3'] = dataset_info
        
        logger.info(f"数据集构建完成: 训练样本 {dataset_info['train_samples']}, "
                   f"验证样本 {dataset_info['eval_samples']}, "
                   f"特征数 {dataset_info['feature_count']}")
    
    def milestone_4_model_training(self):
        """Milestone 4: 模型训练（多模型集成）"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 4: 模型训练（多模型集成）")
        logger.info("=" * 80)
        
        # 加载数据
        X_train, y_train, X_eval, y_eval = load_processed_dataset()
        
        # 创建多模型集成
        ensemble = MultiModelEnsemble(
            models=['lgb', 'xgb', 'cat'],
            ensemble_method='weighted_avg',
            random_state=42,
        )
        
        # 训练
        logger.info("开始训练多模型集成...")
        ensemble.fit(X_train, y_train, X_eval, y_eval)
        
        # 评估
        y_pred_proba = ensemble.predict_proba(X_eval)
        
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        auc = roc_auc_score(y_eval, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred)
        
        # 保存模型
        model_dir = "models/ensemble_auto"
        ensemble.save(model_dir)
        
        self.results['milestone_4'] = {
            'model_dir': model_dir,
            'auc': auc,
            'accuracy': accuracy,
            'f1': f1,
        }
        
        logger.info(f"模型训练完成: AUC={auc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    def milestone_5_backtest(self):
        """Milestone 5: 回测验证（优化版）"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 5: 回测验证（优化版）")
        logger.info("=" * 80)
        
        # 采样股票
        data_dir = CONF.history_data.data_dir
        data_files = list(Path(data_dir).glob("*.csv"))
        sampled_files = np.random.choice(data_files, 
                                         min(self.sample_count, len(data_files)), 
                                         replace=False)
        codes = [f.stem for f in sampled_files]
        
        # 加载策略参数
        params = StrategyParams()
        
        # 执行回测
        logger.info(f"开始回测验证: {len(codes)} 只股票...")
        results = batch_backtest_optimized(
            codes=codes,
            params=params,
            use_parallel=True,
        )
        
        # 统计结果
        if results:
            df_results = pd.DataFrame(results)
            
            backtest_stats = {
                'num_stocks': len(results),
                'avg_return': df_results['return_pct'].mean(),
                'avg_sharpe': df_results['sharpe'].mean(),
                'avg_win_rate': df_results['win_rate'].mean(),
                'max_drawdown': df_results['max_drawdown'].min(),
            }
            
            self.results['milestone_5'] = backtest_stats
            
            logger.info(f"回测完成: 平均收益 {backtest_stats['avg_return']:.2f}%, "
                       f"平均夏普 {backtest_stats['avg_sharpe']:.2f}")
        else:
            logger.warning("回测结果为空")
            self.results['milestone_5'] = {}
    
    def milestone_6_7_optimization(self):
        """Milestone 6-7: 参数优化（贝叶斯 + CMA-ES）"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 6-7: 参数优化（贝叶斯 + CMA-ES）")
        logger.info("=" * 80)
        
        # 贝叶斯优化
        logger.info("开始贝叶斯优化 (TPE)...")
        results_tpe = run_enhanced_optimization(
            algorithm='tpe',
            n_trials=self.optimization_trials,
            multi_objective=False,
        )
        
        # CMA-ES 优化
        logger.info("开始 CMA-ES 优化...")
        results_cmaes = run_enhanced_optimization(
            algorithm='cmaes',
            n_trials=self.optimization_trials,
            multi_objective=False,
        )
        
        # 对比结果
        self.results['milestone_6_7'] = {
            'tpe_best_score': results_tpe.get('best_score', 0.0),
            'tpe_test_score': results_tpe.get('test_score', 0.0),
            'cmaes_best_score': results_cmaes.get('best_score', 0.0),
            'cmaes_test_score': results_cmaes.get('test_score', 0.0),
            'tpe_params': results_tpe.get('best_params', {}),
            'cmaes_params': results_cmaes.get('best_params', {}),
        }
        
        logger.info(f"优化完成: TPE={self.results['milestone_6_7']['tpe_best_score']:.4f}, "
                   f"CMA-ES={self.results['milestone_6_7']['cmaes_best_score']:.4f}")
    
    def milestone_8_performance_evaluation(self):
        """Milestone 8: 性能评估和报告生成"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 8: 性能评估和报告生成")
        logger.info("=" * 80)
        
        # 收集系统指标
        system_metrics = self.monitor.get_system_metrics()
        
        # 生成总结报告
        report = self._generate_summary_report()
        
        # 保存报告
        report_dir = "data/auto_reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(
            report_dir,
            f"auto_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存结果 JSON
        results_path = os.path.join(
            report_dir,
            f"auto_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.results['milestone_8'] = {
            'report_path': report_path,
            'results_path': results_path,
            'system_metrics': system_metrics,
        }
        
        logger.info(f"性能评估完成: 报告已保存至 {report_path}")
    
    def milestone_9_summary(self):
        """Milestone 9: 任务验收和总结"""
        logger.info("\n" + "=" * 80)
        logger.info("Milestone 9: 任务验收和总结")
        logger.info("=" * 80)
        
        # 打印总结
        logger.info("\n自动化运行总结:")
        logger.info(f"  - Milestone 1 (数据更新): {self.results.get('milestone_1', {}).get('data_files_count', 'N/A')} 个数据文件")
        logger.info(f"  - Milestone 2 (特征提取): {self.results.get('milestone_2', {}).get('avg_features_per_stock', 'N/A'):.1f} 维特征/股票")
        logger.info(f"  - Milestone 3 (数据集): {self.results.get('milestone_3', {}).get('train_samples', 'N/A')} 训练样本")
        logger.info(f"  - Milestone 4 (模型训练): AUC={self.results.get('milestone_4', {}).get('auc', 'N/A'):.4f}")
        logger.info(f"  - Milestone 5 (回测): 收益={self.results.get('milestone_5', {}).get('avg_return', 'N/A'):.2f}%")
        logger.info(f"  - Milestone 6-7 (优化): TPE={self.results.get('milestone_6_7', {}).get('tpe_best_score', 'N/A'):.4f}, CMA-ES={self.results.get('milestone_6_7', {}).get('cmaes_best_score', 'N/A'):.4f}")
        
        logger.info("\n✅ 所有任务执行完成！")
    
    def _generate_summary_report(self) -> str:
        """生成总结报告"""
        report_lines = [
            "# 量化交易系统自动化运行报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 执行概要",
            "",
            "| Milestone | 描述 | 状态 | 结果 |",
            "|-----------|------|------|------|",
        ]
        
        # Milestone 1
        m1 = self.results.get('milestone_1', {})
        report_lines.append(f"| 1 | 数据更新和准备 | ✅ | {m1.get('data_files_count', 'N/A')} 个文件 |")
        
        # Milestone 2
        m2 = self.results.get('milestone_2', {})
        report_lines.append(f"| 2 | 特征工程和特征提取 | ✅ | {m2.get('avg_features_per_stock', 'N/A'):.1f} 维特征 |")
        
        # Milestone 3
        m3 = self.results.get('milestone_3', {})
        report_lines.append(f"| 3 | 数据集构建和预处理 | ✅ | {m3.get('train_samples', 'N/A')} 训练样本 |")
        
        # Milestone 4
        m4 = self.results.get('milestone_4', {})
        if m4:
            report_lines.append(f"| 4 | 模型训练（多模型集成） | ✅ | AUC={m4.get('auc', 'N/A'):.4f} |")
        else:
            report_lines.append(f"| 4 | 模型训练（多模型集成） | ⏭️ | 跳过 |")
        
        # Milestone 5
        m5 = self.results.get('milestone_5', {})
        if m5:
            report_lines.append(f"| 5 | 回测验证（优化版） | ✅ | 收益={m5.get('avg_return', 'N/A'):.2f}% |")
        else:
            report_lines.append(f"| 5 | 回测验证（优化版） | ⏭️ | 跳过 |")
        
        # Milestone 6-7
        m67 = self.results.get('milestone_6_7', {})
        if m67:
            report_lines.append(f"| 6-7 | 参数优化（TPE + CMA-ES） | ✅ | TPE={m67.get('tpe_best_score', 'N/A'):.4f} |")
        else:
            report_lines.append(f"| 6-7 | 参数优化（TPE + CMA-ES） | ⏭️ | 跳过 |")
        
        # Milestone 8
        m8 = self.results.get('milestone_8', {})
        report_lines.append(f"| 8 | 性能评估和报告生成 | ✅ | 报告已生成 |")
        
        # Milestone 9
        report_lines.append(f"| 9 | 任务验收和总结 | ✅ | 完成 |")
        
        report_lines.extend([
            "",
            "## 详细结果",
            "",
            "### 数据更新",
            "",
            f"- 数据文件数量: {m1.get('data_files_count', 'N/A')}",
            f"- 数据目录: {m1.get('data_dir', 'N/A')}",
            "",
            "### 特征提取",
            "",
            f"- 采样股票数量: {m2.get('sampled_stocks', 'N/A')}",
            f"- 平均特征数: {m2.get('avg_features_per_stock', 'N/A'):.1f}",
            "",
            "### 数据集",
            "",
            f"- 训练样本数: {m3.get('train_samples', 'N/A')}",
            f"- 验证样本数: {m3.get('eval_samples', 'N/A')}",
            f"- 特征数: {m3.get('feature_count', 'N/A')}",
            f"- 正样本比例: {m3.get('positive_ratio', 'N/A'):.2%}",
            "",
        ])
        
        # 模型训练
        if m4:
            report_lines.extend([
                "### 模型训练（多模型集成）",
                "",
                f"- 模型目录: {m4.get('model_dir', 'N/A')}",
                f"- AUC: {m4.get('auc', 'N/A'):.4f}",
                f"- Accuracy: {m4.get('accuracy', 'N/A'):.4f}",
                f"- F1 Score: {m4.get('f1', 'N/A'):.4f}",
                "",
            ])
        
        # 回测
        if m5:
            report_lines.extend([
                "### 回测验证（优化版）",
                "",
                f"- 回测股票数: {m5.get('num_stocks', 'N/A')}",
                f"- 平均收益率: {m5.get('avg_return', 'N/A'):.2f}%",
                f"- 平均夏普比率: {m5.get('avg_sharpe', 'N/A'):.2f}",
                f"- 平均胜率: {m5.get('avg_win_rate', 'N/A'):.2f}%",
                f"- 最大回撤: {m5.get('max_drawdown', 'N/A'):.2f}%",
                "",
            ])
        
        # 参数优化
        if m67:
            report_lines.extend([
                "### 参数优化（TPE + CMA-ES）",
                "",
                f"- TPE 最佳得分: {m67.get('tpe_best_score', 'N/A'):.4f}",
                f"- TPE 测试得分: {m67.get('tpe_test_score', 'N/A'):.4f}",
                f"- CMA-ES 最佳得分: {m67.get('cmaes_best_score', 'N/A'):.4f}",
                f"- CMA-ES 测试得分: {m67.get('cmaes_test_score', 'N/A'):.4f}",
                "",
            ])
        
        report_lines.extend([
            "---",
            "",
            "*本报告由自动化运行系统生成*"
        ])
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统自动化运行')
    parser.add_argument('--skip-data-update', action='store_true', help='跳过数据更新')
    parser.add_argument('--skip-training', action='store_true', help='跳过模型训练')
    parser.add_argument('--skip-backtest', action='store_true', help='跳过回测')
    parser.add_argument('--skip-optimization', action='store_true', help='跳过参数优化')
    parser.add_argument('--sample-count', type=int, default=100, help='样本股票数量')
    parser.add_argument('--optimization-trials', type=int, default=50, help='优化试验次数')
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = AutomationRunner(
        skip_data_update=args.skip_data_update,
        skip_training=args.skip_training,
        skip_backtest=args.skip_backtest,
        skip_optimization=args.skip_optimization,
        sample_count=args.sample_count,
        optimization_trials=args.optimization_trials,
    )
    
    # 执行自动化运行
    runner.run()


if __name__ == '__main__':
    main()
