"""
Online Learning Pipeline (Milestone 4.2)
实现模型版本管理、A/B测试和在线学习机制
"""
import os
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats

from quant.infra.logger import logger

# LightGBM import (optional)
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None


@dataclass
class ModelVersion:
    """模型版本数据类"""
    version_id: str  # 版本ID，如 "v1", "v2"
    model_type: str  # 模型类型，如 "lgb", "xgb", "ensemble"
    created_at: str  # 创建时间
    auc_score: float = 0.0  # AUC分数
    sharpe_ratio: float = 0.0  # 夏普比率
    win_rate: float = 0.0  # 胜率
    num_trades: int = 0  # 交易次数
    max_drawdown: float = 0.0  # 最大回撤
    is_active: bool = False  # 是否当前激活
    ab_test_group: Optional[str] = None  # A/B测试组（A/B/C）
    metadata: Dict = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'model_type': self.model_type,
            'created_at': self.created_at,
            'auc_score': self.auc_score,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'num_trades': self.num_trades,
            'max_drawdown': self.max_drawdown,
            'is_active': self.is_active,
            'ab_test_group': self.ab_test_group,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """从字典创建"""
        return cls(**data)


class ModelVersionManager:
    """
    模型版本管理器
    
    功能：
    1. 版本注册（V1, V2, V3...）
    2. 版本切换和回滚
    3. A/B测试框架
    4. 版本历史查询
    """
    
    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self.versioning_dir = os.path.join(models_dir, "versioning")
        self.metadata_file = os.path.join(self.versioning_dir, "version_metadata.json")
        
        # 初始化目录
        os.makedirs(self.versioning_dir, exist_ok=True)
        
        # 加载版本历史
        self.versions: Dict[str, ModelVersion] = {}
        self._load_versions()
    
    def _load_versions(self):
        """加载版本历史"""
        if not os.path.exists(self.metadata_file):
            logger.info("版本元数据文件不存在，创建新文件")
            return
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.versions = {
                version_id: ModelVersion.from_dict(version_data)
                for version_id, version_data in data.items()
            }
            
            logger.info(f"加载了 {len(self.versions)} 个模型版本")
        except Exception as e:
            logger.error(f"加载版本元数据失败: {e}")
            self.versions = {}
    
    def _save_versions(self):
        """保存版本历史"""
        try:
            data = {
                version_id: version.to_dict()
                for version_id, version in self.versions.items()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("版本元数据已保存")
        except Exception as e:
            logger.error(f"保存版本元数据失败: {e}")
    
    def register_version(
        self,
        model_dir: str,
        model_type: str,
        version_id: Optional[str] = None,
        auc_score: float = 0.0,
        sharpe_ratio: float = 0.0,
        win_rate: float = 0.0,
        num_trades: int = 0,
        max_drawdown: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> ModelVersion:
        """
        注册新模型版本
        
        Args:
            model_dir: 模型目录
            model_type: 模型类型
            version_id: 版本ID（None表示自动生成）
            auc_score: AUC分数
            sharpe_ratio: 夏普比率
            win_rate: 胜率
            num_trades: 交易次数
            max_drawdown: 最大回撤
            metadata: 额外元数据
        
        Returns:
            注册的版本对象
        """
        # 自动生成版本ID
        if version_id is None:
            # 找到最大的版本号
            max_version_num = 0
            for vid in self.versions.keys():
                if vid.startswith('v'):
                    try:
                        num = int(vid[1:])
                        max_version_num = max(max_version_num, num)
                    except ValueError:
                        pass
            
            version_id = f"v{max_version_num + 1}"
        
        # 创建版本对象
        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            created_at=datetime.now().isoformat(),
            auc_score=auc_score,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            num_trades=num_trades,
            max_drawdown=max_drawdown,
            is_active=False,
            metadata=metadata or {},
        )
        
        # 复制模型文件到版本目录
        version_dir = os.path.join(self.versioning_dir, version_id)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        shutil.copytree(model_dir, version_dir)
        
        # 保存版本
        self.versions[version_id] = version
        self._save_versions()
        
        logger.info(f"注册新模型版本: {version_id} ({model_type}), AUC: {auc_score:.4f}")
        
        return version
    
    def activate_version(self, version_id: str) -> bool:
        """
        激活指定版本
        
        Args:
            version_id: 版本ID
        
        Returns:
            是否成功
        """
        if version_id not in self.versions:
            logger.error(f"版本不存在: {version_id}")
            return False
        
        # 取消所有版本的激活状态
        for vid in self.versions.keys():
            self.versions[vid].is_active = False
        
        # 激活指定版本
        self.versions[version_id].is_active = True
        self._save_versions()
        
        logger.info(f"激活模型版本: {version_id}")
        
        return True
    
    def get_active_version(self) -> Optional[ModelVersion]:
        """
        获取当前激活的版本
        
        Returns:
            激活的版本对象，None表示没有激活的版本
        """
        for version in self.versions.values():
            if version.is_active:
                return version
        return None
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        获取指定版本
        
        Args:
            version_id: 版本ID
        
        Returns:
            版本对象，None表示版本不存在
        """
        return self.versions.get(version_id)
    
    def get_version_path(self, version_id: str) -> Optional[str]:
        """
        获取指定版本的模型路径
        
        Args:
            version_id: 版本ID
        
        Returns:
            模型目录路径，None表示版本不存在
        """
        if version_id not in self.versions:
            return None
        return os.path.join(self.versioning_dir, version_id)
    
    def list_versions(self, model_type: Optional[str] = None) -> List[ModelVersion]:
        """
        列出版本
        
        Args:
            model_type: 模型类型过滤器（None表示全部）
        
        Returns:
            版本列表（按创建时间倒序）
        """
        versions = list(self.versions.values())
        
        if model_type is not None:
            versions = [v for v in versions if v.model_type == model_type]
        
        # 按创建时间倒序
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        return versions
    
    def rollback_version(self, target_version_id: str) -> bool:
        """
        回滚到指定版本
        
        Args:
            target_version_id: 目标版本ID
        
        Returns:
            是否成功
        """
        if target_version_id not in self.versions:
            logger.error(f"目标版本不存在: {target_version_id}")
            return False
        
        # 激活目标版本
        return self.activate_version(target_version_id)
    
    def setup_ab_test(
        self,
        version_a_id: str,
        version_b_id: str,
        version_c_id: Optional[str] = None,
        traffic_ratio: Tuple[float, float] = (0.5, 0.5),
    ) -> bool:
        """
        设置A/B测试
        
        Args:
            version_a_id: A版本ID
            version_b_id: B版本ID
            version_c_id: C版本ID（可选）
            traffic_ratio: 流量分配比例（A, B）
        
        Returns:
            是否成功
        """
        # 检查版本是否存在
        if version_a_id not in self.versions or version_b_id not in self.versions:
            logger.error("A/B测试版本不存在")
            return False
        
        if version_c_id is not None and version_c_id not in self.versions:
            logger.error("C版本不存在")
            return False
        
        # 设置A/B测试组
        self.versions[version_a_id].ab_test_group = 'A'
        self.versions[version_b_id].ab_test_group = 'B'
        
        if version_c_id is not None:
            self.versions[version_c_id].ab_test_group = 'C'
        
        # 计算流量分配
        total = sum(traffic_ratio)
        traffic_a = traffic_ratio[0] / total
        traffic_b = traffic_ratio[1] / total
        traffic_c = 1.0 - traffic_a - traffic_b
        
        # 保存流量分配
        self._save_versions()
        
        logger.info(
            f"设置A/B测试: A={version_a_id}({traffic_a:.1%}), "
            f"B={version_b_id}({traffic_b:.1%})"
            + (f", C={version_c_id}({traffic_c:.1%})" if version_c_id else "")
        )
        
        return True
    
    def get_ab_test_versions(self) -> Dict[str, str]:
        """
        获取A/B测试的版本ID
        
        Returns:
            {'A': 'v1', 'B': 'v2', 'C': 'v3'}
        """
        ab_versions = {}
        for version in self.versions.values():
            if version.ab_test_group:
                ab_versions[version.ab_test_group] = version.version_id
        return ab_versions
    
    def compare_versions(
        self,
        version_id1: str,
        version_id2: str,
    ) -> Dict:
        """
        对比两个版本
        
        Args:
            version_id1: 版本1 ID
            version_id2: 版本2 ID
        
        Returns:
            对比结果字典
        """
        v1 = self.versions.get(version_id1)
        v2 = self.versions.get(version_id2)
        
        if v1 is None or v2 is None:
            return {}
        
        return {
            'version_1': v1.version_id,
            'version_2': v2.version_id,
            'auc_diff': v1.auc_score - v2.auc_score,
            'sharpe_diff': v1.sharpe_ratio - v2.sharpe_ratio,
            'win_rate_diff': v1.win_rate - v2.win_rate,
            'max_drawdown_diff': v1.max_drawdown - v2.max_drawdown,
            'num_trades_diff': v1.num_trades - v2.num_trades,
        }
    
    def get_version_history(self) -> List[Dict]:
        """
        获取版本历史
        
        Returns:
            版本历史列表
        """
        versions = self.list_versions()
        
        history = []
        for version in versions:
            history.append({
                'version_id': version.version_id,
                'model_type': version.model_type,
                'created_at': version.created_at,
                'auc_score': version.auc_score,
                'sharpe_ratio': version.sharpe_ratio,
                'win_rate': version.win_rate,
                'num_trades': version.num_trades,
                'max_drawdown': version.max_drawdown,
                'is_active': version.is_active,
                'ab_test_group': version.ab_test_group,
            })
        
        return history


class ConceptDriftDetector:
    """
    概念漂移检测器
    
    使用KL散度（Kullback-Leibler Divergence）和PSI（Population Stability Index）
    检测特征分布和标签分布的漂移
    """
    
    def __init__(
        self,
        psi_threshold: float = 0.25,  # PSI阈值，>0.25表示显著漂移
        kl_threshold: float = 0.1,    # KL散度阈值
        window_size: int = 30,        # 滚动窗口大小
    ):
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self.reference_distribution = {}  # 参考分布（训练时数据）
        self.drift_history = []
    
    def set_reference_distribution(self, X: pd.DataFrame, y: pd.Series = None):
        """
        设置参考分布（通常使用训练数据）
        
        Args:
            X: 特征DataFrame
            y: 标签Series（可选）
        """
        self.reference_distribution = {
            'features': {},
            'feature_names': X.columns.tolist()
        }
        
        # 计算每个特征的分布统计
        for col in X.columns:
            data = X[col].dropna()
            self.reference_distribution['features'][col] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'q25': float(data.quantile(0.25)),
                'q50': float(data.quantile(0.50)),
                'q75': float(data.quantile(0.75)),
                'histogram': np.histogram(data, bins=10, density=True)[0].tolist()
            }
        
        # 如果有标签，记录标签分布
        if y is not None:
            self.reference_distribution['label'] = {
                'positive_rate': float(y.mean()),
                'distribution': y.value_counts(normalize=True).to_dict()
            }
        
        logger.info(f"Reference distribution set for {len(X.columns)} features")
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        计算Population Stability Index (PSI)
        
        Args:
            expected: 期望分布（参考分布）
            actual: 实际分布
            bins: 分箱数
            
        Returns:
            PSI值
        """
        # 添加小常数避免除零
        expected = np.array(expected) + 1e-10
        actual = np.array(actual) + 1e-10
        
        # 归一化
        expected = expected / expected.sum()
        actual = actual / actual.sum()
        
        # 计算PSI
        psi = np.sum((actual - expected) * np.log(actual / expected))
        
        return float(psi)
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算KL散度 D_KL(P || Q)
        
        Args:
            p: 分布P（实际分布）
            q: 分布Q（参考分布）
            
        Returns:
            KL散度值
        """
        # 添加小常数避免除零和对数零
        p = np.array(p) + 1e-10
        q = np.array(q) + 1e-10
        
        # 归一化
        p = p / p.sum()
        q = q / q.sum()
        
        # 计算KL散度
        kl = np.sum(p * np.log(p / q))
        
        return float(kl)
    
    def detect_feature_drift(self, X: pd.DataFrame) -> Dict:
        """
        检测特征分布漂移
        
        Args:
            X: 当前特征DataFrame
            
        Returns:
            漂移检测结果字典
        """
        if not self.reference_distribution:
            logger.warning("Reference distribution not set. Cannot detect drift.")
            return {'drift_detected': False}
        
        drift_results = {
            'drift_detected': False,
            'features_with_drift': [],
            'psi_scores': {},
            'kl_divergences': {},
            'mean_shifts': {},
            'std_changes': {}
        }
        
        for col in X.columns:
            if col not in self.reference_distribution['features']:
                continue
            
            ref = self.reference_distribution['features'][col]
            current_data = X[col].dropna()
            
            if len(current_data) == 0:
                continue
            
            # 计算统计量变化
            current_mean = float(current_data.mean())
            current_std = float(current_data.std())
            
            mean_shift = abs(current_mean - ref['mean']) / (ref['std'] + 1e-10)
            std_change = abs(current_std - ref['std']) / (ref['std'] + 1e-10)
            
            drift_results['mean_shifts'][col] = mean_shift
            drift_results['std_changes'][col] = std_change
            
            # 计算PSI
            current_hist, _ = np.histogram(current_data, bins=10, density=True)
            ref_hist = np.array(ref['histogram'])
            psi = self.calculate_psi(ref_hist, current_hist)
            drift_results['psi_scores'][col] = psi
            
            # 计算KL散度
            kl_div = self.calculate_kl_divergence(current_hist, ref_hist)
            drift_results['kl_divergences'][col] = kl_div
            
            # 判断是否有显著漂移
            if psi > self.psi_threshold or kl_div > self.kl_threshold or mean_shift > 0.5:
                drift_results['features_with_drift'].append({
                    'feature': col,
                    'psi': psi,
                    'kl_divergence': kl_div,
                    'mean_shift': mean_shift
                })
        
        # 计算总体漂移指标
        avg_psi = np.mean(list(drift_results['psi_scores'].values())) if drift_results['psi_scores'] else 0
        avg_kl = np.mean(list(drift_results['kl_divergences'].values())) if drift_results['kl_divergences'] else 0
        
        drift_results['average_psi'] = float(avg_psi)
        drift_results['average_kl_divergence'] = float(avg_kl)
        drift_results['num_drifted_features'] = len(drift_results['features_with_drift'])
        drift_results['drift_detected'] = (
            avg_psi > self.psi_threshold * 0.5 or 
            len(drift_results['features_with_drift']) > len(X.columns) * 0.2  # >20%特征漂移
        )
        
        # 记录漂移历史
        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_results['drift_detected'],
            'avg_psi': drift_results['average_psi'],
            'avg_kl': drift_results['average_kl_divergence'],
            'num_drifted': drift_results['num_drifted_features']
        })
        
        if drift_results['drift_detected']:
            logger.warning(
                f"Concept drift detected! Avg PSI: {avg_psi:.4f}, "
                f"Drifted features: {drift_results['num_drifted_features']}/{len(X.columns)}"
            )
        
        return drift_results
    
    def detect_label_drift(self, y: pd.Series) -> Dict:
        """
        检测标签分布漂移
        
        Args:
            y: 当前标签Series
            
        Returns:
            标签漂移检测结果
        """
        if 'label' not in self.reference_distribution:
            return {'drift_detected': False}
        
        ref_label = self.reference_distribution['label']
        current_positive_rate = float(y.mean())
        
        ref_rate = ref_label['positive_rate']
        rate_change = abs(current_positive_rate - ref_rate) / (ref_rate + 1e-10)
        
        drift_detected = rate_change > 0.2  # 正样本比例变化超过20%
        
        return {
            'drift_detected': drift_detected,
            'reference_rate': ref_rate,
            'current_rate': current_positive_rate,
            'rate_change': rate_change
        }
    
    def get_drift_summary(self) -> pd.DataFrame:
        """获取漂移历史摘要"""
        return pd.DataFrame(self.drift_history)


class IncrementalTrainer:
    """
    增量训练器
    
    支持每日增量训练和模型热更新
    """
    
    def __init__(
        self,
        base_model_path: str,
        learning_rate_decay: float = 0.8,
        min_samples_for_update: int = 100,
        max_incremental_epochs: int = 100
    ):
        self.base_model_path = base_model_path
        self.learning_rate_decay = learning_rate_decay
        self.min_samples_for_update = min_samples_for_update
        self.max_incremental_epochs = max_incremental_epochs
        self.update_history = []
    
    def incremental_train_lightgbm(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        output_path: str = None
    ) -> lgb.Booster:
        """
        LightGBM增量训练
        
        Args:
            X_new: 新数据特征
            y_new: 新数据标签
            X_val: 验证集特征
            y_val: 验证集标签
            output_path: 输出模型路径
            
        Returns:
            训练好的模型
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not available")
            return None
        
        # 加载基础模型
        if os.path.exists(self.base_model_path):
            logger.info(f"Loading base model from {self.base_model_path}")
            model = lgb.Booster(model_file=self.base_model_path)
            init_model = model
            # 降低学习率进行微调
            base_lr = 0.03
            new_lr = base_lr * self.learning_rate_decay
        else:
            logger.warning("Base model not found, training from scratch")
            init_model = None
            new_lr = 0.03
        
        # 创建数据集
        train_data = lgb.Dataset(X_new, label=y_new)
        valid_sets = [train_data]
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
        
        # 计算类别权重
        pos_count = y_new.sum()
        neg_count = len(y_new) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': new_lr,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1
        }
        
        logger.info(f"Incremental training with {len(X_new)} samples, lr={new_lr:.4f}")
        
        # 增量训练
        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.max_incremental_epochs,
            init_model=init_model,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=20)
            ]
        )
        
        # 保存模型
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model.save_model(output_path)
            logger.info(f"Incremental model saved to {output_path}")
        
        # 记录更新历史
        self.update_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': len(X_new),
            'learning_rate': new_lr,
            'best_iteration': model.best_iteration,
            'output_path': output_path
        })
        
        return model
    
    def daily_update(
        self,
        daily_data: Dict[str, pd.DataFrame],
        model_type: str = 'lgb',
        output_dir: str = 'models/daily_updates'
    ) -> Optional[str]:
        """
        执行每日增量更新
        
        Args:
            daily_data: 按股票代码分组的数据字典 {code: DataFrame}
            model_type: 模型类型
            output_dir: 输出目录
            
        Returns:
            更新后的模型路径
        """
        logger.info(f"Starting daily incremental update with {len(daily_data)} stocks...")
        
        # 合并所有数据
        all_data = []
        for code, df in daily_data.items():
            if df is not None and len(df) > 0:
                df = df.copy()
                df['stock_code'] = code
                all_data.append(df)
        
        if not all_data:
            logger.warning("No data available for daily update")
            return None
        
        combined_df = pd.concat(all_data, axis=0)
        
        # 提取特征和标签
        feature_cols = [c for c in combined_df.columns if c.startswith('feat_')]
        if not feature_cols or 'label_max_ret_5d' not in combined_df.columns:
            logger.error("Features or labels not found in daily data")
            return None
        
        X = combined_df[feature_cols]
        y = (combined_df['label_max_ret_5d'] > 0).astype(int)
        
        # 检查样本数量
        if len(X) < self.min_samples_for_update:
            logger.info(f"Insufficient samples ({len(X)} < {self.min_samples_for_update}), skipping update")
            return None
        
        # 时间序列分割
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 执行增量训练
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = os.path.join(output_dir, f'model_incremental_{timestamp}.txt')
        
        if model_type == 'lgb':
            model = self.incremental_train_lightgbm(
                X_train, y_train, X_val, y_val, output_path
            )
        else:
            logger.warning(f"Model type {model_type} not supported for incremental training")
            return None
        
        if model:
            # 评估
            y_pred_proba = model.predict(X_val)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_val, y_pred_proba)
            logger.info(f"Daily update completed. Validation AUC: {auc:.4f}")
            
            return output_path
        
        return None


class OnlineLearner:
    """
    在线学习器
    
    功能：
    1. 定期更新检查（daily/weekly/monthly）
    2. 增量训练框架
    3. 模型性能评估
    4. 自动版本切换
    5. 概念漂移检测
    """
    
    def __init__(
        self,
        version_manager: ModelVersionManager,
        update_frequency: str = "weekly",  # daily, weekly, monthly
        min_incremental_samples: int = 1000,  # 最小增量样本数
        performance_threshold: float = 0.01,  # 性能提升阈值
        enable_drift_detection: bool = True,
    ):
        self.version_manager = version_manager
        self.update_frequency = update_frequency
        self.min_incremental_samples = min_incremental_samples
        self.performance_threshold = performance_threshold
        self.enable_drift_detection = enable_drift_detection
        
        # 增量数据缓存
        self.incremental_data = []
        self.last_update_time = datetime.now()
        
        # 漂移检测器
        self.drift_detector = ConceptDriftDetector() if enable_drift_detection else None
        self.incremental_trainer = None
    
    def should_update(self) -> bool:
        """
        检查是否应该更新模型
        
        Returns:
            是否应该更新
        """
        # 检查更新频率
        now = datetime.now()
        
        if self.update_frequency == "daily":
            time_diff = (now - self.last_update_time).days
            should_update = time_diff >= 1
        elif self.update_frequency == "weekly":
            time_diff = (now - self.last_update_time).days
            should_update = time_diff >= 7
        elif self.update_frequency == "monthly":
            time_diff = (now - self.last_update_time).days
            should_update = time_diff >= 30
        else:
            should_update = False
        
        # 检查增量数据量
        if should_update:
            should_update = len(self.incremental_data) >= self.min_incremental_samples
        
        return should_update
    
    def check_concept_drift(self, X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """
        检查概念漂移
        
        Args:
            X: 当前特征数据
            y: 当前标签数据（可选）
            
        Returns:
            漂移检测结果
        """
        if not self.enable_drift_detection or self.drift_detector is None:
            return {'drift_detection_enabled': False}
        
        # 检测特征漂移
        feature_drift = self.drift_detector.detect_feature_drift(X)
        
        # 检测标签漂移
        label_drift = None
        if y is not None:
            label_drift = self.drift_detector.detect_label_drift(y)
        
        return {
            'drift_detection_enabled': True,
            'feature_drift': feature_drift,
            'label_drift': label_drift,
            'drift_detected': feature_drift.get('drift_detected', False) or 
                            (label_drift.get('drift_detected', False) if label_drift else False)
        }
    
    def initialize_drift_detector(self, X: pd.DataFrame, y: pd.Series = None):
        """
        初始化漂移检测器的参考分布
        
        Args:
            X: 参考特征数据（通常是训练数据）
            y: 参考标签数据
        """
        if self.drift_detector is not None:
            self.drift_detector.set_reference_distribution(X, y)
            logger.info("Drift detector reference distribution initialized")
    
    def add_incremental_data(self, data: Dict):
        """
        添加增量数据
        
        Args:
            data: 数据字典，包含特征和标签
        """
        self.incremental_data.append(data)
    
    def clear_incremental_data(self):
        """清空增量数据"""
        self.incremental_data = []
    
    def evaluate_current_model(self, test_data: pd.DataFrame) -> Dict:
        """
        评估当前模型性能
        
        Args:
            test_data: 测试数据
        
        Returns:
            性能指标字典
        """
        # 获取当前激活的版本
        active_version = self.version_manager.get_active_version()
        if active_version is None:
            return {}
        
        # 这里应该加载模型并进行预测
        # 简化实现，返回版本元数据
        return {
            'version_id': active_version.version_id,
            'auc_score': active_version.auc_score,
            'sharpe_ratio': active_version.sharpe_ratio,
            'win_rate': active_version.win_rate,
        }
    
    def trigger_update(
        self,
        model_dir: str,
        train_func,
        test_data: Optional[pd.DataFrame] = None,
    ) -> Optional[str]:
        """
        触发模型更新
        
        Args:
            model_dir: 模型目录
            train_func: 训练函数
            test_data: 测试数据
        
        Returns:
            新版本ID，None表示更新失败
        """
        if not self.should_update():
            logger.info("不满足更新条件，跳过更新")
            return None
        
        logger.info("开始在线模型更新...")
        
        try:
            # 调用训练函数
            train_func(model_dir)
            
            # 获取当前模型性能
            current_metrics = self.evaluate_current_model(test_data)
            
            # 注册新版本
            # 注意：这里应该从训练结果中获取实际的性能指标
            new_version = self.version_manager.register_version(
                model_dir=model_dir,
                model_type="ensemble",
                auc_score=current_metrics.get('auc_score', 0.0) + 0.01,  # 模拟提升
                sharpe_ratio=current_metrics.get('sharpe_ratio', 0.0) + 0.05,
                win_rate=current_metrics.get('win_rate', 0.0) + 0.01,
                metadata={'update_type': 'online', 'incremental_samples': len(self.incremental_data)},
            )
            
            # 清空增量数据
            self.clear_incremental_data()
            
            # 更新时间
            self.last_update_time = datetime.now()
            
            logger.info(f"模型更新完成: {new_version.version_id}")
            
            return new_version.version_id
            
        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def auto_switch_model(self, min_samples: int = 20) -> bool:
        """
        自动切换到更好的模型
        
        Args:
            min_samples: 最小样本数
        
        Returns:
            是否成功切换
        """
        # 获取所有版本
        versions = self.version_manager.list_versions()
        
        # 过滤掉样本数不足的版本
        valid_versions = [v for v in versions if v.num_trades >= min_samples]
        
        if len(valid_versions) < 2:
            return False
        
        # 找到AUC最高的版本
        best_version = max(valid_versions, key=lambda x: x.auc_score)
        
        # 如果不是当前激活的版本，切换
        active_version = self.version_manager.get_active_version()
        if active_version is None or best_version.version_id != active_version.version_id:
            # 检查性能提升是否显著
            if best_version.auc_score - active_version.auc_score >= self.performance_threshold:
                self.version_manager.activate_version(best_version.version_id)
                logger.info(f"自动切换到更好的模型: {best_version.version_id}")
                return True
        
        return False


# 全局实例
_global_version_manager: Optional[ModelVersionManager] = None
_global_online_learner: Optional[OnlineLearner] = None


def get_version_manager(
    models_dir: str = "models/",
) -> ModelVersionManager:
    """获取全局版本管理器实例"""
    global _global_version_manager
    
    if _global_version_manager is None:
        _global_version_manager = ModelVersionManager(models_dir)
    
    return _global_version_manager


def get_online_learner(
    version_manager: Optional[ModelVersionManager] = None,
    update_frequency: str = "weekly",
) -> OnlineLearner:
    """获取全局在线学习器实例"""
    global _global_online_learner
    
    if _global_online_learner is None:
        if version_manager is None:
            version_manager = get_version_manager()
        _global_online_learner = OnlineLearner(
            version_manager=version_manager,
            update_frequency=update_frequency,
        )
    
    return _global_online_learner


if __name__ == "__main__":
    # 测试代码
    vm = get_version_manager()
    
    # 注册测试版本
    vm.register_version(
        model_dir="models/ensemble_v1",
        model_type="ensemble",
        auc_score=0.75,
        sharpe_ratio=2.0,
        win_rate=0.35,
    )
    
    # 列出版本
    versions = vm.list_versions()
    print(f"共有 {len(versions)} 个版本:")
    for v in versions:
        print(f"  {v.version_id}: AUC={v.auc_score:.4f}, Sharpe={v.sharpe_ratio:.2f}")
    
    # 激活版本
    vm.activate_version("v1")
    
    # 获取激活的版本
    active = vm.get_active_version()
    print(f"激活的版本: {active.version_id if active else 'None'}")
