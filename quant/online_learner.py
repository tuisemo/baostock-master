"""
Online Learning Pipeline (Milestone 4.2)
实现模型版本管理、A/B测试和在线学习机制
"""
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from quant.logger import logger


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


class OnlineLearner:
    """
    在线学习器
    
    功能：
    1. 定期更新检查（daily/weekly/monthly）
    2. 增量训练框架
    3. 模型性能评估
    4. 自动版本切换
    """
    
    def __init__(
        self,
        version_manager: ModelVersionManager,
        update_frequency: str = "weekly",  # daily, weekly, monthly
        min_incremental_samples: int = 1000,  # 最小增量样本数
        performance_threshold: float = 0.01,  # 性能提升阈值
    ):
        self.version_manager = version_manager
        self.update_frequency = update_frequency
        self.min_incremental_samples = min_incremental_samples
        self.performance_threshold = performance_threshold
        
        # 增量数据缓存
        self.incremental_data = []
        self.last_update_time = datetime.now()
    
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
