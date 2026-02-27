"""
数据缓存和工具函数
用于提升数据加载和预处理性能
"""
import os
import pickle
import hashlib
import joblib
from pathlib import Path
from typing import Optional
from datetime import datetime

from quant.config import CONF
from quant.logger import logger


def get_data_hash(data_dir: str, params_dict: dict) -> str:
    """
    生成数据缓存键

    Args:
        data_dir: 数据目录
        params_dict: 参数字典

    Returns:
        MD5 哈希值
    """
    # 获取数据目录中文件的修改时间
    try:
        files = sorted(Path(data_dir).glob("*.csv"))
        file_mod_times = []
        for f in files:
            if f.name != "stock-list.csv":
                file_mod_times.append(f.stat().st_mtime)

        # 如果没有文件，使用当前时间
        if not file_mod_times:
            file_mod_times = [datetime.now().timestamp()]

        # 生成哈希
        hash_input = f"{data_dir}_{sorted(file_mod_times)}_{sorted(params_dict.items())}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"生成数据哈希失败: {e}")
        # fallback: 使用参数和时间
        hash_input = f"{data_dir}_{params_dict}_{datetime.now().strftime('%Y%m%d')}"
        return hashlib.md5(hash_input.encode()).hexdigest()


class DatasetCache:
    """数据集缓存管理器"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录，默认为 data/cache
        """
        self.cache_dir = Path(cache_dir or "data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = True

    def get_cache_path(self, cache_key: str, suffix: str = ".pkl") -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}{suffix}"

    def save(self, data, cache_key: str, suffix: str = ".pkl") -> Path:
        """
        保存数据到缓存

        Args:
            data: 要缓存的数据
            cache_key: 缓存键
            suffix: 文件后缀

        Returns:
            缓存文件路径
        """
        cache_path = self.get_cache_path(cache_key, suffix)

        try:
            # 使用 joblib 保存（支持大文件）
            joblib.dump(data, cache_path)
            logger.debug(f"数据已缓存到: {cache_path}")
            return cache_path
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
            return cache_path

    def load(self, cache_key: str, suffix: str = ".pkl"):
        """
        从缓存加载数据

        Args:
            cache_key: 缓存键
            suffix: 文件后缀

        Returns:
            缓存的数据，如果不存在或加载失败则返回 None
        """
        if not self.cache_enabled:
            return None

        cache_path = self.get_cache_path(cache_key, suffix)

        if not cache_path.exists():
            return None

        try:
            data = joblib.load(cache_path)
            logger.debug(f"从缓存加载数据: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return None

    def exists(self, cache_key: str, suffix: str = ".pkl") -> bool:
        """检查缓存是否存在"""
        if not self.cache_enabled:
            return False
        return self.get_cache_path(cache_key, suffix).exists()

    def clear(self, pattern: Optional[str] = None):
        """
        清空缓存

        Args:
            pattern: 匹配模式，None 表示清空所有
        """
        try:
            if pattern is None:
                # 清空所有缓存
                for f in self.cache_dir.glob("*"):
                    f.unlink()
                logger.info(f"已清空所有缓存: {self.cache_dir}")
            else:
                # 清空匹配的缓存
                for f in self.cache_dir.glob(pattern):
                    f.unlink()
                logger.info(f"已清空匹配 '{pattern}' 的缓存")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

    def get_cache_size(self) -> dict:
        """获取缓存统计信息"""
        try:
            files = list(self.cache_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            return {
                'cache_dir': str(self.cache_dir),
                'file_count': len(files),
                'total_size_mb': total_size / 1024 / 1024,
                'files': [str(f) for f in files]
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {}


# 全局缓存实例
_global_cache = None


def get_global_cache() -> DatasetCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        cache_dir = os.path.join(CONF.history_data.data_dir, "cache")
        _global_cache = DatasetCache(cache_dir)
    return _global_cache
