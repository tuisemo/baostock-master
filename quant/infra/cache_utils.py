"""
数据缓存和工具函数
用于提升数据加载和预处理性能
"""
import os
import pickle
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
from functools import lru_cache
from collections import OrderedDict
import threading
import gc

from quant.infra.config import CONF
from quant.infra.logger import logger


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


# 特征缓存版本 - 用于缓存失效检查
FEATURE_VERSION = "2.0"


def get_feature_cache_key(stock_code: str, feature_version: str = FEATURE_VERSION) -> str:
    """
    生成特征缓存键

    Args:
        stock_code: 股票代码
        feature_version: 特征版本号

    Returns:
        缓存键字符串
    """
    data_dir = CONF.history_data.data_dir
    stock_path = os.path.join(data_dir, f"{stock_code}.csv")

    # 基于文件修改时间和特征版本生成缓存键
    try:
        if os.path.exists(stock_path):
            mtime = os.path.getmtime(stock_path)
            hash_input = f"{stock_code}_{feature_version}_{mtime}"
        else:
            hash_input = f"{stock_code}_{feature_version}_{datetime.now().strftime('%Y%m%d')}"
    except Exception:
        hash_input = f"{stock_code}_{feature_version}_{datetime.now().strftime('%Y%m%d')}"

    return hashlib.md5(hash_input.encode()).hexdigest()


class FeatureCache:
    """
    特征计算缓存管理器
    用于缓存特征计算结果，避免重复计算
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化特征缓存

        Args:
            cache_dir: 缓存目录，默认为 data/cache/features
        """
        if cache_dir is None:
            cache_dir = os.path.join(CONF.history_data.data_dir, "cache", "features")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = FEATURE_VERSION
        self.enabled = True

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.joblib"

    def get(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        获取缓存的特征数据

        Args:
            stock_code: 股票代码

        Returns:
            缓存的特征DataFrame，如果不存在或版本过期则返回None
        """
        if not self.enabled:
            return None

        cache_key = get_feature_cache_key(stock_code, self.version)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            data = joblib.load(cache_path)
            # 检查版本
            if isinstance(data, dict) and data.get('version') == self.version:
                logger.debug(f"从缓存加载特征: {stock_code}")
                return data['features']
            else:
                # 版本不匹配，删除旧缓存
                cache_path.unlink()
                return None
        except Exception as e:
            logger.warning(f"加载特征缓存失败 {stock_code}: {e}")
            return None

    def set(self, stock_code: str, features_df: pd.DataFrame) -> bool:
        """
        保存特征数据到缓存

        Args:
            stock_code: 股票代码
            features_df: 特征DataFrame

        Returns:
            是否保存成功
        """
        if not self.enabled:
            return False

        cache_key = get_feature_cache_key(stock_code, self.version)
        cache_path = self._get_cache_path(cache_key)

        try:
            data = {
                'version': self.version,
                'stock_code': stock_code,
                'created_at': datetime.now().isoformat(),
                'feature_cols': [c for c in features_df.columns if c.startswith('feat_')],
                'n_features': len([c for c in features_df.columns if c.startswith('feat_')]),
                'features': features_df
            }
            joblib.dump(data, cache_path)
            logger.debug(f"特征已缓存: {stock_code} ({data['n_features']} features)")
            return True
        except Exception as e:
            logger.warning(f"保存特征缓存失败 {stock_code}: {e}")
            return False

    def invalidate(self, pattern: Optional[str] = None):
        """
        使缓存失效

        Args:
            pattern: 股票代码匹配模式，None表示清空所有
        """
        try:
            if pattern is None:
                # 清空所有特征缓存
                for f in self.cache_dir.glob("*.joblib"):
                    f.unlink()
                logger.info(f"已清空所有特征缓存: {self.cache_dir}")
            else:
                # 清空匹配的股票缓存
                # 注意：这里需要遍历所有缓存文件并检查内容
                for f in self.cache_dir.glob("*.joblib"):
                    try:
                        data = joblib.load(f)
                        if isinstance(data, dict) and pattern in data.get('stock_code', ''):
                            f.unlink()
                    except:
                        pass
                logger.info(f"已清空匹配 '{pattern}' 的特征缓存")
        except Exception as e:
            logger.error(f"清空特征缓存失败: {e}")

    def warm_cache(self, stock_codes: List[str],
                   feature_func: Optional[callable] = None,
                   max_workers: int = 4):
        """
        预热缓存 - 提前计算并缓存指定股票的特征

        Args:
            stock_codes: 股票代码列表
            feature_func: 特征计算函数，如果提供则用于计算缺失的特征
            max_workers: 并行工作线程数
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_stock(code):
            if self.get(code) is not None:
                return code, True, "cached"

            if feature_func is not None:
                try:
                    # 这里假设feature_func会返回包含特征的DataFrame
                    df = feature_func(code)
                    if df is not None and len(df) > 0:
                        self.set(code, df)
                        return code, True, "computed"
                except Exception as e:
                    return code, False, str(e)

            return code, False, "no_feature_func"

        logger.info(f"开始预热特征缓存，共 {len(stock_codes)} 只股票")

        processed = 0
        cached = 0
        computed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_stock, code): code for code in stock_codes}

            for future in as_completed(futures):
                code, success, status = future.result()
                processed += 1

                if status == "cached":
                    cached += 1
                elif status == "computed":
                    computed += 1
                    logger.debug(f"特征计算完成: {code}")
                else:
                    failed += 1
                    logger.debug(f"特征计算失败 {code}: {status}")

                if processed % 100 == 0:
                    logger.info(f"预热进度: {processed}/{len(stock_codes)} "
                               f"(cached: {cached}, computed: {computed}, failed: {failed})")

        logger.info(f"特征缓存预热完成: 共 {processed} 只，"
                   f"已缓存 {cached}，新计算 {computed}，失败 {failed}")

    def get_cache_stats(self) -> dict:
        """
        获取特征缓存统计信息

        Returns:
            统计信息字典
        """
        try:
            files = list(self.cache_dir.glob("*.joblib"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            # 分析缓存内容
            versions = {}
            n_features_stats = []

            for f in files:
                try:
                    data = joblib.load(f)
                    if isinstance(data, dict):
                        ver = data.get('version', 'unknown')
                        versions[ver] = versions.get(ver, 0) + 1
                        n_features_stats.append(data.get('n_features', 0))
                except:
                    pass

            return {
                'cache_dir': str(self.cache_dir),
                'file_count': len(files),
                'total_size_mb': total_size / 1024 / 1024,
                'versions': versions,
                'avg_features': np.mean(n_features_stats) if n_features_stats else 0,
                'current_version': self.version
            }
        except Exception as e:
            logger.error(f"获取特征缓存统计失败: {e}")
            return {}

    def clear_outdated(self, keep_versions: Optional[List[str]] = None):
        """
        清理过期版本的缓存

        Args:
            keep_versions: 要保留的版本列表，None表示只保留当前版本
        """
        if keep_versions is None:
            keep_versions = [self.version]

        removed = 0
        try:
            for f in self.cache_dir.glob("*.joblib"):
                try:
                    data = joblib.load(f)
                    if isinstance(data, dict):
                        ver = data.get('version', 'unknown')
                        if ver not in keep_versions:
                            f.unlink()
                            removed += 1
                    else:
                        # 旧格式缓存，删除
                        f.unlink()
                        removed += 1
                except:
                    # 无法读取，删除
                    f.unlink()
                    removed += 1

            logger.info(f"已清理 {removed} 个过期缓存文件")
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")


# 全局特征缓存实例
_global_feature_cache = None


def get_feature_cache() -> FeatureCache:
    """获取全局特征缓存实例"""
    global _global_feature_cache
    if _global_feature_cache is None:
        _global_feature_cache = FeatureCache()
    return _global_feature_cache


# =============================================================================
# Multi-Level Cache Implementation (Phase 9 Performance Optimization)
# =============================================================================

class LRUCache:
    """
    线程安全的 LRU (Least Recently Used) 内存缓存
    用于高频访问数据的快速存取
    """

    def __init__(self, capacity: int = 1000, ttl: Optional[int] = None):
        """
        初始化 LRU 缓存

        Args:
            capacity: 最大缓存条目数
            ttl: 条目的生存时间（秒），None 表示不过期
        """
        self.capacity = capacity
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, Optional[float]]] = OrderedDict()
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None

            value, expiry = self._cache[key]

            # 检查是否过期
            if expiry is not None and time.time() > expiry:
                del self._cache[key]
                self._miss_count += 1
                return None

            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            self._hit_count += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            expiry = time.time() + self.ttl if self.ttl else None

            if key in self._cache:
                # 更新已有值
                self._cache[key] = (value, expiry)
                self._cache.move_to_end(key)
            else:
                # 添加新值
                if len(self._cache) >= self.capacity:
                    # 淘汰最旧的条目
                    self._cache.popitem(last=False)
                self._cache[key] = (value, expiry)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hit_count = 0
            self._miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'capacity': self.capacity,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class MultiLevelCache:
    """
    多级缓存系统
    L1: 内存 LRU 缓存（最快速）
    L2: 磁盘缓存（使用 joblib，持久化）
    """

    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 l1_capacity: int = 500,
                 l1_ttl: Optional[int] = 3600):  # 默认1小时过期
        """
        初始化多级缓存

        Args:
            cache_dir: L2 磁盘缓存目录
            l1_capacity: L1 内存缓存容量
            l1_ttl: L1 缓存过期时间（秒）
        """
        # L1: 内存 LRU 缓存
        self.l1_memory = LRUCache(capacity=l1_capacity, ttl=l1_ttl)
        
        # L2: 磁盘缓存
        self.l2_disk = DatasetCache(cache_dir)
        
        self.enabled = True
        self._lock = threading.RLock()
        self._warmup_keys: set = set()

    def _generate_key(self, *args, **kwargs) -> str:
        """生成确定性缓存键"""
        # 使用参数的哈希值生成键
        key_parts = []
        
        # 处理位置参数
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(hashlib.md5(str(sorted(arg)).encode()).hexdigest()[:16])
            elif isinstance(arg, dict):
                key_parts.append(hashlib.md5(str(sorted(arg.items())).encode()).hexdigest()[:16])
            elif isinstance(arg, pd.DataFrame):
                # 使用 DataFrame 的列名和形状作为键的一部分
                key_parts.append(f"df_{len(arg)}_{hashlib.md5(str(arg.columns.tolist()).encode()).hexdigest()[:8]}")
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:16])
        
        # 处理关键字参数
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(hashlib.md5(str(sorted_kwargs).encode()).hexdigest()[:16])
        
        return "|".join(key_parts)

    def get(self, *args, **kwargs) -> Tuple[Optional[Any], str]:
        """
        获取缓存值
        尝试顺序: L1 -> L2

        Returns:
            Tuple[值, 来源] - 来源为 'l1', 'l2', 或 'miss'
        """
        if not self.enabled:
            return None, 'miss'

        key = self._generate_key(*args, **kwargs)

        # 尝试 L1 缓存
        value = self.l1_memory.get(key)
        if value is not None:
            logger.debug(f"L1 缓存命中: {key[:50]}...")
            return value, 'l1'

        # 尝试 L2 磁盘缓存
        with self._lock:
            value = self.l2_disk.load(key)
            if value is not None:
                # 回填到 L1
                self.l1_memory.set(key, value)
                logger.debug(f"L2 缓存命中: {key[:50]}...")
                return value, 'l2'

        return None, 'miss'

    def set(self, value: Any, *args, **kwargs) -> None:
        """
        设置缓存值
        写入到 L1 和 L2
        """
        if not self.enabled:
            return

        key = self._generate_key(*args, **kwargs)

        with self._lock:
            # 写入 L1
            self.l1_memory.set(key, value)
            
            # 写入 L2（异步保存大对象）
            try:
                self.l2_disk.save(value, key)
            except Exception as e:
                logger.warning(f"L2 缓存保存失败: {e}")

    def invalidate(self, pattern: Optional[str] = None) -> None:
        """使缓存失效"""
        with self._lock:
            self.l1_memory.clear()
            self.l2_disk.clear(pattern)

    def warm_cache(self, keys_values: List[Tuple[tuple, dict, Any]]) -> None:
        """
        预热缓存

        Args:
            keys_values: List of ((args), {kwargs}, value) tuples
        """
        logger.info(f"开始预热缓存，共 {len(keys_values)} 个条目")
        
        for i, (args, kwargs, value) in enumerate(keys_values):
            self.set(value, *args, **kwargs)
            self._warmup_keys.add(self._generate_key(*args, **kwargs))
            
            if (i + 1) % 100 == 0:
                logger.info(f"预热进度: {i + 1}/{len(keys_values)}")

        logger.info(f"缓存预热完成")

    def get_stats(self) -> Dict[str, Any]:
        """获取多级缓存统计信息"""
        l1_stats = self.l1_memory.get_stats()
        l2_stats = self.l2_disk.get_cache_size()
        
        return {
            'l1_memory': l1_stats,
            'l2_disk': l2_stats,
            'enabled': self.enabled,
            'warmup_keys': len(self._warmup_keys)
        }

    def clear_outdated(self, max_age_hours: int = 24) -> int:
        """
        清理过期缓存

        Args:
            max_age_hours: 最大缓存年龄（小时）

        Returns:
            清理的条目数
        """
        # L1 会自动清理过期条目
        # L2 需要手动检查文件修改时间
        removed = 0
        try:
            cache_dir = Path(self.l2_disk.cache_dir)
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for f in cache_dir.glob("*.pkl"):
                try:
                    if current_time - f.stat().st_mtime > max_age_seconds:
                        f.unlink()
                        removed += 1
                except Exception:
                    pass
            
            logger.info(f"清理了 {removed} 个过期缓存文件")
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
        
        return removed


# 全局多级缓存实例
_global_multi_level_cache = None


def get_multi_level_cache() -> MultiLevelCache:
    """获取全局多级缓存实例"""
    global _global_multi_level_cache
    if _global_multi_level_cache is None:
        cache_dir = os.path.join(CONF.history_data.data_dir, "cache")
        _global_multi_level_cache = MultiLevelCache(cache_dir=cache_dir)
    return _global_multi_level_cache


def cached_with_ttl(ttl_seconds: int = 3600, maxsize: int = 128):
    """
    装饰器：为函数添加 TTL 缓存

    Args:
        ttl_seconds: 缓存生存时间
        maxsize: 最大缓存条目数

    Example:
        @cached_with_ttl(ttl_seconds=1800, maxsize=100)
        def expensive_function(arg1, arg2):
            return compute_result(arg1, arg2)
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(capacity=maxsize, ttl=ttl_seconds)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成键
            key_parts = [func.__name__]
            key_parts.extend([str(a) for a in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # 尝试获取缓存
            result = cache.get(key)
            if result is not None:
                return result
            
            # 计算并缓存
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # 附加缓存统计方法
        wrapper.cache_stats = cache.get_stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


# 导入 time 模块（放在文件末尾避免循环导入）
import time
import functools
