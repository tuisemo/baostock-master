"""
全局缓存管理器单例

本模块提供统一的全局缓存管理，替换分散在各个文件中的全局缓存变量：
- _MARKET_INDEX_CACHE (backtester.py, features.py, market_state.py)
- _WEEKLY_SRC_CACHE (backtester.py)
- _SECTOR_DATA_CACHE (features.py)

所有缓存应通过GlobalCache单例访问，以实现：
1. 集中管理
2. 统一的清理机制
3. 避免缓存不一致
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd


class GlobalCache:
    """
    全局缓存管理器单例类
    
    使用单例模式确保整个应用程序使用同一个缓存实例。
    通过属性方法提供类型安全的缓存访问。
    """
    
    _instance: Optional["GlobalCache"] = None
    
    def __new__(cls) -> "GlobalCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset()
        return cls._instance
    
    def _reset(self) -> None:
        """重置所有缓存状态"""
        self._market_index: Optional[pd.DataFrame] = None
        self._weekly_data: Dict[str, pd.DataFrame] = {}
        self._sector_data: Dict[str, pd.DataFrame] = {}
    
    # ========================================================================
    # 市场指数缓存 (market_index)
    # ========================================================================
    
    @property
    def market_index(self) -> Optional[pd.DataFrame]:
        """获取市场指数缓存"""
        return self._market_index
    
    @market_index.setter
    def market_index(self, value: Optional[pd.DataFrame]) -> None:
        """设置市场指数缓存"""
        self._market_index = value
    
    def clear_market_index(self) -> None:
        """清除市场指数缓存"""
        self._market_index = None
    
    # ========================================================================
    # 周线数据缓存 (weekly_data)
    # ========================================================================
    
    @property
    def weekly_data(self) -> Dict[str, pd.DataFrame]:
        """获取周线数据缓存字典"""
        return self._weekly_data
    
    def get_weekly(self, key: str) -> Optional[pd.DataFrame]:
        """获取指定key的周线数据"""
        return self._weekly_data.get(key)
    
    def set_weekly(self, key: str, value: pd.DataFrame) -> None:
        """设置指定key的周线数据"""
        self._weekly_data[key] = value
    
    def clear_weekly(self, key: Optional[str] = None) -> None:
        """
        清除周线数据缓存
        
        Args:
            key: 如果提供，只清除该key；否则清除所有
        """
        if key is not None:
            self._weekly_data.pop(key, None)
        else:
            self._weekly_data.clear()
    
    def clear_oldest_weekly(self) -> None:
        """清除最旧的周线数据（用于保持缓存 bounded）"""
        if self._weekly_data:
            oldest_key = next(iter(self._weekly_data))
            self._weekly_data.pop(oldest_key, None)
    
    @property
    def weekly_count(self) -> int:
        """获取当前周线缓存数量"""
        return len(self._weekly_data)
    
    # ========================================================================
    # 板块数据缓存 (sector_data)
    # ========================================================================
    
    @property
    def sector_data(self) -> Dict[str, pd.DataFrame]:
        """获取板块数据缓存字典"""
        return self._sector_data
    
    def get_sector(self, sector_code: str) -> Optional[pd.DataFrame]:
        """获取指定板块代码的缓存数据"""
        return self._sector_data.get(sector_code)
    
    def set_sector(self, sector_code: str, value: pd.DataFrame) -> None:
        """设置指定板块代码的缓存数据"""
        self._sector_data[sector_code] = value
    
    def clear_sector(self, sector_code: Optional[str] = None) -> None:
        """
        清除板块数据缓存
        
        Args:
            sector_code: 如果提供，只清除该板块；否则清除所有
        """
        if sector_code is not None:
            self._sector_data.pop(sector_code, None)
        else:
            self._sector_data.clear()
    
    # ========================================================================
    # 通用缓存清理
    # ========================================================================
    
    def clear_all(self) -> None:
        """清除所有缓存"""
        self._reset()
    
    def clear_feature_cache(self) -> None:
        """清除特征缓存（预留接口）"""
        # 如有需要，后续可扩展
        pass


# 导出单例实例
GLOBAL_CACHE = GlobalCache()


# ========================================================================
# 便利函数（供其他模块使用）
# ========================================================================

def get_global_cache() -> GlobalCache:
    """获取全局缓存单例"""
    return GLOBAL_CACHE


def clear_market_index_cache() -> None:
    """清除市场指数缓存（向后兼容接口）"""
    GLOBAL_CACHE.clear_market_index()


def clear_weekly_cache() -> None:
    """清除周线数据缓存（向后兼容接口）"""
    GLOBAL_CACHE.clear_weekly()


def clear_sector_cache() -> None:
    """清除板块数据缓存（向后兼容接口）"""
    GLOBAL_CACHE.clear_sector()
