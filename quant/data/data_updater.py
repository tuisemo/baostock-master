import baostock as bs
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Optional, Dict, List, Iterator
import gc
import mmap

from quant.config import CONF
from quant.logger import logger


# =============================================================================
# Phase 9: Lazy Loading and Memory Optimization
# =============================================================================

class LazyDataLoader:
    """
    惰性数据加载器
    按需加载股票数据，支持内存缓存和内存映射
    """
    
    def __init__(self, data_dir: str, memory_cache_size: int = 100):
        """
        初始化惰性加载器
        
        Args:
            data_dir: 数据目录
            memory_cache_size: 内存缓存的最大股票数
        """
        self.data_dir = data_dir
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_size = memory_cache_size
        self._access_order: List[str] = []  # LRU 顺序
        self._total_loaded = 0
        self._total_cache_hits = 0
        
    def _get_file_path(self, code: str) -> str:
        """获取股票数据文件路径"""
        return os.path.join(self.data_dir, f"{code}.csv")
    
    def _load_stock_data(self, code: str) -> Optional[pd.DataFrame]:
        """
        加载股票数据
        
        Args:
            code: 股票代码
            
        Returns:
            DataFrame 或 None
        """
        file_path = self._get_file_path(code)
        
        if not os.path.exists(file_path):
            return None
            
        try:
            # 使用优化的数据类型读取
            df = pd.read_csv(file_path)
            
            if df.empty:
                return None
            
            # 优化数据类型以减少内存使用
            df = self._optimize_dataframe_types(df)
            
            # 转换日期列
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            return df
            
        except Exception as e:
            logger.warning(f"加载股票数据失败 {code}: {e}")
            return None
    
    def _optimize_dataframe_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化 DataFrame 数据类型以减少内存使用
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            优化后的 DataFrame
        """
        # 数值列类型优化
        for col in df.columns:
            if df[col].dtype == 'float64':
                # 尝试转换为 float32
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                # 尝试转换为 int32
                df[col] = df[col].astype('int32')
            elif df[col].dtype == 'object' and col in ['date', 'Date']:
                # 日期列保持为 object，稍后转换为 datetime
                pass
        
        return df
    
    def _add_to_cache(self, code: str, df: pd.DataFrame) -> None:
        """添加数据到缓存（LRU 策略）"""
        # 如果缓存已满，移除最旧的条目
        while len(self._cache) >= self._cache_size and self._cache:
            oldest_code = self._access_order.pop(0)
            if oldest_code in self._cache:
                del self._cache[oldest_code]
                gc.collect()  # 提示垃圾回收
        
        # 添加新条目
        self._cache[code] = df
        self._access_order.append(code)
        self._total_loaded += 1
    
    def _update_access_order(self, code: str) -> None:
        """更新访问顺序"""
        if code in self._access_order:
            self._access_order.remove(code)
        self._access_order.append(code)
        self._total_cache_hits += 1
    
    def get_stock_data(self, code: str) -> Optional[pd.DataFrame]:
        """
        获取股票数据（带缓存）
        
        Args:
            code: 股票代码
            
        Returns:
            DataFrame 或 None
        """
        # 检查缓存
        if code in self._cache:
            self._update_access_order(code)
            logger.debug(f"缓存命中: {code}")
            return self._cache[code]
        
        # 加载数据
        df = self._load_stock_data(code)
        if df is not None:
            self._add_to_cache(code, df)
        
        return df
    
    def get_stock_data_chunked(
        self, 
        code: str, 
        chunk_size: int = 1000,
        columns: Optional[List[str]] = None
    ) -> Iterator[pd.DataFrame]:
        """
        分块加载股票数据（用于大数据集）
        
        Args:
            code: 股票代码
            chunk_size: 每个块的大小
            columns: 指定列（None 表示所有列）
            
        Yields:
            DataFrame 块
            
        Example:
            for chunk in loader.get_stock_data_chunked('sh.600000', chunk_size=500):
                process(chunk)
        """
        file_path = self._get_file_path(code)
        
        if not os.path.exists(file_path):
            return
        
        try:
            # 使用 chunk 迭代器
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=columns):
                # 优化数据类型
                chunk = self._optimize_dataframe_types(chunk)
                yield chunk
                
        except Exception as e:
            logger.warning(f"分块加载失败 {code}: {e}")
    
    def get_stock_data_mmap(self, code: str) -> Optional[pd.DataFrame]:
        """
        使用内存映射加载股票数据
        适用于非常大的文件
        
        Args:
            code: 股票代码
            
        Returns:
            DataFrame 或 None
        """
        file_path = self._get_file_path(code)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            # 对于 CSV 文件，使用 memory_map 选项
            df = pd.read_csv(file_path, memory_map=True)
            return df
            
        except Exception as e:
            logger.warning(f"内存映射加载失败 {code}: {e}")
            return None
    
    def get_cached_codes(self) -> List[str]:
        """获取当前缓存中的所有股票代码"""
        return list(self._cache.keys())
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()
        gc.collect()
        logger.info("数据加载器缓存已清空")
    
    def get_stats(self) -> Dict:
        """获取加载器统计信息"""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self._cache_size,
            'total_loaded': self._total_loaded,
            'total_cache_hits': self._total_cache_hits,
            'hit_rate': self._total_cache_hits / (self._total_loaded + self._total_cache_hits) 
                        if (self._total_loaded + self._total_cache_hits) > 0 else 0.0,
            'cached_codes': list(self._cache.keys())[:10]  # 只显示前10个
        }
    
    def preload_stocks(self, codes: List[str], show_progress: bool = True) -> int:
        """
        预加载多只股票到缓存
        
        Args:
            codes: 股票代码列表
            show_progress: 是否显示进度条
            
        Returns:
            成功加载的股票数
        """
        loaded = 0
        
        iterator = tqdm(codes, desc="预加载股票") if show_progress else codes
        
        for code in iterator:
            if code not in self._cache:
                df = self._load_stock_data(code)
                if df is not None:
                    self._add_to_cache(code, df)
                    loaded += 1
        
        logger.info(f"预加载完成: {loaded}/{len(codes)} 只股票")
        return loaded


# 全局惰性加载器实例
_global_lazy_loader: Optional[LazyDataLoader] = None


def get_lazy_loader() -> LazyDataLoader:
    """获取全局惰性加载器实例"""
    global _global_lazy_loader
    if _global_lazy_loader is None:
        _global_lazy_loader = LazyDataLoader(CONF.history_data.data_dir)
    return _global_lazy_loader


def reset_lazy_loader() -> None:
    """重置全局惰性加载器"""
    global _global_lazy_loader
    if _global_lazy_loader is not None:
        _global_lazy_loader.clear_cache()
    _global_lazy_loader = None


def _fetch_single_stock(code: str, end_date_str: str, default_start_date: str) -> dict:
    """
    在已登录的 baostock 会话中拉取单只股票的历史 K 线数据。
    调用方必须确保已 bs.login()。
    """
    out_file = os.path.join(CONF.history_data.data_dir, f"{code}.csv")
    rs_fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"

    # 判断是否增量更新
    is_incremental = False
    start_date = default_start_date
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file)
            if not existing_df.empty and 'date' in existing_df.columns:
                last_date_str = str(existing_df['date'].max())
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                is_incremental = True
        except Exception:
            pass

    # 已是最新数据则跳过
    if is_incremental and start_date > end_date_str:
        return {"code": code, "status": "skip"}

    rs = bs.query_history_k_data_plus(
        code,
        rs_fields,
        start_date=start_date,
        end_date=end_date_str,
        frequency="d",
        adjustflag="2"  # 前复权
    )

    if rs.error_code != '0':
        return {"code": code, "status": "error", "msg": f"API error_code={rs.error_code}, {rs.error_msg}"}

    data_list = []
    while rs.error_code == '0' and rs.next():
        data_list.append(rs.get_row_data())

    if len(data_list) > 0:
        result_df = pd.DataFrame(data_list, columns=rs.fields)
        if is_incremental:
            result_df.to_csv(out_file, mode='a', header=False, index=False, encoding="utf-8-sig")
            return {"code": code, "status": "updated"}
        else:
            result_df.to_csv(out_file, index=False, encoding="utf-8-sig")
            return {"code": code, "status": "new"}
    else:
        return {"code": code, "status": "empty"}


def _safe_session(action: str = "login") -> bool:
    """安全地执行 baostock login/logout，出错不抛异常。"""
    try:
        if action == "login":
            lg = bs.login()
            return lg.error_code == '0'
        else:
            bs.logout()
            return True
    except Exception:
        return False


def update_history_data():
    """
    拉取并增量更新股票池中所有标的的历史 K 线数据。

    策略：
      - baostock 底层使用单一全局 socket，不支持多线程/多进程并发。
      - 因此采用 **串行顺序** 拉取，保证连接稳定。
      - 对失败的股票做二次重试（重新登录后逐个重试）。
    """
    list_path = os.path.join(CONF.history_data.data_dir, "stock-list.csv")
    if not os.path.exists(list_path):
        logger.error(f"Stock list not found at {list_path}. Please run 'update-list' first.")
        return

    df_list = pd.read_csv(list_path)
    if df_list.empty:
        logger.warning("Stock list is empty. Nothing to update.")
        return

    codes = df_list['code'].tolist()
    if 'sh.000001' not in codes:
        codes.append('sh.000001')

    total = len(codes)
    logger.info(f"Loaded {total} stocks to update historical data (including market index).")

    end_date = datetime.now()
    end_date_str = end_date.strftime("%Y-%m-%d")
    default_start_date = (end_date - timedelta(days=CONF.history_data.default_lookback_days)).strftime("%Y-%m-%d")

    os.makedirs(CONF.history_data.data_dir, exist_ok=True)

    # ===== 第一轮：顺序拉取全量 =====
    if not _safe_session("login"):
        logger.error("Baostock 登录失败，无法更新历史数据。")
        return

    updated_count = 0
    new_count = 0
    empty_count = 0
    error_count = 0
    skip_count = 0
    failed_codes: list[str] = []

    logger.info("第一轮：串行拉取全量数据...")
    for i, code in enumerate(tqdm(codes, desc="拉取历史数据")):
        try:
            res = _fetch_single_stock(code, end_date_str, default_start_date)
        except Exception as e:
            res = {"code": code, "status": "error", "msg": str(e)}
            # 连接可能已损坏，尝试重新建立
            _safe_session("logout")
            time.sleep(1)
            if not _safe_session("login"):
                logger.error(f"重连失败，中止在第 {i}/{total} 只。")
                break

        s = res["status"]
        if s == "updated":
            updated_count += 1
        elif s == "new":
            new_count += 1
        elif s == "empty":
            empty_count += 1
        elif s == "skip":
            skip_count += 1
        elif s == "error":
            error_count += 1
            failed_codes.append(code)
            logger.debug(f"拉取失败 [{code}]: {res.get('msg', 'Unknown')}")

    _safe_session("logout")

    logger.info(
        f"第一轮完成: {new_count} new, {updated_count} updated, "
        f"{empty_count} empty, {skip_count} skipped, {error_count} errors."
    )

    # ===== 第二轮：对失败的代码进行重试 =====
    if failed_codes:
        logger.info(f"第二轮：对 {len(failed_codes)} 只失败股票进行重试（间隔 500ms）...")
        time.sleep(2)  # 等待服务端释放连接

        if not _safe_session("login"):
            logger.error("第二轮重试登录失败，跳过。")
        else:
            retry_success = 0
            still_failed: list[str] = []

            for code in tqdm(failed_codes, desc="重试失败股票"):
                time.sleep(0.5)  # 每个请求间隔 500ms
                try:
                    res = _fetch_single_stock(code, end_date_str, default_start_date)
                    if res["status"] in ("new", "updated", "empty", "skip"):
                        retry_success += 1
                    else:
                        still_failed.append(code)
                        logger.warning(f"二次拉取仍失败 [{code}]: {res.get('msg', 'Unknown')}")
                except Exception as e:
                    still_failed.append(code)
                    logger.warning(f"二次拉取异常 [{code}]: {e}")
                    # 连接损坏时重连
                    _safe_session("logout")
                    time.sleep(1)
                    if not _safe_session("login"):
                        logger.error("重试期间重连失败，中止重试。")
                        still_failed.extend([c for c in failed_codes if c not in still_failed and c != code])
                        break

            _safe_session("logout")

            error_count = error_count - retry_success
            logger.info(
                f"第二轮重试完成: {retry_success}/{len(failed_codes)} 恢复成功。"
                f"最终剩余 {len(still_failed)} 只无法获取。"
            )

            if still_failed:
                logger.info(f"最终失败列表 (前 30 个): {still_failed[:30]}")

    # ===== 最终汇总 =====
    total_success = new_count + updated_count + retry_success if failed_codes else new_count + updated_count
    logger.info(
        f"===== 数据更新完毕 =====\n"
        f"  成功下载/更新: {total_success} 只\n"
        f"  无新数据 (empty): {empty_count} 只\n"
        f"  已是最新 (skip): {skip_count} 只\n"
        f"  最终失败: {error_count} 只"
    )
