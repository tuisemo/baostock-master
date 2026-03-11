"""
回测引擎性能优化模块
包含并行化处理、增量回测、指标缓存、可视化和实时数据接口
"""
import os
import json
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime

from quant.infra.logger import logger
from quant.infra.config import CONF
from quant.core.strategy_params import StrategyParams
from quant.app.backtester import run_backtest, create_strategy

# ============================================================================
# Milestone 5: 并行化处理增强
# ============================================================================

@dataclass
class ParallelTask:
    """并行任务数据类"""
    task_id: str
    code: str
    params: StrategyParams
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    priority: int = 0


class ParallelBacktestExecutor:
    """
    并行回测执行器
    
    功能：
    1. 动态负载均衡
    2. 任务优先级调度
    3. 进度监控
    4. 异常处理和重试
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_threading: bool = False,
        retry_times: int = 2,
    ):
        self.max_workers = max_workers or (os.cpu_count() - 1)
        self.use_threading = use_threading
        self.retry_times = retry_times
        
        self.task_queue = []
        self.results = []
        self.progress = {
            'total': 0,
            'completed': 0,
            'failed': 0,
        }
    
    def add_tasks(self, tasks: List[ParallelTask]):
        """添加任务"""
        self.task_queue.extend(tasks)
        self.progress['total'] = len(self.task_queue)
        logger.info(f"添加了 {len(tasks)} 个并行回测任务")
    
    def execute(self) -> List[Dict]:
        """
        执行并行回测
        
        Returns:
            回测结果列表
        """
        if not self.task_queue:
            return []
        
        # 按优先级排序
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        # 选择执行器
        executor_class = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        
        logger.info(f"开始并行回测: {len(self.task_queue)} 个任务, {self.max_workers} 个工作进程")
        start_time = time.time()
        
        with executor_class(max_workers=self.max_workers) as executor:
            # 提交任务
            futures = {
                executor.submit(
                    self._execute_single_task,
                    task
                ): task for task in self.task_queue
            }
            
            # 收集结果
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result(timeout=300)  # 5 分钟超时
                    if result is not None:
                        self.results.append(result)
                        self.progress['completed'] += 1
                    else:
                        self.progress['failed'] += 1
                except Exception as e:
                    logger.error(f"任务 {task.task_id} 执行失败: {e}")
                    self.progress['failed'] += 1
                
                # 进度日志
                if self.progress['completed'] % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / (self.progress['completed'] / self.progress['total']) - elapsed
                    logger.info(
                        f"并行回测进度: {self.progress['completed']}/{self.progress['total']} "
                        f"({self.progress['completed']/self.progress['total']*100:.1f}%), "
                        f"预计剩余: {eta:.1f}s"
                    )
        
        elapsed = time.time() - start_time
        logger.info(
            f"并行回测完成: {self.progress['completed']} 成功, "
            f"{self.progress['failed']} 失败, "
            f"耗时: {elapsed:.2f}s"
        )
        
        return self.results
    
    def _execute_single_task(self, task: ParallelTask) -> Optional[Dict]:
        """
        执行单个任务（带重试）
        
        Args:
            task: 并行任务
        
        Returns:
            回测结果字典
        """
        for attempt in range(self.retry_times + 1):
            try:
                result = run_backtest(
                    code=task.code,
                    params=task.params,
                    start_date=task.start_date,
                    end_date=task.end_date,
                )
                
                if result is not None:
                    _, stats = result
                    return {
                        'task_id': task.task_id,
                        'code': task.code,
                        'return_pct': stats.get("Return [%]", 0.0),
                        'win_rate': stats.get("Win Rate [%]", 0.0),
                        'max_drawdown': stats.get("Max. Drawdown [%]", 0.0),
                        'num_trades': stats.get("# Trades", 0),
                        'sharpe': stats.get("Sharpe Ratio", 0.0),
                        'equity_final': stats.get("Equity Final [$]", 0.0),
                    }
            except Exception as e:
                if attempt < self.retry_times:
                    logger.debug(f"任务 {task.task_id} 第 {attempt+1} 次重试: {e}")
                    time.sleep(1)
                else:
                    raise
        return None


# ============================================================================
# Milestone 6: 增量回测实现
# ============================================================================

class IncrementalBacktestEngine:
    """
    增量回测引擎
    
    功能：
    1. 指标增量更新
    2. 状态缓存
    3. 增量回测接口
    """
    
    def __init__(self, cache_dir: str = "data/backtest_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.state_cache = {}
    
    def _get_cache_key(self, code: str, params: StrategyParams) -> str:
        """
        获取缓存键
        
        Args:
            code: 股票代码
            params: 策略参数
        
        Returns:
            缓存键
        """
        params_hash = hashlib.md5(
            json.dumps(params.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        return f"{code}_{params_hash}"
    
    def _load_cached_state(self, cache_key: str) -> Optional[Dict]:
        """
        加载缓存的状态
        
        Args:
            cache_key: 缓存键
        
        Returns:
            缓存的状态字典
        """
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"加载缓存失败: {e}")
        
        return None
    
    def _save_cached_state(self, cache_key: str, state: Dict):
        """
        保存缓存的状态
        
        Args:
            cache_key: 缓存键
            state: 状态字典
        """
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.debug(f"保存缓存失败: {e}")
    
    def run_incremental_backtest(
        self,
        code: str,
        params: StrategyParams,
        start_date: str,
        end_date: str,
    ) -> Optional[Dict]:
        """
        运行增量回测
        
        Args:
            code: 股票代码
            params: 策略参数
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            回测结果字典
        """
        cache_key = self._get_cache_key(code, params)
        
        # 尝试加载缓存
        cached_state = self._load_cached_state(cache_key)
        
        # 检查缓存是否有效
        if cached_state and cached_state.get('end_date') >= start_date:
            logger.debug(f"使用缓存的增量回测状态: {cache_key}")
            # 从缓存的结束日期开始增量回测
            incremental_start = cached_state['end_date']
        else:
            logger.debug(f"缓存无效或不存在，执行全量回测: {cache_key}")
            incremental_start = start_date
            cached_state = None
        
        # 执行增量回测
        result = run_backtest(
            code=code,
            params=params,
            start_date=incremental_start,
            end_date=end_date,
        )
        
        if result is not None:
            _, stats = result
            
            # 更新缓存状态
            state = {
                'code': code,
                'params': params.to_dict(),
                'start_date': start_date,
                'end_date': end_date,
                'stats': {
                    'return_pct': stats.get("Return [%]", 0.0),
                    'win_rate': stats.get("Win Rate [%]", 0.0),
                    'max_drawdown': stats.get("Max. Drawdown [%]", 0.0),
                    'num_trades': stats.get("# Trades", 0),
                    'sharpe': stats.get("Sharpe Ratio", 0.0),
                },
                'timestamp': datetime.now().isoformat(),
            }
            
            self._save_cached_state(cache_key, state)
            
            return {
                'code': code,
                'return_pct': stats.get("Return [%]", 0.0),
                'win_rate': stats.get("Win Rate [%]", 0.0),
                'max_drawdown': stats.get("Max. Drawdown [%]", 0.0),
                'num_trades': stats.get("# Trades", 0),
                'sharpe': stats.get("Sharpe Ratio", 0.0),
                'equity_final': stats.get("Equity Final [$]", 0.0),
                'is_incremental': cached_state is not None,
            }
        
        return None


# ============================================================================
# Milestone 7: 指标缓存和可视化
# ============================================================================

class IndicatorCache:
    """
    指标缓存器
    
    功能：
    1. 智能缓存策略
    2. 缓存失效机制
    3. LRU 缓存管理
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
    
    @lru_cache(maxsize=500)
    def get_indicator(
        self,
        code: str,
        indicator_type: str,
        params_dict: str,
    ) -> Optional[pd.Series]:
        """
        获取缓存的指标（带 LRU）
        
        Args:
            code: 股票代码
            indicator_type: 指标类型
            params_dict: 参数字典的字符串表示
        
        Returns:
            指标 Series
        """
        cache_key = f"{code}_{indicator_type}_{params_dict}"
        
        if cache_key in self._cache:
            # 更新访问顺序
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]
        
        return None
    
    def set_indicator(
        self,
        code: str,
        indicator_type: str,
        params_dict: str,
        indicator: pd.Series,
    ):
        """
        设置缓存的指标
        
        Args:
            code: 股票代码
            indicator_type: 指标类型
            params_dict: 参数字典的字符串表示
            indicator: 指标 Series
        """
        cache_key = f"{code}_{indicator_type}_{params_dict}"
        
        # 如果缓存已满，移除最久未使用的项
        if len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[cache_key] = indicator
        self._access_order.append(cache_key)
    
    def invalidate(self, code: str = None):
        """
        失效缓存
        
        Args:
            code: 股票代码，如果为 None 则失效全部
        """
        if code is None:
            self._cache.clear()
            self._access_order.clear()
            logger.debug("失效所有指标缓存")
        else:
            # 失效指定股票的所有指标
            keys_to_remove = [k for k in self._cache if k.startswith(code)]
            for key in keys_to_remove:
                del self._cache[key]
                self._access_order.remove(key)
            logger.debug(f"失效 {code} 的指标缓存: {len(keys_to_remove)} 项")


# 全局指标缓存实例
_global_indicator_cache = IndicatorCache()


def get_indicator_cache() -> IndicatorCache:
    """获取全局指标缓存实例"""
    return _global_indicator_cache


class BacktestVisualizer:
    """
    回测结果可视化器
    
    功能：
    1. 生成可视化图表
    2. 生成分析报告
    3. 导出结果
    """
    
    def __init__(self, output_dir: str = "data/backtest_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        results: List[Dict],
        title: str = "回测分析报告",
    ) -> str:
        """
        生成回测分析报告
        
        Args:
            results: 回测结果列表
            title: 报告标题
        
        Returns:
            Markdown 格式的报告
        """
        if not results:
            return "无回测结果"
        
        df = pd.DataFrame(results)
        
        # 统计摘要
        total_return = df['return_pct'].mean()
        avg_sharpe = df['sharpe'].mean()
        avg_win_rate = df['win_rate'].mean()
        max_drawdown = df['max_drawdown'].min()
        
        # 生成报告
        report_lines = [
            f"# {title}",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**股票数量**: {len(df)}",
            "",
            "## 统计摘要",
            "",
            f"- **平均收益率**: {total_return:.2f}%",
            f"- **平均夏普比率**: {avg_sharpe:.2f}",
            f"- **平均胜率**: {avg_win_rate:.2f}%",
            f"- **最大回撤**: {max_drawdown:.2f}%",
            "",
            "## 详细结果",
            "",
            df.to_markdown(index=False),
            "",
            "---",
            "",
            "*报告由回测引擎自动生成*"
        ]
        
        report = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(
            self.output_dir,
            f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"回测报告已保存: {report_path}")
        
        return report
    
    def plot_equity_curve(self, results: List[Dict]):
        """
        绘制权益曲线（占位函数，实际实现需要安装 matplotlib）
        
        Args:
            results: 回测结果列表
        """
        # 这里可以添加实际的可视化代码
        # 由于环境限制，这里只提供占位实现
        logger.debug("权益曲线可视化功能已调用（占位实现）")


# ============================================================================
# Milestone 8: 实时数据接口
# ============================================================================

class RealTimeDataInterface:
    """
    实时数据接口（占位实现）
    
    功能：
    1. WebSocket 数据推送
    2. 实时指标计算
    3. 实时监控面板
    """
    
    def __init__(self):
        self.is_connected = False
        self.subscribers = {}
    
    def connect(self, url: str):
        """
        连接到实时数据源
        
        Args:
            url: WebSocket URL
        """
        logger.info(f"连接到实时数据源: {url}")
        # 这里可以添加实际的 WebSocket 连接代码
        self.is_connected = True
    
    def subscribe(self, channel: str, callback: Callable):
        """
        订阅实时数据频道
        
        Args:
            channel: 频道名称
            callback: 回调函数
        """
        self.subscribers[channel] = callback
        logger.info(f"订阅频道: {channel}")
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        logger.info("断开实时数据连接")


# ============================================================================
# 综合回测引擎
# ============================================================================

class OptimizedBacktestEngine:
    """
    优化版综合回测引擎
    
    整合所有优化功能：
    1. 并行化处理
    2. 增量回测
    3. 指标缓存
    4. 可视化和报告
    5. 实时数据接口
    """
    
    def __init__(
        self,
        use_parallel: bool = True,
        use_incremental: bool = True,
        use_cache: bool = True,
    ):
        self.use_parallel = use_parallel
        self.use_incremental = use_incremental
        self.use_cache = use_cache
        
        # 初始化各模块
        self.parallel_executor = ParallelBacktestExecutor() if use_parallel else None
        self.incremental_engine = IncrementalBacktestEngine() if use_incremental else None
        self.indicator_cache = get_indicator_cache() if use_cache else None
        self.visualizer = BacktestVisualizer()
    
    def batch_backtest(
        self,
        codes: List[str],
        params: StrategyParams,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """
        批量回测（优化版）
        
        Args:
            codes: 股票代码列表
            params: 策略参数
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            回测结果列表
        """
        results = []
        
        if self.use_parallel and len(codes) > 1:
            # 并行回测
            tasks = [
                ParallelTask(
                    task_id=f"{code}_{i}",
                    code=code,
                    params=params,
                    start_date=start_date,
                    end_date=end_date,
                )
                for i, code in enumerate(codes)
            ]
            
            self.parallel_executor.add_tasks(tasks)
            results = self.parallel_executor.execute()
        
        else:
            # 串行回测（使用增量回测）
            for code in codes:
                if self.use_incremental:
                    result = self.incremental_engine.run_incremental_backtest(
                        code=code,
                        params=params,
                        start_date=start_date,
                        end_date=end_date,
                    )
                else:
                    result = run_backtest(
                        code=code,
                        params=params,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    
                    if result is not None:
                        _, stats = result
                        result = {
                            'code': code,
                            'return_pct': stats.get("Return [%]", 0.0),
                            'win_rate': stats.get("Win Rate [%]", 0.0),
                            'max_drawdown': stats.get("Max. Drawdown [%]", 0.0),
                            'num_trades': stats.get("# Trades", 0),
                            'sharpe': stats.get("Sharpe Ratio", 0.0),
                            'equity_final': stats.get("Equity Final [$]", 0.0),
                        }
                
                if result is not None:
                    results.append(result)
        
        # 生成报告
        if results:
            self.visualizer.generate_report(results)
        
        return results
    
    def clear_cache(self, code: str = None):
        """
        清除缓存
        
        Args:
            code: 股票代码，如果为 None 则清除全部
        """
        if self.indicator_cache:
            self.indicator_cache.invalidate(code)


# 便捷函数
def batch_backtest_optimized(
    codes: List[str],
    params: StrategyParams,
    start_date: str = None,
    end_date: str = None,
    use_parallel: bool = True,
) -> List[Dict]:
    """
    便捷函数：优化的批量回测
    
    Args:
        codes: 股票代码列表
        params: 策略参数
        start_date: 开始日期
        end_date: 结束日期
        use_parallel: 是否使用并行化
    
    Returns:
        回测结果列表
    """
    engine = OptimizedBacktestEngine(use_parallel=use_parallel)
    return engine.batch_backtest(codes, params, start_date, end_date)


if __name__ == "__main__":
    # 测试代码
    from quant.core.strategy_params import StrategyParams
    
    # 采样股票
    data_dir = CONF.history_data.data_dir
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    codes = [f[:-4] for f in files[:10]]  # 取前 10 只股票测试
    
    params = StrategyParams()
    
    # 运行优化版批量回测
    results = batch_backtest_optimized(
        codes=codes,
        params=params,
        use_parallel=True,
    )
    
    print(f"回测完成: {len(results)} 个结果")
    for result in results[:5]:
        print(f"  {result['code']}: 收益={result['return_pct']:.2f}%, "
              f"夏普={result['sharpe']:.2f}")
