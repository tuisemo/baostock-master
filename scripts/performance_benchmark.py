"""
性能基准测试脚本
Phase 9 性能优化验证

测试项目:
1. Walk-forward 优化时间（并行 vs 顺序）
2. 特征计算时间（Numba vs 纯 Python）
3. 缓存命中率
4. 内存使用
"""

import os
import sys
import time
import gc
import json
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psutil

from quant.infra.config import CONF
from quant.infra.logger import logger
from quant.core.strategy_params import StrategyParams


def get_memory_usage() -> float:
    """获取当前进程内存使用（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_walk_forward() -> Dict[str, Any]:
    """
    测试 walk-forward 优化性能
    """
    logger.info("=" * 60)
    logger.info("Benchmark 1: Walk-Forward Optimization")
    logger.info("=" * 60)
    
    from quant.app.optimizer import walk_forward_cv, walk_forward_cv_parallel
    
    # 使用少量股票进行快速测试
    codes = ['sh.600000', 'sh.600001', 'sh.600002']
    params = StrategyParams()
    
    results = {}
    
    # 测试顺序执行
    logger.info("测试顺序 walk-forward...")
    gc.collect()
    mem_before = get_memory_usage()
    start_time = time.time()
    
    try:
        score_seq = walk_forward_cv(
            codes=codes[:1],  # 使用单只股票测试
            params=params,
            n_folds=3,
            parallel=False
        )
        time_seq = time.time() - start_time
        mem_after = get_memory_usage()
        
        results['sequential'] = {
            'time': time_seq,
            'memory_mb': mem_after - mem_before,
            'score': float(score_seq)
        }
        logger.info(f"  顺序执行: {time_seq:.2f}s, 内存: {mem_after - mem_before:.1f}MB")
    except Exception as e:
        logger.error(f"顺序执行失败: {e}")
        results['sequential'] = {'error': str(e)}
    
    # 测试并行执行
    logger.info("测试并行 walk-forward...")
    gc.collect()
    mem_before = get_memory_usage()
    start_time = time.time()
    
    try:
        score_par = walk_forward_cv_parallel(
            codes=codes[:1],
            params=params,
            n_folds=3,
            n_jobs=4
        )
        time_par = time.time() - start_time
        mem_after = get_memory_usage()
        
        results['parallel'] = {
            'time': time_par,
            'memory_mb': mem_after - mem_before,
            'score': float(score_par)
        }
        logger.info(f"  并行执行: {time_par:.2f}s, 内存: {mem_after - mem_before:.1f}MB")
        
        # 计算加速比
        if 'sequential' in results and 'time' in results['sequential']:
            speedup = results['sequential']['time'] / time_par if time_par > 0 else 0
            results['speedup'] = speedup
            logger.info(f"  加速比: {speedup:.2f}x")
            
    except Exception as e:
        logger.error(f"并行执行失败: {e}")
        results['parallel'] = {'error': str(e)}
    
    return results


def benchmark_cache() -> Dict[str, Any]:
    """
    测试多级缓存性能
    """
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark 2: Multi-Level Cache")
    logger.info("=" * 60)
    
    from quant.infra.cache_utils import MultiLevelCache, LRUCache
    
    results = {}
    
    # 测试 LRU 缓存
    logger.info("测试 LRU 缓存...")
    cache = LRUCache(capacity=1000, ttl=60)
    
    # 写入测试
    start = time.time()
    for i in range(1000):
        cache.set(f"key_{i}", np.random.randn(100))
    write_time = time.time() - start
    
    # 读取测试（混合命中和未命中）
    start = time.time()
    hits = 0
    for i in range(2000):
        value = cache.get(f"key_{i % 1500}")  # 1500 次访问只有 1000 个键
        if value is not None:
            hits += 1
    read_time = time.time() - start
    
    stats = cache.get_stats()
    results['lru_cache'] = {
        'write_time_1000': write_time,
        'read_time_2000': read_time,
        'hit_rate': stats['hit_rate'],
        'size': stats['size']
    }
    
    logger.info(f"  写入 1000 条: {write_time:.3f}s")
    logger.info(f"  读取 2000 条: {read_time:.3f}s")
    logger.info(f"  命中率: {stats['hit_rate']:.2%}")
    logger.info(f"  缓存大小: {stats['size']}")
    
    # 测试多级缓存
    logger.info("测试多级缓存...")
    ml_cache = MultiLevelCache(l1_capacity=100, l1_ttl=60)
    
    # 写入测试
    start = time.time()
    for i in range(100):
        ml_cache.set(np.random.randn(50), f"test_{i}")
    ml_write_time = time.time() - start
    
    # 读取测试
    start = time.time()
    ml_hits = 0
    sources = {'l1': 0, 'l2': 0, 'miss': 0}
    for i in range(200):
        value, source = ml_cache.get(f"test_{i % 150}")
        sources[source] += 1
        if value is not None:
            ml_hits += 1
    ml_read_time = time.time() - start
    
    results['multi_level_cache'] = {
        'write_time_100': ml_write_time,
        'read_time_200': ml_read_time,
        'hits': ml_hits,
        'sources': sources
    }
    
    logger.info(f"  写入 100 条: {ml_write_time:.3f}s")
    logger.info(f"  读取 200 条: {ml_read_time:.3f}s")
    logger.info(f"  命中: {ml_hits}/200")
    logger.info(f"  来源分布: L1={sources['l1']}, L2={sources['l2']}, Miss={sources['miss']}")
    
    return results


def benchmark_numba() -> Dict[str, Any]:
    """
    测试 Numba 加速性能
    """
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark 3: Numba Acceleration")
    logger.info("=" * 60)
    
    from quant.infra.numba_accelerator import (
        compute_rsi_numba,
        compute_macd_numba,
        compute_bollinger_bands_numba,
        NUMBA_AVAILABLE
    )
    
    results = {'numba_available': NUMBA_AVAILABLE}
    
    if not NUMBA_AVAILABLE:
        logger.warning("Numba 不可用，跳过测试")
        return results
    
    # 生成测试数据
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(10000) * 0.01) + 100
    
    # 测试 RSI
    logger.info("测试 RSI 计算...")
    
    # Numba 版本（包含编译时间）
    start = time.time()
    rsi_numba = compute_rsi_numba(prices, period=14)
    time_numba_first = time.time() - start
    
    # Numba 版本（已编译）
    start = time.time()
    rsi_numba = compute_rsi_numba(prices, period=14)
    time_numba_cached = time.time() - start
    
    # 纯 Python 版本
    def compute_rsi_python(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean().values
        avg_losses = pd.Series(losses).rolling(window=period).mean().values
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    start = time.time()
    rsi_python = compute_rsi_python(prices, period=14)
    time_python = time.time() - start
    
    results['rsi'] = {
        'numba_first_run': time_numba_first,
        'numba_cached': time_numba_cached,
        'python': time_python,
        'speedup_vs_first': time_python / time_numba_first if time_numba_first > 0 else 0,
        'speedup_vs_cached': time_python / time_numba_cached if time_numba_cached > 0 else 0
    }
    
    logger.info(f"  Numba (首次): {time_numba_first:.4f}s")
    logger.info(f"  Numba (已编译): {time_numba_cached:.4f}s")
    logger.info(f"  Python: {time_python:.4f}s")
    logger.info(f"  加速比: {results['rsi']['speedup_vs_cached']:.2f}x")
    
    # 测试 MACD
    logger.info("测试 MACD 计算...")
    start = time.time()
    macd, signal, hist = compute_macd_numba(prices, fast_period=12, slow_period=26, signal_period=9)
    time_macd = time.time() - start
    
    results['macd'] = {'time': time_macd}
    logger.info(f"  Numba MACD: {time_macd:.4f}s")
    
    # 测试布林带
    logger.info("测试布林带计算...")
    start = time.time()
    upper, middle, lower = compute_bollinger_bands_numba(prices, period=20, std_dev=2.0)
    time_bb = time.time() - start
    
    results['bollinger_bands'] = {'time': time_bb}
    logger.info(f"  Numma Bollinger: {time_bb:.4f}s")
    
    return results


def benchmark_lazy_loading() -> Dict[str, Any]:
    """
    测试惰性加载性能
    """
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark 4: Lazy Data Loading")
    logger.info("=" * 60)
    
    from quant.data.data_updater import LazyDataLoader
    
    data_dir = CONF.history_data.data_dir
    loader = LazyDataLoader(data_dir, memory_cache_size=50)
    
    results = {}
    
    # 获取股票列表
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'stock-list.csv']
    test_codes = [f[:-4] for f in all_files[:20]]  # 使用前20只
    
    if not test_codes:
        logger.warning("没有可用的测试股票")
        return results
    
    # 测试首次加载
    logger.info(f"测试首次加载 {len(test_codes)} 只股票...")
    start = time.time()
    for code in test_codes:
        df = loader.get_stock_data(code)
    first_load_time = time.time() - start
    
    # 测试缓存命中
    logger.info("测试缓存命中...")
    start = time.time()
    for code in test_codes:
        df = loader.get_stock_data(code)
    cached_load_time = time.time() - start
    
    stats = loader.get_stats()
    
    results['lazy_loading'] = {
        'first_load_time': first_load_time,
        'cached_load_time': cached_load_time,
        'speedup': first_load_time / cached_load_time if cached_load_time > 0 else 0,
        'cache_stats': stats
    }
    
    logger.info(f"  首次加载: {first_load_time:.2f}s")
    logger.info(f"  缓存加载: {cached_load_time:.2f}s")
    logger.info(f"  加速比: {results['lazy_loading']['speedup']:.2f}x")
    logger.info(f"  缓存命中率: {stats['hit_rate']:.2%}")
    
    return results


def benchmark_feature_extraction() -> Dict[str, Any]:
    """
    测试特征提取性能
    """
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark 5: Feature Extraction")
    logger.info("=" * 60)
    
    from quant.features.features import extract_features, extract_features_with_cache
    from quant.data.data_updater import LazyDataLoader
    
    data_dir = CONF.history_data.data_dir
    loader = LazyDataLoader(data_dir)
    
    results = {}
    
    # 加载测试数据
    test_code = 'sh.600000'
    df = loader.get_stock_data(test_code)
    
    if df is None:
        logger.warning(f"无法加载测试股票 {test_code}")
        return results
    
    # 测试特征提取（首次）
    logger.info("测试特征提取（首次）...")
    gc.collect()
    mem_before = get_memory_usage()
    start = time.time()
    
    try:
        features = extract_features(df)
        time_first = time.time() - start
        mem_after = get_memory_usage()
        
        results['first_extraction'] = {
            'time': time_first,
            'memory_mb': mem_after - mem_before,
            'num_features': len([c for c in features.columns if c.startswith('feat_')])
        }
        logger.info(f"  耗时: {time_first:.2f}s")
        logger.info(f"  内存: {mem_after - mem_before:.1f}MB")
        logger.info(f"  特征数: {results['first_extraction']['num_features']}")
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        results['first_extraction'] = {'error': str(e)}
    
    return results


def run_all_benchmarks() -> Dict[str, Any]:
    """
    运行所有基准测试
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 9 PERFORMANCE BENCHMARK")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
    }
    
    # Run benchmarks
    all_results['walk_forward'] = benchmark_walk_forward()
    all_results['cache'] = benchmark_cache()
    all_results['numba'] = benchmark_numba()
    all_results['lazy_loading'] = benchmark_lazy_loading()
    all_results['feature_extraction'] = benchmark_feature_extraction()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)
    
    # Walk-forward speedup
    if 'walk_forward' in all_results:
        wf = all_results['walk_forward']
        if 'speedup' in wf:
            logger.info(f"Walk-Forward 加速比: {wf['speedup']:.2f}x")
    
    # Cache hit rate
    if 'cache' in all_results and 'lru_cache' in all_results['cache']:
        cache = all_results['cache']['lru_cache']
        logger.info(f"缓存命中率: {cache.get('hit_rate', 0):.2%}")
    
    # Numba speedup
    if 'numba' in all_results and 'rsi' in all_results['numba']:
        numba = all_results['numba']['rsi']
        logger.info(f"Numba RSI 加速比: {numba.get('speedup_vs_cached', 0):.2f}x")
    
    # Lazy loading speedup
    if 'lazy_loading' in all_results and 'lazy_loading' in all_results['lazy_loading']:
        lazy = all_results['lazy_loading']['lazy_loading']
        logger.info(f"惰性加载加速比: {lazy.get('speedup', 0):.2f}x")
    
    return all_results


def save_benchmark_report(results: Dict[str, Any], output_path: str = None):
    """
    保存基准测试报告
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"performance_benchmark_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n基准测试报告已保存: {output_path}")


if __name__ == "__main__":
    # 配置日志
    import logging
    logger.setLevel(logging.INFO)
    
    # 运行测试
    results = run_all_benchmarks()
    
    # 保存报告
    save_benchmark_report(results)
    
    logger.info("\n基准测试完成!")
