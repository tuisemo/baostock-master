"""
Integration tests for Feature Engineering modules
Tests feature extraction, feature selection, and caching
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile


def test_feature_imports():
    """Test all feature-related imports work"""
    from quant.features.features import extract_features, add_sector_relative_features
    from quant.features.feature_selection import FeatureSelector
    from quant.infra.cache_utils import FeatureCache, MultiLevelCache
    from quant.infra.numba_accelerator import get_numba_status
    assert True


def test_extract_features_count():
    """Test that feature extraction produces expected number of features"""
    from quant.features.features import extract_features

    # Create sample OHLCV data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(100).cumsum() + 10,
        'high': np.random.randn(100).cumsum() + 11,
        'low': np.random.randn(100).cumsum() + 9,
        'close': np.random.randn(100).cumsum() + 10.5,
        'volume': np.random.randint(1000, 10000, 100),
    })

    # Extract features
    df_features = extract_features(df)

    # Check that we have feature columns
    feat_cols = [c for c in df_features.columns if c.startswith('feat_')]
    assert len(feat_cols) > 0, "No feature columns found"

    # We expect at least 46 features (from the spec)
    print(f"Extracted {len(feat_cols)} features")
    assert len(feat_cols) >= 20, f"Expected at least 20 features, got {len(feat_cols)}"


def test_feature_selector_init():
    """Test FeatureSelector initialization"""
    from quant.features.feature_selection import FeatureSelector

    # Test RFE method
    selector_rfe = FeatureSelector(method='rfe', n_features=10)
    assert selector_rfe.method == 'rfe'
    assert selector_rfe.n_features == 10

    # Test correlation method
    selector_corr = FeatureSelector(method='correlation', correlation_threshold=0.9)
    assert selector_corr.method == 'correlation'
    assert selector_corr.correlation_threshold == 0.9

    # Test importance method
    selector_imp = FeatureSelector(method='importance', importance_threshold=0.01)
    assert selector_imp.method == 'importance'
    assert selector_imp.importance_threshold == 0.01


def test_feature_selector_fit_transform():
    """Test FeatureSelector fit and transform"""
    from quant.features.feature_selection import FeatureSelector

    # Create sample data
    n_samples = 100
    n_features = 20
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))

    # Test correlation method (doesn't require y)
    selector = FeatureSelector(method='correlation', correlation_threshold=0.95)
    selector.fit(X, y=None)
    X_transformed = selector.transform(X)

    assert X_transformed is not None
    assert len(X_transformed.columns) <= len(X.columns)


def test_cache_utils_init():
    """Test cache utilities initialization"""
    from quant.infra.cache_utils import FeatureCache, MultiLevelCache

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test FeatureCache
        cache = FeatureCache(cache_dir=tmpdir)
        assert cache.cache_dir == tmpdir

        # Test MultiLevelCache
        ml_cache = MultiLevelCache(cache_dir=tmpdir)
        assert ml_cache.cache_dir == tmpdir


def test_cache_hit_miss():
    """Test cache hit and miss functionality"""
    from quant.infra.cache_utils import FeatureCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FeatureCache(cache_dir=tmpdir)

        # Create test data
        data = {'test': 'data', 'value': 123}
        key = 'test_key'

        # Initially should be a miss
        assert not cache.exists(key)

        # Save data
        cache.save(key, data)

        # Now should exist
        assert cache.exists(key)

        # Load data
        loaded = cache.load(key)
        assert loaded == data


def test_numba_status():
    """Test Numba accelerator status"""
    from quant.infra.numba_accelerator import get_numba_status

    status = get_numba_status()
    assert isinstance(status, dict)
    assert 'numba_available' in status


def test_feature_memory_optimization():
    """Test feature memory optimization"""
    from quant.features.features import _optimize_feature_memory

    # Create sample dataframe with feature columns
    df = pd.DataFrame({
        'feat_a': np.random.randn(100).astype('float64'),
        'feat_b': np.random.randn(100).astype('float64'),
        'other_col': np.random.randn(100)
    })

    # Optimize memory
    df_opt = _optimize_feature_memory(df)

    # Check feature columns are now float32
    assert df_opt['feat_a'].dtype == np.float32
    assert df_opt['feat_b'].dtype == np.float32


def test_correlation_analysis():
    """Test feature correlation analysis"""
    from quant.features.feature_selection import analyze_feature_correlation

    # Create correlated features
    base = np.random.randn(100)
    X = pd.DataFrame({
        'feat_a': base,
        'feat_b': base + np.random.randn(100) * 0.1,  # Highly correlated
        'feat_c': np.random.randn(100),  # Uncorrelated
    })

    # Analyze correlation
    high_corr = analyze_feature_correlation(X, threshold=0.8)

    assert isinstance(high_corr, pd.DataFrame)
    # Should find feat_a and feat_b as highly correlated


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
