"""
Integration tests for Strategy Optimization modules
Tests dynamic thresholds, signal fusion, and multi-timeframe confirmation
"""

import pytest
import pandas as pd
import numpy as np


def test_strategy_imports():
    """Test all strategy-related imports work"""
    from quant.core.adaptive_strategy import get_dynamic_params_with_thresholds
    from quant.core.adaptive_strategy import get_market_thresholds, MARKET_STATE_THRESHOLDS
    assert True


def test_market_thresholds():
    """Test market state thresholds configuration"""
    from quant.core.adaptive_strategy import get_market_thresholds, MARKET_STATE_THRESHOLDS

    # Test all defined market states
    expected_states = [
        'strong_bull', 'bull_momentum', 'bull_volume', 'weak_bull',
        'sideways_low_vol', 'sideways_high_vol',
        'weak_bear', 'bear_momentum', 'bear_panic', 'strong_bear'
    ]

    for state in expected_states:
        thresholds = get_market_thresholds(state)
        assert 'ai_threshold' in thresholds
        assert 'position_mult' in thresholds
        assert 'stop_mult' in thresholds


def test_dynamic_params_with_thresholds():
    """Test dynamic parameter adjustment based on market state"""
    from quant.core.adaptive_strategy import get_dynamic_params_with_thresholds
    from quant.core.strategy_params import StrategyParams

    # Create base parameters
    base_params = StrategyParams(
        ai_prob_threshold=0.35,
        vol_up_ratio=1.5,
        rsi_cooled_max=55
    )

    # Test different market states
    market_states = ['strong_bull', 'sideways_low_vol', 'strong_bear']

    for state in market_states:
        adjusted_params, thresholds = get_dynamic_params_with_thresholds(base_params, state)
        assert adjusted_params is not None
        assert isinstance(thresholds, dict)


def test_threshold_variations_by_market_state():
    """Test that thresholds vary appropriately by market state"""
    from quant.core.adaptive_strategy import get_market_thresholds

    # Strong bull should have lower AI threshold (more aggressive)
    strong_bull = get_market_thresholds('strong_bull')
    strong_bear = get_market_thresholds('strong_bear')

    # Bull markets should have lower thresholds (easier to enter)
    assert strong_bull['ai_threshold'] < strong_bear['ai_threshold']

    # Bull markets should have larger position sizes
    assert strong_bull['position_mult'] > strong_bear['position_mult']


def test_signal_fusion_weights():
    """Test signal fusion weights from config"""
    from quant.infra.config import CONF

    # Check if signal fusion weights exist in config
    if hasattr(CONF.strategy, 'signal_fusion_weights'):
        weights = CONF.strategy.signal_fusion_weights
        assert 'trend' in weights
        assert 'reversion' in weights
        assert sum(weights.values()) > 0


def test_multi_timeframe_config():
    """Test multi-timeframe configuration"""
    from quant.infra.config import CONF

    # Check if multi-timeframe config exists
    if hasattr(CONF.strategy, 'multi_timeframe'):
        mt_config = CONF.strategy.multi_timeframe
        assert 'enabled' in mt_config
        assert 'weekly_confirmation' in mt_config


def test_volatility_adjusted_stop():
    """Test volatility-adjusted stop calculation"""
    from quant.core.adaptive_strategy import calculate_volatility_adjusted_stop, get_market_thresholds

    base_stop = 2.0
    atr_history = [2.0] * 25
    market_state = 'sideways_low_vol'
    base_mult = get_market_thresholds(market_state)["stop_mult"]

    # High volatility should tighten stops (smaller multiplier).
    adjusted = calculate_volatility_adjusted_stop(base_stop, 4.0, atr_history, market_state)

    assert adjusted > 0
    assert adjusted < base_mult

    # Low volatility should widen stops slightly (larger multiplier).
    adjusted_low = calculate_volatility_adjusted_stop(base_stop, 0.8, atr_history, market_state)
    assert adjusted_low > 0
    assert adjusted_low > base_mult


def test_market_state_thresholds_config():
    """Test market state thresholds from config.yaml"""
    from quant.infra.config import CONF

    # Verify config has market_state_thresholds
    assert hasattr(CONF.strategy, 'market_state_thresholds')

    thresholds = CONF.strategy.market_state_thresholds
    assert 'strong_bull' in thresholds
    assert 'sideways' in thresholds
    assert 'strong_bear' in thresholds

    # Verify each has required fields
    for state, params in thresholds.items():
        assert 'ai_threshold' in params
        assert 'position_mult' in params
        assert 'stop_mult' in params


def test_signal_scorer():
    """Test SignalScorer initialization if available"""
    try:
        from quant.core.signal_scorer import SignalScorer
        from quant.core.strategy_params import StrategyParams

        scorer = SignalScorer(StrategyParams())
        assert scorer is not None
    except ImportError:
        pytest.skip("SignalScorer not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
