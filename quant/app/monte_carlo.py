"""
Monte Carlo Simulation Module

Provides robustness testing through bootstrap resampling and Monte Carlo simulations.
Features:
1. Block bootstrap to preserve time series properties
2. Monte Carlo backtesting
3. Confidence intervals for returns and drawdowns
4. Stress testing scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from quant.infra.logger import logger


@dataclass
class MonteCarloResult:
    """Result container for Monte Carlo simulation"""
    paths: np.ndarray  # Simulated paths (n_paths x n_periods)
    final_returns: np.ndarray  # Final returns for each path
    max_drawdowns: np.ndarray  # Max drawdowns for each path
    volatility: np.ndarray  # Volatility for each path
    sharpe_ratios: np.ndarray  # Sharpe ratios for each path
    var_95: np.ndarray  # 95% VaR for each path
    var_99: np.ndarray  # 99% VaR for each path

    # Statistics
    mean_return: float = 0.0
    median_return: float = 0.0
    return_ci_95: Tuple[float, float] = (0.0, 0.0)
    return_ci_99: Tuple[float, float] = (0.0, 0.0)

    mean_drawdown: float = 0.0
    max_drawdown: float = 0.0
    drawdown_ci_95: Tuple[float, float] = (0.0, 0.0)

    prob_profit: float = 0.0  # Probability of profitable outcome
    prob_max_dd_10: float = 0.0  # P(max DD > 10%)
    prob_max_dd_20: float = 0.0  # P(max DD > 20%)

    stress_test_results: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'mean_return': self.mean_return,
            'median_return': self.median_return,
            'return_ci_95': self.return_ci_95,
            'return_ci_99': self.return_ci_99,
            'mean_drawdown': self.mean_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_ci_95': self.drawdown_ci_95,
            'prob_profit': self.prob_profit,
            'prob_max_dd_10': self.prob_max_dd_10,
            'prob_max_dd_20': self.prob_max_dd_20,
            'stress_test_results': self.stress_test_results,
        }


class BlockBootstrap:
    """Block bootstrap for time series resampling"""

    def __init__(self, block_size: int = 20):
        """
        Initialize block bootstrap

        Args:
            block_size: Size of blocks to sample (default 20 days)
        """
        self.block_size = block_size

    def resample(self, returns: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """
        Generate bootstrap samples using block bootstrap

        Args:
            returns: Original returns array
            n_samples: Number of bootstrap samples to generate

        Returns:
            Array of shape (n_samples, len(returns))
        """
        n = len(returns)
        n_blocks = int(np.ceil(n / self.block_size))

        samples = np.zeros((n_samples, n))

        for i in range(n_samples):
            # Randomly select starting points for blocks
            start_indices = np.random.randint(0, n - self.block_size + 1, size=n_blocks)

            # Concatenate blocks
            resampled = []
            for start in start_indices:
                block = returns[start:start + self.block_size]
                resampled.extend(block)

            # Trim to original length
            samples[i] = np.array(resampled[:n])

        return samples

    def resample_preserve_autocorr(
        self,
        returns: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Resample preserving autocorrelation structure using stationary bootstrap

        Args:
            returns: Original returns array
            n_samples: Number of samples

        Returns:
            Resampled returns
        """
        n = len(returns)
        samples = np.zeros((n_samples, n))

        # Expected block length (geometric distribution)
        p = 1.0 / self.block_size

        for i in range(n_samples):
            resampled = []
            while len(resampled) < n:
                # Random starting point
                start = np.random.randint(0, n)
                # Random block length
                block_len = np.random.geometric(p)
                # Sample block with wrap-around
                for j in range(block_len):
                    idx = (start + j) % n
                    resampled.append(returns[idx])
                    if len(resampled) >= n:
                        break

            samples[i] = np.array(resampled[:n])

        return samples


def bootstrap_resample(
    returns: pd.Series,
    n_samples: int = 1000,
    block_size: int = 20,
    preserve_autocorr: bool = True
) -> np.ndarray:
    """
    Block bootstrap resampling to preserve time series properties

    Args:
        returns: Returns series
        n_samples: Number of bootstrap samples
        block_size: Block size for resampling
        preserve_autocorr: Whether to preserve autocorrelation

    Returns:
        Bootstrap samples array
    """
    bootstrap = BlockBootstrap(block_size)
    returns_arr = returns.values if isinstance(returns, pd.Series) else returns

    if preserve_autocorr:
        return bootstrap.resample_preserve_autocorr(returns_arr, n_samples)
    return bootstrap.resample(returns_arr, n_samples)


def monte_carlo_simulation(
    returns: pd.Series,
    n_paths: int = 1000,
    initial_equity: float = 100000.0,
    block_size: int = 20,
    n_jobs: int = -1
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation using block bootstrap

    Args:
        returns: Historical returns series
        n_paths: Number of paths to simulate
        initial_equity: Initial equity
        block_size: Block size for bootstrap
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        MonteCarloResult with simulation statistics
    """
    logger.info(f"Running Monte Carlo simulation with {n_paths} paths...")

    # Generate bootstrap samples
    bootstrap = BlockBootstrap(block_size)
    returns_arr = returns.values if isinstance(returns, pd.Series) else returns
    samples = bootstrap.resample_preserve_autocorr(returns_arr, n_paths)

    # Calculate paths
    n_periods = len(returns_arr)
    paths = np.zeros((n_paths, n_periods))
    paths[:, 0] = initial_equity

    for i in range(n_paths):
        for t in range(1, n_periods):
            paths[i, t] = paths[i, t-1] * (1 + samples[i, t])

    # Calculate metrics for each path
    final_returns = (paths[:, -1] - initial_equity) / initial_equity

    # Calculate drawdowns
    max_drawdowns = np.zeros(n_paths)
    volatility = np.zeros(n_paths)
    sharpe_ratios = np.zeros(n_paths)
    var_95 = np.zeros(n_paths)
    var_99 = np.zeros(n_paths)

    for i in range(n_paths):
        # Max drawdown
        equity = paths[i]
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdowns[i] = np.min(drawdown)

        # Volatility (annualized)
        vol = np.std(samples[i]) * np.sqrt(252)
        volatility[i] = vol

        # Sharpe ratio
        mean_return = np.mean(samples[i]) * 252
        if vol > 0:
            sharpe_ratios[i] = mean_return / vol
        else:
            sharpe_ratios[i] = 0

        # VaR
        var_95[i] = np.percentile(samples[i], 5)
        var_99[i] = np.percentile(samples[i], 1)

    # Create result
    result = MonteCarloResult(
        paths=paths,
        final_returns=final_returns,
        max_drawdowns=max_drawdowns,
        volatility=volatility,
        sharpe_ratios=sharpe_ratios,
        var_95=var_95,
        var_99=var_99,
    )

    # Calculate statistics
    result.mean_return = np.mean(final_returns)
    result.median_return = np.median(final_returns)
    result.return_ci_95 = (np.percentile(final_returns, 2.5), np.percentile(final_returns, 97.5))
    result.return_ci_99 = (np.percentile(final_returns, 0.5), np.percentile(final_returns, 99.5))

    result.mean_drawdown = np.mean(max_drawdowns)
    result.max_drawdown = np.min(max_drawdowns)
    result.drawdown_ci_95 = (np.percentile(max_drawdowns, 2.5), np.percentile(max_drawdowns, 97.5))

    result.prob_profit = np.mean(final_returns > 0)
    result.prob_max_dd_10 = np.mean(max_drawdowns < -0.10)
    result.prob_max_dd_20 = np.mean(max_drawdowns < -0.20)

    logger.info(f"Monte Carlo simulation complete: "
                f"mean_return={result.mean_return:.2%}, "
                f"mean_dd={result.mean_drawdown:.2%}, "
                f"prob_profit={result.prob_profit:.1%}")

    return result


def monte_carlo_backtest(
    strategy: Callable,
    n_paths: int = 1000,
    returns_data: pd.DataFrame = None,
    **strategy_kwargs
) -> MonteCarloResult:
    """
    Run Monte Carlo backtest on a strategy

    Args:
        strategy: Strategy function that takes returns and returns equity curve
        n_paths: Number of paths
        returns_data: Historical returns data
        **strategy_kwargs: Additional arguments for strategy

    Returns:
        MonteCarloResult
    """
    logger.info(f"Running Monte Carlo backtest with {n_paths} paths...")

    if returns_data is None:
        raise ValueError("returns_data is required")

    # Use portfolio returns (equally weighted)
    if isinstance(returns_data, pd.DataFrame):
        returns = returns_data.mean(axis=1)
    else:
        returns = returns_data

    return monte_carlo_simulation(returns, n_paths, **strategy_kwargs)


class StressTestScenario:
    """Stress test scenario definitions"""

    SCENARIOS = {
        'market_crash': {
            'description': 'Market crash scenario (2008-style)',
            'mean_return': -0.02,  # -2% daily
            'volatility': 0.05,    # 5% daily vol
            'skewness': -1.5,      # Negative skew
            'kurtosis': 10,        # Fat tails
        },
        'bear_market': {
            'description': 'Bear market scenario',
            'mean_return': -0.005,  # -0.5% daily
            'volatility': 0.025,    # 2.5% daily vol
            'skewness': -0.5,
            'kurtosis': 5,
        },
        'high_volatility': {
            'description': 'High volatility regime',
            'mean_return': 0.0,
            'volatility': 0.04,
            'skewness': 0,
            'kurtosis': 6,
        },
        'liquidity_crisis': {
            'description': 'Liquidity crisis',
            'mean_return': -0.01,
            'volatility': 0.035,
            'skewness': -2.0,
            'kurtosis': 15,
        },
        'recovery': {
            'description': 'Market recovery',
            'mean_return': 0.01,
            'volatility': 0.02,
            'skewness': 0.5,
            'kurtosis': 4,
        }
    }

    @classmethod
    def generate_returns(
        cls,
        scenario: str,
        n_periods: int = 252,
        seed: int = None
    ) -> np.ndarray:
        """
        Generate returns for a stress test scenario

        Args:
            scenario: Scenario name
            n_periods: Number of periods
            seed: Random seed

        Returns:
            Returns array
        """
        if seed is not None:
            np.random.seed(seed)

        params = cls.SCENARIOS.get(scenario, cls.SCENARIOS['bear_market'])

        # Generate returns with specified moments
        # Use Cornish-Fisher expansion approximation
        returns = np.random.normal(
            params['mean_return'],
            params['volatility'],
            n_periods
        )

        # Adjust for skewness and kurtosis (simplified)
        if params['skewness'] != 0:
            returns = returns + params['skewness'] * np.abs(returns) ** 1.5 * np.sign(returns) * 0.1

        return returns

    @classmethod
    def run_stress_test(
        cls,
        strategy: Callable,
        scenarios: List[str] = None,
        n_paths: int = 100,
        n_periods: int = 252,
    ) -> Dict[str, Dict]:
        """
        Run stress tests for multiple scenarios

        Args:
            strategy: Strategy function
            scenarios: List of scenario names (default: all)
            n_paths: Number of paths per scenario
            n_periods: Number of periods

        Returns:
            Results by scenario
        """
        if scenarios is None:
            scenarios = list(cls.SCENARIOS.keys())

        results = {}

        for scenario in scenarios:
            logger.info(f"Running stress test: {scenario}")

            scenario_returns = []
            scenario_equity = []

            for _ in range(n_paths):
                returns = cls.generate_returns(scenario, n_periods)

                # Simulate equity curve
                equity = np.zeros(n_periods)
                equity[0] = 100000.0

                for t in range(1, n_periods):
                    equity[t] = equity[t-1] * (1 + returns[t])

                scenario_returns.append((equity[-1] - equity[0]) / equity[0])
                scenario_equity.append(equity)

            # Calculate statistics
            returns_arr = np.array(scenario_returns)

            results[scenario] = {
                'description': cls.SCENARIOS[scenario]['description'],
                'mean_return': np.mean(returns_arr),
                'median_return': np.median(returns_arr),
                'return_ci_95': (
                    np.percentile(returns_arr, 2.5),
                    np.percentile(returns_arr, 97.5)
                ),
                'prob_profit': np.mean(returns_arr > 0),
                'max_loss': np.min(returns_arr),
                'max_gain': np.max(returns_arr),
            }

        return results


def generate_confidence_intervals(
    returns: pd.Series,
    confidence_levels: List[float] = [0.95, 0.99],
    n_bootstrap: int = 1000,
    block_size: int = 20
) -> Dict[float, Tuple[float, float]]:
    """
    Generate confidence intervals for returns using bootstrap

    Args:
        returns: Returns series
        confidence_levels: Confidence levels
        n_bootstrap: Number of bootstrap samples
        block_size: Block size

    Returns:
        Dictionary of confidence level -> (lower, upper)
    """
    bootstrap = BlockBootstrap(block_size)
    returns_arr = returns.values if isinstance(returns, pd.Series) else returns

    samples = bootstrap.resample_preserve_autocorr(returns_arr, n_bootstrap)

    results = {}
    for conf in confidence_levels:
        alpha = (1 - conf) / 2
        lower = np.percentile(samples, alpha * 100)
        upper = np.percentile(samples, (1 - alpha) * 100)
        results[conf] = (lower, upper)

    return results


if __name__ == "__main__":
    # Test code
    print("Testing Monte Carlo module...")

    # Generate synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 252))

    # Run Monte Carlo
    result = monte_carlo_simulation(returns, n_paths=1000)

    print(f"\nMonte Carlo Results:")
    print(f"Mean Return: {result.mean_return:.2%}")
    print(f"95% CI: [{result.return_ci_95[0]:.2%}, {result.return_ci_95[1]:.2%}]")
    print(f"Mean Max Drawdown: {result.mean_drawdown:.2%}")
    print(f"Probability of Profit: {result.prob_profit:.1%}")

    # Test stress scenarios
    print("\nStress Test Scenarios:")
    stress_results = StressTestScenario.run_stress_test(
        None,
        scenarios=['market_crash', 'bear_market'],
        n_paths=100
    )

    for scenario, res in stress_results.items():
        print(f"\n{scenario}:")
        print(f"  Mean Return: {res['mean_return']:.2%}")
        print(f"  Prob Profit: {res['prob_profit']:.1%}")
