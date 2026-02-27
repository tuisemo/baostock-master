import os
import yaml
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    level: str = "INFO"
    file: str = "data/quant.log"


@dataclass
class FilterConfig:
    keep_star_market: bool = False
    min_market_cap_billion: float = 30.0
    min_turnover_amount_wan: float = 10000.0
    max_turnover_rate_pct: float = 20.0
    min_turnover_rate_pct: float = 1.0
    min_pe: float = 0.0
    max_pe: float = 80.0
    min_pb: float = 0.5


@dataclass
class HistoryDataConfig:
    default_lookback_days: int = 500
    data_dir: str = "data"


@dataclass
class AnalyzerWeights:
    trend: float = 0.4
    reversion: float = 0.3
    volume: float = 0.3


@dataclass
class AnalyzerConfig:
    weights: AnalyzerWeights = field(default_factory=AnalyzerWeights)
    ma_short: int = 5
    ma_long: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_length: int = 14
    rsi_buy_threshold: float = 40.0
    rsi_sell_threshold: float = 70.0
    bbands_length: int = 20
    bbands_std: float = 2.0
    atr_length: int = 14
    atr_multiplier: float = 2.0


@dataclass
class StrategyConfig:
    vol_up_ratio: float = 1.35
    rsi_cooled_max: float = 55.0
    pullback_ma_tolerance: float = 1.02
    negative_bias_pct: float = 0.95
    rsi_oversold: float = 35.0
    bbands_lower_bias: float = 1.02
    rsi_oversold_extreme: float = 25.0
    trail_atr_mult: float = 1.8
    take_profit_pct: float = 0.06
    breakeven_trigger: float = 0.04
    breakeven_buffer: float = 1.005
    w_pullback_ma: float = 2.0
    w_macd_cross: float = 1.0
    w_vol_up: float = 1.0
    w_rsi_rebound: float = 2.0
    w_green_candle: float = 1.0


@dataclass
class OptimizerConfig:
    sample_count: int = 200
    max_rounds: int = 5
    convergence_threshold: float = 0.005
    walk_forward_splits: int = 5
    train_ratio: float = 0.7
    objective: str = "sharpe_adj"
    results_dir: str = "data/optimize_results"


@dataclass
class AppConfig:
    log: LogConfig = field(default_factory=LogConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    history_data: HistoryDataConfig = field(default_factory=HistoryDataConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    analyzer_data = data.get("analyzer", {})
    weights_data = analyzer_data.pop("weights", {})
    analyzer_cfg = AnalyzerConfig(
        ma_short=analyzer_data.get("ma_short", 5),
        ma_long=analyzer_data.get("ma_long", 20),
        macd_fast=analyzer_data.get("macd_fast", 12),
        macd_slow=analyzer_data.get("macd_slow", 26),
        macd_signal=analyzer_data.get("macd_signal", 9),
        rsi_length=analyzer_data.get("rsi_length", 14),
        rsi_buy_threshold=analyzer_data.get("rsi_buy_threshold", 40.0),
        rsi_sell_threshold=analyzer_data.get("rsi_sell_threshold", 70.0),
        bbands_length=analyzer_data.get("bbands_length", 20),
        bbands_std=analyzer_data.get("bbands_std", 2.0),
        atr_length=analyzer_data.get("atr_length", 14),
        atr_multiplier=analyzer_data.get("atr_multiplier", 2.0),
    )
    analyzer_cfg.weights = AnalyzerWeights(
        trend=weights_data.get("trend", 0.4),
        reversion=weights_data.get("reversion", 0.3),
        volume=weights_data.get("volume", 0.3),
    )
    return AppConfig(
        log=LogConfig(
            level=data.get("log", {}).get("level", "INFO"),
            file=data.get("log", {}).get("file", "data/quant.log")
        ),
        filter=FilterConfig(
            keep_star_market=data.get("filter", {}).get("keep_star_market", False),
            min_market_cap_billion=data.get("filter", {}).get("min_market_cap_billion", 50.0),
            min_turnover_amount_wan=data.get("filter", {}).get("min_turnover_amount_wan", 10000.0),
            max_turnover_rate_pct=data.get("filter", {}).get("max_turnover_rate_pct", 20.0),
            min_turnover_rate_pct=data.get("filter", {}).get("min_turnover_rate_pct", 2.0),
            min_pe=data.get("filter", {}).get("min_pe", 0.0),
            max_pe=data.get("filter", {}).get("max_pe", 100.0),
            min_pb=data.get("filter", {}).get("min_pb", 0.5)
        ),
        history_data=HistoryDataConfig(
            default_lookback_days=data.get("history_data", {}).get("default_lookback_days", 500),
            data_dir=data.get("history_data", {}).get("data_dir", "data")
        ),
        analyzer=analyzer_cfg,
        strategy=StrategyConfig(
            vol_up_ratio=data.get("strategy", {}).get("vol_up_ratio", 1.35),
            rsi_cooled_max=data.get("strategy", {}).get("rsi_cooled_max", 55.0),
            pullback_ma_tolerance=data.get("strategy", {}).get("pullback_ma_tolerance", 1.02),
            negative_bias_pct=data.get("strategy", {}).get("negative_bias_pct", 0.95),
            rsi_oversold=data.get("strategy", {}).get("rsi_oversold", 35.0),
            bbands_lower_bias=data.get("strategy", {}).get("bbands_lower_bias", 1.02),
            rsi_oversold_extreme=data.get("strategy", {}).get("rsi_oversold_extreme", 25.0),
            trail_atr_mult=data.get("strategy", {}).get("trail_atr_mult", 1.8),
            take_profit_pct=data.get("strategy", {}).get("take_profit_pct", 0.06),
            breakeven_trigger=data.get("strategy", {}).get("breakeven_trigger", 0.04),
            breakeven_buffer=data.get("strategy", {}).get("breakeven_buffer", 1.005),
            w_pullback_ma=data.get("strategy", {}).get("w_pullback_ma", 2.0),
            w_macd_cross=data.get("strategy", {}).get("w_macd_cross", 1.0),
            w_vol_up=data.get("strategy", {}).get("w_vol_up", 1.0),
            w_rsi_rebound=data.get("strategy", {}).get("w_rsi_rebound", 2.0),
            w_green_candle=data.get("strategy", {}).get("w_green_candle", 1.0)
        ),
        optimizer=OptimizerConfig(
            sample_count=data.get("optimizer", {}).get("sample_count", 200),
            max_rounds=data.get("optimizer", {}).get("max_rounds", 5),
            convergence_threshold=data.get("optimizer", {}).get("convergence_threshold", 0.005),
            walk_forward_splits=data.get("optimizer", {}).get("walk_forward_splits", 5),
            train_ratio=data.get("optimizer", {}).get("train_ratio", 0.7),
            objective=data.get("optimizer", {}).get("objective", "sharpe_adj"),
            results_dir=data.get("optimizer", {}).get("results_dir", "data/optimize_results")
        ),
    )


CONF = load_config()
