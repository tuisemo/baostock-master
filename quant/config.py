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
    min_market_cap_billion: float = 50.0
    min_turnover_amount_wan: float = 10000.0
    max_turnover_rate_pct: float = 20.0
    min_turnover_rate_pct: float = 2.0


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
    trail_atr_mult: float = 1.8
    take_profit_pct: float = 0.06
    breakeven_trigger: float = 0.04
    breakeven_buffer: float = 1.005


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
    analyzer_cfg = AnalyzerConfig(**analyzer_data)
    analyzer_cfg.weights = AnalyzerWeights(**weights_data)

    return AppConfig(
        log=LogConfig(**data.get("log", {})),
        filter=FilterConfig(**data.get("filter", {})),
        history_data=HistoryDataConfig(**data.get("history_data", {})),
        analyzer=analyzer_cfg,
        strategy=StrategyConfig(**data.get("strategy", {})),
        optimizer=OptimizerConfig(**data.get("optimizer", {})),
    )


CONF = load_config()
