from __future__ import annotations

import yaml
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant.config import AppConfig


@dataclass
class StrategyParams:
    """Flat container for all tunable strategy parameters."""

    # --- Analyzer: Moving Averages ---
    ma_short: int = 5
    ma_long: int = 20

    # --- Analyzer: MACD ---
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- Analyzer: RSI ---
    rsi_length: int = 14
    rsi_buy_threshold: float = 40.0
    rsi_sell_threshold: float = 70.0

    # --- Analyzer: Bollinger Bands ---
    bbands_length: int = 20
    bbands_std: float = 2.0

    # --- Analyzer: ATR ---
    atr_length: int = 14
    atr_multiplier: float = 2.0

    # --- Analyzer: Weights ---
    weight_trend: float = 0.4
    weight_reversion: float = 0.3
    weight_volume: float = 0.3

    # Signal Scoring Weights (Phase 3 Nonlinearity)
    w_pullback_ma: float = 2.0
    w_macd_cross: float = 1.0
    w_vol_up: float = 1.0
    w_rsi_rebound: float = 2.0
    w_green_candle: float = 1.0

    # --- Strategy: Entry ---
    vol_up_ratio: float = 1.35
    rsi_cooled_max: float = 55.0
    pullback_ma_tolerance: float = 1.02
    negative_bias_pct: float = 0.95
    rsi_oversold: float = 35.0
    bbands_lower_bias: float = 1.02
    rsi_oversold_extreme: float = 22.0

    # --- Strategy: Exit ---
    trail_atr_mult: float = 1.8
    take_profit_pct: float = 0.06
    breakeven_trigger: float = 0.04
    breakeven_buffer: float = 1.005

    # --- Phase 9: AI Model Gate ---
    ai_prob_threshold: float = 0.35

    # --- Position Sizing ---
    position_size: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, int | float]) -> StrategyParams:
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategyParams:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_app_config(cls, conf: AppConfig) -> StrategyParams:
        a = conf.analyzer
        strategy_data = getattr(conf, "strategy", None)

        kwargs: dict[str, int | float] = {
            "ma_short": a.ma_short,
            "ma_long": a.ma_long,
            "macd_fast": a.macd_fast,
            "macd_slow": a.macd_slow,
            "macd_signal": a.macd_signal,
            "rsi_length": a.rsi_length,
            "rsi_buy_threshold": a.rsi_buy_threshold,
            "rsi_sell_threshold": a.rsi_sell_threshold,
            "bbands_length": a.bbands_length,
            "bbands_std": a.bbands_std,
            "atr_length": a.atr_length,
            "atr_multiplier": a.atr_multiplier,
            "weight_trend": a.weights.trend,
            "weight_reversion": a.weights.reversion,
            "weight_volume": a.weights.volume,
        }

        if strategy_data is not None:
            for name in (
                "vol_up_ratio", "rsi_cooled_max", "pullback_ma_tolerance",
                "negative_bias_pct", "rsi_oversold", "trail_atr_mult",
                "take_profit_pct", "breakeven_trigger", "breakeven_buffer",
                "w_pullback_ma", "w_macd_cross", "w_vol_up", "w_rsi_rebound", "w_green_candle",
                "bbands_lower_bias", "rsi_oversold_extreme", "ai_prob_threshold", "position_size"
            ):
                val = getattr(strategy_data, name, None)
                if val is not None:
                    kwargs[name] = val

        return cls(**kwargs)


PARAM_SPACE: dict[str, tuple[float, float, float]] = {
    # (min, max, step)
    "ma_short": (3, 10, 1),
    "ma_long": (10, 60, 5),
    "macd_fast": (8, 16, 2),
    "macd_slow": (20, 34, 2),
    "macd_signal": (5, 13, 2),
    "rsi_length": (10, 20, 2),
    "rsi_buy_threshold": (25, 50, 5),
    "rsi_sell_threshold": (60, 85, 5),
    "bbands_length": (14, 30, 2),
    "bbands_std": (1.5, 3.0, 0.25),
    "atr_length": (10, 20, 2),
    "atr_multiplier": (1.0, 3.0, 0.25),
    "weight_trend": (0.1, 0.7, 0.1),
    "weight_reversion": (0.1, 0.6, 0.1),
    "weight_volume": (0.1, 0.5, 0.1),
    
    # Nonlinear Weights
    "w_pullback_ma": (0.5, 5.0, 0.5),
    "w_macd_cross": (0.5, 3.0, 0.5),
    "w_vol_up": (0.5, 3.0, 0.5),
    "w_rsi_rebound": (0.5, 5.0, 0.5),
    "w_green_candle": (0.5, 3.0, 0.5),

    "vol_up_ratio": (1.1, 1.8, 0.05),
    "rsi_cooled_max": (45, 65, 5),
    "pullback_ma_tolerance": (1.00, 1.05, 0.005),
    "negative_bias_pct": (0.80, 0.98, 0.01),
    "rsi_oversold": (25, 45, 5),
    "bbands_lower_bias": (0.90, 1.10, 0.01),
    "rsi_oversold_extreme": (10, 30, 2),
    "trail_atr_mult": (1.0, 2.5, 0.1),
    "take_profit_pct": (0.03, 0.12, 0.01),
    "breakeven_trigger": (0.02, 0.06, 0.005),
    "breakeven_buffer": (1.002, 1.010, 0.001),
    
    # AI ML Gate
    "ai_prob_threshold": (0.1, 0.5, 0.05),
}
