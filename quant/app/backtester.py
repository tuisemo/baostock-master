from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from tqdm import tqdm

from quant.features.analyzer import calculate_indicators, classify_market_state_enhanced as classify_market_state
from quant.core.adaptive_strategy import get_dynamic_params_v10 as get_dynamic_params
from quant.infra.config import CONF
from quant.infra.logger import logger


# ===== Slippage Model (Milestone 5.2) =====
class SlippageModel:
    """
    Realistic slippage model for backtesting

    Features:
    1. Market cap tier-based base slippage
    2. Volume impact cost
    3. Liquidity-based position limits
    """

    # Base slippage by market cap tier
    BASE_SLIPPAGE = {
        'large': 0.0005,   # Large cap: 0.05%
        'mid': 0.001,      # Mid cap: 0.1%
        'small': 0.002,    # Small cap: 0.2%
        'micro': 0.005     # Micro cap: 0.5%
    }

    # Volume impact parameters
    VOLUME_IMPACT_FACTOR = 0.0001  # Impact factor per unit of volume ratio
    MAX_VOLUME_IMPACT = 0.002      # Max 0.2% additional slippage from volume

    # Market cap thresholds (in billions CNY)
    MARKET_CAP_THRESHOLDS = {
        'large': 100,   # > 100B
        'mid': 30,      # 30B - 100B
        'small': 5,     # 5B - 30B
        'micro': 0      # < 5B
    }

    def __init__(self, config: dict = None):
        """
        Initialize slippage model

        Args:
            config: Configuration dict with optional overrides:
                - base_slippage: dict of tier -> slippage
                - volume_impact_factor: float
                - max_volume_impact: float
        """
        if config:
            self.base_slippage = config.get('base_slippage', self.BASE_SLIPPAGE.copy())
            self.volume_impact_factor = config.get('volume_impact_factor', self.VOLUME_IMPACT_FACTOR)
            self.max_volume_impact = config.get('max_volume_impact', self.MAX_VOLUME_IMPACT)
        else:
            self.base_slippage = self.BASE_SLIPPAGE.copy()
            self.volume_impact_factor = self.VOLUME_IMPACT_FACTOR
            self.max_volume_impact = self.MAX_VOLUME_IMPACT

    def get_market_cap_tier(self, market_cap_billion: float) -> str:
        """Determine market cap tier

        Args:
            market_cap_billion: Market cap in billions CNY

        Returns:
            Tier name ('large', 'mid', 'small', 'micro')
        """
        if market_cap_billion >= self.MARKET_CAP_THRESHOLDS['large']:
            return 'large'
        elif market_cap_billion >= self.MARKET_CAP_THRESHOLDS['mid']:
            return 'mid'
        elif market_cap_billion >= self.MARKET_CAP_THRESHOLDS['small']:
            return 'small'
        return 'micro'

    def get_slippage(
        self,
        volume: float,
        avg_volume: float,
        market_cap_tier: str,
        trade_value: float = None
    ) -> float:
        """Calculate realistic slippage

        Args:
            volume: Current trading volume
            avg_volume: Average trading volume (e.g., 20-day average)
            market_cap_tier: Market cap tier ('large', 'mid', 'small', 'micro')
            trade_value: Value of the trade (optional, for market impact)

        Returns:
            Slippage as decimal (e.g., 0.001 = 0.1%)
        """
        # Base slippage by market cap tier
        base = self.base_slippage.get(market_cap_tier, self.base_slippage['small'])

        # Volume impact: higher volume relative to average = more slippage
        volume_ratio = volume / max(avg_volume, 1)
        volume_impact = min(self.max_volume_impact, volume_ratio * self.volume_impact_factor)

        # Market impact from trade size (if provided)
        market_impact = 0.0
        if trade_value and avg_volume > 0:
            # Estimate price from average volume (simplified)
            trade_ratio = trade_value / (avg_volume * 10)  # Assume avg price ~10
            market_impact = min(self.max_volume_impact, trade_ratio * self.volume_impact_factor * 0.5)

        total_slippage = base + volume_impact + market_impact
        logger.debug(f"Slippage: base={base:.4f}, volume_impact={volume_impact:.4f}, "
                    f"market_impact={market_impact:.4f}, total={total_slippage:.4f}")
        return total_slippage

    def get_liquidity_based_limit(
        self,
        avg_volume: float,
        market_cap_tier: str,
        max_participation_pct: float = 0.1
    ) -> float:
        """Calculate liquidity-based position limit

        Args:
            avg_volume: Average daily volume
            market_cap_tier: Market cap tier
            max_participation_pct: Max percentage of daily volume to trade

        Returns:
            Maximum position value
        """
        # Tier-based liquidity adjustment
        liquidity_multiplier = {
            'large': 1.0,
            'mid': 0.8,
            'small': 0.5,
            'micro': 0.2
        }.get(market_cap_tier, 0.5)

        # Assume avg price ~10 CNY for estimation
        avg_price = 10.0
        max_position = avg_volume * avg_price * max_participation_pct * liquidity_multiplier

        return max_position

    def apply_slippage_to_price(
        self,
        price: float,
        is_buy: bool,
        slippage: float
    ) -> float:
        """Apply slippage to price

        Args:
            price: Original price
            is_buy: True for buy, False for sell
            slippage: Slippage value

        Returns:
            Adjusted price
        """
        if is_buy:
            # Buy at higher price
            return price * (1 + slippage)
        else:
            # Sell at lower price
            return price * (1 - slippage)


# Global slippage model instance
_slippage_model: SlippageModel = None


def get_slippage_model(config: dict = None) -> SlippageModel:
    """Get global slippage model instance"""
    global _slippage_model
    if _slippage_model is None:
        _slippage_model = SlippageModel(config)
    return _slippage_model

# ===== AI Model (Phase 8) =====
_AI_MODEL = None
_AI_MODEL_PATH = "models/alpha_lgbm.txt"
_AI_MODEL_LOAD_ATTEMPTED = False
_ENSEMBLE_MODEL = None
_ENSEMBLE_MODEL_PATH = "models/ensemble_v1"
_ENSEMBLE_MODEL_LOAD_ATTEMPTED = False

# Tiered confidence factor thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.65
CONFIDENCE_MEDIUM_THRESHOLD = 0.45
CONFIDENCE_LOW_THRESHOLD = 0.30

def _get_ai_model():
    """Lazy-load the LightGBM model singleton. Returns None if model is unavailable or corrupted."""
    global _AI_MODEL, _AI_MODEL_LOAD_ATTEMPTED
    if _AI_MODEL_LOAD_ATTEMPTED:
        return _AI_MODEL  # 已尝试过加载（成功返回模型，失败返回 None）
    _AI_MODEL_LOAD_ATTEMPTED = True

    import os
    import warnings
    if not os.path.exists(_AI_MODEL_PATH):
        logger.debug(f"AI 模型文件不存在: {_AI_MODEL_PATH}，将使用纯规则引擎。")
        return None

    try:
        import lightgbm as lgb
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 添加显式参数以确保正确加载 (通过params字典)
            _AI_MODEL = lgb.Booster(model_file=_AI_MODEL_PATH, params={"num_threads": 1})
        # 验证模型是否可用
        _AI_MODEL.num_feature()
        logger.info(f"AI 模型已加载: {_AI_MODEL_PATH} ({_AI_MODEL.num_feature()} features)")
    except Exception as e:
        logger.warning(f"AI 模型文件损坏或不兼容，将使用纯规则引擎: {e}")
        _AI_MODEL = None

    return _AI_MODEL

def _get_ensemble_model():
    """
    Lazy-load the Ensemble model (LightGBM+XGBoost+CatBoost).
    Supports both MultiModelEnsemble and legacy EnsembleLGBM.
    """
    global _ENSEMBLE_MODEL, _ENSEMBLE_MODEL_LOAD_ATTEMPTED
    if _ENSEMBLE_MODEL_LOAD_ATTEMPTED:
        return _ENSEMBLE_MODEL
    _ENSEMBLE_MODEL_LOAD_ATTEMPTED = True
    import os
    if not os.path.exists(_ENSEMBLE_MODEL_PATH):
        logger.debug(f"Ensemble not found: {_ENSEMBLE_MODEL_PATH}")
        return None
    
    # Check if this is a MultiModelEnsemble or legacy EnsembleLGBM
    ensemble_metadata_path = os.path.join(_ENSEMBLE_MODEL_PATH, 'ensemble_metadata.json')
    
    try:
        if os.path.exists(ensemble_metadata_path):
            # Try loading as MultiModelEnsemble first
            try:
                from quant.core.ensemble_trainer import MultiModelEnsemble
                _ENSEMBLE_MODEL = MultiModelEnsemble.load(_ENSEMBLE_MODEL_PATH)
                logger.info(f"MultiModelEnsemble loaded: {_ENSEMBLE_MODEL_PATH} "
                           f"(models: {_ENSEMBLE_MODEL.models})")
            except Exception as e1:
                # Fall back to EnsembleLGBM
                logger.debug(f"MultiModelEnsemble load failed: {e1}, trying EnsembleLGBM")
                from quant.core.trainer import EnsembleLGBM
                _ENSEMBLE_MODEL = EnsembleLGBM.load(_ENSEMBLE_MODEL_PATH)
                logger.info(f"EnsembleLGBM loaded: {_ENSEMBLE_MODEL_PATH}")
        else:
            # Legacy path - try EnsembleLGBM
            from quant.core.trainer import EnsembleLGBM
            _ENSEMBLE_MODEL = EnsembleLGBM.load(_ENSEMBLE_MODEL_PATH)
            logger.info(f"EnsembleLGBM loaded: {_ENSEMBLE_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"Ensemble load failed: {e}")
        _ENSEMBLE_MODEL = None
    
    return _ENSEMBLE_MODEL


def get_tiered_confidence_factor(
    ai_confidence: float,
    ensemble_disagreement: float = None,
    use_ensemble: bool = False
) -> tuple[float, str]:
    """
    计算分层置信度因子和交易档位
    
    Args:
        ai_confidence: AI模型置信度 (0-1)
        ensemble_disagreement: 集成模型分歧度 (0-1, optional)
        use_ensemble: 是否使用集成模型
        
    Returns:
        tuple: (confidence_factor, tier)
    """
    # 基础置信度分层
    if ai_confidence >= CONFIDENCE_HIGH_THRESHOLD:
        base_factor = 1.0
        tier = "high"
    elif ai_confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        base_factor = 0.6
        tier = "medium"
    elif ai_confidence >= CONFIDENCE_LOW_THRESHOLD:
        base_factor = 0.3
        tier = "low"
    else:
        return 0.0, "block"  # 低于最低阈值，阻止交易
    
    # 集成模型分歧惩罚
    if use_ensemble and ensemble_disagreement is not None:
        # 分歧度 > 0.2 开始惩罚，> 0.4 严重惩罚
        if ensemble_disagreement > 0.4:
            base_factor *= 0.5  # 严重惩罚
            tier = f"{tier}_high_disagreement"
        elif ensemble_disagreement > 0.2:
            base_factor *= 0.8  # 轻度惩罚
            tier = f"{tier}_disagreement"
    
    return base_factor, tier


def get_ensemble_prediction_and_disagreement(X: pd.DataFrame) -> tuple[float, float]:
    """
    获取集成模型预测结果和模型间分歧度
    
    Args:
        X: 特征DataFrame
        
    Returns:
        tuple: (ensemble_probability, disagreement_score)
    """
    ensemble = _get_ensemble_model()
    if ensemble is None:
        return 0.5, 0.0  # 无集成模型时返回中性值
    
    try:
        # 检查模型类型
        if hasattr(ensemble, 'trained_models'):
            # MultiModelEnsemble
            predictions = []
            for model_name in ensemble.models:
                model = ensemble.trained_models.get(model_name)
                if model is None:
                    continue
                
                # 根据模型类型获取预测
                if model_name == 'lgb':
                    pred = model.predict(X)
                elif model_name == 'xgb':
                    import xgboost as xgb
                    dtest = xgb.DMatrix(X)
                    pred = model.predict(dtest)
                elif model_name == 'cat':
                    pred = model.predict_proba(X)[:, 1]
                else:
                    continue
                
                predictions.append(pred)
            
            if not predictions:
                return 0.5, 0.0
            
            # 计算平均预测
            predictions = np.array(predictions)
            ensemble_proba = np.mean(predictions, axis=0)
            
            # 计算分歧度 (标准差)
            disagreement = np.std(predictions, axis=0)
            
            return float(ensemble_proba[0]) if len(ensemble_proba) > 0 else 0.5, float(disagreement[0])
        
        elif hasattr(ensemble, 'models'):
            # EnsembleLGBM
            predictions = []
            for model in ensemble.models:
                pred = model.predict(X, num_iteration=model.best_iteration)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            ensemble_proba = np.mean(predictions, axis=0)
            disagreement = np.std(predictions, axis=0)
            
            return float(ensemble_proba[0]) if len(ensemble_proba) > 0 else 0.5, float(disagreement[0])
        
        else:
            return 0.5, 0.0
            
    except Exception as e:
        logger.warning(f"Ensemble prediction failed: {e}")
        return 0.5, 0.0

if TYPE_CHECKING:
    from quant.core.strategy_params import StrategyParams


def _build_column_names(p: StrategyParams) -> dict[str, str]:
    return {
        "sma_s": f"SMA_{p.ma_short}",
        "sma_l": f"SMA_{p.ma_long}",
        "macd_h": f"MACDh_{p.macd_fast}_{p.macd_slow}_{p.macd_signal}",
        "rsi": f"RSI_{p.rsi_length}",
        "bb_lower": f"BBL_{p.bbands_length}_{p.bbands_std}",
        "bb_upper": f"BBU_{p.bbands_length}_{p.bbands_std}",
        "obv": "OBV",
        "atr": f"ATRr_{p.atr_length}",
    }


_MARKET_INDEX_CACHE = None

def get_market_index() -> pd.DataFrame | None:
    global _MARKET_INDEX_CACHE
    if _MARKET_INDEX_CACHE is not None:
        return _MARKET_INDEX_CACHE

    file_path = os.path.join(CONF.history_data.data_dir, "sh.000001.csv")
    if not os.path.exists(file_path):
        return None

    df_idx = pd.read_csv(file_path)
    if df_idx.empty or "date" not in df_idx.columns:
        return None

    df_idx.rename(columns={"date": "Date"}, inplace=True)
    df_idx["Date"] = pd.to_datetime(df_idx["Date"])
    df_idx.set_index("Date", inplace=True)
    df_idx.sort_index(inplace=True)
    
    close_col = "close" if "close" in df_idx.columns else "Close"
    close_s = pd.to_numeric(df_idx[close_col], errors="coerce").astype(float)

    # Base trend context
    ma20 = close_s.rolling(window=20).mean()
    ma60 = close_s.rolling(window=60).mean()
    df_idx["MA20"] = ma20
    df_idx["MA60"] = ma60
    df_idx["market_uptrend"] = close_s > ma20

    # Precompute a per-date market state to avoid look-ahead bias from "latest-only" classification.
    try:
        from quant.core.market_classifier import default_thresholds
        thr = default_thresholds()
    except Exception:
        thr = {
            "trend_strength_bull": 0.02,
            "trend_strength_bear": -0.02,
            "volatility_high": 0.025,
            "volatility_low": 0.015,
            "roc_strong": 0.05,
            "volume_ratio_high": 1.5,
        }

    trend_strength = (ma20 - ma60) / ma60.replace(0, np.nan)
    returns = close_s.pct_change()
    volatility = returns.rolling(window=20).std()
    roc_20 = close_s.pct_change(20)

    # Momentum acceleration proxy (matches classifier's 60d window logic)
    mom5 = close_s.pct_change(5)
    mom5_short = mom5.rolling(window=10).mean()
    mom5_long = mom5.shift(50).rolling(window=10).mean()
    mom_acc = mom5_short - mom5_long

    vol_col = "volume" if "volume" in df_idx.columns else ("Volume" if "Volume" in df_idx.columns else None)
    if vol_col is not None:
        vol_s = pd.to_numeric(df_idx[vol_col], errors="coerce").astype(float)
        vol_ma20 = vol_s.rolling(window=20).mean()
        volume_ratio = vol_s / vol_ma20.replace(0, np.nan)
    else:
        volume_ratio = pd.Series(1.0, index=df_idx.index, dtype=float)

    state = pd.Series("sideways_low_vol", index=df_idx.index, dtype=object)

    bull = trend_strength > float(thr.get("trend_strength_bull", 0.02))
    bear = trend_strength < float(thr.get("trend_strength_bear", -0.02))
    sideways = ~(bull | bear)

    sideways_high_vol = sideways & (volatility > float(thr.get("volatility_high", 0.025)))
    state[sideways_high_vol] = "sideways_high_vol"
    state[sideways & ~sideways_high_vol] = "sideways_low_vol"

    strong_bull_cond = bull & (volatility < float(thr.get("volatility_low", 0.015))) & (roc_20 > float(thr.get("roc_strong", 0.05)))
    bull_momentum = strong_bull_cond & (mom_acc > 0.001) & (volume_ratio > float(thr.get("volume_ratio_high", 1.5)))
    bull_volume = strong_bull_cond & ~bull_momentum & (volume_ratio > float(thr.get("volume_ratio_high", 1.5)))
    strong_bull = strong_bull_cond & ~bull_momentum & ~bull_volume
    weak_bull = bull & ~strong_bull_cond

    state[weak_bull] = "weak_bull"
    state[strong_bull] = "strong_bull"
    state[bull_volume] = "bull_volume"
    state[bull_momentum] = "bull_momentum"

    strong_bear_cond = bear & ((trend_strength < float(thr.get("trend_strength_bear", -0.02))) | ((trend_strength < -0.01) & (volatility > float(thr.get("volatility_high", 0.025)))))
    bear_panic = strong_bear_cond & (volume_ratio > float(thr.get("volume_ratio_high", 1.5))) & (volatility > 0.03)
    bear_momentum = strong_bear_cond & ~bear_panic & (mom_acc < -0.001)
    strong_bear = strong_bear_cond & ~bear_panic & ~bear_momentum
    weak_bear = bear & ~strong_bear_cond

    state[weak_bear] = "weak_bear"
    state[strong_bear] = "strong_bear"
    state[bear_momentum] = "bear_momentum"
    state[bear_panic] = "bear_panic"

    df_idx["market_state"] = state
    df_idx["market_volatility"] = volatility
    
    _MARKET_INDEX_CACHE = df_idx
    return _MARKET_INDEX_CACHE

def _resolve_params(params: StrategyParams | None) -> StrategyParams:
    from quant.core.strategy_params import StrategyParams as SP

    if params is not None:
        return params
    return SP.from_app_config(CONF)


def evaluate_buy_signals(
    price: float,
    open_p: float,
    low_p: float,
    sma_l_1: float,
    sma_l_3: float | None,
    sma_s_1: float,
    macd_h_1: float,
    macd_h_2: float,
    rsi_1: float,
    bb_lower_1: float,
    vol_1: float,
    vol_2: float,
    has_vol_slope: bool,
    vol_slope_1: float,
    has_mom_div: bool,
    mom_div_1: float,
    market_uptrend: bool,
    p: StrategyParams,
    weekly_data: dict = None,
) -> tuple[bool, bool, bool, dict]:
    """
    Enhanced buy signal evaluation with multi-timeframe support and detailed scoring.
    
    Args:
        weekly_data: Optional weekly timeframe data for confirmation
        
    Returns:
        Tuple of (signal_pullback, signal_rebound, signal_trend_breakout, signal_details)
    """
    # Optional multi-timeframe controls from config.yaml
    mt_cfg = getattr(getattr(CONF, "strategy", None), "multi_timeframe", {}) or {}
    mt_enabled = bool(mt_cfg.get("enabled", True)) if isinstance(mt_cfg, dict) else True
    weekly_enabled = bool(mt_cfg.get("weekly_confirmation", True)) if isinstance(mt_cfg, dict) else True
    alignment_threshold = float(mt_cfg.get("alignment_threshold", 0.0)) if isinstance(mt_cfg, dict) else 0.0

    macd_golden_cross = macd_h_2 <= 0 and macd_h_1 > 0
    is_green_candle = price > open_p

    if has_vol_slope and vol_slope_1 > 0.1:
        vol_up = True
    else:
        vol_up = vol_1 > vol_2 * p.vol_up_ratio

    mom_ok = True
    if has_mom_div and mom_div_1 <= -0.02:
        mom_ok = False

    # === Multi-Timeframe Confirmation ===
    timeframe_alignment_score = 1.0  # Base score
    if mt_enabled and weekly_enabled and weekly_data is not None:
        weekly_uptrend = weekly_data.get('uptrend', True)
        weekly_rsi = weekly_data.get('rsi', 50)
        weekly_macd_positive = weekly_data.get('macd_positive', True)
        
        # Check alignment between daily and weekly
        if market_uptrend == weekly_uptrend:
            timeframe_alignment_score += 0.3
        if weekly_rsi > 50:
            timeframe_alignment_score += 0.2
        if weekly_macd_positive:
            timeframe_alignment_score += 0.2
        
        # Penalize if weekly is strongly bearish
        if not weekly_uptrend and weekly_rsi < 40:
            timeframe_alignment_score -= 0.5

    timeframe_block = False
    if mt_enabled and weekly_enabled and weekly_data is not None and alignment_threshold > 0:
        timeframe_block = timeframe_alignment_score < alignment_threshold

    # === 左侧交易评估 ===
    is_bb_dip = price < bb_lower_1 * p.bbands_lower_bias
    is_rsi_dip = rsi_1 < p.rsi_oversold_extreme
    
    signal_pullback = False
    signal_rebound = False
    
    # Component scores for signal fusion
    trend_score = 0.0
    reversion_score = 0.0
    volume_score = 0.0
    pattern_score = 0.0
    
    if is_bb_dip or is_rsi_dip:
        # 止跌形态验证（收阳线或长下影线），作为强烈加分项
        lower_shadow = min(open_p, price) - low_p
        body = abs(price - open_p)
        has_bottoming_sign = is_green_candle or (lower_shadow > body * 1.5 and lower_shadow > 0)
            
        # 左侧接飞刀算分系统
        score = 0.0
        if has_bottoming_sign: 
            score += 1.0       # 有止跌形态直接+1分
            pattern_score += 0.8
        if is_bb_dip: 
            score += p.w_pullback_ma    # 复用回调权重为布林下轨突刺分
            reversion_score += 0.7
        if is_rsi_dip: 
            score += p.w_rsi_rebound   # 复用超卖权重为极度恐慌分
            reversion_score += 0.6
        if is_green_candle: 
            score += p.w_green_candle
            pattern_score += 0.5
        if vol_up: 
            score += p.w_vol_up
            volume_score += 0.8
        if macd_golden_cross or macd_h_1 > macd_h_2: 
            score += p.w_macd_cross
            trend_score += 0.6
        if mom_ok: 
            score += 1.0
            trend_score += 0.4
        
        # 大盘熊市时，左侧入局门槛提高 - 优化后门槛适中
        pass_threshold = 1.2 if market_uptrend else 3.0
        signal_pullback = (score >= pass_threshold) and is_bb_dip
        signal_rebound = (score >= pass_threshold) and is_rsi_dip

    # === 右侧交易评估 (牛市专属) ===
    signal_trend_breakout = False
    if market_uptrend:
        is_above_ma = price > sma_s_1
        # Balanced RSI range for better signal quality
        is_rsi_health = 42 < rsi_1 < 72
        # Added volume confirmation requirement
        if macd_golden_cross and is_above_ma and vol_up and is_rsi_health and is_green_candle:
            signal_trend_breakout = True
            trend_score = max(trend_score, 0.8)
            volume_score = max(volume_score, 0.7)
            pattern_score = max(pattern_score, 0.6)

    # === Signal Fusion Enhancement ===
    # Weighted voting for signal strength
    weights_cfg = getattr(getattr(CONF, "strategy", None), "signal_fusion_weights", {}) or {}
    weights = weights_cfg if isinstance(weights_cfg, dict) and weights_cfg else {
        "trend": 0.3,
        "reversion": 0.3,
        "volume": 0.2,
        "pattern": 0.2,
    }
    # Fill missing keys + normalize to sum to 1
    for k in ("trend", "reversion", "volume", "pattern"):
        weights.setdefault(k, 0.0)
    w_sum = float(sum(weights.values()))
    if w_sum > 0:
        weights = {k: float(v) / w_sum for k, v in weights.items()}
    else:
        weights = {"trend": 0.3, "reversion": 0.3, "volume": 0.2, "pattern": 0.2}
    composite_score = (
        trend_score * weights['trend'] +
        reversion_score * weights['reversion'] +
        volume_score * weights['volume'] +
        pattern_score * weights['pattern']
    ) * timeframe_alignment_score

    # Signal strength categorization - higher thresholds for better quality
    if composite_score >= 0.75:
        signal_strength = 'strong'
    elif composite_score >= 0.50:
        signal_strength = 'medium'
    else:
        signal_strength = 'weak'

    # If weekly alignment is explicitly required and fails, block the signal entirely.
    if timeframe_block:
        signal_pullback = False
        signal_rebound = False
        signal_trend_breakout = False
        signal_strength = "weak"

    signal_details = {
        'trend_score': trend_score,
        'reversion_score': reversion_score,
        'volume_score': volume_score,
        'pattern_score': pattern_score,
        'composite_score': composite_score,
        'timeframe_alignment': timeframe_alignment_score,
        'timeframe_block': timeframe_block,
        'alignment_threshold': alignment_threshold,
        'fusion_weights': weights,
        'signal_strength': signal_strength,
    }

    return signal_pullback, signal_rebound, signal_trend_breakout, signal_details


def get_weekly_confirmation(code: str, current_date: str, data_dir: str = 'data') -> dict | None:
    """
    Get weekly timeframe confirmation data for a stock.
    
    Args:
        code: Stock code
        current_date: Current date string
        data_dir: Data directory path
        
    Returns:
        Dictionary with weekly trend data or None
    """
    try:
        import pandas as pd
        import os
        
        file_path = os.path.join(data_dir, f"{code}.csv")
        if not os.path.exists(file_path):
            return None
            
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 50:
            return None
        
        # Convert date column
        date_col = 'date' if 'date' in df.columns else 'Date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Get current date
        current = pd.to_datetime(current_date)
        
        # Filter to data up to current date
        df = df[df[date_col] <= current]
        
        if len(df) < 20:
            return None
        
        # Resample to weekly
        df.set_index(date_col, inplace=True)
        weekly = df.resample('W').last().dropna()
        
        if len(weekly) < 5:
            return None
        
        # Calculate weekly indicators
        close = weekly['close'].values
        weekly_sma20 = weekly['close'].rolling(20).mean().iloc[-1]
        weekly_sma5 = weekly['close'].rolling(5).mean().iloc[-1]
        
        # Weekly RSI
        delta = weekly['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        weekly_rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Weekly MACD
        ema12 = weekly['close'].ewm(span=12).mean()
        ema26 = weekly['close'].ewm(span=26).mean()
        weekly_macd = (ema12 - ema26).iloc[-1]
        
        return {
            'uptrend': weekly_sma5 > weekly_sma20 if not pd.isna(weekly_sma20) else True,
            'rsi': weekly_rsi if not pd.isna(weekly_rsi) else 50,
            'macd_positive': weekly_macd > 0 if not pd.isna(weekly_macd) else True,
            'trend_strength': abs(weekly_sma5 / weekly_sma20 - 1) if weekly_sma20 > 0 else 0,
        }
    except Exception as e:
        logger.debug(f"Weekly confirmation failed for {code}: {e}")
        return None


def get_dynamic_ai_threshold(
    market_state: str,
    base_threshold: float = 0.35,
    volatility_regime: str = 'normal'
) -> float:
    """
    Get dynamic AI threshold based on market state.
    
    Args:
        market_state: Current market state
        base_threshold: Base AI threshold
        volatility_regime: Volatility regime ('low', 'normal', 'high')
        
    Returns:
        Dynamic AI threshold
    """
    from quant.core.adaptive_strategy import get_market_thresholds
    
    thresholds = get_market_thresholds(market_state)
    state_threshold = thresholds.get('ai_threshold', base_threshold)
    
    # Volatility adjustment
    if volatility_regime == 'high':
        # Higher threshold in high volatility (more selective)
        adjusted = state_threshold * 1.1
    elif volatility_regime == 'low':
        # Lower threshold in low volatility (more permissive)
        adjusted = state_threshold * 0.95
    else:
        adjusted = state_threshold
    
    # Clamp to valid range
    return max(0.15, min(0.70, adjusted))


def create_strategy(params: StrategyParams) -> type[Strategy]:
    cols = _build_column_names(params)

    class _Strategy(Strategy):
        _p = params
        _base_p = params  # Keep original params for reference
        _cols = cols
        _market_state = "sideways"  # Default
        _dynamic_p = params  # Dynamic params based on market state

        def init(self):
            p = self._p
            c = self._cols
            def _get_col(col_name):
                # Use self.data instead of self.data.df for backtesting library compatibility
                return self.data.df[col_name].to_numpy() if col_name in self.data.df.columns else self.data.df["Close"].to_numpy()

            self.sma_s = self.I(_get_col, c["sma_s"], name="sma_s")
            self.sma_l = self.I(_get_col, c["sma_l"], name="sma_l")
            self.macd_h = self.I(_get_col, c["macd_h"], name="macd_h")
            self.rsi = self.I(_get_col, c["rsi"], name="rsi")
            self.bb_lower = self.I(_get_col, c["bb_lower"], name="bbl")
            self.bb_upper = self.I(_get_col, c["bb_upper"], name="bbu")
            self.obv = self.I(_get_col, c["obv"], name="obv")
            self.atr = self.I(_get_col, c["atr"], name="atr")

            self.current_stop_loss = 0.0
            self.current_trade_type = ""
            self._has_vol_slope = "vol_slope" in self.data.df.columns
            self._has_mom_div = "momentum_divergence" in self.data.df.columns
            if self._has_vol_slope:
                self.vol_slope = self.I(_get_col, "vol_slope", name="vol_slope")
            if self._has_mom_div:
                self.mom_div = self.I(_get_col, "momentum_divergence", name="mom_div")

            # Trading cooldown tracking
            self.last_trade_bar = -999  # Track last trade bar for cooldown
            self.last_exit_bar = -999   # Track last exit bar for cooldown
            self.min_hold_days = getattr(CONF.strategy, 'min_hold_days', 3)  # Minimum hold days
            self.signal_cooldown_days = getattr(CONF.strategy, 'signal_cooldown_days', 5)  # Signal cooldown
            self.current_entry_bar = -999  # Track entry bar of current position

            # Initialize market state
            # Cache index data for per-bar market context (avoids look-ahead bias)
            self._idx_df = get_market_index()
            self._market_state = "sideways_low_vol"
            self._dynamic_p = self._base_p

        def next(self):
            if pd.isna(self.sma_s[-1]) or pd.isna(self.rsi[-1]) or pd.isna(self.atr[-1]):
                return

            price = self.data.Close[-1]
            current_bar = len(self.data.Close) - 1

            # Update market context for the current bar (no look-ahead).
            current_date = self.data.index[-1]
            market_uptrend = True  # Default if index is unavailable
            new_market_state = getattr(self, "_market_state", "sideways_low_vol")

            idx_df = getattr(self, "_idx_df", None)
            if idx_df is None:
                idx_df = get_market_index()
                self._idx_df = idx_df

            if idx_df is not None:
                idx_loc = idx_df.index.get_indexer([current_date], method="pad")[0]
                if idx_loc != -1:
                    market_uptrend = bool(idx_df.iloc[idx_loc].get("market_uptrend", True))
                    new_market_state = str(idx_df.iloc[idx_loc].get("market_state", new_market_state))

            if new_market_state != getattr(self, "_market_state", "sideways_low_vol"):
                self._market_state = new_market_state
                self._dynamic_p = get_dynamic_params(self._base_p, self._market_state)

            p = self._dynamic_p  # Use dynamic params

            if self.position:
                # Minimum hold days check
                if hasattr(self, 'current_entry_bar') and self.current_entry_bar > 0:
                    days_held = current_bar - self.current_entry_bar
                    if days_held < self.min_hold_days:
                        # Skip exit checks during minimum hold period (except stop loss)
                        pass  # Continue to regular exit checks below
                trail_stop = price - p.trail_atr_mult * self.atr[-1]
                if trail_stop > self.current_stop_loss:
                    self.current_stop_loss = trail_stop

                if price <= self.current_stop_loss:
                    self.last_exit_bar = current_bar
                    self.position.close()
                    return

                # Time-Decay Exit (If held too long with minimal return)
                days_held = len(self.data) - self.trades[0].entry_bar
                if days_held >= p.max_hold_days and self.position.pl_pct < p.max_hold_min_return:
                    self.last_exit_bar = current_bar
                    self.position.close()
                    return

                # Differentiated Take Profit Logic
                tp_pct = p.take_profit_pct_right if self.current_trade_type == "right" else p.take_profit_pct_left
                if self.position.pl_pct >= tp_pct:
                    # 利润达到初步目标后，不立刻清仓，而是收紧移动止损，放飞利润
                    if self.current_trade_type == "right":
                        tight_stop = price - 1.5 * self.atr[-1]
                    else:
                        tight_stop = price - 0.8 * self.atr[-1]
                        
                    if tight_stop > self.current_stop_loss:
                        self.current_stop_loss = tight_stop

                if self.position.pl_pct >= 0.02:
                    # 提早激活移动保护
                    if self.current_trade_type == "right":
                        protect_stop = price - 1.5 * self.atr[-1]
                    else:
                        protect_stop = price - 1.0 * self.atr[-1]
                        
                    if protect_stop > self.current_stop_loss:
                        self.current_stop_loss = protect_stop

                if self.position.pl_pct >= p.breakeven_trigger and len(self.trades) > 0:
                    breakeven_stop = self.trades[0].entry_price * p.breakeven_buffer
                    if self.current_stop_loss < breakeven_stop:
                        self.current_stop_loss = breakeven_stop

                # Enhanced Exit Signals - Active Take-Profit Mechanisms
                
                # 1. RSI Overheating (existing, but with dynamic thresholds)
                rsi_limit = p.rsi_overbought_right if self.current_trade_type == "right" else p.rsi_overbought_left
                if self.rsi[-1] >= rsi_limit:
                    self.last_exit_bar = current_bar
                    self.position.close()
                    return
                
                # 2. Volume-Price Divergence (volume shrinks while price stalls)
                if hasattr(self, 'vol_slope') and len(self.data) >= 5:
                    recent_slope = np.mean(self.vol_slope[-5:])
                    recent_price_change = (self.data.Close[-1] - self.data.Close[-5]) / self.data.Close[-5]
                    if recent_slope < -0.05 and 0 < recent_price_change < 0.02:
                        self.last_exit_bar = current_bar
                        self.position.close()
                        return
                
                # 3. MACD Top Divergence
                if len(self.macd_h) >= 10:
                    recent_macd = self.macd_h[-5:].tolist()
                    recent_highs = self.data.High[-5:].tolist()
                    if (recent_macd[0] > recent_macd[-1] and
                        recent_highs[0] > recent_highs[-1]):
                        self.last_exit_bar = current_bar
                        self.position.close()
                        return

                # 4. Timeout with Minimal Return (Time-based exit)
                if days_held >= p.max_hold_days and self.position.pl_pct < p.max_hold_min_return:
                    self.last_exit_bar = current_bar
                    self.position.close()
                    return
                
                # 5. Consecutive Small Losses Warning (reduce position on next trade)
                if len(self.trades) >= 3:
                    last_3_trades = self.trades[-3:]
                    consecutive_losses = all(t.pl_pct < 0 for t in last_3_trades)
                    if consecutive_losses:
                        # Signal to reduce exposure (handled in position sizing)
                        pass
                
                return

            sma_l_3 = self.sma_l[-3] if len(self.sma_l) >= 3 else None
            vol_slope_1 = self.vol_slope[-1] if self._has_vol_slope else 0.0
            mom_div_1 = self.mom_div[-1] if self._has_mom_div else 0.0

            # ===== 大盘情绪过滤 (Market Regime Filter) =====
            # market_uptrend is already computed for this bar above (no look-ahead).
            # if not market_uptrend:
            #     return
            # ===============================================

            # Get weekly timeframe confirmation (if available)
            weekly_data = None
            try:
                from quant.infra.config import CONF
                current_date_str = str(self.data.index[-1])
                weekly_data = get_weekly_confirmation(
                    getattr(self, '_code', ''),
                    current_date_str,
                    CONF.history_data.data_dir
                )
            except Exception:
                weekly_data = None
            
            signal_pullback, signal_rebound, signal_trend_breakout, signal_details = evaluate_buy_signals(
                price=price,
                open_p=self.data.Open[-1],
                low_p=self.data.Low[-1],
                sma_l_1=self.sma_l[-1],
                sma_l_3=sma_l_3,
                sma_s_1=self.sma_s[-1],
                macd_h_1=self.macd_h[-1],
                macd_h_2=self.macd_h[-2],
                rsi_1=self.rsi[-1],
                bb_lower_1=self.bb_lower[-1],
                vol_1=self.data.Volume[-1],
                vol_2=self.data.Volume[-2],
                has_vol_slope=self._has_vol_slope,
                vol_slope_1=vol_slope_1,
                has_mom_div=self._has_mom_div,
                mom_div_1=mom_div_1,
                market_uptrend=market_uptrend,
                p=p,
                weekly_data=weekly_data,
            )

            # --- Enhanced Signal Scoring with Fusion ---
            # Use signal_details for more accurate scoring
            total_score = signal_details.get('composite_score', 0.0)
            signal_strength = signal_details.get('signal_strength', 'weak')
            
            # Legacy score calculation for backward compatibility
            if total_score == 0.0:
                if signal_pullback: total_score += 0.4
                if signal_rebound: total_score += 0.4
                if signal_trend_breakout: total_score += 0.5
            
            has_rule_signal = signal_pullback or signal_rebound or signal_trend_breakout
            if not has_rule_signal:
                return

            # ===== Signal Cooldown Check =====
            # Prevent over-trading by enforcing cooldown period after exit
            bars_since_exit = current_bar - self.last_exit_bar
            if bars_since_exit < self.signal_cooldown_days:
                logger.debug(f"Signal cooldown active: {bars_since_exit}/{self.signal_cooldown_days} bars since last exit")
                return

            # ===== AI 模型概率门控 (Phase 8) =====
            ai_model = _get_ai_model()
            ensemble_model = _get_ensemble_model()
            
            # 检查是否使用集成模型
            use_ensemble = ensemble_model is not None
            
            ai_confidence = 0.5
            ensemble_disagreement = None
            
            if use_ensemble:
                # 使用集成模型获取预测和分歧度
                feat_cols = [c for c in self.data.df.columns if c.startswith('feat_')]
                if not feat_cols:
                    return
                bar_idx = len(self.data.Close) - 1
                feat_row = self.data.df.iloc[bar_idx][feat_cols]
                if feat_row.isna().any():
                    return
                X_pred = feat_row.values.reshape(1, -1)
                ensemble_proba, disagreement = get_ensemble_prediction_and_disagreement(
                    pd.DataFrame(X_pred, columns=feat_cols)
                )
                ai_confidence = ensemble_proba
                ensemble_disagreement = disagreement
            elif ai_model is not None:
                # 使用单一AI模型
                feat_cols = [c for c in self.data.df.columns if c.startswith('feat_')]
                if not feat_cols:
                    return
                bar_idx = len(self.data.Close) - 1
                feat_row = self.data.df.iloc[bar_idx][feat_cols]
                if feat_row.isna().any():
                    return
                ai_confidence = float(ai_model.predict(feat_row.values.reshape(1, -1))[0])
            
            # AI概率门控检查 - Dynamic threshold based on market state
            # Get current market state for dynamic threshold
            try:
                current_market_state = getattr(self, '_market_state', 'sideways')
                ai_thresh = get_dynamic_ai_threshold(
                    market_state=current_market_state,
                    base_threshold=p.ai_prob_threshold,
                    volatility_regime='normal'  # Could be detected from ATR
                )
            except Exception:
                # Fallback to original logic
                ai_thresh = p.bear_market_ai_threshold if not market_uptrend else p.ai_prob_threshold
            
            if ai_confidence < ai_thresh:
                return  # AI 预测未来不佳，拒绝开仓
            # ============================================

            # 3-Tier AI Position Sizing with Ensemble Disagreement Penalty
            # 使用分层置信度因子和集成模型分歧惩罚
            confidence_factor, tier = get_tiered_confidence_factor(
                ai_confidence=ai_confidence,
                ensemble_disagreement=ensemble_disagreement,
                use_ensemble=use_ensemble
            )
            
            # 如果置信度档位是block，阻止交易
            if tier == "block":
                return

            # Expected value (EV) gate (align with scan_today_signal)
            try:
                atr_val = float(self.atr[-1])
            except Exception:
                atr_val = float("nan")

            if not np.isfinite(atr_val) or atr_val <= 0 or price <= 0:
                return

            target_r = (float(p.ai_target_atr_mult) * atr_val) / float(price)
            stop_r = (float(p.ai_stop_loss_atr_mult) * atr_val) / float(price)
            cost_r = 2.0 * (float(getattr(p, "commission_pct", 0.0)) + float(getattr(p, "slippage_pct", 0.0)))
            ev_pct = (ai_confidence * target_r - (1.0 - ai_confidence) * stop_r - cost_r) * 100.0
            if ev_pct < float(getattr(p, "min_expected_value_pct", 0.0)):
                return
             
            disagreement_str = f"{ensemble_disagreement:.3f}" if ensemble_disagreement is not None else "N/A"
            logger.debug(f"AI confidence: {ai_confidence:.3f}, tier: {tier}, "
                        f"factor: {confidence_factor:.2f}, disagreement: {disagreement_str}")
            
            # Base sizing (ATR-risk based if enabled)
            if ai_model is not None or use_ensemble:
                if hasattr(p, 'atr_risk_per_trade') and p.atr_risk_per_trade > 0 and self.atr[-1] > 0:
                    risk_amt = self.equity * p.atr_risk_per_trade
                    risk_per_share = 2.0 * self.atr[-1]
                    shares = risk_amt / risk_per_share
                    fractional_size = min(0.99, (shares * price) / self.equity)
                    pos_size = fractional_size * confidence_factor
                else:
                    pos_size = p.position_size * confidence_factor
            else:
                pos_size = p.position_size

            # 4. [Friction Defense] Minimum Trade Value Check (A-share 5 RMB rule)
            target_value = self.equity * pos_size
            if target_value < 5000:
                # Cost of 5 RMB on < 5000 RMB is > 0.1%, which kills small-cap strategy edge
                return

            self.buy(size=pos_size)
            self.current_entry_bar = current_bar  # Track entry bar
            if signal_trend_breakout:
                # Use tighter stop for trend breakout (right side)
                self.current_stop_loss = price - 2.0 * self.atr[-1]
                self.current_trade_type = "right"
            else:
                # Use config trail_atr_mult for pullback/rebound signals
                self.current_stop_loss = price - p.trail_atr_mult * self.atr[-1]
                self.current_trade_type = "left"

    _Strategy.__name__ = "MultiFactorStrategy"
    _Strategy.__qualname__ = "MultiFactorStrategy"
    return _Strategy


def _load_and_prepare(code: str, params: StrategyParams) -> pd.DataFrame | None:
    file_path = os.path.join(CONF.history_data.data_dir, f"{code}.csv")
    if not os.path.exists(file_path):
        logger.error(f"历史数据文件不存在: {file_path}")
        return None

    df = pd.read_csv(file_path)
    if df.empty or len(df) < 50:
        return None

    df = calculate_indicators(df, params)

    # ===== Phase 8: Add ML features for AI model inference =====
    try:
        from quant.features.features import extract_features
        df = extract_features(df)
    except Exception:
        pass  # Gracefully degrade if features module has issues

    # Check data quality after indicators
    non_nan_count = len(df.dropna(subset=[f"SMA_{params.ma_long}", f"MACDh_{params.macd_fast}_{params.macd_slow}_{params.macd_signal}", f"RSI_{params.rsi_length}", f"ATRr_{params.atr_length}"]))
    if non_nan_count < 10:
        logger.debug(f"[{code}] 计算指标后有效数据不足 ({non_nan_count} 行)，跳过回测。")
        return None

    rename_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df.rename(columns=rename_map, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        return None
    return df


def run_backtest(
    code: str, params: StrategyParams | None = None,
    start_date: str | None = None, end_date: str | None = None
) -> tuple[Backtest, pd.Series] | None:
    params = _resolve_params(params)
    df = _load_and_prepare(code, params)
    if df is None:
        return None

    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    if df.empty or len(df) < 10:
        return None

    strategy_cls = create_strategy(params)
    # Execution friction: Backtesting.py exposes only a single `commission` rate.
    # We approximate slippage by folding it into commission (both are per-side rates).
    commission = float(getattr(params, "commission_pct", 0.001))
    slippage = float(getattr(params, "slippage_pct", 0.0))
    bt = Backtest(
        df,
        strategy_cls,
        cash=100_000,
        commission=(commission + slippage),
        trade_on_close=True,
        finalize_trades=True,
        exclusive_orders=True,
    )
    stats = bt.run()
    
    if stats['# Trades'] == 0:
        logger.debug(f"[{code}] 回测完成但未产生交易。数据量: {len(df)}")
    
    return bt, stats


def scan_today_signal(
    code: str, params: StrategyParams | None = None, target_date: str | None = None
) -> dict | None:
    from quant.infra.config import CONF
    params = _resolve_params(params)
    file_path = os.path.join(CONF.history_data.data_dir, f"{code}.csv")
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    
    if target_date is not None:
        if "date" in df.columns:
            df = df[df["date"] <= target_date].copy()
        elif "Date" in df.columns:
            df = df[df["Date"] <= target_date].copy()
            
    if df.empty or len(df) < 50:
        return None

    df = calculate_indicators(df, params)
    
    # ===== Phase 8: Add ML features for AI model inference =====
    try:
        from quant.features.features import extract_features
        df = extract_features(df)
    except Exception as e:
        logger.debug(f"[{code}] Feature extraction in scan failed: {e}")
        
    if df.empty:
        return None

    cols = _build_column_names(params)
    required = [cols["sma_s"], cols["sma_l"], cols["macd_h"], cols["rsi"], cols["bb_lower"], cols["atr"]]
    if not all(c in df.columns for c in required):
        return None

    row_1 = df.iloc[-1]
    row_2 = df.iloc[-2]
    row_3 = df.iloc[-3] if len(df) >= 3 else row_2

    price = row_1["close"]
    sma_l_1 = row_1[cols["sma_l"]]
    sma_l_3 = row_3[cols["sma_l"]]
    sma_s_1 = row_1[cols["sma_s"]]
    macd_h_1 = row_1[cols["macd_h"]]
    macd_h_2 = row_2[cols["macd_h"]]
    rsi_val = row_1[cols["rsi"]]
    vol_1 = row_1.get("volume", row_1.get("Volume", 0))
    vol_2 = row_2.get("volume", row_2.get("Volume", 0))

    has_vol_slope = "vol_slope" in df.columns
    vol_slope_1 = row_1["vol_slope"] if has_vol_slope else 0.0
    has_mom_div = "momentum_divergence" in df.columns
    mom_div_1 = row_1["momentum_divergence"] if has_mom_div else 0.0

    # ===== 大盘情绪过滤 (Market Regime Filter) =====
    current_date_str = row_1.get("date")
    market_uptrend = True
    market_state = "sideways_low_vol"
    idx_df = get_market_index()
    if idx_df is not None and current_date_str is not None:
        try:
            # Handle both string and datetime date formats
            current_date_ts = pd.to_datetime(current_date_str)
            idx_loc = idx_df.index.get_indexer([current_date_ts], method='pad')[0]
            if idx_loc != -1 and idx_loc < len(idx_df):
                market_uptrend = bool(idx_df.iloc[idx_loc]["market_uptrend"])
                market_state = str(idx_df.iloc[idx_loc].get("market_state", market_state))
        except Exception as e:
            logger.debug(f"Market trend check failed: {e}")
            pass

    # if not market_uptrend:
    #     return None
    # ===============================================

    # Get weekly timeframe confirmation for scan_today_signal (config-controlled)
    weekly_data = None
    mt_cfg = getattr(getattr(CONF, "strategy", None), "multi_timeframe", {}) or {}
    mt_enabled = bool(mt_cfg.get("enabled", True)) if isinstance(mt_cfg, dict) else True
    weekly_enabled = bool(mt_cfg.get("weekly_confirmation", True)) if isinstance(mt_cfg, dict) else True
    if mt_enabled and weekly_enabled and current_date_str is not None:
        try:
            weekly_data = get_weekly_confirmation(
                code,
                str(current_date_str),
                CONF.history_data.data_dir
            )
        except Exception:
            weekly_data = None
    
    signal_pullback, signal_rebound, signal_trend_breakout, signal_details = evaluate_buy_signals(
        price=price,
        open_p=row_1["open"],
        low_p=row_1["low"],
        sma_l_1=sma_l_1,
        sma_l_3=sma_l_3,
        sma_s_1=sma_s_1,
        macd_h_1=macd_h_1,
        macd_h_2=macd_h_2,
        rsi_1=rsi_val,
        bb_lower_1=row_1[cols["bb_lower"]],
        vol_1=vol_1,
        vol_2=vol_2,
        has_vol_slope=has_vol_slope,
        vol_slope_1=vol_slope_1,
        has_mom_div=has_mom_div,
        mom_div_1=mom_div_1,
        market_uptrend=market_uptrend,
        p=params,
        weekly_data=weekly_data,
    )

    signal_type = ""
    if signal_pullback:
        signal_type = "布林带极度下杀反弹 (左侧)"
    elif signal_rebound:
        signal_type = "超卖恐慌底部 (左侧)"
    elif signal_trend_breakout:
        signal_type = "均线放量金叉 (右侧)"
        
    if not signal_type:
        return None
        
    # ===== AI Gate + EV Filter (align with backtest entry logic) =====
    ai_model = _get_ai_model()
    ensemble_model = _get_ensemble_model()
    use_ensemble = ensemble_model is not None

    model_present = use_ensemble or (ai_model is not None)
    ai_confidence = 0.5
    ensemble_disagreement = None
    ai_model_type = "ensemble" if use_ensemble else ("lgbm" if ai_model is not None else "rule")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if model_present:
        if not feat_cols:
            return None
        feat_row = df.iloc[-1][feat_cols]
        if feat_row.isna().any():
            return None
        if use_ensemble:
            X_pred = feat_row.values.reshape(1, -1)
            ai_confidence, ensemble_disagreement = get_ensemble_prediction_and_disagreement(
                pd.DataFrame(X_pred, columns=feat_cols)
            )
        elif ai_model is not None:
            ai_confidence = float(ai_model.predict(feat_row.values.reshape(1, -1))[0])

    # Dynamic AI threshold based on market state
    try:
        ai_thresh = get_dynamic_ai_threshold(
            market_state=market_state,
            base_threshold=params.ai_prob_threshold,
            volatility_regime="normal",
        )
    except Exception:
        ai_thresh = params.ai_prob_threshold

    if ai_confidence < ai_thresh:
        return None

    confidence_factor, tier = get_tiered_confidence_factor(
        ai_confidence=ai_confidence,
        ensemble_disagreement=ensemble_disagreement,
        use_ensemble=use_ensemble,
    )
    if tier == "block":
        return None

    # Expected value (EV) gate (percent)
    atr_raw = row_1[cols["atr"]]
    if pd.isna(atr_raw) or pd.isna(price) or price <= 0:
        return None
    atr_val = float(atr_raw)
    if np.isnan(atr_val) or atr_val <= 0:
        return None

    target_r = (params.ai_target_atr_mult * atr_val) / float(price)
    stop_r = (params.ai_stop_loss_atr_mult * atr_val) / float(price)
    cost_r = 2.0 * (params.commission_pct + params.slippage_pct)
    ev_pct = (ai_confidence * target_r - (1.0 - ai_confidence) * stop_r - cost_r) * 100.0

    if ev_pct < params.min_expected_value_pct:
        return None
    # ============================================

    # Calculate extra fields for analyzer compatibility (Sector rotation / Ranking)
    mom_20 = 0.0
    if len(df) >= 20:
        mom_20 = (price - df.iloc[-20]["close"]) / df.iloc[-20]["close"]
    
    # Use enhanced signal scoring from signal_details
    total_score = signal_details.get('composite_score', 0.0)
    if total_score == 0.0:
        # Legacy fallback
        if signal_pullback: total_score += 0.4
        if signal_rebound: total_score += 0.4
        if signal_trend_breakout: total_score += 0.5
    
    signal_strength = signal_details.get('signal_strength', 'weak')

    buy_score = float(total_score) * float(ai_confidence) * float(confidence_factor)

    return {
        "code": code,
        "date": str(row_1["date"]).split(" ")[0] if "date" in row_1 else "",
        "close": round(float(price), 2),
        "total_score": round(total_score, 3),
        "buy_score": round(buy_score, 4),
        "signal_strength": signal_strength,
        "signal_type": signal_type,
        "rsi": round(float(rsi_val), 2),
        "mom_20": round(float(mom_20), 4),
        "volume_ratio": round(float(vol_1 / vol_2) if vol_2 > 0 else 0.0, 2),
        "ai_prob": round(float(ai_confidence), 4),
        "ai_threshold": round(float(ai_thresh), 4),
        "ai_tier": tier,
        "ensemble_disagreement": round(float(ensemble_disagreement), 4) if ensemble_disagreement is not None else None,
        "market_state": market_state,
        "market_uptrend": bool(market_uptrend),
        "atr": round(float(atr_val), 4),
        "atr_pct": round(float(atr_val / float(price) * 100.0), 3),
        "expected_value_pct": round(float(ev_pct), 3),
        "ai_model_type": ai_model_type,
    }


def batch_backtest(
    codes: list[str], params: StrategyParams | None = None,
    start_date: str | None = None, end_date: str | None = None
) -> pd.DataFrame:
    params = _resolve_params(params)
    records: list[dict] = []

    for code in tqdm(codes, desc="批量回测"):
        try:
            result = run_backtest(code, params, start_date=start_date, end_date=end_date)
            if result is None:
                continue
            _, stats = result
            num_trades = stats.get("# Trades", 0)
            if num_trades == 0:
                continue
            records.append(
                {
                    "code": code,
                    "return_pct": stats.get("Return [%]", 0.0),
                    "win_rate": stats.get("Win Rate [%]", 0.0),
                    "max_drawdown": stats.get("Max. Drawdown [%]", 0.0),
                    "num_trades": num_trades,
                    "sharpe": stats.get("Sharpe Ratio", 0.0),
                    "equity_final": stats.get("Equity Final [$]", 0.0),
                }
            )
        except Exception as e:
            logger.debug(f"回测异常 {code}: {e}")

    return pd.DataFrame(records)
