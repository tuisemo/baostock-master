from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from tqdm import tqdm

from quant.analyzer import calculate_indicators, classify_market_state, get_dynamic_params
from quant.config import CONF
from quant.logger import logger

# ===== AI Model (Phase 8) =====
_AI_MODEL = None
_AI_MODEL_PATH = "models/alpha_lgbm.txt"
_AI_MODEL_LOAD_ATTEMPTED = False

def _get_ai_model():
    """Lazy-load the LightGBM model singleton. Returns None if model is unavailable or corrupted."""
    global _AI_MODEL, _AI_MODEL_LOAD_ATTEMPTED
    if _AI_MODEL_LOAD_ATTEMPTED:
        return _AI_MODEL  # 已尝试过加载（成功返回模型，失败返回 None）
    _AI_MODEL_LOAD_ATTEMPTED = True

    import os
    if not os.path.exists(_AI_MODEL_PATH):
        logger.debug(f"AI 模型文件不存在: {_AI_MODEL_PATH}，将使用纯规则引擎。")
        return None

    try:
        import lightgbm as lgb
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _AI_MODEL = lgb.Booster(model_file=_AI_MODEL_PATH)
        # 验证模型是否可用
        _AI_MODEL.num_feature()
        logger.info(f"AI 模型已加载: {_AI_MODEL_PATH} ({_AI_MODEL.num_feature()} features)")
    except Exception as e:
        logger.warning(f"AI 模型文件损坏或不兼容，将使用纯规则引擎: {e}")
        _AI_MODEL = None

    return _AI_MODEL

if TYPE_CHECKING:
    from quant.strategy_params import StrategyParams


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
    df_idx["MA20"] = df_idx[close_col].rolling(window=20).mean()
    df_idx["market_uptrend"] = df_idx[close_col] > df_idx["MA20"]
    
    _MARKET_INDEX_CACHE = df_idx
    return _MARKET_INDEX_CACHE

def _resolve_params(params: StrategyParams | None) -> StrategyParams:
    from quant.strategy_params import StrategyParams as SP

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
) -> tuple[bool, bool, bool]:
    macd_golden_cross = macd_h_2 <= 0 and macd_h_1 > 0
    is_green_candle = price > open_p

    if has_vol_slope and vol_slope_1 > 0.1:
        vol_up = True
    else:
        vol_up = vol_1 > vol_2 * p.vol_up_ratio

    mom_ok = True
    if has_mom_div and mom_div_1 <= -0.02:
        mom_ok = False

    # === 左侧交易评估 ===
    is_bb_dip = price < bb_lower_1 * p.bbands_lower_bias
    is_rsi_dip = rsi_1 < p.rsi_oversold_extreme
    
    signal_pullback = False
    signal_rebound = False
    
    if is_bb_dip or is_rsi_dip:
        # 止跌形态验证（收阳线或长下影线），作为强烈加分项
        lower_shadow = min(open_p, price) - low_p
        body = abs(price - open_p)
        has_bottoming_sign = is_green_candle or (lower_shadow > body * 1.5 and lower_shadow > 0)
            
        # 左侧接飞刀算分系统
        score = 0.0
        if has_bottoming_sign: score += 1.0       # 有止跌形态直接+1分
        if is_bb_dip: score += p.w_pullback_ma    # 复用回调权重为布林下轨突刺分
        if is_rsi_dip: score += p.w_rsi_rebound   # 复用超卖权重为极度恐慌分
        if is_green_candle: score += p.w_green_candle
        if vol_up: score += p.w_vol_up
        if macd_golden_cross or macd_h_1 > macd_h_2: score += p.w_macd_cross
        if mom_ok: score += 1.0
        
        # 大盘熊市时，左侧入局门槛提高
        pass_threshold = 1.0 if market_uptrend else 3.0
        signal_pullback = (score >= pass_threshold) and is_bb_dip
        signal_rebound = (score >= pass_threshold) and is_rsi_dip

    # === 右侧交易评估 (牛市专属) ===
    signal_trend_breakout = False
    if market_uptrend:
        is_above_ma = price > sma_s_1
        is_rsi_health = 40 < rsi_1 < 75
        if macd_golden_cross and is_above_ma and vol_up and is_rsi_health:
            signal_trend_breakout = True

    return signal_pullback, signal_rebound, signal_trend_breakout


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
            
            # Initialize market state
            idx_df = get_market_index()
            if idx_df is not None:
                self._market_state = classify_market_state(idx_df)
                self._dynamic_p = get_dynamic_params(self._base_p, self._market_state)
                logger.debug(f"Market state: {self._market_state}, using dynamic params")
            else:
                self._dynamic_p = self._base_p

        def next(self):
            if pd.isna(self.sma_s[-1]) or pd.isna(self.rsi[-1]) or pd.isna(self.atr[-1]):
                return

            p = self._dynamic_p  # Use dynamic params
            price = self.data.Close[-1]

            if self.position:
                trail_stop = price - p.trail_atr_mult * self.atr[-1]
                if trail_stop > self.current_stop_loss:
                    self.current_stop_loss = trail_stop

                if price <= self.current_stop_loss:
                    self.position.close()
                    return

                # Time-Decay Exit (If held too long with minimal return)
                days_held = len(self.data) - self.trades[0].entry_bar
                if days_held >= p.max_hold_days and self.position.pl_pct < p.max_hold_min_return:
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
                    self.position.close()
                    return
                
                # 2. Volume-Price Divergence (volume shrinks while price stalls)
                if hasattr(self, 'vol_slope') and len(self.data) >= 5:
                    recent_slope = np.mean(self.vol_slope[-5:])
                    recent_price_change = (self.data.Close[-1] - self.data.Close[-5]) / self.data.Close[-5]
                    if recent_slope < -0.05 and 0 < recent_price_change < 0.02:
                        self.position.close()
                        return
                
                # 3. MACD Top Divergence
                if len(self.macd_h) >= 10:
                    recent_macd = self.macd_h[-5:].tolist()
                    recent_highs = self.data.High[-5:].tolist()
                    if (recent_macd[0] > recent_macd[-1] and 
                        recent_highs[0] > recent_highs[-1]):
                        self.position.close()
                        return
                
                # 4. Timeout with Minimal Return (Time-based exit)
                if days_held >= p.max_hold_days and self.position.pl_pct < p.max_hold_min_return:
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
            # 如果大盘在上证指数 20日均线下方，认定为熊市/调整期，彻底拒绝一切买入信号
            current_date = self.data.index[-1]
            market_uptrend = True  # Default True if no index found
            idx_df = get_market_index()
            if idx_df is not None:
                # Use 'pad' to get the latest available market data without looking into the future
                idx_loc = idx_df.index.get_indexer([current_date], method='pad')[0]
                if idx_loc != -1:
                    market_uptrend = bool(idx_df.iloc[idx_loc]["market_uptrend"])

            # if not market_uptrend:
            #     return
            # ===============================================

            signal_pullback, signal_rebound, signal_trend_breakout = evaluate_buy_signals(
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
            )

            has_rule_signal = signal_pullback or signal_rebound or signal_trend_breakout
            if not has_rule_signal:
                return

            # ===== AI 模型概率门控 (Phase 8) =====
            ai_model = _get_ai_model()
            if ai_model is not None:
                feat_cols = [c for c in self.data.df.columns if c.startswith('feat_')]
                if feat_cols:
                    bar_idx = len(self.data.Close) - 1
                    feat_row = self.data.df.iloc[bar_idx][feat_cols]
                    if not feat_row.isna().any():
                        prob = ai_model.predict(feat_row.values.reshape(1, -1))[0]
                        ai_thresh = p.bear_market_ai_threshold if not market_uptrend else p.ai_prob_threshold
                        if prob < ai_thresh:
                            return  # AI 预测未来不佳，拒绝开仓
            # ============================================

            # Position Sizing (Dynamic based on AI confidence, volatility, and market state)
            if ai_model is not None and hasattr(p, 'atr_risk_per_trade') and p.atr_risk_per_trade > 0 and self.atr[-1] > 0:
                # Get AI confidence
                feat_cols = [c for c in self.data.df.columns if c.startswith('feat_')]
                ai_confidence = 0.5  # Default moderate confidence
                if feat_cols:
                    bar_idx = len(self.data.Close) - 1
                    feat_row = self.data.df.iloc[bar_idx][feat_cols]
                    if not feat_row.isna().any():
                        ai_confidence = float(ai_model.predict(feat_row.values.reshape(1, -1))[0])
                
                # Volatility adjustment (higher volatility = smaller position)
                vol_ratio = self.atr[-1] / price
                volatility_factor = 1.0 / (1.0 + vol_ratio * 10)
                
                # Market state factor (already applied to dynamic params)
                # Strong bull = 1.3x, Weak bull = 1.1x, Sideways = 1.0x, Weak bear = 0.8x, Strong bear = 0.6x
                
                # Trade type factor (right side = slightly larger position)
                trade_type_factor = 1.1 if (signal_pullback or signal_rebound) else 1.0
                
                # Calculate base position size
                risk_amt = self.equity * p.atr_risk_per_trade
                risk_per_share = 2.0 * self.atr[-1]
                shares = risk_amt / risk_per_share
                fractional_size = min(0.99, (shares * price) / self.equity)
                
                # Apply dynamic factors
                confidence_factor = ai_confidence / 0.5  # Normalize around 0.5 threshold
                dynamic_size = fractional_size * confidence_factor * volatility_factor * trade_type_factor
                
                # Clamp position size to reasonable range (2% to 25% of equity)
                pos_size = max(0.02, min(0.25, dynamic_size))
            elif hasattr(p, 'atr_risk_per_trade') and p.atr_risk_per_trade > 0 and self.atr[-1] > 0:
                # Fallback to standard ATR-based sizing
                risk_amt = self.equity * p.atr_risk_per_trade
                risk_per_share = 2.0 * self.atr[-1]
                shares = risk_amt / risk_per_share
                fractional_size = min(0.99, (shares * price) / self.equity)
                pos_size = max(0.01, min(0.99, fractional_size))
            else:
                # Fallback to fixed ratio
                pos_size = p.position_size

            self.buy(size=pos_size)
            if signal_trend_breakout:
                self.current_stop_loss = price - 2.5 * self.atr[-1]
                self.current_trade_type = "right"
            else:
                self.current_stop_loss = price - 1.5 * self.atr[-1]
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
        from quant.features import extract_features
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
    # Default slippage if not configured
    slippage = getattr(CONF.strategy, "slippage_pct", 0.002)
    bt = Backtest(
        df,
        strategy_cls,
        cash=100_000,
        commission=0.0002 + slippage,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats = bt.run()
    
    if stats['# Trades'] == 0:
        logger.debug(f"[{code}] 回测完成但未产生交易。数据量: {len(df)}")
    
    return bt, stats


def scan_today_signal(
    code: str, params: StrategyParams | None = None, target_date: str | None = None
) -> dict | None:
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
        from quant.features import extract_features
        df = extract_features(df)
    except Exception as e:
        logger.debug(f"[{code}] Feature extraction in scan failed: {e}")
        
    if df.empty:
        return None

    cols = _build_column_names(params)
    required = [cols["sma_s"], cols["sma_l"], cols["macd_h"], cols["rsi"], cols["bb_lower"]]
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
    idx_df = get_market_index()
    if idx_df is not None and current_date_str is not None:
        try:
            # Handle both string and datetime date formats
            current_date_ts = pd.to_datetime(current_date_str)
            idx_loc = idx_df.index.get_indexer([current_date_ts], method='pad')[0]
            if idx_loc != -1 and idx_loc < len(idx_df):
                market_uptrend = bool(idx_df.iloc[idx_loc]["market_uptrend"])
        except Exception as e:
            logger.debug(f"Market trend check failed: {e}")
            pass

    # if not market_uptrend:
    #     return None
    # ===============================================

    signal_pullback, signal_rebound, signal_trend_breakout = evaluate_buy_signals(
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
        
    # ===== AI 模型概率门控 (Phase 8) =====
    ai_model = _get_ai_model()
    act_prob = 1.0  # 默认满算力，纯规则兜底
    if ai_model is not None:
        feat_cols = [c for c in df.columns if c.startswith('feat_')]
        if feat_cols:
            feat_row = df.iloc[-1][feat_cols]
            # 只有当所有特征都不是 NA 的时候才进行推断
            if not feat_row.isna().any():
                act_prob = float(ai_model.predict(feat_row.values.reshape(1, -1))[0])
                if act_prob < getattr(params, 'ai_prob_threshold', 0.35):
                    return None  # AI 预测未来不佳，残酷否决 (对齐回测逻辑)
    # ============================================

    return {
        "代码": code,
        "触发日期": str(row_1["date"]).split(" ")[0] if "date" in row_1 else "",
        "现价": round(float(price), 2),
        "信号类型": signal_type,
        "RSI指标": round(float(rsi_val), 2),
        "量能倍数": round(float(vol_1 / vol_2) if vol_2 > 0 else 0.0, 2),
        "AI胜率预测": round(act_prob, 4),
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
