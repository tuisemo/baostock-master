from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from tqdm import tqdm

from quant.analyzer import calculate_indicators
from quant.config import CONF
from quant.logger import logger

if TYPE_CHECKING:
    from quant.strategy_params import StrategyParams


def _build_column_names(p: StrategyParams) -> dict[str, str]:
    return {
        "sma_s": f"SMA_{p.ma_short}",
        "sma_l": f"SMA_{p.ma_long}",
        "macd_h": f"MACDh_{p.macd_fast}_{p.macd_slow}_{p.macd_signal}",
        "rsi": f"RSI_{p.rsi_length}",
        "bb_lower": f"BBL_{p.bbands_length}_{p.bbands_std}_{p.bbands_std}",
        "bb_upper": f"BBU_{p.bbands_length}_{p.bbands_std}_{p.bbands_std}",
        "obv": "OBV",
        "atr": f"ATRr_{p.atr_length}",
    }


def _resolve_params(params: StrategyParams | None) -> StrategyParams:
    from quant.strategy_params import StrategyParams as SP

    if params is not None:
        return params
    return SP.from_app_config(CONF)


def create_strategy(params: StrategyParams) -> type[Strategy]:
    cols = _build_column_names(params)

    class _Strategy(Strategy):
        _p = params
        _cols = cols

        def init(self):
            p = self._p
            c = self._cols
            self.sma_s = self.data[c["sma_s"]]
            self.sma_l = self.data[c["sma_l"]]
            self.macd_h = self.data[c["macd_h"]]
            self.rsi = self.data[c["rsi"]]
            self.bb_lower = self.data[c["bb_lower"]]
            self.bb_upper = self.data[c["bb_upper"]]
            self.obv = self.data[c["obv"]]
            self.atr = self.data[c["atr"]]
            self.current_stop_loss = 0.0
            self._has_vol_slope = "vol_slope" in self.data.df.columns
            self._has_mom_div = "momentum_divergence" in self.data.df.columns
            if self._has_vol_slope:
                self.vol_slope = self.data["vol_slope"]
            if self._has_mom_div:
                self.mom_div = self.data["momentum_divergence"]

        def next(self):
            if pd.isna(self.sma_s[-1]) or pd.isna(self.rsi[-1]) or pd.isna(self.atr[-1]):
                return

            p = self._p
            price = self.data.Close[-1]

            if self.position:
                trail_stop = price - p.trail_atr_mult * self.atr[-1]
                if trail_stop > self.current_stop_loss:
                    self.current_stop_loss = trail_stop

                if price <= self.current_stop_loss:
                    self.position.close()
                    return

                if self.position.pl_pct >= p.take_profit_pct:
                    self.position.close()
                    return

                if self.position.pl_pct >= p.breakeven_trigger and len(self.trades) > 0:
                    breakeven_stop = self.trades[0].entry_price * p.breakeven_buffer
                    if self.current_stop_loss < breakeven_stop:
                        self.current_stop_loss = breakeven_stop

                if price >= self.bb_upper[-1]:
                    self.position.close()
                elif price < self.sma_s[-1] and self.data.Close[-2] < self.sma_s[-2]:
                    self.position.close()
                return

            is_ma_long_up = False
            if len(self.sma_l) >= 3:
                is_ma_long_up = self.sma_l[-1] > self.sma_l[-3]

            macd_golden_cross = self.macd_h[-2] <= 0 and self.macd_h[-1] > 0
            is_green_candle = price > self.data.Open[-1]

            if self._has_vol_slope and self.vol_slope[-1] > 0.1:
                vol_up = True
            else:
                vol_up = self.data.Volume[-1] > self.data.Volume[-2] * p.vol_up_ratio

            rsi_cooled = self.rsi[-1] < p.rsi_cooled_max

            mom_ok = True
            if self._has_mom_div and self.mom_div[-1] <= -0.02:
                mom_ok = False

            # 均线回踩信波段买点打分机制
            near_ma_long = self.data.Low[-1] <= self.sma_l[-1] * p.pullback_ma_tolerance
            pullback_score = 0
            if is_ma_long_up: pullback_score += 1
            if near_ma_long: pullback_score += 2 # 回踩均线是核心点，权重较高
            if is_green_candle: pullback_score += 1
            if macd_golden_cross or self.macd_h[-1] > self.macd_h[-2]: pullback_score += 1 # 放宽MACD条件：金叉或红柱变长/绿柱缩短
            if vol_up: pullback_score += 1
            if rsi_cooled: pullback_score += 1
            if mom_ok: pullback_score += 1
            
            signal_pullback = pullback_score >= 5

            # 乖离率超卖波段打分机制
            macd_turn_up = self.macd_h[-1] > self.macd_h[-2]
            negative_bias = price < self.sma_s[-1] * p.negative_bias_pct
            rsi_oversold = self.rsi[-1] < p.rsi_oversold

            rebound_score = 0
            if negative_bias: rebound_score += 2 # 核心点
            if rsi_oversold: rebound_score += 2 # 核心点
            if is_green_candle: rebound_score += 1
            if vol_up: rebound_score += 1
            if macd_turn_up: rebound_score += 1
            if mom_ok: rebound_score += 1
            
            signal_rebound = rebound_score >= 5

            if signal_pullback or signal_rebound:
                self.buy()
                self.current_stop_loss = price - p.atr_multiplier * self.atr[-1]

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
    code: str, params: StrategyParams | None = None
) -> tuple[Backtest, pd.Series] | None:
    params = _resolve_params(params)
    df = _load_and_prepare(code, params)
    if df is None:
        return None

    strategy_cls = create_strategy(params)
    bt = Backtest(
        df,
        strategy_cls,
        cash=100_000,
        commission=0.0002,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats = bt.run()
    
    if stats['# Trades'] == 0:
        logger.debug(f"[{code}] 回测完成但未产生交易。数据量: {len(df)}")
    
    return bt, stats


def scan_today_signal(
    code: str, params: StrategyParams | None = None
) -> dict | None:
    params = _resolve_params(params)
    file_path = os.path.join(CONF.history_data.data_dir, f"{code}.csv")
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    if df.empty or len(df) < 50:
        return None

    df = calculate_indicators(df, params)
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

    is_ma_long_up = sma_l_1 > sma_l_3
    macd_golden_cross = macd_h_2 <= 0 and macd_h_1 > 0
    is_green_candle = price > row_1["open"]

    has_vol_slope = "vol_slope" in df.columns
    if has_vol_slope and row_1["vol_slope"] > 0.1:
        vol_up = True
    else:
        vol_up = vol_1 > vol_2 * params.vol_up_ratio

    rsi_cooled = rsi_val < params.rsi_cooled_max

    mom_ok = True
    if "momentum_divergence" in df.columns and row_1["momentum_divergence"] <= -0.02:
        mom_ok = False

    # 均线回踩打分
    near_ma_long = row_1["low"] <= sma_l_1 * params.pullback_ma_tolerance
    pullback_score = 0
    if is_ma_long_up: pullback_score += 1
    if near_ma_long: pullback_score += 2
    if is_green_candle: pullback_score += 1
    if macd_golden_cross or macd_h_1 > macd_h_2: pullback_score += 1
    if vol_up: pullback_score += 1
    if rsi_cooled: pullback_score += 1
    if mom_ok: pullback_score += 1
    
    signal_pullback = pullback_score >= 5

    macd_turn_up = macd_h_1 > macd_h_2
    negative_bias = price < sma_s_1 * params.negative_bias_pct
    rsi_oversold = rsi_val < params.rsi_oversold
    
    # 乖离反弹打分
    rebound_score = 0
    if negative_bias: rebound_score += 2
    if rsi_oversold: rebound_score += 2
    if is_green_candle: rebound_score += 1
    if vol_up: rebound_score += 1
    if macd_turn_up: rebound_score += 1
    if mom_ok: rebound_score += 1

    signal_rebound = rebound_score >= 5

    signal_type = ""
    if signal_pullback:
        signal_type = "均线回踩波段买点"
    elif signal_rebound:
        signal_type = "乖离率超卖波段"
    else:
        return None

    return {
        "代码": code,
        "触发日期": str(row_1["date"]).split(" ")[0] if "date" in row_1 else "",
        "现价": round(float(price), 2),
        "信号类型": signal_type,
        "RSI指标": round(float(rsi_val), 2),
        "量能倍数": round(float(vol_1 / vol_2) if vol_2 > 0 else 0.0, 2),
    }


def batch_backtest(
    codes: list[str], params: StrategyParams | None = None
) -> pd.DataFrame:
    params = _resolve_params(params)
    records: list[dict] = []

    for code in tqdm(codes, desc="批量回测"):
        try:
            result = run_backtest(code, params)
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
