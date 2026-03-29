"""
Crypto Scalper — 5-Minute Binance Futures Backtester
=====================================================
$10k capital, BTC/ETH/SOL, long+short on Binance USDT-M futures.

Strategy thesis:
  Short-term mean-reversion + momentum bursts on 5m candles.
  Crypto markets are dominated by retail and bots. After sharp moves,
  price tends to revert (mean-reversion). But when volume surges with
  trend, continuation is likely (momentum burst). The model learns
  which regime is active and sizes accordingly.

Key design choices for scalping:
  - 5m candles (not 1m — too noisy; not 15m — too slow to compound)
  - 1-4 bar horizon (5-20 min holding period)
  - Binance futures fees: 0.04% taker (conservative)
  - Tight stops: 1.5x ATR (scalpers can't afford large drawdowns)
  - High win rate target: >55% needed to overcome fees
"""

import sys
import time
import warnings
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Scalper")


# =========================================================================
# CONFIGURATION
# =========================================================================

@dataclass
class Config:
    # Assets — will try USDT pairs, fallback to USD pairs (Binance.US)
    assets: List[str] = field(default_factory=lambda: [
        "ADAUSDT", "DOGEUSDT", "ETHUSDT",
    ])
    # (fallback_symbols removed — CCXT handles symbol resolution automatically)

    # Data
    interval: str = "5m"
    lookback_days: int = 90       # 3 months of 5m data
    min_bars: int = 500

    # Prediction / holding
    prediction_horizon: int = 3   # 3 bars = 15 min forward return
    max_hold_bars: int = 12       # Force exit after 1 hour (let winners run)
    min_hold_bars: int = 1        # Hold at least 1 bar (5 min)

    # Feature windows (tuned for 5m scalping)
    feature_windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24, 48])
    # 3 bars = 15m, 6 = 30m, 12 = 1h, 24 = 2h, 48 = 4h

    # Train/test split
    train_pct: float = 0.70

    # Binance futures fees (taker, conservative)
    fee_rate: float = 0.0004      # 0.04% per side
    slippage_rate: float = 0.0003 # 0.03% slippage (realistic for alts)

    # Signal thresholds (lower = more trades = more compounding)
    signal_threshold_long: float = 0.2
    signal_threshold_short: float = -0.2

    # Position sizing
    max_position_pct: float = 0.25  # 25% of capital per trade (aggressive scalping)
    leverage: int = 5               # 5x leverage

    # Risk management
    atr_stop_multiplier: float = 1.5  # Tighter stops — cut losers fast
    atr_period: int = 14
    max_daily_loss_pct: float = 0.08  # 8% max daily loss
    max_open_trades: int = 4          # More concurrent positions

    # Feature selection
    max_features: int = 15
    rolling_norm_window: int = 96     # 96 bars = 8 hours rolling normalization

    # Capital
    capital: float = 10000.0

    # Time window offset: shift entire window back by this many days
    # 0 = most recent data, 90 = start from 90 days ago, etc.
    end_offset_days: int = 0


# =========================================================================
# DATA FETCHER — Binance.US REST API (US-friendly, no key needed, real data)
# =========================================================================

# Internal symbol -> Binance.US symbol
BINANCEUS_SYMBOL_MAP = {
    "BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
    "DOGEUSDT": "DOGEUSD", "SUIUSDT": "SUIUSD", "AVAXUSDT": "AVAXUSD",
    "LINKUSDT": "LINKUSD", "PEPEUSDT": "PEPEUSD", "WIFUSDT": "WIFUSD",
    "BNBUSDT": "BNBUSD", "XRPUSDT": "XRPUSD", "ADAUSDT": "ADAUSD",
    "HYPEUSDT": "HYPEUSD", "HBARUSDT": "HBARUSD", "DOTUSDT": "DOTUSD",
    "SHIBUSDT": "SHIBUSD", "ATOMUSDT": "ATOMUSD", "RENDERUSDT": "RENDERUSD",
    "XLMUSDT": "XLMUSD", "BONKUSDT": "BONKUSD", "TRUMPUSDT": "TRUMPUSD",
}

INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300,
    "15m": 900, "1h": 3600, "4h": 14400,
    "1d": 86400,
}


class BinanceUSDataFetcher:
    """Fetch historical OHLCV from Binance.com REST API (no API key, full liquidity USDT pairs)."""

    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, config: Config):
        self.config = config

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch historical klines for a symbol from Binance.com (USDT pairs)."""
        import requests

        # Binance.com uses USDT pairs directly — no symbol conversion needed
        bus_symbol = symbol  # ADAUSDT, DOGEUSDT, BTCUSDT etc.

        end_ms = int(time.time() * 1000) - (self.config.end_offset_days * 86_400_000)
        start_ms = end_ms - (self.config.lookback_days * 86_400_000)

        log.info(f"Fetching {symbol} {self.config.interval} from Binance.com "
                 f"({self.config.lookback_days} days)...")

        all_candles = []
        cursor = start_ms

        while cursor < end_ms:
            params = {
                "symbol": bus_symbol,
                "interval": self.config.interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }

            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            candles = resp.json()

            if not candles:
                break

            all_candles.extend(candles)
            # Move cursor past last candle
            cursor = int(candles[-1][0]) + 1
            if len(candles) < 1000:
                break

            time.sleep(0.1)  # Rate limiting

        if not all_candles:
            raise ValueError(f"No data returned for {symbol} ({bus_symbol})")

        # Binance kline format: [openTime, open, high, low, close, volume,
        #   closeTime, quoteVolume, numTrades, takerBuyVolume, takerBuyQuoteVolume, ignore]
        df = pd.DataFrame(all_candles, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume",
            "taker_buy_quote_volume", "ignore",
        ])

        df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
        df.set_index("open_time", inplace=True)
        df.drop(columns=["close_time", "ignore"], inplace=True)

        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                     "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)

        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        log.info(f"  {symbol}: {len(df)} bars, "
                 f"{df.index[0]} to {df.index[-1]}")
        return df

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for symbol in self.config.assets:
            try:
                data[symbol] = self.fetch(symbol)
            except Exception as e:
                log.error(f"Failed to fetch {symbol}: {e}")
        return data


# Aliases for backward compatibility (paper_trader.py imports these names)
BybitDataFetcher = BinanceUSDataFetcher
BinanceDataFetcher = BinanceUSDataFetcher
CoinbaseDataFetcher = BinanceUSDataFetcher


# =========================================================================
# SCALPING FEATURE ENGINE
# =========================================================================

class ScalpingFeatureEngine:
    """
    Features specifically designed for 5-minute scalping.

    Key differences from swing trading features:
    1. Microstructure signals (taker buy ratio, trade count momentum)
    2. Very short windows (3-48 bars = 15min to 4h)
    3. Mean-reversion emphasis (z-scores, Bollinger overshoot)
    4. Volume profile signals (volume spikes predict reversals)
    """

    def __init__(self, config: Config):
        self.config = config
        self.feature_names: List[str] = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        opn = df["open"]
        volume = df["volume"]
        trades = df["trades"].astype(float)
        taker_buy_vol = df["taker_buy_volume"]
        quote_vol = df["quote_volume"]

        for w in self.config.feature_windows:
            # === PRICE MOMENTUM ===
            features[f"ret_{w}"] = close.pct_change(w)
            features[f"logret_{w}"] = np.log(close / close.shift(w))

            # === MEAN REVERSION (key for scalping) ===
            roll_mean = close.rolling(w).mean()
            roll_std = close.rolling(w).std().replace(0, np.nan)
            features[f"zscore_{w}"] = (close - roll_mean) / roll_std

            # Bollinger band position
            bb_upper = roll_mean + 2 * roll_std
            bb_lower = roll_mean - 2 * roll_std
            bb_width = (bb_upper - bb_lower).replace(0, np.nan)
            features[f"bb_pos_{w}"] = (close - bb_lower) / bb_width

            # Distance from VWAP (mean-reversion anchor)
            vwap = (close * volume).rolling(w).sum() / volume.rolling(w).sum().replace(0, np.nan)
            features[f"vwap_dev_{w}"] = (close - vwap) / vwap

            # === MOMENTUM STRENGTH ===
            # RSI-like: up/down ratio
            changes = close.diff()
            up = changes.clip(lower=0).rolling(w).mean()
            down = (-changes.clip(upper=0)).rolling(w).mean().replace(0, np.nan)
            rsi = up / (up + down)
            features[f"rsi_{w}"] = rsi - 0.5  # Center at 0

            # Rate of change
            features[f"roc_{w}"] = close / close.shift(w) - 1

            # Signed efficiency ratio (trend quality + direction)
            net_move = close - close.shift(w)
            total_path = changes.abs().rolling(w).sum().replace(0, np.nan)
            features[f"signed_eff_{w}"] = net_move / total_path

            # === VOLATILITY ===
            log_ret = np.log(close / close.shift(1))
            features[f"vol_{w}"] = log_ret.rolling(w).std()

            # Volatility expansion/contraction
            if w >= 6:
                short_vol = log_ret.rolling(3).std()
                long_vol = log_ret.rolling(w).std().replace(0, np.nan)
                features[f"vol_ratio_3_{w}"] = short_vol / long_vol

            # ATR normalized
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            features[f"atr_norm_{w}"] = tr.rolling(w).mean() / close

            # === VOLUME MICROSTRUCTURE (unique to scalping) ===
            # Volume spike detection
            vol_mean = volume.rolling(w).mean().replace(0, np.nan)
            features[f"vol_spike_{w}"] = volume / vol_mean - 1

            # Taker buy ratio (buy pressure vs total)
            taker_ratio = taker_buy_vol / volume.replace(0, np.nan)
            features[f"taker_buy_ratio_{w}"] = taker_ratio.rolling(w).mean() - 0.5

            # Trade count momentum (activity surge = incoming volatility)
            trade_mean = trades.rolling(w).mean().replace(0, np.nan)
            features[f"trade_surge_{w}"] = trades / trade_mean - 1

            # Quote volume per trade (avg trade size — institutions vs retail)
            avg_trade_size = quote_vol / trades.replace(0, np.nan)
            avg_trade_mean = avg_trade_size.rolling(w).mean().replace(0, np.nan)
            features[f"avg_trade_size_{w}"] = avg_trade_size / avg_trade_mean - 1

            # Buy volume divergence (price down but buying up = reversal signal)
            price_dir = close.pct_change(w)
            buy_dir = taker_buy_vol.pct_change(w)
            features[f"buy_divergence_{w}"] = buy_dir - price_dir

            # === CROSS-WINDOW ===
            if w > self.config.feature_windows[0]:
                short_w = self.config.feature_windows[0]
                sma_short = close.rolling(short_w).mean()
                sma_long = close.rolling(w).mean()
                features[f"sma_cross_{short_w}_{w}"] = (sma_short - sma_long) / sma_long

        # === CANDLE PATTERNS ===
        body = (close - opn).abs()
        total_range = (high - low).replace(0, np.nan)
        features["body_ratio"] = body / total_range
        features["upper_wick"] = (high - pd.concat([close, opn], axis=1).max(axis=1)) / total_range
        features["lower_wick"] = (pd.concat([close, opn], axis=1).min(axis=1) - low) / total_range
        features["intrabar_dir"] = (close - opn) / opn

        # Doji detection (small body = indecision = potential reversal)
        features["doji"] = (body / total_range < 0.1).astype(float)

        # Consecutive direction
        bar_dir = np.sign(close - opn)
        features["consec_dir_3"] = bar_dir.rolling(3).sum() / 3
        features["consec_dir_5"] = bar_dir.rolling(5).sum() / 5

        # === TIME FEATURES ===
        if hasattr(df.index, "hour"):
            features["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        if hasattr(df.index, "dayofweek"):
            features["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            features["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        self.feature_names = list(features.columns)
        return features

    def normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        w = self.config.rolling_norm_window
        roll_mean = features.rolling(w, min_periods=20).mean()
        roll_std = features.rolling(w, min_periods=20).std().replace(0, np.nan)
        normed = (features - roll_mean) / roll_std
        return normed.clip(-3, 3)

    def select_features(self, features: pd.DataFrame,
                        forward_returns: pd.Series,
                        max_features: int = 15) -> List[str]:
        """Select features by IC score with stability filtering."""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        feat = features.reset_index(drop=True)
        fwd = forward_returns.reset_index(drop=True)
        min_len = min(len(feat), len(fwd))
        feat = feat.iloc[:min_len].dropna(axis=1, how="all")
        fwd = fwd.iloc[:min_len]

        # Compute IC for each feature
        ic_scores = {}
        for col in feat.columns:
            mask = feat[col].notna() & fwd.notna()
            s = feat.loc[mask, col].values
            f = fwd[mask].values
            if len(s) < 100:
                continue
            ic, _ = scipy_stats.spearmanr(s, f)
            if not np.isnan(ic) and abs(ic) > 0.005:
                ic_scores[col] = ic

        candidates = list(ic_scores.keys())
        if len(candidates) < 3:
            return candidates

        # Cluster correlated features
        feat_sub = feat[candidates].dropna()
        if len(feat_sub) < 100:
            return sorted(candidates, key=lambda x: abs(ic_scores[x]), reverse=True)[:max_features]

        corr = feat_sub.corr(method="spearman").abs()
        dist = 1.0 - corr.values
        np.fill_diagonal(dist, 0)
        dist = np.clip((dist + dist.T) / 2, 0, None)
        condensed = squareform(dist, checks=False)

        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=0.3, criterion="distance")

        cluster_map: Dict[int, List[str]] = {}
        for name, label in zip(candidates, labels):
            cluster_map.setdefault(label, []).append(name)

        # Top 2 per cluster by |IC|
        selected = []
        for members in cluster_map.values():
            ranked = sorted(members, key=lambda x: abs(ic_scores[x]), reverse=True)
            selected.extend(ranked[:2])

        selected.sort(key=lambda x: abs(ic_scores[x]), reverse=True)
        selected = selected[:max_features]

        log.info(f"Feature selection: {len(ic_scores)} candidates -> "
                 f"{len(cluster_map)} clusters -> {len(selected)} selected")
        for name in selected[:10]:
            log.info(f"  {name:30s} IC={ic_scores[name]:+.4f}")

        return selected


# =========================================================================
# REGIME DETECTOR — Avoids getting destroyed in trends
# =========================================================================

class RegimeDetector:
    """
    Detects whether the market is CHOPPY (good for mean-reversion)
    or TRENDING (bad — mean-reversion gets killed).

    Uses Efficiency Ratio (ER):
      - ER near 0 = choppy (price goes nowhere despite lots of movement)
      - ER near 1 = strong trend (price moves in one direction consistently)

    Also detects volatility explosions (flash crashes, liquidation cascades)
    where ALL strategies should sit out.
    """

    def __init__(self, config: Config):
        self.config = config

    def compute_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        changes = close.diff().abs()

        regime = pd.DataFrame(index=df.index)

        # --- Efficiency Ratio (24 bars = 2 hours) ---
        for w in [12, 24, 48]:
            net_move = (close - close.shift(w)).abs()
            total_path = changes.rolling(w).sum().replace(0, np.nan)
            er = (net_move / total_path).fillna(0.5)
            regime[f"er_{w}"] = er

        # Primary ER for decisions (2-hour window)
        er = regime["er_24"]

        # --- Trend Filter ---
        # SMA slope: which direction is the market going?
        sma_24 = close.rolling(24).mean()  # 2-hour SMA
        sma_48 = close.rolling(48).mean()  # 4-hour SMA
        sma_slope = (sma_24 - sma_24.shift(6)) / close  # Normalized slope

        # +1 = uptrend, -1 = downtrend, 0 = unclear
        trend = pd.Series(0, index=df.index, dtype=int)
        trend[sma_slope > 0.001] = 1
        trend[sma_slope < -0.001] = -1
        regime["trend"] = trend

        # Stronger trend confirmation: SMA cross
        regime["sma_cross"] = (sma_24 > sma_48).astype(int) * 2 - 1

        # --- Volatility Circuit Breaker ---
        # Compare current volatility to recent average
        log_ret = np.log(close / close.shift(1))
        current_vol = log_ret.rolling(6).std()    # Last 30 min vol
        normal_vol = log_ret.rolling(96).std()     # Last 8 hour vol
        vol_ratio = current_vol / normal_vol.replace(0, np.nan)
        regime["vol_ratio"] = vol_ratio.fillna(1.0)

        # --- Position Scalar ---
        # 1.0 = full size, 0.0 = don't trade
        scalar = pd.Series(1.0, index=df.index)

        # Reduce size in trends but never fully sit out — mean-reversion still works
        scalar[er > 0.60] = 0.5    # Half size in strong trends
        scalar[er > 0.80] = 0.25   # Quarter size in very strong trends (never zero)

        # Kill switch: volatility explosion (3x normal = something is wrong)
        scalar[vol_ratio > 3.0] = 0.10  # Minimal size, don't fully block
        scalar[vol_ratio > 2.5] = np.minimum(scalar[vol_ratio > 2.5], 0.25)

        regime["scalar"] = scalar

        return regime


# =========================================================================
# CROSS-ASSET ENGINE
# =========================================================================

class CrossAssetEngine:
    """BTC leads alts. Use BTC signals to predict ETH/SOL."""

    def generate(self, btc_df: pd.DataFrame, asset_df: pd.DataFrame,
                 all_data: Dict[str, pd.DataFrame], this_asset: str) -> pd.DataFrame:
        features = pd.DataFrame(index=asset_df.index)

        if this_asset != "BTCUSDT" and btc_df is not None:
            btc_close = btc_df["close"].reindex(asset_df.index, method="ffill")
            btc_ret = np.log(btc_close / btc_close.shift(1))

            for lag in [1, 2, 3, 6]:
                features[f"btc_ret_lag{lag}"] = btc_ret.shift(lag)
                features[f"btc_ret4_lag{lag}"] = btc_close.pct_change(4).shift(lag)

            features["btc_mom12"] = btc_close.pct_change(12)
            features["btc_mom24"] = btc_close.pct_change(24)

            # Relative strength vs BTC
            asset_ret = asset_df["close"].pct_change(12)
            features["rel_vs_btc_12"] = asset_ret - btc_close.pct_change(12)

            # BTC volume spike (leads alt moves)
            btc_vol = btc_df["volume"].reindex(asset_df.index, method="ffill")
            btc_vol_mean = btc_vol.rolling(24).mean().replace(0, np.nan)
            features["btc_vol_spike"] = btc_vol / btc_vol_mean - 1

        # Cross-sectional rank
        for w in [6, 12, 24]:
            rets = {}
            for sym, df in all_data.items():
                r = df["close"].pct_change(w).reindex(asset_df.index)
                rets[sym] = r
            rets_df = pd.DataFrame(rets)
            if this_asset in rets_df.columns:
                rank = rets_df.rank(axis=1, pct=True)[this_asset]
                features[f"cs_rank_{w}"] = rank - 0.5

        return features


# =========================================================================
# BACKTESTER (Scalping-specific)
# =========================================================================

class ScalpingBacktester:
    """
    Multi-position bar-by-bar backtester for scalping:
    - Multiple concurrent trades per coin (max_open_trades)
    - Re-entry allowed on same bar after exit
    - Trailing ATR stops, max hold, daily loss limit
    - Leverage-adjusted returns with proper fee accounting
    """

    def __init__(self, config: Config):
        self.config = config

    def run(self, df: pd.DataFrame, signals: pd.Series,
            regime: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        close = df["close"]
        n = len(close)

        signals = signals.reindex(close.index).fillna(0)

        # Regime data (scalar, trend)
        if regime is not None:
            regime = regime.reindex(close.index).ffill()
            regime_scalar = regime["scalar"].fillna(1.0).values
            trend = regime["trend"].fillna(0).values
        else:
            regime_scalar = np.ones(n)
            trend = np.zeros(n)

        # Combined friction: fee + slippage per side
        friction = self.config.fee_rate + self.config.slippage_rate

        # ATR for stops
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.config.atr_period).mean()
        atr = atr.reindex(close.index).ffill().fillna(close * 0.005)

        equity = np.ones(n)
        trades = []
        day_start_equity = 1.0
        current_day = None
        regime_blocks = 0

        # Track multiple open positions as a list of dicts
        open_positions = []  # Each: {direction, size, entry_price, stop_price, best_price, bars_held}

        close_vals = close.values
        atr_vals = atr.values

        for i in range(1, n):
            equity[i] = equity[i - 1]
            bar_date = close.index[i]
            bar_day = bar_date.date() if hasattr(bar_date, "date") else bar_date

            if bar_day != current_day:
                current_day = bar_day
                day_start_equity = equity[i]

            price = close_vals[i]
            prev_price = close_vals[i - 1]
            atr_val = atr_vals[i]

            # Daily loss check — close ALL positions if hit
            daily_pnl = (equity[i] - day_start_equity) / day_start_equity
            if daily_pnl < -self.config.max_daily_loss_pct and open_positions:
                for pos in open_positions:
                    pnl = pos["size"] * (price / pos["entry_price"] - 1) * self.config.leverage
                    if pos["direction"] < 0:
                        pnl = -pnl
                    cost = abs(pos["size"]) * friction
                    equity[i] *= (1 + pnl - cost)
                    trades.append({"bar": i, "reason": "daily_limit", "pnl": pnl - cost})
                open_positions = []
                continue

            # Process all open positions: update stops, check exits, mark-to-market
            surviving = []
            for pos in open_positions:
                pos["bars_held"] += 1

                # Update trailing stop
                if pos["direction"] > 0:
                    pos["best_price"] = max(pos["best_price"], price)
                    trail_stop = pos["best_price"] - self.config.atr_stop_multiplier * atr_val
                    pos["stop_price"] = max(pos["stop_price"], trail_stop)
                else:
                    pos["best_price"] = min(pos["best_price"], price)
                    trail_stop = pos["best_price"] + self.config.atr_stop_multiplier * atr_val
                    pos["stop_price"] = min(pos["stop_price"], trail_stop)

                # Check stop
                stopped = False
                if pos["direction"] > 0 and price < pos["stop_price"]:
                    exit_price = pos["stop_price"]
                    stopped = True
                elif pos["direction"] < 0 and price > pos["stop_price"]:
                    exit_price = price
                    stopped = True

                # Check max hold
                max_hold = pos["bars_held"] >= self.config.max_hold_bars

                if stopped or max_hold:
                    if stopped:
                        ep = exit_price
                        reason = "stop"
                    else:
                        ep = price
                        reason = "max_hold"

                    if pos["direction"] > 0:
                        pnl = pos["size"] * (ep / pos["entry_price"] - 1) * self.config.leverage
                    else:
                        pnl = pos["size"] * (pos["entry_price"] / ep - 1) * self.config.leverage

                    cost = abs(pos["size"]) * friction  # Exit fee
                    equity[i] *= (1 + pnl - cost)
                    trades.append({"bar": i, "reason": reason, "pnl": pnl - cost})
                else:
                    # Mark-to-market (no fee, just equity tracking)
                    if pos["direction"] > 0:
                        pnl = pos["size"] * (price / prev_price - 1) * self.config.leverage
                    else:
                        pnl = pos["size"] * (prev_price / price - 1) * self.config.leverage
                    equity[i] *= (1 + pnl)
                    surviving.append(pos)

            open_positions = surviving

            # Signal-based entry — allowed if under max concurrent positions
            if len(open_positions) < self.config.max_open_trades:
                sig = signals.iloc[i - 1]  # Previous bar's signal (no lookahead)
                rs = regime_scalar[i - 1]
                tr_dir = trend[i - 1]

                # REGIME GATE
                if rs <= 0:
                    regime_blocks += 1
                elif tr_dir > 0 and sig < 0:
                    regime_blocks += 1
                elif tr_dir < 0 and sig > 0:
                    regime_blocks += 1
                else:
                    entered = False
                    if sig > self.config.signal_threshold_long:
                        pos_size = min(sig, 1.0) * self.config.max_position_pct * rs
                        open_positions.append({
                            "direction": 1,
                            "size": pos_size,
                            "entry_price": price,
                            "stop_price": price - self.config.atr_stop_multiplier * atr_val,
                            "best_price": price,
                            "bars_held": 0,
                        })
                        cost = pos_size * friction  # Entry fee
                        equity[i] *= (1 - cost)
                        trades.append({"bar": i, "reason": "entry_long", "size": pos_size})
                        entered = True

                    elif sig < self.config.signal_threshold_short:
                        pos_size = abs(max(sig, -1.0)) * self.config.max_position_pct * rs
                        open_positions.append({
                            "direction": -1,
                            "size": pos_size,
                            "entry_price": price,
                            "stop_price": price + self.config.atr_stop_multiplier * atr_val,
                            "best_price": price,
                            "bars_held": 0,
                        })
                        cost = pos_size * friction  # Entry fee
                        equity[i] *= (1 - cost)
                        trades.append({"bar": i, "reason": "entry_short", "size": pos_size})
                        entered = True

        # Statistics
        eq_series = pd.Series(equity, index=close.index)
        returns = eq_series.pct_change().dropna()

        running_max = eq_series.cummax()
        drawdown = (eq_series - running_max) / running_max
        max_dd = drawdown.min()

        total_return = eq_series.iloc[-1] / eq_series.iloc[0] - 1

        # Annualized Sharpe (5m bars: ~105,120 bars/year)
        bars_per_year = 365.25 * 24 * 12  # 5-min bars per year
        sharpe = (returns.mean() / returns.std() * np.sqrt(bars_per_year)
                  if returns.std() > 0 else 0)

        # Trade stats
        closed_trades = [t for t in trades if "pnl" in t]
        pnls = [t["pnl"] for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_return": total_return,
            "total_return_dollar": total_return * self.config.capital,
            "max_drawdown": max_dd,
            "max_drawdown_dollar": max_dd * self.config.capital,
            "sharpe": sharpe,
            "n_trades": len(closed_trades),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
            "avg_trade": np.mean(pnls) if pnls else 0,
            "regime_blocks": regime_blocks,
            "equity_curve": eq_series,
            "trades": trades,
            "daily_returns": returns,
        }


# =========================================================================
# SIGNAL ENGINE
# =========================================================================

class SignalEngine:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.weights = np.zeros(len(feature_names))

    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    def score(self, features: pd.DataFrame, rolling_window: int = 96) -> pd.Series:
        feat_matrix = features[self.feature_names].values
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)
        raw = feat_matrix @ self.weights

        # Features are already rolling-z-scored by feature_engine.normalize(),
        # so a second rolling normalization here collapses signals to zero as
        # the live buffer becomes homogeneous.  Just clip the raw weighted sum.
        raw_series = pd.Series(raw, index=features.index)
        return raw_series.clip(-1, 1).fillna(0.0).rename("signal")


# =========================================================================
# OPTIMIZER (IC-weighted, same proven approach as ClaudeTrader)
# =========================================================================

class Optimizer:
    def __init__(self, config: Config):
        self.config = config

    def optimize(self, asset_data: Dict[str, Dict],
                 feature_names: List[str]) -> np.ndarray:
        n_features = len(feature_names)
        log.info(f"Optimizing {n_features} feature weights across "
                 f"{len(asset_data)} assets...")

        # IC-weighted approach (robust, doesn't overfit)
        ic_weights = np.zeros(n_features)
        for feat_idx, feat_name in enumerate(feature_names):
            ics = []
            for asset, data in asset_data.items():
                features = data["features"]
                ohlcv = data["ohlcv"]
                split = data["split_idx"]

                train_feat = features[feat_name].iloc[:split]
                h = self.config.prediction_horizon
                fwd_ret = ohlcv["close"].iloc[:split].pct_change(h).shift(-h)

                valid = train_feat.notna() & fwd_ret.notna()
                if valid.sum() < 50:
                    continue

                ic, _ = scipy_stats.spearmanr(
                    train_feat[valid].values, fwd_ret[valid].values
                )
                if not np.isnan(ic):
                    ics.append(ic)

            if ics:
                ic_weights[feat_idx] = np.mean(ics)

        # Normalize
        max_abs = np.max(np.abs(ic_weights))
        if max_abs > 0:
            ic_weights = ic_weights / max_abs

        # Zero out weak weights
        mask = np.abs(ic_weights) < 0.05
        ic_weights[mask] = 0.0
        n_active = np.sum(~mask)
        log.info(f"Active features: {n_active}/{n_features}")

        for name, w in sorted(zip(feature_names, ic_weights),
                               key=lambda x: abs(x[1]), reverse=True):
            if abs(w) > 0.05:
                log.info(f"  {name:30s} w={w:+.4f}")

        return ic_weights


# =========================================================================
# MAIN
# =========================================================================

def main():
    # CLI: python scalper.py [offset_days] [coin_set]
    # coin_set: "default", "alts", "degens", or comma-separated symbols
    offset = 0
    coin_arg = "default"

    for arg in sys.argv[1:]:
        if arg.isdigit():
            offset = int(arg)
        else:
            coin_arg = arg

    COIN_SETS = {
        "default": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "alts":    ["SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"],
        "degens":  ["DOGEUSDT", "PEPEUSDT", "WIFUSDT", "SUIUSDT"],
        "mid":     ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"],
        "top":     ["ETHUSDT", "SOLUSDT", "DOGEUSDT"],
    }

    if coin_arg in COIN_SETS:
        assets = COIN_SETS[coin_arg]
    else:
        assets = [s.strip().upper() for s in coin_arg.split(",")]
        # Ensure USDT suffix
        assets = [s if s.endswith("USDT") else s + "USDT" for s in assets]

    config = Config(end_offset_days=offset, assets=assets)

    period_label = "most recent 90 days" if offset == 0 else f"90 days ending {offset} days ago"
    print("=" * 70)
    print("  Crypto Scalper — 5min Binance Futures Backtest")
    print(f"  Assets: BTC, ETH, SOL | Capital: $10,000 | Leverage: 5x")
    print(f"  Period: {period_label}")
    print("=" * 70)
    print()

    # === 1. FETCH DATA ===
    print("[1/5] Fetching 5-minute data from Binance.US...")
    fetcher = BinanceUSDataFetcher(config)
    ohlcv_data = fetcher.fetch_all()

    if len(ohlcv_data) < 1:
        print("ERROR: No data fetched. Check internet connection.")
        sys.exit(1)

    for sym, df in ohlcv_data.items():
        print(f"  {sym}: {len(df)} bars ({len(df)/12/24:.0f} days)")

    # === 2. GENERATE FEATURES ===
    print("\n[2/5] Generating scalping features...")
    fe = ScalpingFeatureEngine(config)
    cae = CrossAssetEngine()
    btc_df = ohlcv_data.get("BTCUSDT")

    asset_features: Dict[str, pd.DataFrame] = {}
    asset_fwd: Dict[str, pd.Series] = {}

    for sym, df in ohlcv_data.items():
        raw = fe.generate(df)
        normed = fe.normalize(raw)

        # Cross-asset features
        cross = cae.generate(btc_df, df, ohlcv_data, sym)
        cross_normed = fe.normalize(cross)
        combined = pd.concat([normed, cross_normed], axis=1)

        asset_features[sym] = combined
        h = config.prediction_horizon
        asset_fwd[sym] = df["close"].pct_change(h).shift(-h)

        print(f"  {sym}: {len(combined.columns)} features")

    # === 3. SPLIT FIRST, then select features on TRAINING DATA ONLY ===
    # This prevents look-ahead bias — feature selection never sees test data
    print("\n[3/5] Selecting features (training data only — no look-ahead)...")

    # Split each asset into train/test BEFORE feature selection
    train_features = {}
    train_fwd = {}
    optimizer_data = {}
    for sym in ohlcv_data:
        features = asset_features[sym]
        ohlcv = ohlcv_data[sym]
        common = features.index.intersection(ohlcv.index)
        features = features.loc[common]
        ohlcv = ohlcv.loc[common]
        split = int(len(features) * config.train_pct)

        train_features[sym] = features.iloc[:split]
        train_fwd[sym] = asset_fwd[sym].reindex(features.index).iloc[:split]

        optimizer_data[sym] = {
            "features": features,
            "ohlcv": ohlcv,
            "split_idx": split,
        }

    # Feature selection on TRAINING data only
    pooled_train_feat = pd.concat([train_features[s] for s in ohlcv_data], axis=0)
    pooled_train_fwd = pd.concat([train_fwd[s] for s in ohlcv_data], axis=0)

    selected = fe.select_features(pooled_train_feat, pooled_train_fwd,
                                   max_features=config.max_features)

    # Force-include key cross-asset features if available
    forced = ["btc_ret_lag1", "btc_mom12", "cs_rank_12", "rel_vs_btc_12"]
    for f in forced:
        if f in pooled_train_feat.columns and f not in selected:
            selected.append(f)

    # Trim to available
    for sym in asset_features:
        avail = [f for f in selected if f in asset_features[sym].columns]
        asset_features[sym] = asset_features[sym][avail]
    selected = list(asset_features[list(asset_features.keys())[0]].columns)

    # Update optimizer_data with trimmed features
    for sym in ohlcv_data:
        optimizer_data[sym]["features"] = asset_features[sym].loc[optimizer_data[sym]["features"].index.intersection(asset_features[sym].index)]

    print(f"  {len(selected)} features selected")

    # === 4. OPTIMIZE (training data only) ===
    print(f"\n[4/5] Optimizing weights (training data only)...")
    optimizer = Optimizer(config)
    weights = optimizer.optimize(optimizer_data, selected)

    # === 5. BACKTEST ===
    print(f"\n[5/5] Running backtest (test period only)...")
    signal_engine = SignalEngine(selected)
    signal_engine.set_weights(weights)
    backtester = ScalpingBacktester(config)
    regime_detector = RegimeDetector(config)

    all_results = {}
    for sym in ohlcv_data:
        data = optimizer_data[sym]
        features = data["features"]
        ohlcv = data["ohlcv"]
        split = data["split_idx"]

        # TEST PERIOD ONLY (out-of-sample)
        test_features = features.iloc[split:]
        test_ohlcv = ohlcv.iloc[split:]

        signals = signal_engine.score(test_features,
                                       rolling_window=config.rolling_norm_window)
        regime_data = regime_detector.compute_regime(test_ohlcv)
        results = backtester.run(test_ohlcv, signals, regime=regime_data)
        all_results[sym] = results

        test_days = len(test_ohlcv) / (12 * 24)
        print(f"\n  {sym} (test: {test_days:.0f} days):")
        print(f"    Return: {results['total_return']:+.2%} "
              f"(${results['total_return_dollar']:+,.0f})")
        print(f"    Max DD: {results['max_drawdown']:.2%} "
              f"(${results['max_drawdown_dollar']:,.0f})")
        print(f"    Sharpe: {results['sharpe']:.2f}")
        print(f"    Trades: {results['n_trades']}")
        print(f"    Win rate: {results['win_rate']:.1%}")
        print(f"    Avg win: {results['avg_win']:.4%} | "
              f"Avg loss: {results['avg_loss']:.4%}")
        print(f"    Profit factor: {results['profit_factor']:.2f}")
        print(f"    Avg trade: {results['avg_trade']:.4%}")
        print(f"    Regime blocks: {results['regime_blocks']} "
              f"(entries blocked by regime/trend filter)")

    # === PORTFOLIO SUMMARY ===
    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY (Out-of-Sample)")
    print("=" * 70)

    total_dollar_return = sum(r["total_return_dollar"] for r in all_results.values())
    avg_return = np.mean([r["total_return"] for r in all_results.values()])
    worst_dd = min(r["max_drawdown"] for r in all_results.values())
    total_trades = sum(r["n_trades"] for r in all_results.values())
    avg_winrate = np.mean([r["win_rate"] for r in all_results.values()])
    avg_sharpe = np.mean([r["sharpe"] for r in all_results.values()])

    print(f"\n  Avg return per asset: {avg_return:+.2%}")
    print(f"  Est. portfolio P&L:   ${total_dollar_return/len(all_results):+,.0f} "
          f"(equal-weight, $10k)")
    print(f"  Worst drawdown:       {worst_dd:.2%}")
    print(f"  Avg Sharpe:           {avg_sharpe:.2f}")
    print(f"  Total trades:         {total_trades}")
    print(f"  Avg win rate:         {avg_winrate:.1%}")

    # Profitability assessment
    print("\n" + "-" * 70)
    if avg_return > 0 and avg_sharpe > 0.5:
        print("VERDICT: Positive out-of-sample edge detected.")
        print("  Next step: paper trade for 2 weeks to validate in real-time.")
    elif avg_return > 0:
        print("VERDICT: Marginal edge. Sharpe is low — may not survive live fees.")
        print("  Consider: wider stops, longer hold period, or fewer trades.")
    else:
        print("VERDICT: No edge found in this period.")
        print("  The strategy lost money out-of-sample after fees.")
        print("  This is honest — most short-term strategies fail.")

    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
