"""
Hyperliquid Testnet Trader V6 — Optimized for live HL execution
===============================================================================
V6 improvements over V5 (from backtest sweep + live trading analysis):
  - Removed taker_buy_ratio features (HL estimates at 0.5*vol = garbage data)
  - Wider ATR stops: 2.5x (was 1.5x — trades were getting stopped on noise)
  - Dropped ETH from trading (consistently negative in all configs)
  - Dynamic per-coin momentum allocation (ride winners, cut losers)
  - Server-side stop-loss safety net on Hyperliquid

Architecture:
  FROZEN: Signal model loaded from frozen_weights.json (11 months, L2-normalized)
  DYNAMIC: Position sizing (per-coin momentum multiplier 0.25x-1.5x)

Usage:
    python hyperliquid_trader.py                  # Default: ADA/DOGE/ETH
    python hyperliquid_trader.py SOL,BTC,ETH      # Custom coins
"""

import sys
import os
import json
import time
import threading
import signal as sig_module
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scalper import (
    Config, BinanceUSDataFetcher, ScalpingFeatureEngine,
    CrossAssetEngine, RegimeDetector, SignalEngine, Optimizer,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HLTrader")

# Suppress noisy SDK logs
logging.getLogger("hyperliquid").setLevel(logging.WARNING)


# =========================================================================
# CONFIG
# =========================================================================

CONFIG_FILE = Path("hl_config.json")
STATE_FILE = Path("hl_state.json")
TRADE_LOG = Path("hl_trades.json")
LOG_FILE = Path("hl_trader.log")

# Map internal symbols to Hyperliquid perp names
# Hyperliquid uses short names like "BTC", "ETH", "SOL" etc.
# NOTE: Testnet has fewer coins than mainnet. Some use "k" prefix (kPEPE, kBONK, kSHIB).
# Verified available on testnet: BTC, ETH, SOL, ADA, DOGE, AVAX, SUI, BNB,
#   HBAR, ATOM, XLM, TRUMP, HYPE, kPEPE, kBONK, kSHIB
# NOT on testnet: LINK, XRP, DOT
HL_SYMBOL_MAP = {
    "BTCUSDT": "BTC",   "ETHUSDT": "ETH",   "SOLUSDT": "SOL",
    "DOGEUSDT": "DOGE",  "ADAUSDT": "ADA",   "AVAXUSDT": "AVAX",
    "SUIUSDT": "SUI",    "BNBUSDT": "BNB",   "HYPEUSDT": "HYPE",
    "HBARUSDT": "HBAR",  "ATOMUSDT": "ATOM",  "XLMUSDT": "XLM",
    "TRUMPUSDT": "TRUMP",
    "PEPEUSDT": "kPEPE",  "BONKUSDT": "kBONK", "SHIBUSDT": "kSHIB",
}

# Coins blocked from trading (volatility too low, ATR within noise)
BLOCKED_COINS = {"HBARUSDT"}

# Cooldown after stop-out: don't re-enter for this many bars (6 bars = 30 min)
COOLDOWN_BARS = 2


# =========================================================================
# WALLET SETUP
# =========================================================================

# =========================================================================
# HYPERLIQUID HISTORICAL DATA FETCHER
# =========================================================================

class HyperliquidDataFetcher:
    """Fetch historical OHLCV from Hyperliquid testnet API (no key needed)."""

    BASE_URL = "https://api.hyperliquid-testnet.xyz/info"

    def __init__(self, config: Config):
        self.config = config

    def fetch(self, symbol: str) -> pd.DataFrame:
        import requests

        hl_coin = HL_SYMBOL_MAP.get(symbol)
        if not hl_coin:
            raise ValueError(f"No HL mapping for {symbol}")

        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (self.config.lookback_days * 86_400_000)

        log.info(f"Fetching {symbol} ({hl_coin}) {self.config.interval} from Hyperliquid testnet "
                 f"({self.config.lookback_days} days)...")

        all_candles = []
        cursor = start_ms

        while cursor < end_ms:
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": hl_coin,
                    "interval": self.config.interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                },
            }
            resp = requests.post(self.BASE_URL, json=payload, timeout=15)
            resp.raise_for_status()
            candles = resp.json()

            if not candles:
                break

            all_candles.extend(candles)
            # Move cursor past last candle
            cursor = int(candles[-1]["t"]) + 1
            if len(candles) < 5000:
                break

            time.sleep(0.1)

        if not all_candles:
            raise ValueError(f"No data returned for {symbol} ({hl_coin})")

        rows = []
        for c in all_candles:
            close_f = float(c["c"])
            vol_f = float(c["v"])
            rows.append({
                "open_time": pd.Timestamp(c["t"], unit="ms"),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": close_f,
                "volume": vol_f,
                "quote_volume": close_f * vol_f,
                "trades": int(c.get("n", 0)),
                "taker_buy_volume": vol_f * 0.5,       # estimated
                "taker_buy_quote_volume": close_f * vol_f * 0.5,
            })

        df = pd.DataFrame(rows)
        df.set_index("open_time", inplace=True)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        log.info(f"  {symbol}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
        return df

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for symbol in self.config.assets:
            try:
                data[symbol] = self.fetch(symbol)
            except Exception as e:
                log.error(f"Failed to fetch {symbol}: {e}")
        return data


# =========================================================================
# WALLET SETUP
# =========================================================================

def load_wallet():
    """Load wallet from env var or config file."""
    import eth_account

    # Try env var first
    key = os.environ.get("HL_PRIVATE_KEY")

    # Try config file
    if not key and CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            key = cfg.get("private_key")
        except Exception:
            pass

    if not key:
        # Generate new wallet and save
        acct = eth_account.Account.create()
        key = acct.key.hex()
        cfg = {
            "private_key": key,
            "wallet_address": acct.address,
            "note": "Fund this address at https://app.hyperliquid-testnet.xyz/drip",
        }
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        print(f"\n  NEW WALLET GENERATED:")
        print(f"  Address:     {acct.address}")
        print(f"  Private key: saved to {CONFIG_FILE}")
        print(f"\n  Fund it at: https://app.hyperliquid-testnet.xyz/drip")
        print(f"  Or:          https://faucet.quicknode.com/hyperliquid/testnet\n")
        return eth_account.Account.from_key(key)

    if not key.startswith("0x"):
        key = "0x" + key

    wallet = eth_account.Account.from_key(key)
    print(f"  Wallet loaded: {wallet.address}")
    return wallet


# =========================================================================
# HYPERLIQUID EXECUTION LAYER
# =========================================================================

class HyperliquidExecutor:
    """Handles all communication with Hyperliquid testnet."""

    def __init__(self, wallet, leverage: int = 5):
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.api import API
        from hyperliquid.utils.constants import TESTNET_API_URL

        self.base_url = TESTNET_API_URL
        self.wallet = wallet
        self.address = wallet.address
        self.leverage = leverage

        log.info(f"Connecting to Hyperliquid testnet...")

        # Fetch perp meta first, then pass it to Info to avoid spot metadata crash on testnet
        api = API(TESTNET_API_URL)
        meta = api.post("/info", {"type": "meta"})

        # Build a minimal spot_meta to prevent the SDK from fetching (and crashing on) real spot data
        minimal_spot_meta = {"universe": [], "tokens": []}

        self.info = Info(self.base_url, skip_ws=True, meta=meta, spot_meta=minimal_spot_meta)
        self.exchange = Exchange(wallet, self.base_url, meta=meta, spot_meta=minimal_spot_meta)

        # Verify connection + get account state
        try:
            equity = self.get_equity()
            log.info(f"  Connected! Account equity: ${equity:,.2f}")
            if equity == 0:
                log.warning(f"  Account has $0 — fund it first!")
                log.warning(f"  Address: {self.address}")
                log.warning(f"  Faucet:  https://app.hyperliquid-testnet.xyz/drip")
        except Exception as e:
            log.error(f"  Connection failed: {e}")
            raise

        # Set leverage for all traded coins
        self._leverage_set = set()

    def _ensure_leverage(self, hl_coin: str):
        """Set leverage for a coin (only once per session)."""
        if hl_coin in self._leverage_set:
            return
        try:
            self.exchange.update_leverage(self.leverage, hl_coin, is_cross=True)
            self._leverage_set.add(hl_coin)
            log.info(f"  Set {hl_coin} leverage to {self.leverage}x cross")
        except Exception as e:
            log.warning(f"  Failed to set leverage for {hl_coin}: {e}")

    def get_account_state(self) -> dict:
        """Get full account state: equity, positions, margin.
        Retries on transient connection drops (HL testnet occasionally resets sessions)."""
        import requests as _req
        for attempt in range(3):
            try:
                return self.info.user_state(self.address)
            except Exception as e:
                if attempt == 2:
                    raise
                log.warning(f"get_account_state attempt {attempt+1} failed ({e}) — retrying...")
                time.sleep(1 + attempt)

    def _get_spot_usdc(self) -> Tuple[float, float]:
        """Get USDC balance from spot (for unified accounts).
        Returns (total, hold) where hold = margin locked for perps."""
        try:
            import requests
            r = requests.post(f"{self.base_url}/info",
                json={"type": "spotClearinghouseState", "user": self.address}, timeout=10)
            data = r.json()
            for bal in data.get("balances", []):
                if bal["coin"] == "USDC":
                    return float(bal["total"]), float(bal.get("hold", 0))
        except Exception:
            pass
        return 0.0, 0.0

    def get_equity(self) -> float:
        """Compute total equity for unified account.
        equity = (spot_usdc_total - spot_hold) + perp_accountValue
        When no position: hold=0, accountValue=0, so equity = spot_total.
        When position open: hold = margin, accountValue = margin + unrealized PnL."""
        state = self.get_account_state()
        perp_value = float(state["marginSummary"]["accountValue"])
        spot_total, spot_hold = self._get_spot_usdc()
        return (spot_total - spot_hold) + perp_value

    def get_positions(self) -> List[dict]:
        """Get all open positions."""
        state = self.get_account_state()
        positions = []
        for ap in state.get("assetPositions", []):
            pos = ap["position"]
            szi = float(pos["szi"])
            if szi == 0:
                continue
            positions.append({
                "coin": pos["coin"],
                "size": szi,
                "side": "long" if szi > 0 else "short",
                "entry_price": float(pos.get("entryPx", 0)),
                "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                "margin_used": float(pos.get("marginUsed", 0)),
                "liquidation_px": float(pos["liquidationPx"]) if pos.get("liquidationPx") else None,
            })
        return positions

    def get_mid_price(self, hl_coin: str) -> Optional[float]:
        """Get current mid price for a coin."""
        try:
            mids = self.info.all_mids()
            return float(mids.get(hl_coin, 0))
        except Exception:
            return None

    @staticmethod
    def _extract_fill(result: dict) -> Optional[Tuple[float, float]]:
        """Extract (avgPx, totalSz) from exchange order result."""
        try:
            statuses = result["response"]["data"]["statuses"]
            fill = statuses[0].get("filled")
            if fill:
                return float(fill["avgPx"]), float(fill["totalSz"])
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def market_open(self, hl_coin: str, is_buy: bool, size: float) -> dict:
        """Open a position with a market order."""
        self._ensure_leverage(hl_coin)

        # Round size to proper decimals for this asset
        sz_decimals = self.info.asset_to_sz_decimals.get(
            self.info.coin_to_asset.get(hl_coin, -1), 3
        )
        size = round(size, sz_decimals)

        if size <= 0:
            return {"status": "error", "msg": "Size too small after rounding"}

        try:
            result = self.exchange.market_open(hl_coin, is_buy, size, slippage=0.01)
            log.info(f"  ORDER: {'BUY' if is_buy else 'SELL'} {size} {hl_coin} | result: {result}")
            fill = self._extract_fill(result)
            return {"status": "ok", "result": result,
                    "fill_price": fill[0] if fill else None,
                    "fill_size": fill[1] if fill else None}
        except Exception as e:
            log.error(f"  ORDER FAILED: {e}")
            return {"status": "error", "msg": str(e)}

    def market_close(self, hl_coin: str, size: Optional[float] = None) -> dict:
        """Close a position with a market order."""
        try:
            result = self.exchange.market_close(hl_coin, sz=size, slippage=0.01)
            log.info(f"  CLOSE: {hl_coin} size={size} | result: {result}")
            fill = self._extract_fill(result)
            return {"status": "ok", "result": result,
                    "fill_price": fill[0] if fill else None,
                    "fill_size": fill[1] if fill else None}
        except Exception as e:
            log.error(f"  CLOSE FAILED: {e}")
            return {"status": "error", "msg": str(e)}

    def get_open_orders(self) -> List[dict]:
        return self.info.open_orders(self.address)

    def cancel_all_orders(self, hl_coin: str):
        """Cancel all open orders for a coin."""
        orders = self.get_open_orders()
        for order in orders:
            if order["coin"] == hl_coin:
                try:
                    self.exchange.cancel(hl_coin, order["oid"])
                except Exception:
                    pass

    def place_stop_loss(self, hl_coin: str, is_buy: bool, size: float,
                        stop_price: float) -> dict:
        """Place a server-side stop-loss order on Hyperliquid.
        This protects the position even if the bot goes offline.
        Args:
            is_buy: True to buy (close a short), False to sell (close a long)
            size: position size in coins
            stop_price: trigger price
        """
        sz_decimals = self.info.asset_to_sz_decimals.get(
            self.info.coin_to_asset.get(hl_coin, -1), 3
        )
        size = round(size, sz_decimals)
        if size <= 0:
            return {"status": "error", "msg": "Size too small"}

        # Round price to 5 significant figures to satisfy HL's float_to_wire
        stop_price = float(f"{stop_price:.5g}")

        try:
            result = self.exchange.order(
                name=hl_coin,
                is_buy=is_buy,
                sz=size,
                limit_px=stop_price,
                order_type={"trigger": {
                    "triggerPx": stop_price,
                    "isMarket": True,
                    "tpsl": "sl",
                }},
                reduce_only=True,
            )
            log.info(f"  STOP ORDER: {hl_coin} trigger @ ${stop_price:,.4f} | result: {result}")
            return {"status": "ok", "result": result}
        except Exception as e:
            log.error(f"  STOP ORDER FAILED: {hl_coin} @ ${stop_price:,.4f} — {e}")
            return {"status": "error", "msg": str(e)}

    def update_stop_loss(self, hl_coin: str, is_buy: bool, size: float,
                         new_stop_price: float) -> dict:
        """Cancel existing stop orders for this coin and place a new one."""
        self.cancel_all_orders(hl_coin)
        return self.place_stop_loss(hl_coin, is_buy, size, new_stop_price)


# =========================================================================
# BINANCE.US WEBSOCKET STREAM — Real-time 5m candles, push-based
#
# Why Binance.US over Hyperliquid's own candle feed:
#   - HL candle API has NO taker_buy_volume — it estimates 0.5*vol (useless)
#   - Binance.US provides REAL taker_buy_volume from actual trade flow
#   - Both use UTC-aligned 5m intervals → candles close at the same instant
#   - HL perp prices track Binance spot within milliseconds (arbitrage)
#   - WebSocket push (not polling) → signal fires the moment the bar closes
# =========================================================================

# Internal USDT symbol -> Binance.US USD symbol (already defined in scalper.py
# as BINANCEUS_SYMBOL_MAP — mirrored here for the WebSocket symbol mapping)
_BUS_SYMBOL_MAP = {v: k for k, v in {
    "BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
    "DOGEUSDT": "DOGEUSD", "ADAUSDT": "ADAUSD", "SUIUSDT": "SUIUSD",
    "AVAXUSDT": "AVAXUSD", "LINKUSDT": "LINKUSD", "PEPEUSDT": "PEPEUSD",
    "WIFUSDT": "WIFUSD", "BNBUSDT": "BNBUSD", "XRPUSDT": "XRPUSD",
    "HYPEUSDT": "HYPEUSD", "HBARUSDT": "HBARUSD", "DOTUSDT": "DOTUSD",
    "SHIBUSDT": "SHIBUSD", "ATOMUSDT": "ATOMUSD", "XLMUSDT": "XLMUSD",
    "BONKUSDT": "BONKUSD", "TRUMPUSDT": "TRUMPUSD",
}.items()}  # "BTCUSD" -> "BTCUSDT"


class BinanceUSWebSocketStream:
    """
    Real-time 5m OHLCV via Binance.com WebSocket kline stream (primary).

    Binance.com advantages over Binance.US:
    - Full liquidity on ADA/DOGE/ETH/BTC USDT pairs (vs near-zero on Binance.US)
    - Real taker_buy_volume (field V) — all volume features fully populated
    - Same USDT symbols as HL perps — zero symbol translation needed
    - Push-based: candle close event arrives within ~50ms of the bar closing
    - Auto-reconnects on disconnect with exponential backoff
    """

    WS_BASE = "wss://stream.binance.com"
    REST_BASE = "https://api.binance.com/api/v3"

    def __init__(self, symbols: List[str], on_candle_close):
        self.symbols = symbols          # Internal symbols: ["ADAUSDT", "DOGEUSDT", ...]
        self.on_candle_close = on_candle_close
        self._running = False
        self._ws = None
        self._thread = None
        self._reconnect_delay = 5       # seconds, doubles on repeated failures

    def _to_stream(self, internal: str) -> str:
        """ADAUSDT -> adausdt  (Binance.com lowercase stream name, USDT pairs)"""
        return internal.lower()

    def _build_url(self) -> str:
        streams = "/".join(f"{self._to_stream(s)}@kline_5m" for s in self.symbols)
        return f"{self.WS_BASE}/stream?streams={streams}"

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # Combined stream format: {"stream": "adausdt@kline_5m", "data": {...}}
            payload = data.get("data", data)
            k = payload.get("k")
            if not k:
                return

            if not k.get("x", False):
                return  # Candle still forming — ignore mid-bar updates

            internal = k["s"]  # "ADAUSDT" — already our internal symbol, no mapping needed
            if internal not in self.symbols:
                return

            # Standard candle dict expected by _append_candle()
            candle = {
                "t": k["t"],    # open time ms
                "o": k["o"],    # open
                "h": k["h"],    # high
                "l": k["l"],    # low
                "c": k["c"],    # close
                "v": k["v"],    # base volume (REAL)
                "q": k["q"],    # quote volume / USDT volume (REAL)
                "n": k["n"],    # trade count (REAL)
                "V": k["V"],    # taker buy base volume (REAL — not 0.5*vol)
                "Q": k["Q"],    # taker buy quote volume (REAL)
                "s": internal,
                "x": True,
            }
            self.on_candle_close(internal, candle)

        except Exception as e:
            log.error(f"WebSocket message parse error: {e}")

    def _on_error(self, ws, error):
        log.error(f"Binance.US WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        log.warning(f"Binance.com WebSocket closed (code={close_status_code}) — "
                    f"reconnecting in {self._reconnect_delay}s...")
        if self._running:
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 60)
            self._connect()

    def _on_open(self, ws):
        self._reconnect_delay = 5  # Reset backoff on successful connect
        symbols_str = ", ".join(self._to_stream(s).upper() for s in self.symbols)
        log.info(f"Binance.com WebSocket CONNECTED — 5m klines: {symbols_str}")

    def _connect(self):
        import websocket as ws_lib
        url = self._build_url()
        log.info(f"Connecting to Binance.com WebSocket: {url}")
        self._ws = ws_lib.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def start(self) -> bool:
        """Test REST connectivity, then start WebSocket thread."""
        import requests
        sym = self._to_stream(self.symbols[0]).upper()
        try:
            resp = requests.get(f"{self.REST_BASE}/ticker/price",
                                params={"symbol": sym}, timeout=8)
            resp.raise_for_status()
            price = float(resp.json()["price"])
            print(f"  Binance.com WebSocket: {sym} @ ${price:,.4f} — ready")
        except Exception as e:
            print(f"  Binance.com REST check failed ({sym}): {e}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._connect, daemon=True)
        self._thread.start()

        # Give it a moment to establish the connection before returning
        time.sleep(2)
        return True

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass


# =========================================================================
# HYPERLIQUID TRADER — Plugs V4 strategy into real testnet execution
# =========================================================================

class HyperliquidTrader:
    """
    Same V4 scalper logic, but:
    - Opens/closes positions on Hyperliquid testnet
    - Reads equity/positions from the exchange
    - Tracks local state for stop management (HL doesn't have native trailing stops)
    """

    def __init__(self, config: Config, signal_engine: SignalEngine,
                 regime_detector: RegimeDetector, feature_engine: ScalpingFeatureEngine,
                 cross_engine: CrossAssetEngine, feature_names: List[str],
                 executor: HyperliquidExecutor):
        self.config = config
        self.signal_engine = signal_engine
        self.regime_detector = regime_detector
        self.feature_engine = feature_engine
        self.cross_engine = cross_engine
        self.feature_names = feature_names
        self.executor = executor

        self.friction = config.fee_rate + config.slippage_rate

        # Local stop tracking (HL doesn't do trailing stops natively)
        # Key: hl_coin -> {stop_price, best_price, entry_price, side, bars_held, entry_time, size_usd}
        self.tracked_positions: Dict[str, dict] = {}

        # Candle buffers (need ~200 bars for features + rolling norm)
        self.candle_buffers: Dict[str, pd.DataFrame] = {}
        self.buffer_size = 300

        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.starting_equity = executor.get_equity()
        self.peak_equity = self.starting_equity
        self.day_start_equity = self.starting_equity
        self.current_day = None

        # Trade log
        self.trades: List[dict] = []

        # Dynamic allocation: per-coin momentum tracking
        self.coin_pnl_history: Dict[str, List[float]] = {}
        self.momentum_lookback = 10       # Recent trades to consider
        self.momentum_max_mult = 1.5      # Hot streak ceiling
        self.momentum_min_mult = 0.25     # Bleeding floor

        # Cooldown tracking: bars remaining before re-entry allowed after stop-out
        self.stop_cooldown: Dict[str, int] = {}

        # Load saved state
        self._load_state()

        # Fix: if saved starting_equity is 0, re-fetch from exchange
        if self.starting_equity <= 0:
            self.starting_equity = self.executor.get_equity()

    def _save_state(self):
        state = {
            "tracked_positions": self.tracked_positions,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "starting_equity": self.starting_equity,
            "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "coin_pnl_history": self.coin_pnl_history,
            "stop_cooldown": self.stop_cooldown,
            "saved_at": datetime.utcnow().isoformat(),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text())
            self.tracked_positions = state.get("tracked_positions", {})
            self.total_trades = state.get("total_trades", 0)
            self.winning_trades = state.get("winning_trades", 0)
            self.total_pnl = state.get("total_pnl", 0.0)
            self.starting_equity = state.get("starting_equity", self.starting_equity)
            self.peak_equity = state.get("peak_equity", self.peak_equity)
            self.day_start_equity = state.get("day_start_equity", self.day_start_equity)
            self.coin_pnl_history = state.get("coin_pnl_history", {})
            self.stop_cooldown = state.get("stop_cooldown", {})
            if self.tracked_positions:
                print(f"  Resumed: tracking {len(self.tracked_positions)} positions from saved state")
                # Re-place server-side stops for all tracked positions (safety net on restart)
                self._sync_server_stops()
        except Exception as e:
            print(f"  Warning: could not load state: {e}")

    def _sync_server_stops(self):
        """Place server-side stop-loss orders for all tracked positions.
        Called on startup to ensure protection even if bot crashed mid-session."""
        for hl_coin, tp in self.tracked_positions.items():
            try:
                pos_size = tp["size_usd"] / tp["entry_price"]
                is_buy = tp["side"] == "short"  # buy to close short, sell to close long
                self.executor.update_stop_loss(
                    hl_coin, is_buy=is_buy, size=pos_size,
                    new_stop_price=tp["stop_price"])
                print(f"  Server stop synced: {hl_coin} {tp['side'].upper()} "
                      f"@ ${tp['stop_price']:,.4f}")
            except Exception as e:
                print(f"  WARNING: Failed to sync stop for {hl_coin}: {e}")

    def _log_trade(self, trade: dict):
        self.trades.append(trade)
        log_data = []
        if TRADE_LOG.exists():
            try:
                log_data = json.loads(TRADE_LOG.read_text())
            except Exception:
                pass
        log_data.append(trade)
        TRADE_LOG.write_text(json.dumps(log_data, indent=2))

    def warmup_buffers(self, ohlcv_data: Dict[str, pd.DataFrame]):
        for sym, df in ohlcv_data.items():
            self.candle_buffers[sym] = df.tail(self.buffer_size).copy()
            print(f"  {sym}: {len(self.candle_buffers[sym])} bars loaded")

    def _append_candle(self, symbol: str, candle: dict) -> bool:
        ts = pd.Timestamp(candle["t"], unit="ms")
        new_row = pd.DataFrame([{
            "open": float(candle["o"]),
            "high": float(candle["h"]),
            "low": float(candle["l"]),
            "close": float(candle["c"]),
            "volume": float(candle["v"]),
            "quote_volume": float(candle["q"]),
            "trades": int(candle["n"]),
            "taker_buy_volume": float(candle["V"]),
            "taker_buy_quote_volume": float(candle["Q"]),
        }], index=[ts])

        if symbol not in self.candle_buffers:
            self.candle_buffers[symbol] = new_row
        else:
            buf = self.candle_buffers[symbol]
            if ts in buf.index:
                return False
            self.candle_buffers[symbol] = pd.concat([buf, new_row]).tail(self.buffer_size)
        return True

    def _compute_atr(self, df: pd.DataFrame) -> float:
        close = df["close"]
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.config.atr_period).mean()
        return atr.iloc[-1] if len(atr) > 0 else close.iloc[-1] * 0.005

    def _get_momentum_multiplier(self, hl_coin: str) -> float:
        """Per-coin momentum sizing: ride winners, cut losers.
        Uses win rate over last N trades mapped to [0.25x, 1.5x]."""
        history = self.coin_pnl_history.get(hl_coin, [])
        if len(history) < 3:
            return 1.0  # Not enough data, use neutral

        recent = history[-self.momentum_lookback:]
        win_rate = sum(1 for p in recent if p > 0) / len(recent)

        # Map win_rate [0,1] -> momentum_score [-1, +1] centered at 0.5
        score = (win_rate - 0.5) * 2

        if score >= 0:
            mult = 1.0 + score * (self.momentum_max_mult - 1.0)
        else:
            mult = 1.0 + score * (1.0 - self.momentum_min_mult)

        return np.clip(mult, self.momentum_min_mult, self.momentum_max_mult)

    def _get_atr_multiplier(self, hl_coin: str) -> float:
        """Dynamic ATR stop multiplier: tighten stops for losing coins, keep wide for winners.
        Range: 1.2x (cold streak) to 2.5x (hot streak). Default 2.5x until 3 trades."""
        history = self.coin_pnl_history.get(hl_coin, [])
        if len(history) < 3:
            return self.config.atr_stop_multiplier  # No data yet — use default

        recent = history[-self.momentum_lookback:]
        win_rate = sum(1 for p in recent if p > 0) / len(recent)

        # Map win_rate [0,1] to ATR multiplier [1.2, 2.5]
        atr_min, atr_max = 1.2, self.config.atr_stop_multiplier
        mult = atr_min + win_rate * (atr_max - atr_min)
        return round(mult, 2)

    def _get_signal_and_regime(self, symbol: str):
        buf = self.candle_buffers.get(symbol)
        if buf is None or len(buf) < 100:
            return 0.0, 0.0, 0, 0.0

        raw = self.feature_engine.generate(buf)
        normed = self.feature_engine.normalize(raw)

        all_data = self.candle_buffers
        btc_df = all_data.get("BTCUSDT")
        cross = self.cross_engine.generate(btc_df, buf, all_data, symbol)
        cross_normed = self.feature_engine.normalize(cross)
        combined = pd.concat([normed, cross_normed], axis=1)

        avail = [f for f in self.feature_names if f in combined.columns]
        combined = combined[avail]

        signal = self.signal_engine.score(combined, rolling_window=self.config.rolling_norm_window)
        regime = self.regime_detector.compute_regime(buf)

        latest_signal = signal.iloc[-1] if len(signal) > 0 else 0.0
        latest_scalar = regime["scalar"].iloc[-1] if len(regime) > 0 else 1.0
        latest_trend = int(regime["trend"].iloc[-1]) if len(regime) > 0 else 0
        atr = self._compute_atr(buf)

        return latest_signal, latest_scalar, latest_trend, atr

    def _sync_positions(self):
        """Sync local tracking with actual exchange positions."""
        hl_positions = self.executor.get_positions()
        hl_coins_with_pos = {p["coin"] for p in hl_positions}

        # Remove tracked positions that no longer exist on exchange
        # (manually closed via web UI, or liquidated)
        for coin in list(self.tracked_positions.keys()):
            if coin not in hl_coins_with_pos:
                tp = self.tracked_positions.pop(coin)
                print(f"  Position {coin} {tp['side']} no longer on exchange — removed from tracking")

    def _close_position_on_exchange(self, hl_coin: str, reason: str):
        """Close a position on Hyperliquid and log the trade."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        tp = self.tracked_positions.get(hl_coin)

        result = self.executor.market_close(hl_coin)

        if result["status"] == "ok" and tp:
            # Use actual fill price from exchange, not mid price
            exit_price = result.get("fill_price") or self.executor.get_mid_price(hl_coin) or tp.get("entry_price", 0)
            entry_price = tp.get("entry_price", 0)
            size_usd = tp.get("size_usd", 0)

            if entry_price > 0:
                if tp["side"] == "long":
                    raw_pnl_pct = (exit_price / entry_price - 1) * self.config.leverage
                else:
                    raw_pnl_pct = (entry_price / exit_price - 1) * self.config.leverage

                # Subtract fees: taker fee on entry + exit, applied to notional
                # HL testnet fees: 0.045% taker / 0.015% maker (we use market orders = taker)
                fee_pct = 2 * 0.00045 * self.config.leverage  # entry + exit fees on leveraged notional
                pnl_pct = raw_pnl_pct - fee_pct
                dollar_pnl = pnl_pct * (size_usd / self.config.leverage)  # P&L on margin
            else:
                pnl_pct = 0
                dollar_pnl = 0

            self.total_trades += 1
            if pnl_pct > 0:
                self.winning_trades += 1
            self.total_pnl += dollar_pnl

            # Track per-coin P&L for momentum allocation
            if hl_coin not in self.coin_pnl_history:
                self.coin_pnl_history[hl_coin] = []
            self.coin_pnl_history[hl_coin].append(pnl_pct)

            trade_record = {
                "time": now,
                "coin": hl_coin,
                "side": tp["side"],
                "entry": entry_price,
                "exit": exit_price,
                "pnl_pct": f"{pnl_pct:.4%}",
                "pnl_dollar": f"${dollar_pnl:+,.2f}",
                "fees_pct": f"{fee_pct:.4%}",
                "reason": reason,
                "bars_held": tp.get("bars_held", 0),
            }
            self._log_trade(trade_record)

            sign = "+" if pnl_pct > 0 else ""
            print(f"  EXIT {hl_coin} {tp['side'].upper()} | {reason} | "
                  f"fill: ${exit_price:,.4f} | PnL: {sign}${dollar_pnl:,.2f} ({pnl_pct:.3%})")

            self.tracked_positions.pop(hl_coin, None)
        elif result["status"] == "error":
            print(f"  EXIT FAILED for {hl_coin}: {result['msg']}")

    def process_candle(self, symbol: str, candle: dict):
        """Process a newly closed 5m candle — check exits, then entries."""
        is_new = self._append_candle(symbol, candle)
        if not is_new:
            return

        hl_coin = HL_SYMBOL_MAP.get(symbol)
        if not hl_coin:
            return

        close = float(candle["c"])
        high = float(candle["h"])
        low = float(candle["l"])
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        # Day tracking
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day = today
            self.day_start_equity = self.executor.get_equity()

        # Update peak equity
        current_equity = self.executor.get_equity()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # --- Daily loss check ---
        daily_pnl = (current_equity - self.day_start_equity) / self.day_start_equity if self.day_start_equity > 0 else 0
        if daily_pnl < -self.config.max_daily_loss_pct and self.tracked_positions:
            print(f"  DAILY LOSS LIMIT HIT ({daily_pnl:.2%}) — closing all positions")
            for coin in list(self.tracked_positions.keys()):
                self.executor.cancel_all_orders(coin)  # Cancel server-side stops first
                self._close_position_on_exchange(coin, "daily_limit")
            self._save_state()
            return

        # Compute signal once for both exit TP and entry decisions
        signal, regime_scalar, trend, atr = self._get_signal_and_regime(symbol)

        # --- DECREMENT COOLDOWN ---
        if hl_coin in self.stop_cooldown:
            self.stop_cooldown[hl_coin] -= 1
            if self.stop_cooldown[hl_coin] <= 0:
                del self.stop_cooldown[hl_coin]
                log.info(f"  {hl_coin} cooldown expired — entries allowed")

        # --- CHECK EXITS for this symbol's position ---
        tp = self.tracked_positions.get(hl_coin)
        if tp:
            tp["bars_held"] += 1

            exited = False
            exit_reason = ""
            atr_mult = tp.get("atr_multiplier", self.config.atr_stop_multiplier)

            if tp["side"] == "long":
                if close > tp["best_price"]:
                    tp["best_price"] = close
                trail_stop = tp["best_price"] - atr_mult * atr
                if trail_stop > tp["stop_price"]:
                    old_stop = tp["stop_price"]
                    tp["stop_price"] = trail_stop
                    pos_size = tp["size_usd"] / tp["entry_price"]
                    self.executor.update_stop_loss(hl_coin, is_buy=False,
                                                   size=pos_size, new_stop_price=trail_stop)
                    log.info(f"  TRAIL {hl_coin} stop updated: ${old_stop:,.4f} -> ${trail_stop:,.4f}")

                if low <= tp["stop_price"]:
                    exited = True
                    exit_reason = "stop"
                elif close >= tp["entry_price"] + 2.5 * atr:
                    exited = True
                    exit_reason = "take_profit"
                elif tp["bars_held"] >= 2 and signal < 0:
                    exited = True
                    exit_reason = "signal_reversal"
                elif tp["bars_held"] >= self.config.max_hold_bars:
                    exited = True
                    exit_reason = "max_hold"

            elif tp["side"] == "short":
                if close < tp["best_price"]:
                    tp["best_price"] = close
                trail_stop = tp["best_price"] + atr_mult * atr
                if trail_stop < tp["stop_price"]:
                    old_stop = tp["stop_price"]
                    tp["stop_price"] = trail_stop
                    pos_size = tp["size_usd"] / tp["entry_price"]
                    self.executor.update_stop_loss(hl_coin, is_buy=True,
                                                   size=pos_size, new_stop_price=trail_stop)
                    log.info(f"  TRAIL {hl_coin} stop updated: ${old_stop:,.4f} -> ${trail_stop:,.4f}")

                if high >= tp["stop_price"]:
                    exited = True
                    exit_reason = "stop"
                elif close <= tp["entry_price"] - 2.5 * atr:
                    exited = True
                    exit_reason = "take_profit"
                elif tp["bars_held"] >= 2 and signal > 0:
                    exited = True
                    exit_reason = "signal_reversal"
                elif tp["bars_held"] >= self.config.max_hold_bars:
                    exited = True
                    exit_reason = "max_hold"

            if exited:
                # Cancel server-side stop before closing (we're handling the exit)
                self.executor.cancel_all_orders(hl_coin)
                self._close_position_on_exchange(hl_coin, exit_reason)
                # Cooldown after stop-out to prevent immediate re-entry death spiral
                if exit_reason == "stop":
                    self.stop_cooldown[hl_coin] = COOLDOWN_BARS
                    log.info(f"  {hl_coin} cooldown set: {COOLDOWN_BARS} bars after stop-out")

        # --- CHECK ENTRY ---
        # Count current HL positions
        hl_positions = self.executor.get_positions()
        if len(hl_positions) >= self.config.max_open_trades:
            self._save_state()
            return

        # Don't enter if already have a position in this coin
        if hl_coin in self.tracked_positions:
            self._save_state()
            return

        # Daily loss check for new entries
        if daily_pnl < -self.config.max_daily_loss_pct:
            self._save_state()
            return

        # Cooldown check: don't re-enter after stop-out
        if hl_coin in self.stop_cooldown:
            self._save_state()
            return

        # Signal and regime already computed above (used for exit TP decisions too)

        # Regime gate
        if regime_scalar <= 0:
            return

        # REMOVED: Hard trend filter that blocked counter-trend entries.
        # The mean-reversion model WANTS counter-trend entries (all weights negative).
        # ER-based regime scalar already reduces size in trending markets.

        # Volatility filter: skip if market is too quiet (ATR < 0.3% of price)
        # Catches ranging/low-vol coins like HBAR/ADA where stops are within noise
        if close > 0 and (atr / close) < 0.003:
            return

        equity = self.executor.get_equity()

        if signal > self.config.signal_threshold_long:
            momentum_mult = self._get_momentum_multiplier(hl_coin)
            atr_mult = self._get_atr_multiplier(hl_coin)
            pos_pct = min(signal, 1.0) * self.config.max_position_pct * regime_scalar * momentum_mult
            notional = equity * pos_pct * self.config.leverage
            size = notional / close  # Size in coins

            result = self.executor.market_open(hl_coin, is_buy=True, size=size)
            if result["status"] == "ok":
                fill_price = result.get("fill_price") or close
                stop = fill_price - atr_mult * atr
                self.tracked_positions[hl_coin] = {
                    "side": "long",
                    "entry_price": fill_price,
                    "stop_price": stop,
                    "best_price": fill_price,
                    "bars_held": 0,
                    "entry_time": now,
                    "size_usd": notional,
                    "atr_multiplier": atr_mult,
                }
                # Place server-side stop-loss on HL (protects if bot goes offline)
                fill_size = result.get("fill_size") or size
                self.executor.place_stop_loss(hl_coin, is_buy=False, size=fill_size,
                                              stop_price=stop)
                print(f"  ENTRY {hl_coin} LONG @ ${fill_price:,.4f} (fill) | "
                      f"size: ${notional:,.0f} ({size:.4f} coins) | "
                      f"stop: ${stop:,.4f} | sig: {signal:.3f} | regime: {regime_scalar:.2f} | mom: {momentum_mult:.2f}x | atr: {atr_mult:.1f}x")
            self._save_state()

        elif signal < self.config.signal_threshold_short:
            momentum_mult = self._get_momentum_multiplier(hl_coin)
            atr_mult = self._get_atr_multiplier(hl_coin)
            pos_pct = min(abs(signal), 1.0) * self.config.max_position_pct * regime_scalar * momentum_mult
            notional = equity * pos_pct * self.config.leverage
            size = notional / close

            result = self.executor.market_open(hl_coin, is_buy=False, size=size)
            if result["status"] == "ok":
                fill_price = result.get("fill_price") or close
                stop = fill_price + atr_mult * atr
                self.tracked_positions[hl_coin] = {
                    "side": "short",
                    "entry_price": fill_price,
                    "stop_price": stop,
                    "best_price": fill_price,
                    "bars_held": 0,
                    "entry_time": now,
                    "size_usd": notional,
                    "atr_multiplier": atr_mult,
                }
                # Place server-side stop-loss on HL (protects if bot goes offline)
                fill_size = result.get("fill_size") or size
                self.executor.place_stop_loss(hl_coin, is_buy=True, size=fill_size,
                                              stop_price=stop)
                print(f"  ENTRY {hl_coin} SHORT @ ${fill_price:,.4f} (fill) | "
                      f"size: ${notional:,.0f} ({size:.4f} coins) | "
                      f"stop: ${stop:,.4f} | sig: {signal:.3f} | regime: {regime_scalar:.2f} | mom: {momentum_mult:.2f}x | atr: {atr_mult:.1f}x")
            self._save_state()

    def print_status(self):
        equity = self.executor.get_equity()
        dd = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        total_ret = (equity - self.starting_equity) / self.starting_equity if self.starting_equity > 0 else 0

        print(f"\n{'='*65}")
        print(f"  HYPERLIQUID V6 TRADER — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*65}")
        real_pnl = equity - self.starting_equity
        print(f"  Equity:     ${equity:,.2f} ({total_ret:+.2%})")
        print(f"  Peak:       ${self.peak_equity:,.2f}")
        print(f"  Drawdown:   {dd:.2%}")
        print(f"  Real P&L:   ${real_pnl:+,.2f} (equity vs start)")
        print(f"  Trades:     {self.total_trades} (win rate: {win_rate:.1%})")

        # Show live positions from exchange
        positions = self.executor.get_positions()
        print(f"  Positions:  {len(positions)} open on Hyperliquid")

        if positions:
            print(f"\n  Live Positions (from exchange):")
            for pos in positions:
                unrealized = pos["unrealized_pnl"]
                liq = f"${pos['liquidation_px']:,.2f}" if pos["liquidation_px"] else "N/A"
                print(f"    {pos['coin']} {pos['side'].upper()} | "
                      f"size: {abs(pos['size']):.4f} @ ${pos['entry_price']:,.4f} | "
                      f"unrealized: ${unrealized:+,.2f} | liq: {liq}")

        # Show tracked stop info
        if self.tracked_positions:
            print(f"\n  Stop Tracking:")
            for coin, tp in self.tracked_positions.items():
                print(f"    {coin} {tp['side'].upper()} | "
                      f"stop: ${tp['stop_price']:,.4f} | "
                      f"best: ${tp['best_price']:,.4f} | "
                      f"bars: {tp['bars_held']}")

        # Show momentum multipliers
        if self.coin_pnl_history:
            print(f"\n  Momentum Allocation:")
            for hl_coin, history in sorted(self.coin_pnl_history.items()):
                recent = history[-self.momentum_lookback:]
                mult = self._get_momentum_multiplier(hl_coin)
                wr = sum(1 for p in recent if p > 0) / len(recent) if recent else 0
                print(f"    {hl_coin}: {len(recent)} trades, WR: {wr:.0%}, sizing: {mult:.2f}x")

        print(f"{'='*65}\n")


# =========================================================================
# MAIN
# =========================================================================

def main():
    # File logging
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(fh)

    import builtins
    _orig_print = builtins.print
    def _logging_print(*args, **kwargs):
        _orig_print(*args, **kwargs, flush=True)
        msg = " ".join(str(a) for a in args)
        if msg.strip():
            log.info(msg.strip())
    builtins.print = _logging_print

    coin_arg = sys.argv[1] if len(sys.argv) > 1 else "default"

    COIN_SETS = {
        "default": ["DOGEUSDT", "XLMUSDT", "SUIUSDT"],
        "top":     ["DOGEUSDT", "XLMUSDT", "SUIUSDT"],
        "alts":    ["SOLUSDT", "XRPUSDT", "AVAXUSDT"],
    }

    if coin_arg in COIN_SETS:
        assets = COIN_SETS[coin_arg]
    else:
        assets = [s.strip().upper() for s in coin_arg.split(",")]
        assets = [s if s.endswith("USDT") else s + "USDT" for s in assets]

    # Verify all coins have HL mapping
    for a in assets:
        if a not in HL_SYMBOL_MAP:
            print(f"ERROR: {a} has no Hyperliquid mapping. Available: {list(HL_SYMBOL_MAP.keys())}")
            sys.exit(1)

    # Remove blocked coins (low volatility, consistently untradeable)
    blocked = [a for a in assets if a in BLOCKED_COINS]
    for a in blocked:
        print(f"  WARNING: {a} is blocked (low volatility). Removing from coin list.")
        assets.remove(a)
    if not assets:
        print("ERROR: No tradeable coins remaining after removing blocked coins.")
        sys.exit(1)

    # Always include BTC + ETH for cross-asset features (even if not trading them)
    fetch_assets = list(assets)
    if "BTCUSDT" not in fetch_assets:
        fetch_assets.insert(0, "BTCUSDT")
    if "ETHUSDT" not in fetch_assets:
        fetch_assets.append("ETHUSDT")

    config = Config(assets=fetch_assets, lookback_days=30,
                    signal_threshold_long=0.3, signal_threshold_short=-0.3,
                    atr_stop_multiplier=4.0, rolling_norm_window=48)

    print("=" * 65)
    print("  HYPERLIQUID V6 TRADER")
    print(f"  Trading: {', '.join(a.replace('USDT','') for a in assets)}")
    print(f"  Capital: testnet USDC | Leverage: {config.leverage}x")
    print(f"  Max positions: {config.max_open_trades}")
    print(f"  Model: Frozen weights (11-month, L2-normalized) + dynamic momentum allocation")
    print(f"  Execution: Hyperliquid Testnet (real orders)")
    print(f"  Data feed: Binance.com WebSocket (full liquidity, real taker_buy_volume, ~50ms latency)")
    print("=" * 65)

    # === 1. LOAD WALLET & CONNECT ===
    print("\n[1/5] Loading wallet & connecting to Hyperliquid testnet...")
    wallet = load_wallet()
    executor = HyperliquidExecutor(wallet, leverage=config.leverage)

    equity = executor.get_equity()
    if equity == 0:
        print("\n  ERROR: Your testnet account has $0.")
        print(f"  Fund address {wallet.address} first:")
        print(f"    1. https://app.hyperliquid-testnet.xyz/drip")
        print(f"    2. https://faucet.quicknode.com/hyperliquid/testnet")
        print(f"\n  Then re-run this script.")
        sys.exit(1)

    # === 2. LOAD FROZEN WEIGHTS FROM FILE ===
    FROZEN_WEIGHTS_FILE = Path("frozen_weights.json")

    if not FROZEN_WEIGHTS_FILE.exists():
        print(f"\nERROR: {FROZEN_WEIGHTS_FILE} not found.")
        print(f"  Run 'python train_frozen_weights.py' first to generate it.")
        sys.exit(1)

    print(f"\n[2/5] Loading frozen weights from {FROZEN_WEIGHTS_FILE}...")
    frozen = json.loads(FROZEN_WEIGHTS_FILE.read_text())
    selected = frozen["features"]
    weights = np.array(frozen["weights"])
    print(f"  Loaded {len(selected)} features, {frozen['training_windows']} windows "
          f"(trained {frozen['trained_at']})")
    for name, w in sorted(zip(selected, weights), key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"    {name:30s} w={w:+.6f}")

    signal_engine = SignalEngine(selected)
    signal_engine.set_weights(weights)
    regime_detector = RegimeDetector(config)
    fe = ScalpingFeatureEngine(config)
    cae = CrossAssetEngine()

    # === 3. FETCH WARMUP DATA (recent 30 days for buffer initialization) ===
    print("\n[3/5] Fetching warmup data (30 days from Binance.US)...")
    warmup_config = Config(assets=fetch_assets, lookback_days=30, end_offset_days=0)
    warmup_data = BinanceUSDataFetcher(warmup_config).fetch_all()
    for sym, df in warmup_data.items():
        print(f"    {sym}: {len(df)} bars")
    ohlcv_data = warmup_data

    # === 4. INITIALIZE TRADER ===
    print("\n[4/5] Initializing Hyperliquid trader...")

    trader = HyperliquidTrader(
        config=config,
        signal_engine=signal_engine,
        regime_detector=regime_detector,
        feature_engine=fe,
        cross_engine=cae,
        feature_names=selected,
        executor=executor,
    )
    trader.warmup_buffers(ohlcv_data)

    # Set leverage for all traded coins
    for a in assets:
        hl_coin = HL_SYMBOL_MAP[a]
        executor._ensure_leverage(hl_coin)

    # === 5. CONNECT TO LIVE PRICE FEED ===
    print("\n[5/5] Connecting to live price feed...")

    candle_count = [0]
    status_interval = 12  # Print status every 12 candles (1 hour)
    candle_lock = threading.Lock()

    def on_candle(symbol, candle_data):
        with candle_lock:
            ts = datetime.utcfromtimestamp(candle_data["t"] / 1000)
            close_price = float(candle_data["c"])
            vol = float(candle_data["v"])
            print(f"\n  [{ts.strftime('%H:%M')}] {symbol.replace('USDT','')} closed @ ${close_price:,.4f} (vol: {vol:,.0f})")

            # Non-traded coins (BTC, ETH) only update buffers for cross-asset features
            if symbol not in assets:
                trader._append_candle(symbol, candle_data)
                return

            trader.process_candle(symbol, candle_data)

            candle_count[0] += 1
            if candle_count[0] % status_interval == 0:
                trader._sync_positions()
                trader.print_status()

    all_poll_symbols = fetch_assets

    # Binance.US WebSocket — push-based, real taker_buy_volume, ~50ms candle-close latency.
    # HL perp prices track Binance spot within milliseconds; 5m candles close at the same
    # UTC timestamps on both exchanges, so signals fire at exactly the right moment for HL.
    stream = BinanceUSWebSocketStream(all_poll_symbols, on_candle)
    connected = stream.start()

    if not connected:
        print("ERROR: Could not connect to Binance.US WebSocket.")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    print(f"  Data source: Binance.US WebSocket (real-time, push-based)")
    trader.print_status()
    print("Waiting for candles (new 5m candle every 5 minutes)...")
    print("Press Ctrl+C to stop.\n")

    def shutdown(signum, frame):
        print("\n\nShutting down Hyperliquid trader...")
        stream.stop()
        trader._save_state()
        trader.print_status()
        print("  NOTE: Open positions remain on Hyperliquid testnet.")
        print(f"  Manage them at: https://app.hyperliquid-testnet.xyz/trade")
        sys.exit(0)

    sig_module.signal(sig_module.SIGINT, shutdown)
    sig_module.signal(sig_module.SIGTERM, shutdown)

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
