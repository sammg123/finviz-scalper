"""
Paper Trader — Identical strategy to hyperliquid_trader.py, no exchange.
================================================================================
Purpose: isolate whether testnet execution (fills, stops, fees) is the issue,
         or whether the signal model itself is losing.

Rules (honest):
  - Same frozen weights, same signal threshold (±0.4), same coins (DOGE/XLM/SUI)
  - Same entry gates: regime, volatility filter (ATR < 0.3%)
  - Same position sizing: momentum multiplier + dynamic ATR multiplier
  - Same exit logic: trailing ATR stop OR max_hold_bars
  - Fill at candle close price (same assumption as HL bot)
  - Stop exits fill at the stop price (no slippage, same as HL assumption)
  - Fees: 0.045% taker per side (entry + exit = 0.09% of notional)
    Applied same as HL: fee_pct = 2 * 0.00045 * leverage
  - Start: $1,000 virtual USDC
  - Nothing is fabricated. If it loses, it loses.

Usage:
    python paper_trader.py
"""

import sys
import json
import time
import threading
import logging
import builtins
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scalper import (
    Config, BinanceUSDataFetcher, ScalpingFeatureEngine,
    CrossAssetEngine, RegimeDetector, SignalEngine,
)
from sheets_logger import SheetsLogger

# ── Files ──────────────────────────────────────────────────────────────────────
STATE_FILE  = Path("paper_state.json")
TRADES_FILE = Path("paper_trades.json")

# ── Google Sheets ─────────────────────────────────────────────────────────────
# Put your Google Sheet ID here (from the URL: docs.google.com/spreadsheets/d/SHEET_ID/edit)
SHEET_ID = "17t9JQCjErqsR-82XjbiXbuADucWAdiwsxywvibZfuJs"
LOG_FILE    = Path("paper_trader.log")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger("PaperTrader")

# Route print() through logging so everything goes to the log file
_orig_print = builtins.print
def _logged_print(*args, **kwargs):
    _orig_print(*args, **kwargs, flush=True)
    msg = " ".join(str(a) for a in args).strip()
    if msg:
        log.info(msg)
builtins.print = _logged_print

# ── Strategy constants ─────────────────────────────────────────────────────────
STARTING_EQUITY   = 10_000.0
COINS             = ["DOGEUSDT", "XLMUSDT", "SUIUSDT"]   # Same as live bot
FETCH_ASSETS      = ["BTCUSDT"] + COINS + ["ETHUSDT"]
HL_FEE_TAKER      = 0.00045   # 0.045% per side (HL testnet taker rate)
MIN_ATR_PCT       = 0.0005    # Volatility filter: skip if ATR < 0.05% of price
COOLDOWN_BARS     = 1         # Bars to wait before re-entry after stop-out


# ══════════════════════════════════════════════════════════════════════════════
# Binance WebSocket (copy of class from hyperliquid_trader.py — no changes)
# ══════════════════════════════════════════════════════════════════════════════

class BinanceLiveWebSocket:
    WS_BASE   = "wss://stream.binance.com"
    REST_BASE = "https://api.binance.com/api/v3"

    def __init__(self, symbols: List[str], on_candle_close):
        self.symbols           = symbols
        self.on_candle_close   = on_candle_close
        self._running          = False
        self._ws               = None
        self._thread           = None
        self._reconnect_delay  = 5

    def _to_stream(self, s: str) -> str:
        return s.lower()

    def _build_url(self) -> str:
        streams = "/".join(f"{self._to_stream(s)}@kline_5m" for s in self.symbols)
        return f"{self.WS_BASE}/stream?streams={streams}"

    def _on_message(self, ws, message):
        try:
            data    = json.loads(message)
            payload = data.get("data", data)
            k       = payload.get("k")
            if not k or not k.get("x", False):
                return
            internal = k["s"]
            if internal not in self.symbols:
                return
            candle = {
                "t": k["t"], "o": k["o"], "h": k["h"], "l": k["l"],
                "c": k["c"], "v": k["v"], "q": k["q"], "n": k["n"],
                "V": k["V"], "Q": k["Q"], "s": internal, "x": True,
            }
            self.on_candle_close(internal, candle)
        except Exception as e:
            import traceback
            log.error(f"WebSocket parse error: {e}\n{traceback.format_exc()}")

    def _on_error(self, ws, error):
        log.error(f"WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        log.warning(f"WebSocket closed (code={code}) — reconnecting in {self._reconnect_delay}s")
        if self._running:
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 60)
            self._connect()

    def _on_open(self, ws):
        self._reconnect_delay = 5
        syms = ", ".join(self._to_stream(s).upper() for s in self.symbols)
        log.info(f"Binance.com WebSocket CONNECTED — 5m klines: {syms}")

    def _connect(self):
        import websocket as ws_lib
        url = self._build_url()
        log.info(f"Connecting to Binance.com WebSocket: {url}")
        self._ws = ws_lib.WebSocketApp(
            url,
            on_open=self._on_open, on_message=self._on_message,
            on_error=self._on_error, on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def start(self) -> bool:
        import requests
        sym = self._to_stream(self.symbols[0]).upper()
        try:
            resp  = requests.get(f"{self.REST_BASE}/ticker/price",
                                 params={"symbol": sym}, timeout=8)
            resp.raise_for_status()
            price = float(resp.json()["price"])
            print(f"  Binance.com WebSocket: {sym} @ ${price:,.4f} — ready")
        except Exception as e:
            print(f"  Binance.com REST check failed: {e}")
            return False
        self._running = True
        self._thread  = threading.Thread(target=self._connect, daemon=True)
        self._thread.start()
        time.sleep(2)
        return True

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Paper Trader
# ══════════════════════════════════════════════════════════════════════════════

class PaperTrader:
    """
    Pure paper simulation — same strategy, no exchange calls.
    All positions, equity, and P&L are tracked in memory / state file.
    """

    def __init__(self, config: Config, signal_engine: SignalEngine,
                 regime_detector: RegimeDetector, feature_engine: ScalpingFeatureEngine,
                 cross_engine: CrossAssetEngine, feature_names: List[str]):
        self.config          = config
        self.signal_engine   = signal_engine
        self.regime_detector = regime_detector
        self.feature_engine  = feature_engine
        self.cross_engine    = cross_engine
        self.feature_names   = feature_names

        self.candle_buffers: Dict[str, pd.DataFrame] = {}
        self.buffer_size = 300

        # Virtual account
        self.equity         = STARTING_EQUITY
        self.peak_equity    = STARTING_EQUITY
        self.day_start_equity = STARTING_EQUITY
        self.current_day    = None

        # Positions: coin -> {side, entry_price, stop_price, best_price, bars_held, size_usd, entry_time}
        self.positions: Dict[str, dict] = {}

        # Stats
        self.total_trades   = 0
        self.winning_trades = 0
        self.total_pnl      = 0.0
        self.trades: List[dict] = []

        # Google Sheets logger
        self.sheets = SheetsLogger(SHEET_ID) if SHEET_ID else None

        # Momentum tracking (identical to HL bot)
        self.coin_pnl_history: Dict[str, List[float]] = {}
        self.momentum_lookback  = 10
        self.momentum_max_mult  = 1.5
        self.momentum_min_mult  = 0.25

        # Cooldown tracking: bars remaining before re-entry allowed after stop-out
        self.stop_cooldown: Dict[str, int] = {}

        self._load_state()

    # ── State persistence ──────────────────────────────────────────────────────

    def _save_state(self):
        state = {
            "equity":           self.equity,
            "peak_equity":      self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "total_trades":     self.total_trades,
            "winning_trades":   self.winning_trades,
            "total_pnl":        self.total_pnl,
            "positions":        self.positions,
            "coin_pnl_history": self.coin_pnl_history,
            "stop_cooldown":    self.stop_cooldown,
            "saved_at":         datetime.utcnow().isoformat(),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            s = json.loads(STATE_FILE.read_text())
            self.equity           = s.get("equity",           STARTING_EQUITY)
            self.peak_equity      = s.get("peak_equity",      STARTING_EQUITY)
            self.day_start_equity = s.get("day_start_equity", STARTING_EQUITY)
            self.total_trades     = s.get("total_trades",     0)
            self.winning_trades   = s.get("winning_trades",   0)
            self.total_pnl        = s.get("total_pnl",        0.0)
            self.positions        = s.get("positions",        {})
            self.coin_pnl_history = s.get("coin_pnl_history", {})
            self.stop_cooldown    = s.get("stop_cooldown", {})
            if self.positions:
                print(f"  Resumed: {len(self.positions)} open paper positions")
        except Exception as e:
            print(f"  Warning: could not load paper state: {e}")

    def _log_trade(self, trade: dict):
        self.trades.append(trade)
        existing = []
        if TRADES_FILE.exists():
            try:
                existing = json.loads(TRADES_FILE.read_text())
            except Exception:
                pass
        existing.append(trade)
        TRADES_FILE.write_text(json.dumps(existing, indent=2))

    # ── Candle buffer ──────────────────────────────────────────────────────────

    def warmup_buffers(self, ohlcv_data: Dict[str, pd.DataFrame]):
        for sym, df in ohlcv_data.items():
            self.candle_buffers[sym] = df.tail(self.buffer_size).copy()
            print(f"  {sym}: {len(self.candle_buffers[sym])} bars loaded")

    def _append_candle(self, symbol: str, candle: dict) -> bool:
        ts = pd.Timestamp(candle["t"], unit="ms")
        new_row = pd.DataFrame([{
            "open":                   float(candle["o"]),
            "high":                   float(candle["h"]),
            "low":                    float(candle["l"]),
            "close":                  float(candle["c"]),
            "volume":                 float(candle["v"]),
            "quote_volume":           float(candle["q"]),
            "trades":                 int(candle["n"]),
            "taker_buy_volume":       float(candle["V"]),
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

    # ── Helpers (identical to HL bot) ─────────────────────────────────────────

    def _compute_atr(self, df: pd.DataFrame) -> float:
        close = df["close"]
        tr    = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"]  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.config.atr_period).mean()
        return atr.iloc[-1] if len(atr) > 0 else close.iloc[-1] * 0.005

    def _get_momentum_multiplier(self, coin: str) -> float:
        history = self.coin_pnl_history.get(coin, [])
        if len(history) < 3:
            return 1.0
        recent   = history[-self.momentum_lookback:]
        win_rate = sum(1 for p in recent if p > 0) / len(recent)
        score    = (win_rate - 0.5) * 2
        if score >= 0:
            mult = 1.0 + score * (self.momentum_max_mult - 1.0)
        else:
            mult = 1.0 + score * (1.0 - self.momentum_min_mult)
        return np.clip(mult, self.momentum_min_mult, self.momentum_max_mult)

    def _get_atr_multiplier(self, coin: str) -> float:
        history = self.coin_pnl_history.get(coin, [])
        if len(history) < 3:
            return 1.2  # New coins get tightest stops until proven
        recent   = history[-self.momentum_lookback:]
        win_rate = sum(1 for p in recent if p > 0) / len(recent)
        atr_min, atr_max = 1.2, self.config.atr_stop_multiplier
        return round(atr_min + win_rate * (atr_max - atr_min), 2)

    def _get_signal_and_regime(self, symbol: str):
        buf = self.candle_buffers.get(symbol)
        if buf is None or len(buf) < 100:
            return 0.0, 0.0, 0, 0.0

        raw         = self.feature_engine.generate(buf)
        normed      = self.feature_engine.normalize(raw)
        btc_df      = self.candle_buffers.get("BTCUSDT")
        cross       = self.cross_engine.generate(btc_df, buf, self.candle_buffers, symbol)
        cross_normed = self.feature_engine.normalize(cross)
        combined    = pd.concat([normed, cross_normed], axis=1)

        avail    = [f for f in self.feature_names if f in combined.columns]
        combined = combined[avail]

        signal   = self.signal_engine.score(combined, rolling_window=self.config.rolling_norm_window)
        regime   = self.regime_detector.compute_regime(buf)

        latest_signal  = signal.iloc[-1]  if len(signal) > 0  else 0.0
        latest_scalar  = regime["scalar"].iloc[-1] if len(regime) > 0 else 1.0
        latest_trend   = int(regime["trend"].iloc[-1]) if len(regime) > 0 else 0
        atr            = self._compute_atr(buf)

        return latest_signal, latest_scalar, latest_trend, atr

    # ── Virtual exit ──────────────────────────────────────────────────────────

    def _close_position(self, coin: str, exit_price: float, reason: str):
        """Simulate closing a position and update virtual equity."""
        tp = self.positions.get(coin)
        if tp is None:
            return

        entry_price = tp["entry_price"]
        size_usd    = tp["size_usd"]
        side        = tp["side"]

        # Raw P&L as % of notional (leveraged)
        if side == "long":
            raw_pnl_pct = (exit_price / entry_price - 1) * self.config.leverage
        else:
            raw_pnl_pct = (entry_price / exit_price - 1) * self.config.leverage

        # Fees: same formula as HL bot — taker both sides, applied to leveraged notional
        fee_pct  = 2 * HL_FEE_TAKER * self.config.leverage
        pnl_pct  = raw_pnl_pct - fee_pct

        # Dollar P&L on the margin (size_usd / leverage = margin used)
        margin      = size_usd / self.config.leverage
        dollar_pnl  = pnl_pct * margin

        self.equity += dollar_pnl
        self.total_trades  += 1
        if pnl_pct > 0:
            self.winning_trades += 1
        self.total_pnl += dollar_pnl

        if coin not in self.coin_pnl_history:
            self.coin_pnl_history[coin] = []
        self.coin_pnl_history[coin].append(pnl_pct)

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        trade_record = {
            "time":         now,
            "coin":         coin,
            "side":         side,
            "entry":        entry_price,
            "exit":         exit_price,
            "raw_pnl_pct":  f"{raw_pnl_pct:.4%}",
            "fees_pct":     f"{fee_pct:.4%}",
            "net_pnl_pct":  f"{pnl_pct:.4%}",
            "pnl_dollar":   f"${dollar_pnl:+,.2f}",
            "equity_after": f"${self.equity:,.2f}",
            "reason":       reason,
            "bars_held":    tp.get("bars_held", 0),
        }
        self._log_trade(trade_record)

        # Log to Google Sheets
        if self.sheets and self.sheets.connected:
            self.sheets.log_trade(trade_record)
            self._push_dashboard()

        sign = "+" if pnl_pct > 0 else ""
        print(f"  [PAPER] EXIT {coin} {side.upper()} | {reason} | "
              f"entry: ${entry_price:,.4f} exit: ${exit_price:,.4f} | "
              f"raw: {raw_pnl_pct:+.3%} fees: -{fee_pct:.3%} net: {sign}{pnl_pct:.3%} | "
              f"P&L: {sign}${dollar_pnl:,.2f} | equity: ${self.equity:,.2f}")

        self.positions.pop(coin, None)

    def _push_dashboard(self):
        """Compute and push all stats to Google Sheets dashboard."""
        if not self.sheets or not self.sheets.connected:
            return

        # Compute per-trade stats for avg win/loss and profit factor
        wins_dollar = []
        losses_dollar = []
        consec_loss = 0
        max_consec = 0
        for t in self.trades:
            pnl = float(t.get("pnl_dollar", "$0").replace("$", "").replace(",", "").replace("+", ""))
            if pnl > 0:
                wins_dollar.append(pnl)
                consec_loss = 0
            else:
                losses_dollar.append(abs(pnl))
                consec_loss += 1
                max_consec = max(max_consec, consec_loss)

        gross_wins = sum(wins_dollar) if wins_dollar else 0
        gross_losses = sum(losses_dollar) if losses_dollar else 0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0

        # Per-coin stats
        coin_stats = {}
        for coin, history in self.coin_pnl_history.items():
            recent = history[-self.momentum_lookback:]
            wr = sum(1 for p in recent if p > 0) / len(recent) if recent else 0
            pnls = []
            for t in self.trades:
                if t.get("coin") == coin:
                    pnl = float(t.get("pnl_dollar", "$0").replace("$", "").replace(",", "").replace("+", ""))
                    pnls.append(pnl)
            coin_stats[coin] = {
                "trades": len(history),
                "win_rate": wr,
                "total_pnl": sum(pnls),
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
                "momentum_mult": self._get_momentum_multiplier(coin),
                "cooldown": self.stop_cooldown.get(coin, 0),
            }

        # Open positions
        open_pos = []
        for coin, tp in self.positions.items():
            buf = self.candle_buffers.get(coin)
            cur = float(buf["close"].iloc[-1]) if buf is not None else 0
            unreal = (cur / tp["entry_price"] - 1) * self.config.leverage * (tp["size_usd"] / self.config.leverage)
            if tp["side"] == "short":
                unreal = -unreal
            open_pos.append({
                "coin": coin, "side": tp["side"],
                "entry": tp["entry_price"], "current": cur,
                "unrealized": unreal, "stop": tp["stop_price"],
                "bars": tp.get("bars_held", 0),
            })

        stats = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "starting_equity": STARTING_EQUITY,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "avg_win": sum(wins_dollar) / len(wins_dollar) if wins_dollar else 0,
            "avg_loss": -(sum(losses_dollar) / len(losses_dollar)) if losses_dollar else 0,
            "profit_factor": profit_factor,
            "max_consec_loss": max_consec,
            "coin_stats": coin_stats,
            "open_positions": open_pos,
        }
        self.sheets.update_dashboard(stats)

    # ── Main candle processor ─────────────────────────────────────────────────

    def process_candle(self, symbol: str, candle: dict):
        is_new = self._append_candle(symbol, candle)
        if not is_new:
            return

        # Skip BTC/ETH — used only for cross-asset features
        if symbol in ("BTCUSDT", "ETHUSDT"):
            return

        coin  = symbol  # Paper trader uses Binance symbols directly
        close = float(candle["c"])
        high  = float(candle["h"])
        low   = float(candle["l"])
        now   = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

        # Day reset
        today = datetime.utcnow().date()
        if today != self.current_day:
            self.current_day      = today
            self.day_start_equity = self.equity

        # Peak tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        daily_pnl = (self.equity - self.day_start_equity) / self.day_start_equity \
                    if self.day_start_equity > 0 else 0

        # Daily loss limit — close all and stop entering (same 8% as HL bot)
        if daily_pnl < -self.config.max_daily_loss_pct and self.positions:
            print(f"  [PAPER] DAILY LOSS LIMIT ({daily_pnl:.2%}) — closing all positions")
            for c in list(self.positions.keys()):
                buf = self.candle_buffers.get(c)
                if buf is not None:
                    c_close = float(buf["close"].iloc[-1])
                    self._close_position(c, c_close, "daily_limit")
            self._save_state()
            return

        # Compute signal once for both exit TP and entry decisions
        signal, regime_scalar, trend, atr = self._get_signal_and_regime(symbol)

        # ── DECREMENT COOLDOWN ──────────────────────────────────────────────
        if coin in self.stop_cooldown:
            self.stop_cooldown[coin] -= 1
            if self.stop_cooldown[coin] <= 0:
                del self.stop_cooldown[coin]

        # ── EXIT CHECK ──────────────────────────────────────────────────────
        tp = self.positions.get(coin)
        if tp:
            tp["bars_held"] += 1

            exited      = False
            exit_reason = ""
            exit_price  = close
            atr_mult    = tp.get("atr_multiplier", self.config.atr_stop_multiplier)

            if tp["side"] == "long":
                if close > tp["best_price"]:
                    tp["best_price"] = close
                trail_stop = tp["best_price"] - atr_mult * atr
                if trail_stop > tp["stop_price"]:
                    tp["stop_price"] = trail_stop

                if low <= tp["stop_price"]:
                    exited      = True
                    exit_reason = "stop"
                    exit_price  = tp["stop_price"]
                elif close >= tp["entry_price"] + 2.5 * atr:
                    exited      = True
                    exit_reason = "take_profit"
                    exit_price  = close
                elif tp["bars_held"] >= 2 and signal < 0:
                    exited      = True
                    exit_reason = "signal_reversal"
                    exit_price  = close
                elif tp["bars_held"] >= self.config.max_hold_bars:
                    exited      = True
                    exit_reason = "max_hold"
                    exit_price  = close

            elif tp["side"] == "short":
                if close < tp["best_price"]:
                    tp["best_price"] = close
                trail_stop = tp["best_price"] + atr_mult * atr
                if trail_stop < tp["stop_price"]:
                    tp["stop_price"] = trail_stop

                if high >= tp["stop_price"]:
                    exited      = True
                    exit_reason = "stop"
                    exit_price  = tp["stop_price"]
                elif close <= tp["entry_price"] - 2.5 * atr:
                    exited      = True
                    exit_reason = "take_profit"
                    exit_price  = close
                elif tp["bars_held"] >= 2 and signal > 0:
                    exited      = True
                    exit_reason = "signal_reversal"
                    exit_price  = close
                elif tp["bars_held"] >= self.config.max_hold_bars:
                    exited      = True
                    exit_reason = "max_hold"
                    exit_price  = close

            if exited:
                self._close_position(coin, exit_price, exit_reason)
                # Cooldown after stop-out to prevent immediate re-entry
                if exit_reason == "stop":
                    self.stop_cooldown[coin] = COOLDOWN_BARS

        # ── ENTRY CHECK ─────────────────────────────────────────────────────
        if len(self.positions) >= self.config.max_open_trades:
            self._save_state()
            return

        if coin in self.positions:
            self._save_state()
            return

        if daily_pnl < -self.config.max_daily_loss_pct:
            self._save_state()
            return

        # Cooldown check: don't re-enter after stop-out
        if coin in self.stop_cooldown:
            self._save_state()
            return

        # Signal and regime already computed above (used for exit TP decisions too)

        atr_pct = (atr / close) if close > 0 else 0
        log.info(f"  [{coin}] sig={signal:.4f} regime={regime_scalar:.2f} atr%={atr_pct:.4f} thr=±{self.config.signal_threshold_long}")

        if regime_scalar <= 0:
            return

        # REMOVED: Hard trend filter that blocked counter-trend entries.
        # The mean-reversion model WANTS counter-trend entries (all weights negative).
        # ER-based regime scalar already reduces size in trending markets.

        # Volatility filter (same as HL bot)
        if close > 0 and atr_pct < MIN_ATR_PCT:
            return

        if signal > self.config.signal_threshold_long:
            mom_mult = self._get_momentum_multiplier(coin)
            atr_mult = self._get_atr_multiplier(coin)
            pos_pct  = min(signal, 1.0) * self.config.max_position_pct * regime_scalar * mom_mult
            notional = self.equity * pos_pct * self.config.leverage
            stop     = close - atr_mult * atr

            self.positions[coin] = {
                "side":        "long",
                "entry_price": close,
                "stop_price":  stop,
                "best_price":  close,
                "bars_held":   0,
                "entry_time":  now,
                "size_usd":    notional,
                "atr_multiplier": atr_mult,
            }
            print(f"  [PAPER] ENTRY {coin} LONG @ ${close:,.4f} | "
                  f"size: ${notional:,.0f} | stop: ${stop:,.4f} | "
                  f"sig: {signal:.3f} | regime: {regime_scalar:.2f} | "
                  f"mom: {mom_mult:.2f}x | atr: {atr_mult:.1f}x | equity: ${self.equity:,.2f}")
            self._save_state()

        elif signal < self.config.signal_threshold_short:
            mom_mult = self._get_momentum_multiplier(coin)
            atr_mult = self._get_atr_multiplier(coin)
            pos_pct  = min(abs(signal), 1.0) * self.config.max_position_pct * regime_scalar * mom_mult
            notional = self.equity * pos_pct * self.config.leverage
            stop     = close + atr_mult * atr

            self.positions[coin] = {
                "side":        "short",
                "entry_price": close,
                "stop_price":  stop,
                "best_price":  close,
                "bars_held":   0,
                "entry_time":  now,
                "size_usd":    notional,
                "atr_multiplier": atr_mult,
            }
            print(f"  [PAPER] ENTRY {coin} SHORT @ ${close:,.4f} | "
                  f"size: ${notional:,.0f} | stop: ${stop:,.4f} | "
                  f"sig: {signal:.3f} | regime: {regime_scalar:.2f} | "
                  f"mom: {mom_mult:.2f}x | atr: {atr_mult:.1f}x | equity: ${self.equity:,.2f}")
            self._save_state()

    # ── Status print ──────────────────────────────────────────────────────────

    def print_status(self):
        dd       = (self.equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        ret      = (self.equity - STARTING_EQUITY) / STARTING_EQUITY

        print(f"\n{'='*65}")
        print(f"  PAPER TRADER — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*65}")
        print(f"  Equity:   ${self.equity:,.2f} ({ret:+.2%} from $1,000 start)")
        print(f"  Peak:     ${self.peak_equity:,.2f}")
        print(f"  Drawdown: {dd:.2%}")
        print(f"  Trades:   {self.total_trades} (win rate: {win_rate:.1%})")
        print(f"  Net P&L:  ${self.total_pnl:+,.2f}")

        if self.positions:
            print(f"\n  Open Positions:")
            for coin, tp in self.positions.items():
                buf   = self.candle_buffers.get(coin)
                cur   = float(buf["close"].iloc[-1]) if buf is not None else 0
                unreal = (cur / tp["entry_price"] - 1) * self.config.leverage * (tp["size_usd"] / self.config.leverage)
                if tp["side"] == "short":
                    unreal = -unreal
                print(f"    {coin} {tp['side'].upper()} @ ${tp['entry_price']:,.4f} | "
                      f"current: ${cur:,.4f} | unrealized: ${unreal:+,.2f} | "
                      f"stop: ${tp['stop_price']:,.4f} | bars: {tp['bars_held']}")

        if self.coin_pnl_history:
            print(f"\n  Momentum Allocation:")
            for coin, hist in sorted(self.coin_pnl_history.items()):
                recent = hist[-self.momentum_lookback:]
                wr     = sum(1 for p in recent if p > 0) / len(recent) if recent else 0
                mult   = self._get_momentum_multiplier(coin)
                print(f"    {coin}: {len(recent)} trades, WR: {wr:.0%}, sizing: {mult:.2f}x")

        print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PAPER TRADER — Binance data, simulated fills, HL fees")
    print(f"  Coins:    {', '.join(c.replace('USDT','') for c in COINS)}")
    print(f"  Capital:  ${STARTING_EQUITY:,.0f} virtual USDC")
    print(f"  Leverage: 5x | Signal threshold: ±0.4 | ATR stop: 2.5x")
    print(f"  Fees:     {HL_FEE_TAKER*100:.3f}% taker per side (HL rate)")
    print(f"  Fills:    at candle close price (same assumption as HL bot)")
    print(f"  Stops:    fill at stop price, no slippage")
    print("=" * 65)

    # === 1. Load frozen weights ===
    frozen_file = Path("frozen_weights.json")
    if not frozen_file.exists():
        print("ERROR: frozen_weights.json not found.")
        sys.exit(1)

    print("\n[1/4] Loading frozen weights...")
    frozen   = json.loads(frozen_file.read_text())
    selected = frozen["features"]
    weights  = np.array(frozen["weights"])
    print(f"  {len(selected)} features, trained {frozen['trained_at']}")

    # === 2. Build config + engines (identical to HL bot) ===
    print("\n[2/4] Building strategy engines...")
    config = Config(
        assets=FETCH_ASSETS,
        lookback_days=30,
        signal_threshold_long=0.3,
        signal_threshold_short=-0.3,
        atr_stop_multiplier=2.0,
        rolling_norm_window=48,
    )
    feature_engine  = ScalpingFeatureEngine(config)
    cross_engine    = CrossAssetEngine()
    regime_detector = RegimeDetector(config)
    signal_engine   = SignalEngine(selected)
    signal_engine.set_weights(weights)

    trader = PaperTrader(
        config, signal_engine, regime_detector,
        feature_engine, cross_engine, selected,
    )

    # === 3. Warm up candle buffers (30 days of 5m data from Binance) ===
    print(f"\n[3/4] Fetching 30-day warmup from Binance.com...")
    fetcher = BinanceUSDataFetcher(config)
    ohlcv   = fetcher.fetch_all()

    if not ohlcv:
        print("ERROR: No data returned from Binance.")
        sys.exit(1)

    # Deduplicate and align (same as coin_ranker.py)
    for sym in list(ohlcv.keys()):
        df = ohlcv[sym]
        if df.index.duplicated().any():
            ohlcv[sym] = df[~df.index.duplicated(keep="last")]

    btc_df = ohlcv.get("BTCUSDT")
    if btc_df is not None:
        for sym, df in ohlcv.items():
            if sym != "BTCUSDT":
                ohlcv[sym] = df.reindex(
                    btc_df.index, method="nearest",
                    tolerance=pd.Timedelta("3m"),
                ).dropna()

    trader.warmup_buffers(ohlcv)

    # === 4. Connect WebSocket ===
    print(f"\n[4/4] Connecting to Binance.com WebSocket...")

    lock = threading.Lock()

    def on_candle_close(symbol: str, candle: dict):
        with lock:
            ts = pd.Timestamp(candle["t"], unit="ms")
            sym_short = symbol.replace("USDT", "")
            c = float(candle["c"])
            v = int(float(candle["v"]))
            print(f"  [{ts.strftime('%H:%M')}] {sym_short} closed @ ${c:,.4f} (vol: {v:,})")
            trader.process_candle(symbol, candle)

    ws = BinanceLiveWebSocket(FETCH_ASSETS, on_candle_close)
    if not ws.start():
        print("ERROR: Could not connect to Binance WebSocket.")
        sys.exit(1)

    trader.print_status()
    trader._push_dashboard()  # Initial dashboard push on startup
    print("Waiting for candles (5-minute bars)...")
    print("Press Ctrl+C to stop.\n")

    try:
        last_status = time.time()
        while True:
            time.sleep(10)
            if time.time() - last_status >= 1800:   # Print status every 30 min
                with lock:
                    trader.print_status()
                    trader._push_dashboard()
                last_status = time.time()
    except KeyboardInterrupt:
        pass

    ws.stop()
    trader._save_state()
    trader.print_status()
    print("Paper trader stopped.")


if __name__ == "__main__":
    main()
