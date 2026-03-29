"""
Strategy Validation Test
========================
Tests the frozen-weight scalping model across multiple parameter variants
to find which configuration (if any) produces a real out-of-sample edge.

Variants tested:
  A. Baseline: frozen weights, threshold ±0.4 (paper_trader config), regime on
  B. Lower threshold ±0.3
  C. Lower threshold ±0.2
  D. Threshold ±0.4, regime detector OFF
  E. Threshold ±0.4, 6-bar cooldown after stops
  F. INVERTED weights (momentum instead of mean-reversion)
"""

import sys
import json
import time
import warnings
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import strategy components from scalper.py
from scalper import (
    Config,
    BinanceUSDataFetcher,
    ScalpingFeatureEngine,
    CrossAssetEngine,
    RegimeDetector,
    SignalEngine,
    ScalpingBacktester,
)


# ═══════════════════════════════════════════════════════════════════════════
# Modified backtester with cooldown support
# ═══════════════════════════════════════════════════════════════════════════

class BacktesterWithCooldown(ScalpingBacktester):
    """Extended backtester that adds N-bar cooldown after stop-outs."""

    def __init__(self, config: Config, cooldown_bars: int = 0):
        super().__init__(config)
        self.cooldown_bars = cooldown_bars

    def run(self, df, signals, regime=None):
        if self.cooldown_bars <= 0:
            return super().run(df, signals, regime)

        # Full reimplementation with cooldown logic
        close = df["close"]
        n = len(close)
        signals = signals.reindex(close.index).fillna(0)

        if regime is not None:
            regime = regime.reindex(close.index).ffill()
            regime_scalar = regime["scalar"].fillna(1.0).values
            trend = regime["trend"].fillna(0).values
        else:
            regime_scalar = np.ones(n)
            trend = np.zeros(n)

        friction = self.config.fee_rate + self.config.slippage_rate

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
        open_positions = []
        cooldown_remaining = 0  # bars until re-entry allowed

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

            # Decrement cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

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

            surviving = []
            for pos in open_positions:
                pos["bars_held"] += 1

                if pos["direction"] > 0:
                    pos["best_price"] = max(pos["best_price"], price)
                    trail_stop = pos["best_price"] - self.config.atr_stop_multiplier * atr_val
                    pos["stop_price"] = max(pos["stop_price"], trail_stop)
                else:
                    pos["best_price"] = min(pos["best_price"], price)
                    trail_stop = pos["best_price"] + self.config.atr_stop_multiplier * atr_val
                    pos["stop_price"] = min(pos["stop_price"], trail_stop)

                stopped = False
                if pos["direction"] > 0 and price < pos["stop_price"]:
                    exit_price = pos["stop_price"]
                    stopped = True
                elif pos["direction"] < 0 and price > pos["stop_price"]:
                    exit_price = price
                    stopped = True

                max_hold = pos["bars_held"] >= self.config.max_hold_bars

                if stopped or max_hold:
                    ep = exit_price if stopped else price
                    reason = "stop" if stopped else "max_hold"

                    if pos["direction"] > 0:
                        pnl = pos["size"] * (ep / pos["entry_price"] - 1) * self.config.leverage
                    else:
                        pnl = pos["size"] * (pos["entry_price"] / ep - 1) * self.config.leverage

                    cost = abs(pos["size"]) * friction
                    equity[i] *= (1 + pnl - cost)
                    trades.append({"bar": i, "reason": reason, "pnl": pnl - cost})

                    # Set cooldown after stop
                    if stopped:
                        cooldown_remaining = self.cooldown_bars
                else:
                    if pos["direction"] > 0:
                        pnl = pos["size"] * (price / prev_price - 1) * self.config.leverage
                    else:
                        pnl = pos["size"] * (prev_price / price - 1) * self.config.leverage
                    equity[i] *= (1 + pnl)
                    surviving.append(pos)

            open_positions = surviving

            # Entry -- same as parent but with cooldown gate
            if len(open_positions) < self.config.max_open_trades and cooldown_remaining <= 0:
                sig = signals.iloc[i - 1]
                rs = regime_scalar[i - 1]
                tr_dir = trend[i - 1]

                if rs <= 0:
                    regime_blocks += 1
                elif tr_dir > 0 and sig < 0:
                    regime_blocks += 1
                elif tr_dir < 0 and sig > 0:
                    regime_blocks += 1
                else:
                    if sig > self.config.signal_threshold_long:
                        pos_size = min(sig, 1.0) * self.config.max_position_pct * rs
                        open_positions.append({
                            "direction": 1, "size": pos_size,
                            "entry_price": price,
                            "stop_price": price - self.config.atr_stop_multiplier * atr_val,
                            "best_price": price, "bars_held": 0,
                        })
                        cost = pos_size * friction
                        equity[i] *= (1 - cost)
                        trades.append({"bar": i, "reason": "entry_long", "size": pos_size})

                    elif sig < self.config.signal_threshold_short:
                        pos_size = abs(max(sig, -1.0)) * self.config.max_position_pct * rs
                        open_positions.append({
                            "direction": -1, "size": pos_size,
                            "entry_price": price,
                            "stop_price": price + self.config.atr_stop_multiplier * atr_val,
                            "best_price": price, "bars_held": 0,
                        })
                        cost = pos_size * friction
                        equity[i] *= (1 - cost)
                        trades.append({"bar": i, "reason": "entry_short", "size": pos_size})

        # Stats (same as parent)
        eq_series = pd.Series(equity, index=close.index)
        returns = eq_series.pct_change().dropna()
        running_max = eq_series.cummax()
        drawdown = (eq_series - running_max) / running_max
        max_dd = drawdown.min()
        total_return = eq_series.iloc[-1] / eq_series.iloc[0] - 1
        bars_per_year = 365.25 * 24 * 12
        sharpe = (returns.mean() / returns.std() * np.sqrt(bars_per_year)
                  if returns.std() > 0 else 0)
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


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def generate_features_and_signals(
    ohlcv_data: Dict[str, pd.DataFrame],
    config: Config,
    feature_names: List[str],
    weights: np.ndarray,
) -> Dict[str, Dict]:
    """Generate features and signals for all coins using frozen weights."""
    fe = ScalpingFeatureEngine(config)
    cae = CrossAssetEngine()
    btc_df = ohlcv_data.get("BTCUSDT")

    results = {}
    for sym, df in ohlcv_data.items():
        if sym == "BTCUSDT":
            continue  # BTC is reference only

        raw = fe.generate(df)
        normed = fe.normalize(raw)
        cross = cae.generate(btc_df, df, ohlcv_data, sym)
        cross_normed = fe.normalize(cross)
        combined = pd.concat([normed, cross_normed], axis=1)

        # Select only the frozen feature columns
        avail = [f for f in feature_names if f in combined.columns]
        combined = combined[avail]

        # Build signal engine with frozen weights
        sig_engine = SignalEngine(avail)
        # Map frozen weights to available features
        w = np.zeros(len(avail))
        for idx, fname in enumerate(avail):
            if fname in feature_names:
                fi = feature_names.index(fname)
                if fi < len(weights):
                    w[idx] = weights[fi]
        sig_engine.set_weights(w)

        signals = sig_engine.score(combined, rolling_window=config.rolling_norm_window)

        results[sym] = {
            "ohlcv": df,
            "features": combined,
            "signals": signals,
        }

    return results


def run_variant(
    name: str,
    ohlcv_data: Dict[str, pd.DataFrame],
    config: Config,
    feature_names: List[str],
    weights: np.ndarray,
    use_regime: bool = True,
    cooldown_bars: int = 0,
    coins: List[str] = None,
) -> Dict[str, Any]:
    """Run one variant configuration and return results."""
    if coins is None:
        coins = ["DOGEUSDT", "XLMUSDT", "SUIUSDT"]

    prepped = generate_features_and_signals(ohlcv_data, config, feature_names, weights)

    regime_detector = RegimeDetector(config) if use_regime else None
    if cooldown_bars > 0:
        backtester = BacktesterWithCooldown(config, cooldown_bars)
    else:
        backtester = ScalpingBacktester(config)

    per_coin = {}
    for sym in coins:
        if sym not in prepped:
            continue
        data = prepped[sym]
        df = data["ohlcv"]
        signals = data["signals"]

        regime_data = None
        if use_regime and regime_detector is not None:
            regime_data = regime_detector.compute_regime(df)

        result = backtester.run(df, signals, regime=regime_data)
        per_coin[sym] = result

    # Aggregate
    if not per_coin:
        return {"name": name, "per_coin": {}, "agg": {}}

    agg = {
        "total_return": np.mean([r["total_return"] for r in per_coin.values()]),
        "total_return_dollar": np.mean([r["total_return_dollar"] for r in per_coin.values()]),
        "max_drawdown": min(r["max_drawdown"] for r in per_coin.values()),
        "sharpe": np.mean([r["sharpe"] for r in per_coin.values()]),
        "n_trades": sum(r["n_trades"] for r in per_coin.values()),
        "win_rate": np.mean([r["win_rate"] for r in per_coin.values()]) if per_coin else 0,
        "avg_trade": np.mean([r["avg_trade"] for r in per_coin.values()]) if per_coin else 0,
        "profit_factor": np.mean([r["profit_factor"] for r in per_coin.values()]) if per_coin else 0,
    }

    return {"name": name, "per_coin": per_coin, "agg": agg}


def print_result(res: Dict):
    """Pretty-print results for one variant."""
    name = res["name"]
    agg = res["agg"]
    per_coin = res["per_coin"]

    print(f"\n{'='*72}")
    print(f"  VARIANT: {name}")
    print(f"{'='*72}")

    if not per_coin:
        print("  No data / no trades.")
        return

    for sym, r in per_coin.items():
        coin = sym.replace("USDT", "")
        print(f"\n  {coin}:")
        print(f"    Return:  {r['total_return']:+.2%}  (${r['total_return_dollar']:+,.0f})")
        print(f"    Max DD:  {r['max_drawdown']:.2%}")
        print(f"    Sharpe:  {r['sharpe']:.2f}")
        print(f"    Trades:  {r['n_trades']}")
        print(f"    Win rate: {r['win_rate']:.1%}")
        print(f"    Avg win:  {r['avg_win']:.4%}  |  Avg loss: {r['avg_loss']:.4%}")
        print(f"    Profit factor: {r['profit_factor']:.2f}")
        print(f"    Avg trade: {r['avg_trade']:.4%}")
        if r.get('regime_blocks', 0) > 0:
            print(f"    Regime blocks: {r['regime_blocks']}")

    print(f"\n  --- AGGREGATE ---")
    print(f"    Avg return:     {agg['total_return']:+.2%} (${agg['total_return_dollar']:+,.0f})")
    print(f"    Worst DD:       {agg['max_drawdown']:.2%}")
    print(f"    Avg Sharpe:     {agg['sharpe']:.2f}")
    print(f"    Total trades:   {agg['n_trades']}")
    print(f"    Avg win rate:   {agg['win_rate']:.1%}")
    print(f"    Avg profit factor: {agg['profit_factor']:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  STRATEGY VALIDATION TEST")
    print("  Testing frozen-weight model across parameter variants")
    print(f"  Run time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 72)

    # ── 1. Load frozen weights ──
    frozen_path = Path("frozen_weights.json")
    if not frozen_path.exists():
        frozen_path = Path(__file__).parent / "frozen_weights.json"
    frozen = json.loads(frozen_path.read_text())
    feature_names = frozen["features"]
    weights = np.array(frozen["weights"])

    print(f"\n[1/4] Frozen weights loaded:")
    print(f"  Version: {frozen['version']}")
    print(f"  Trained: {frozen['trained_at']}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Round-trip friction: {frozen['round_trip_friction']}")
    print(f"  All weights negative: {all(w <= 0 for w in weights)}")
    print(f"  Weight range: [{min(weights):.4f}, {max(weights):.4f}]")

    # ── 2. Fetch data ──
    COINS = ["DOGEUSDT", "XLMUSDT", "SUIUSDT"]
    FETCH_ASSETS = ["BTCUSDT"] + COINS + ["ETHUSDT"]

    print(f"\n[2/4] Fetching 30 days of 5m data from Binance.com...")
    config = Config(
        assets=FETCH_ASSETS,
        lookback_days=30,
        signal_threshold_long=0.4,
        signal_threshold_short=-0.4,
        atr_stop_multiplier=2.5,  # paper_trader uses 2.5
    )
    fetcher = BinanceUSDataFetcher(config)

    try:
        ohlcv_data = fetcher.fetch_all()
    except Exception as e:
        print(f"\n  ERROR fetching data: {e}")
        print(f"  If you see HTTP 451, Binance.com is geo-blocked.")
        print(f"  Try with a VPN or use Binance.US endpoints.")
        sys.exit(1)

    if not ohlcv_data:
        print("  ERROR: No data returned. Check connectivity.")
        sys.exit(1)

    print(f"\n  Data fetched successfully:")
    for sym, df in ohlcv_data.items():
        days = len(df) / (12 * 24)
        print(f"    {sym}: {len(df)} bars ({days:.1f} days) | "
              f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    # ── 3. Run all variants ──
    print(f"\n[3/4] Running backtest variants...")
    all_results = []

    # --- Variant A: Baseline (paper_trader config) ---
    print("\n  Running A: Baseline (threshold ±0.4, regime ON)...")
    cfg_a = Config(
        assets=FETCH_ASSETS, lookback_days=30,
        signal_threshold_long=0.4,
        signal_threshold_short=-0.4,
        atr_stop_multiplier=2.5,
    )
    res_a = run_variant("A: Baseline (±0.4, regime ON)", ohlcv_data, cfg_a,
                        feature_names, weights, use_regime=True, coins=COINS)
    all_results.append(res_a)

    # --- Variant B: Lower threshold ±0.3 ---
    print("  Running B: Threshold ±0.3...")
    cfg_b = Config(
        assets=FETCH_ASSETS, lookback_days=30,
        signal_threshold_long=0.3,
        signal_threshold_short=-0.3,
        atr_stop_multiplier=2.5,
    )
    res_b = run_variant("B: Lower threshold (±0.3, regime ON)", ohlcv_data, cfg_b,
                        feature_names, weights, use_regime=True, coins=COINS)
    all_results.append(res_b)

    # --- Variant C: Lower threshold ±0.2 ---
    print("  Running C: Threshold ±0.2...")
    cfg_c = Config(
        assets=FETCH_ASSETS, lookback_days=30,
        signal_threshold_long=0.2,
        signal_threshold_short=-0.2,
        atr_stop_multiplier=2.5,
    )
    res_c = run_variant("C: Lower threshold (±0.2, regime ON)", ohlcv_data, cfg_c,
                        feature_names, weights, use_regime=True, coins=COINS)
    all_results.append(res_c)

    # --- Variant D: Threshold ±0.4, NO regime ---
    print("  Running D: No regime detector...")
    cfg_d = Config(
        assets=FETCH_ASSETS, lookback_days=30,
        signal_threshold_long=0.4,
        signal_threshold_short=-0.4,
        atr_stop_multiplier=2.5,
    )
    res_d = run_variant("D: No regime (±0.4, regime OFF)", ohlcv_data, cfg_d,
                        feature_names, weights, use_regime=False, coins=COINS)
    all_results.append(res_d)

    # --- Variant E: Threshold ±0.4, with cooldown ---
    print("  Running E: 6-bar cooldown after stops...")
    cfg_e = Config(
        assets=FETCH_ASSETS, lookback_days=30,
        signal_threshold_long=0.4,
        signal_threshold_short=-0.4,
        atr_stop_multiplier=2.5,
    )
    res_e = run_variant("E: Cooldown 6-bar (±0.4, regime ON)", ohlcv_data, cfg_e,
                        feature_names, weights, use_regime=True, cooldown_bars=6, coins=COINS)
    all_results.append(res_e)

    # --- Variant F: INVERTED weights (momentum strategy) ---
    print("  Running F: Inverted weights (momentum)...")
    inverted_weights = -1.0 * weights
    cfg_f = Config(
        assets=FETCH_ASSETS, lookback_days=30,
        signal_threshold_long=0.4,
        signal_threshold_short=-0.4,
        atr_stop_multiplier=2.5,
    )
    res_f = run_variant("F: INVERTED weights (momentum)", ohlcv_data, cfg_f,
                        feature_names, inverted_weights, use_regime=True, coins=COINS)
    all_results.append(res_f)

    # ── 4. Print all results ──
    print(f"\n\n[4/4] Results:")
    for res in all_results:
        print_result(res)

    # ── COMPARISON TABLE ──
    print(f"\n\n{'='*100}")
    print(f"  SUMMARY COMPARISON TABLE")
    print(f"{'='*100}")
    header = f"{'Variant':<42} {'Return':>8} {'Dollar':>9} {'MaxDD':>8} {'Sharpe':>7} {'Trades':>7} {'WinRate':>8} {'PF':>6}"
    print(header)
    print("-" * 100)

    for res in all_results:
        a = res["agg"]
        if not a:
            print(f"  {res['name']:<40}  -- no data --")
            continue
        line = (f"  {res['name']:<40} {a['total_return']:>+7.2%} "
                f"${a['total_return_dollar']:>+8,.0f} "
                f"{a['max_drawdown']:>7.2%} "
                f"{a['sharpe']:>6.2f} "
                f"{a['n_trades']:>6d} "
                f"{a['win_rate']:>7.1%} "
                f"{a['profit_factor']:>5.2f}")
        print(line)

    # ── VERDICT ──
    print(f"\n{'='*100}")
    print("  VERDICT")
    print(f"{'='*100}")

    # Check if any variant has positive return
    positive = [r for r in all_results if r["agg"] and r["agg"]["total_return"] > 0]
    best = max(all_results, key=lambda r: r["agg"]["total_return"] if r["agg"] else -999)
    worst = min(all_results, key=lambda r: r["agg"]["total_return"] if r["agg"] else 999)

    if not positive:
        print("\n  NO variant produced positive returns on this 30-day window.")
        print("  The frozen weights do not have an out-of-sample edge in recent data.")
    else:
        print(f"\n  {len(positive)}/{len(all_results)} variants were profitable.")
        print(f"  Best: {best['name']}")
        if best["agg"]:
            print(f"    Return: {best['agg']['total_return']:+.2%}, "
                  f"Sharpe: {best['agg']['sharpe']:.2f}, "
                  f"Win rate: {best['agg']['win_rate']:.1%}")

    # Compare mean-reversion vs momentum
    mr_return = res_a["agg"]["total_return"] if res_a["agg"] else 0
    mom_return = res_f["agg"]["total_return"] if res_f["agg"] else 0

    print(f"\n  Mean-reversion (original) return: {mr_return:+.2%}")
    print(f"  Momentum (inverted) return:       {mom_return:+.2%}")
    if mom_return > mr_return:
        print(f"  --> INVERTED weights outperform! The model may have the wrong sign.")
    elif mr_return > mom_return:
        print(f"  --> Original mean-reversion direction is correct.")
    else:
        print(f"  --> Both are similar -- model may not have any real signal.")

    # Check regime impact
    regime_on = res_a["agg"]["total_return"] if res_a["agg"] else 0
    regime_off = res_d["agg"]["total_return"] if res_d["agg"] else 0
    print(f"\n  With regime filter:    {regime_on:+.2%}")
    print(f"  Without regime filter: {regime_off:+.2%}")
    if regime_off > regime_on:
        print(f"  --> Regime filter HURTS performance. Consider removing it.")
    else:
        print(f"  --> Regime filter helps (or at least doesn't hurt).")

    # Check cooldown
    base_return = res_a["agg"]["total_return"] if res_a["agg"] else 0
    cool_return = res_e["agg"]["total_return"] if res_e["agg"] else 0
    print(f"\n  Without cooldown: {base_return:+.2%}")
    print(f"  With 6-bar cooldown: {cool_return:+.2%}")
    if cool_return > base_return:
        print(f"  --> Cooldown HELPS. Avoiding re-entry after stops is beneficial.")
    else:
        print(f"  --> Cooldown does not help.")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
