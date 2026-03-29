"""
Coin Ranker — 11-month backtest to find the best coins to trade alongside DOGE
================================================================================
Tests each candidate coin individually using the frozen model weights.
Ranks by Sharpe ratio and total return over 11 months of 5m data.

Usage:
    python coin_ranker.py
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from scalper import (
    Config, BinanceUSDataFetcher, ScalpingFeatureEngine,
    CrossAssetEngine, RegimeDetector, SignalEngine, ScalpingBacktester,
)

# ── Candidates (all available on HL testnet) ──
CANDIDATES = [
    "DOGEUSDT",   # baseline — already in use
    "SOLUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
    "XLMUSDT",
    "HBARUSDT",
    "AVAXUSDT",
    "SUIUSDT",
]

# Always fetch BTC + ETH for cross-asset features
ALWAYS_FETCH = ["BTCUSDT", "ETHUSDT"]

# 11 months of 30-day windows (same as frozen weight training)
MONTHS = 11
LOOKBACK_DAYS = 30

FROZEN_WEIGHTS_FILE = Path("frozen_weights.json")


def load_frozen_model():
    data = json.loads(FROZEN_WEIGHTS_FILE.read_text())
    return data["features"], np.array(data["weights"])


def backtest_coin(symbol: str, frozen_features, frozen_weights) -> dict:
    """Run 11-month backtest on a single coin using frozen weights."""
    all_returns = []
    all_trades = []
    total_bars = 0

    for month_i in range(MONTHS):
        end_offset = month_i * LOOKBACK_DAYS

        config = Config(
            assets=[symbol] + ALWAYS_FETCH,
            lookback_days=LOOKBACK_DAYS,
            end_offset_days=end_offset,
            signal_threshold_long=0.4,
            signal_threshold_short=-0.4,
            atr_stop_multiplier=2.5,
        )

        fetcher = BinanceUSDataFetcher(config)
        try:
            ohlcv = fetcher.fetch_all()
        except Exception as e:
            print(f"    [{symbol}] month {month_i+1} fetch failed: {e}")
            continue

        if symbol not in ohlcv or len(ohlcv[symbol]) < 200:
            continue

        # Deduplicate
        for s in list(ohlcv.keys()):
            df = ohlcv[s]
            if df.index.duplicated().any():
                ohlcv[s] = df[~df.index.duplicated(keep="last")]

        # Align to BTC index
        btc_df = ohlcv.get("BTCUSDT")
        if btc_df is not None:
            for s, df in ohlcv.items():
                if s != "BTCUSDT":
                    ohlcv[s] = df.reindex(
                        btc_df.index, method="nearest",
                        tolerance=pd.Timedelta("3m")
                    ).dropna()

        df = ohlcv[symbol]
        btc_df = ohlcv.get("BTCUSDT")

        fe = ScalpingFeatureEngine(config)
        cae = CrossAssetEngine()

        raw = fe.generate(df)
        normed = fe.normalize(raw)
        cross = cae.generate(btc_df, df, ohlcv, symbol)
        cross_normed = fe.normalize(cross)
        combined = pd.concat([normed, cross_normed], axis=1)

        # Align features to frozen model
        avail = [f for f in frozen_features if f in combined.columns]
        if len(avail) < len(frozen_features) * 0.5:
            continue

        feat_df = combined.reindex(columns=frozen_features, fill_value=0.0)

        # Signal
        se = SignalEngine(frozen_features)
        se.set_weights(frozen_weights)
        signals = se.score(feat_df)

        # Regime
        rd = RegimeDetector(config)
        regime = rd.compute_regime(df)

        # Backtest
        bt = ScalpingBacktester(config)
        result = bt.run(df, signals, regime)

        if result and result.get("trades"):
            all_returns.extend(result.get("daily_returns", pd.Series()).tolist())
            all_trades.extend([t for t in result["trades"] if "pnl" in t])
            total_bars += len(df)

    if not all_trades:
        return {"symbol": symbol, "total_return": 0, "sharpe": 0,
                "win_rate": 0, "n_trades": 0, "avg_pnl": 0}

    # Aggregate metrics
    wins = [t for t in all_trades if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(all_trades) if all_trades else 0
    avg_pnl = np.mean([t["pnl"] for t in all_trades]) if all_trades else 0

    # Compute equity curve from trades
    equity = 1.0
    for t in all_trades:
        equity *= (1 + t.get("pnl", 0))
    total_return = equity - 1.0

    # Sharpe from bar returns
    if all_returns and len(all_returns) > 10:
        r = np.array(all_returns)
        sharpe = (np.mean(r) / (np.std(r) + 1e-10)) * np.sqrt(12 * 30 * 24 * 12)
    else:
        sharpe = 0.0

    return {
        "symbol": symbol,
        "total_return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_trades": len(all_trades),
        "avg_pnl": avg_pnl,
    }


def main():
    print("=" * 65)
    print("  COIN RANKER — 11-month backtest with frozen V6 weights")
    print(f"  Candidates: {', '.join(CANDIDATES)}")
    print(f"  Signal threshold: ±0.4 | ATR stop: 2.5x")
    print("=" * 65)

    frozen_features, frozen_weights = load_frozen_model()
    print(f"\nLoaded frozen model: {len(frozen_features)} features\n")

    results = []
    for sym in CANDIDATES:
        print(f"  Backtesting {sym} ({MONTHS} months)...", end=" ", flush=True)
        t0 = time.time()
        r = backtest_coin(sym, frozen_features, frozen_weights)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.0f}s) | trades={r['n_trades']} | "
              f"WR={r['win_rate']*100:.0f}% | "
              f"ret={r['total_return']*100:+.1f}% | "
              f"sharpe={r['sharpe']:.2f}")
        results.append(r)

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print("\n" + "=" * 65)
    print("  RANKING (by Sharpe ratio)")
    print("=" * 65)
    print(f"  {'Coin':<12} {'Sharpe':>7} {'Total Ret':>10} {'Win Rate':>9} {'Trades':>7} {'Avg PnL':>9}")
    print("  " + "-" * 60)
    for r in results:
        sym = r['symbol'].replace('USDT', '')
        mark = " <-- already trading" if r['symbol'] == "DOGEUSDT" else ""
        print(f"  {sym:<12} {r['sharpe']:>7.2f} "
              f"{r['total_return']*100:>9.1f}% "
              f"{r['win_rate']*100:>8.0f}% "
              f"{r['n_trades']:>7} "
              f"{r['avg_pnl']*100:>8.2f}%"
              f"{mark}")

    # Recommendation
    non_doge = [r for r in results if r['symbol'] != 'DOGEUSDT']
    top2 = [r for r in non_doge if r['sharpe'] > 0 and r['n_trades'] >= 5][:2]

    print("\n" + "=" * 65)
    print("  RECOMMENDATION")
    print("=" * 65)
    print(f"  Keep: DOGE")
    if top2:
        for r in top2:
            print(f"  Add:  {r['symbol'].replace('USDT','')} "
                  f"(sharpe={r['sharpe']:.2f}, WR={r['win_rate']*100:.0f}%)")
    else:
        print("  No other coins cleared the bar — run DOGE only")


if __name__ == "__main__":
    main()
