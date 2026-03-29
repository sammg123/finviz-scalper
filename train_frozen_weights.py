"""
Train V6 Frozen Weights — 11 months, fee-aware optimization
================================================================================
IC optimization uses net P&L (after HL fees + slippage) as the target variable,
not raw returns. This means features that predict small moves (eaten by fees)
get penalized, and features predicting large tradeable moves get rewarded.

Usage:
    python train_frozen_weights.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

from scalper import (
    Config, BinanceUSDataFetcher, ScalpingFeatureEngine,
    CrossAssetEngine,
)

# ── HL Fee Structure ──
LEVERAGE = 5
HL_TAKER_FEE = 0.00045       # 0.045% per side
HL_SLIPPAGE = 0.0003          # 0.03% per side
# Round-trip friction (both sides, leveraged):
#   (2 * taker_fee + 2 * slippage) * leverage
ROUND_TRIP_FRICTION = (2 * HL_TAKER_FEE + 2 * HL_SLIPPAGE) * LEVERAGE  # = 0.0075

# Assets: same as live trader
FETCH_ASSETS = ["BTCUSDT", "ADAUSDT", "DOGEUSDT", "ETHUSDT"]

# 11 monthly windows
TRAINING_WINDOWS = [
    {"end_offset_days": i * 30, "lookback_days": 30, "label": f"Month {i+1}"}
    for i in range(11)
]

OUTPUT_FILE = Path("frozen_weights.json")


def compute_net_pnl(fwd_ret: pd.Series, leverage: int, friction: float) -> pd.Series:
    """
    Convert raw forward returns to net P&L after fees.

    For a long trade: pnl = ret * leverage - friction
    For a short trade: pnl = -ret * leverage - friction
    Generalized: pnl = |ret| * leverage - friction (positive if move covers fees)
    Preserve sign: net = sign(ret) * max(0, |ret| * leverage - friction)

    Small moves that don't cover fees become 0 (not profitable either direction).
    Large moves keep their sign but are reduced by friction.
    """
    abs_ret = fwd_ret.abs()
    gross_pnl = abs_ret * leverage
    net_magnitude = (gross_pnl - friction).clip(lower=0)
    return np.sign(fwd_ret) * net_magnitude


def fee_aware_optimize(asset_data, feature_names, prediction_horizon,
                       leverage, friction):
    """
    IC-weighted optimization using net P&L (after fees) as target.

    This penalizes features that correlate with small moves (eaten by fees)
    and rewards features that predict large, tradeable moves.
    """
    n_features = len(feature_names)
    ic_weights = np.zeros(n_features)

    for feat_idx, feat_name in enumerate(feature_names):
        ics = []
        for asset, data in asset_data.items():
            features = data["features"]
            ohlcv = data["ohlcv"]
            split = data["split_idx"]

            train_feat = features[feat_name].iloc[:split]
            h = prediction_horizon
            raw_fwd = ohlcv["close"].iloc[:split].pct_change(h).shift(-h)

            # Convert to net P&L after fees
            net_fwd = compute_net_pnl(raw_fwd, leverage, friction)

            valid = train_feat.notna() & net_fwd.notna() & (net_fwd != 0)
            if valid.sum() < 50:
                continue

            ic, _ = scipy_stats.spearmanr(
                train_feat[valid].values, net_fwd[valid].values
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

    return ic_weights


def main():
    print("=" * 70)
    print("V6 FROZEN WEIGHT TRAINING — 11 months, FEE-AWARE")
    print("=" * 70)
    print(f"Assets: {FETCH_ASSETS}")
    print(f"Windows: {len(TRAINING_WINDOWS)} x 30 days")
    print(f"Leverage: {LEVERAGE}x")
    print(f"HL taker fee: {HL_TAKER_FEE*100:.3f}%/side")
    print(f"Slippage: {HL_SLIPPAGE*100:.3f}%/side")
    print(f"Round-trip friction: {ROUND_TRIP_FRICTION*100:.2f}% "
          f"(moves < {ROUND_TRIP_FRICTION/LEVERAGE*100:.3f}% raw are unprofitable)\n")

    all_weights = []
    canonical_features = None
    actual_features = None

    for win_idx, win in enumerate(TRAINING_WINDOWS):
        print(f"\n--- Window {win_idx+1}/{len(TRAINING_WINDOWS)}: {win['label']} "
              f"(offset={win['end_offset_days']}d) ---")

        train_config = Config(
            assets=FETCH_ASSETS,
            lookback_days=win["lookback_days"],
            end_offset_days=win["end_offset_days"],
        )

        fetcher = BinanceUSDataFetcher(train_config)
        try:
            ohlcv_window = fetcher.fetch_all()
        except Exception as e:
            print(f"  SKIP: fetch failed — {e}")
            continue

        if len(ohlcv_window) < 2:
            print(f"  SKIP: insufficient data ({len(ohlcv_window)} assets)")
            continue

        for sym, df in ohlcv_window.items():
            print(f"  {sym}: {len(df)} bars")

        # Deduplicate and align to BTC index
        for sym in list(ohlcv_window.keys()):
            df = ohlcv_window[sym]
            if df.index.duplicated().any():
                ohlcv_window[sym] = df[~df.index.duplicated(keep="last")]

        btc_df = ohlcv_window.get("BTCUSDT")
        if btc_df is not None:
            common_idx = btc_df.index
            for sym, df in ohlcv_window.items():
                if sym != "BTCUSDT":
                    ohlcv_window[sym] = df.reindex(
                        common_idx, method="nearest", tolerance=pd.Timedelta("3m")
                    ).dropna()

        # Feature engineering
        fe = ScalpingFeatureEngine(train_config)
        cae = CrossAssetEngine()
        btc_df = ohlcv_window.get("BTCUSDT")

        asset_features = {}
        asset_fwd = {}
        for sym, df in ohlcv_window.items():
            raw = fe.generate(df)
            normed = fe.normalize(raw)
            cross = cae.generate(btc_df, df, ohlcv_window, sym)
            cross_normed = fe.normalize(cross)
            combined = pd.concat([normed, cross_normed], axis=1)
            asset_features[sym] = combined
            h = train_config.prediction_horizon
            asset_fwd[sym] = df["close"].pct_change(h).shift(-h)

        # Feature selection: first window sets the canonical feature list
        if canonical_features is None:
            pooled_feat = pd.concat([asset_features[s] for s in ohlcv_window], axis=0)
            pooled_fwd = pd.concat([asset_fwd[s] for s in ohlcv_window], axis=0)
            canonical_features = fe.select_features(
                pooled_feat, pooled_fwd, max_features=train_config.max_features
            )
            for f in ["btc_ret_lag1", "btc_mom12", "cs_rank_12", "rel_vs_btc_12"]:
                if f in pooled_feat.columns and f not in canonical_features:
                    canonical_features.append(f)
            canonical_features = [f for f in canonical_features
                                  if "taker_buy_ratio" not in f]

        for sym in asset_features:
            avail = [f for f in canonical_features if f in asset_features[sym].columns]
            asset_features[sym] = asset_features[sym][avail]

        actual_features = list(asset_features[list(asset_features.keys())[0]].columns)

        # Build optimizer data
        optimizer_data = {}
        for sym in ohlcv_window:
            features = asset_features[sym]
            ohlcv = ohlcv_window[sym]
            common = features.index.intersection(ohlcv.index)
            features = features.loc[common]
            ohlcv = ohlcv.loc[common]
            optimizer_data[sym] = {
                "features": features,
                "ohlcv": ohlcv,
                "split_idx": len(features),
            }

        try:
            w = fee_aware_optimize(
                optimizer_data, actual_features,
                prediction_horizon=train_config.prediction_horizon,
                leverage=LEVERAGE,
                friction=ROUND_TRIP_FRICTION,
            )
            n_active = np.sum(np.abs(w) > 0.05)
            all_weights.append(w)
            print(f"  Trained: {len(actual_features)} features, active={n_active}")

            # Show top weights for this window
            for name, wt in sorted(zip(actual_features, w),
                                    key=lambda x: abs(x[1]), reverse=True)[:5]:
                if abs(wt) > 0.05:
                    print(f"    {name:30s} w={wt:+.4f}")
        except Exception as e:
            print(f"  SKIP: optimization failed — {e}")
            continue

    if not all_weights:
        print("\nERROR: All windows failed. Cannot produce weights.")
        sys.exit(1)

    # ── Average across windows ──
    avg_weights = np.mean(all_weights, axis=0)

    # ── Normalize: L2 unit norm ──
    l2_norm = np.linalg.norm(avg_weights)
    if l2_norm > 0:
        normalized_weights = avg_weights / l2_norm
    else:
        normalized_weights = avg_weights

    # Zero out noise (|w| < 0.02 after normalization)
    noise_mask = np.abs(normalized_weights) < 0.02
    normalized_weights[noise_mask] = 0.0
    l2_norm = np.linalg.norm(normalized_weights)
    if l2_norm > 0:
        normalized_weights = normalized_weights / l2_norm

    # ── Print results ──
    print("\n" + "=" * 70)
    print(f"FROZEN MODEL (FEE-AWARE): {len(all_weights)}/{len(TRAINING_WINDOWS)} windows")
    print(f"Features: {len(actual_features)}, "
          f"Active: {np.sum(np.abs(normalized_weights) > 0)}")
    print(f"Round-trip friction: {ROUND_TRIP_FRICTION*100:.2f}%")
    print(f"L2 norm: {np.linalg.norm(normalized_weights):.6f}")
    print("=" * 70)

    print("\nWeight table (sorted by magnitude):")
    for name, w in sorted(zip(actual_features, normalized_weights),
                           key=lambda x: abs(x[1]), reverse=True):
        marker = " ***" if abs(w) > 0.3 else ""
        print(f"  {name:30s}  w={w:+.6f}{marker}")

    # ── Save to JSON ──
    output = {
        "version": "v6_fee_aware",
        "training_windows": len(all_weights),
        "training_months": len(TRAINING_WINDOWS),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "assets": FETCH_ASSETS,
        "normalization": "L2_unit_norm",
        "leverage": LEVERAGE,
        "hl_taker_fee_per_side": HL_TAKER_FEE,
        "slippage_per_side": HL_SLIPPAGE,
        "round_trip_friction": ROUND_TRIP_FRICTION,
        "features": actual_features,
        "weights": normalized_weights.tolist(),
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
