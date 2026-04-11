"""
Order-book micro-structure features.

Computed from real-time WebSocket snapshots cached in Redis.
Features: bid-ask spread, imbalance, depth, pressure, weighted mid-price.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


def compute_orderbook_features(ob: Dict[str, Any]) -> Dict[str, float]:
    """
    From a raw orderbook dict {bids: [[p,q],...], asks: [[p,q],...]}
    compute micro-structure features.
    """
    bids: List[List[float]] = ob.get("bids", [])
    asks: List[List[float]] = ob.get("asks", [])

    if not bids or not asks:
        return _empty_features()

    bid_prices = np.array([b[0] for b in bids], dtype=np.float64)
    bid_sizes = np.array([b[1] for b in bids], dtype=np.float64)
    ask_prices = np.array([a[0] for a in asks], dtype=np.float64)
    ask_sizes = np.array([a[1] for a in asks], dtype=np.float64)

    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    mid = (best_bid + best_ask) / 2

    features = {}

    # ── Spread ───────────────────────────────────────
    features["spread"] = best_ask - best_bid
    features["spread_bps"] = (best_ask - best_bid) / (mid + 1e-15) * 10000

    # ── Weighted mid-price ───────────────────────────
    total_bid_vol = bid_sizes.sum()
    total_ask_vol = ask_sizes.sum()
    features["weighted_mid"] = (
        best_bid * total_ask_vol + best_ask * total_bid_vol
    ) / (total_bid_vol + total_ask_vol + 1e-15)

    # ── Imbalance (top N levels) ─────────────────────
    for n in [1, 3, 5, 10]:
        bv = bid_sizes[:n].sum()
        av = ask_sizes[:n].sum()
        features[f"imbalance_{n}"] = (bv - av) / (bv + av + 1e-15)

    # ── Depth (total notional within 0.5%, 1%, 2%) ──
    for pct in [0.005, 0.01, 0.02]:
        bid_mask = bid_prices >= mid * (1 - pct)
        ask_mask = ask_prices <= mid * (1 + pct)
        bd = (bid_prices[bid_mask] * bid_sizes[bid_mask]).sum()
        ad = (ask_prices[ask_mask] * ask_sizes[ask_mask]).sum()
        pct_label = str(int(pct * 100))
        features[f"bid_depth_{pct_label}pct"] = bd
        features[f"ask_depth_{pct_label}pct"] = ad
        features[f"depth_ratio_{pct_label}pct"] = bd / (ad + 1e-15)

    # ── Pressure (cumulative volume gradient) ────────
    features["bid_pressure"] = _volume_gradient(bid_sizes)
    features["ask_pressure"] = _volume_gradient(ask_sizes)

    # ── Large-order detection ────────────────────────
    bid_mean = bid_sizes.mean()
    ask_mean = ask_sizes.mean()
    features["large_bid_count"] = float(np.sum(bid_sizes > 3 * bid_mean))
    features["large_ask_count"] = float(np.sum(ask_sizes > 3 * ask_mean))
    features["whale_imbalance"] = (
        features["large_bid_count"] - features["large_ask_count"]
    ) / (features["large_bid_count"] + features["large_ask_count"] + 1e-15)

    return features


def _volume_gradient(sizes: np.ndarray) -> float:
    """Measure how concentrated volume is at the top of book."""
    if len(sizes) < 2:
        return 0.0
    cumsum = np.cumsum(sizes)
    total = cumsum[-1]
    if total <= 0:
        return 0.0
    # Fraction of volume in top 3 levels vs total
    top3 = cumsum[min(2, len(cumsum) - 1)]
    return float(top3 / total)


def _empty_features() -> Dict[str, float]:
    """Return zeroed feature dict when orderbook is unavailable."""
    return {
        "spread": 0.0, "spread_bps": 0.0, "weighted_mid": 0.0,
        "imbalance_1": 0.0, "imbalance_3": 0.0, "imbalance_5": 0.0,
        "imbalance_10": 0.0,
        "bid_depth_0pct": 0.0, "ask_depth_0pct": 0.0, "depth_ratio_0pct": 0.0,
        "bid_depth_1pct": 0.0, "ask_depth_1pct": 0.0, "depth_ratio_1pct": 0.0,
        "bid_depth_2pct": 0.0, "ask_depth_2pct": 0.0, "depth_ratio_2pct": 0.0,
        "bid_pressure": 0.0, "ask_pressure": 0.0,
        "large_bid_count": 0.0, "large_ask_count": 0.0, "whale_imbalance": 0.0,
    }
