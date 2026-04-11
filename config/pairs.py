"""
Pair universe definitions & correlation groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class PairConfig:
    symbol: str
    base: str
    quote: str = "USDT"
    max_leverage: float = 10.0
    tick_size: float = 0.01
    lot_size: float = 0.001
    min_notional: float = 10.0
    correlation_group: str = "default"


# ── Default Universe ─────────────────────────────────────
DEFAULT_PAIRS: Dict[str, PairConfig] = {
    "BTC/USDT": PairConfig(
        symbol="BTC/USDT", base="BTC", max_leverage=20.0,
        tick_size=0.1, lot_size=0.001, min_notional=10.0,
        correlation_group="btc_group",
    ),
    "ETH/USDT": PairConfig(
        symbol="ETH/USDT", base="ETH", max_leverage=20.0,
        tick_size=0.01, lot_size=0.01, min_notional=10.0,
        correlation_group="eth_group",
    ),
    "SOL/USDT": PairConfig(
        symbol="SOL/USDT", base="SOL", max_leverage=10.0,
        tick_size=0.01, lot_size=0.1, min_notional=10.0,
        correlation_group="alt_group",
    ),
    "XRP/USDT": PairConfig(
        symbol="XRP/USDT", base="XRP", max_leverage=10.0,
        tick_size=0.0001, lot_size=1.0, min_notional=10.0,
        correlation_group="alt_group",
    ),
    "TON/USDT": PairConfig(
        symbol="TON/USDT", base="TON", max_leverage=5.0,
        tick_size=0.001, lot_size=0.1, min_notional=10.0,
        correlation_group="alt_group",
    ),
}


@dataclass
class PairUniverse:
    """Manages the tradable pair universe with correlation constraints."""
    pairs: Dict[str, PairConfig] = field(default_factory=lambda: dict(DEFAULT_PAIRS))
    max_correlated: int = 3

    def add_pair(self, cfg: PairConfig) -> None:
        self.pairs[cfg.symbol] = cfg

    def remove_pair(self, symbol: str) -> None:
        self.pairs.pop(symbol, None)

    @property
    def symbols(self) -> List[str]:
        return list(self.pairs.keys())

    def correlation_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for sym, cfg in self.pairs.items():
            groups.setdefault(cfg.correlation_group, []).append(sym)
        return groups

    def can_open(self, symbol: str, open_positions: Set[str]) -> bool:
        """Check correlation constraint before opening a new position."""
        cfg = self.pairs.get(symbol)
        if cfg is None:
            return False
        group = cfg.correlation_group
        same_group_open = sum(
            1 for s in open_positions
            if s in self.pairs and self.pairs[s].correlation_group == group
        )
        return same_group_open < self.max_correlated
