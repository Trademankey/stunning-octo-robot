"""
Auto-Discovery Engine — scans exchange for all tradable pairs.

Filters by:
  - Minimum 24h volume (liquidity)
  - Spread threshold
  - Quote currency (USDT)
  - Market type (futures / spot)
  - Excludes stablecoins, leveraged tokens

Dynamically builds the pair universe at boot and refreshes periodically.
Also fetches real account balance for position sizing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import ccxt.async_support as ccxt_async
import structlog

from config.pairs import PairConfig, PairUniverse
from config.settings import get_settings

log = structlog.get_logger(__name__)

EXCLUDE_BASES: Set[str] = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "USDD",
    "WBTC", "WETH", "STETH",
    "BTCUP", "BTCDOWN", "ETHUP", "ETHDOWN",
    "BULL", "BEAR", "HALF",
}

CORRELATION_GROUPS = {
    "BTC": "btc_group", "ETH": "eth_group",
    "SOL": "sol_group", "AVAX": "sol_group", "NEAR": "sol_group", "SUI": "sol_group",
    "DOT": "alt_l1", "ADA": "alt_l1", "ATOM": "alt_l1", "APT": "alt_l1",
    "LINK": "oracle", "DOGE": "meme", "SHIB": "meme", "PEPE": "meme",
    "FLOKI": "meme", "WIF": "meme", "BONK": "meme",
    "XRP": "payment", "XLM": "payment", "TON": "messaging",
    "ARB": "l2", "OP": "l2", "MATIC": "l2", "STRK": "l2",
}


@dataclass
class DiscoveredPair:
    symbol: str
    base: str
    quote: str
    volume_24h_usd: float
    spread_bps: float
    price: float
    tick_size: float
    lot_size: float
    min_notional: float
    max_leverage: float
    correlation_group: str


class AutoDiscovery:
    """Scans exchange for tradable pairs, filters, and builds PairUniverse."""

    def __init__(
        self,
        min_volume_usd: float = 100_000,
        max_spread_bps: float = 20.0,
        max_pairs: int = 25,
        quote: str = "USDT",
    ):
        self._settings = get_settings()
        self.min_volume_usd = min_volume_usd
        self.max_spread_bps = max_spread_bps
        self.max_pairs = max_pairs
        self.quote = quote
        self._exchange: Optional[ccxt_async.Exchange] = None

    async def start(self) -> None:
        exchange_cls = getattr(ccxt_async, self._settings.exchange.id)
        self._exchange = exchange_cls({
            "apiKey": self._settings.exchange.api_key.get_secret_value(),
            "secret": self._settings.exchange.api_secret.get_secret_value(),
            "sandbox": self._settings.exchange.sandbox,
            "enableRateLimit": True,
            "options": {"defaultType": self._settings.market_type},
        })
        await self._exchange.load_markets()
        log.info(
            "autodiscovery.started",
            exchange=self._settings.exchange.id,
            total_markets=len(self._exchange.markets),
        )

    async def stop(self) -> None:
        if self._exchange:
            await self._exchange.close()

    async def discover(self) -> Tuple[List[DiscoveredPair], PairUniverse]:
        """Scan exchange, filter, rank by volume, return top pairs + universe."""
        if self._exchange is None:
            raise RuntimeError("AutoDiscovery not started")

        # ── 1. Filter markets ────────────────────────
        candidates: List[Dict] = []
        for symbol, market in self._exchange.markets.items():
            if not market.get("active", False):
                continue
            if market.get("quote") != self.quote:
                continue

            if self._settings.market_type == "futures":
                if market.get("type") not in ("swap", "future"):
                    continue
                if not market.get("linear", True):
                    continue
            else:
                if market.get("type") != "spot":
                    continue

            base = market.get("base", "")
            if base.upper() in EXCLUDE_BASES:
                continue
            if any(t in base.upper() for t in ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "2L", "2S")):
                continue

            candidates.append(market)

        log.info("autodiscovery.candidates", count=len(candidates))

        # ── 2. Fetch tickers ─────────────────────────
        tickers = {}
        try:
            tickers = await self._exchange.fetch_tickers(
                [m["symbol"] for m in candidates[:200]]
            )
        except Exception as exc:
            log.warning("autodiscovery.batch_tickers_failed", error=str(exc))
            for m in candidates[:50]:
                try:
                    t = await self._exchange.fetch_ticker(m["symbol"])
                    tickers[m["symbol"]] = t
                    await asyncio.sleep(0.05)
                except Exception:
                    pass

        # ── 3. Score and filter ──────────────────────
        discovered: List[DiscoveredPair] = []
        for market in candidates:
            symbol = market["symbol"]
            ticker = tickers.get(symbol)
            if not ticker:
                continue

            vol_usd = float(ticker.get("quoteVolume") or 0)
            if vol_usd < self.min_volume_usd:
                continue

            bid = float(ticker.get("bid") or 0)
            ask = float(ticker.get("ask") or 0)
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
            spread_bps = ((ask - bid) / mid * 10000) if mid > 0 else 999
            if spread_bps > self.max_spread_bps:
                continue

            base = market.get("base", "")
            precision = market.get("precision", {})
            limits = market.get("limits", {})

            tick_size = 10 ** (-precision.get("price", 2)) if precision.get("price") else 0.01
            lot_size = 10 ** (-precision.get("amount", 3)) if precision.get("amount") else 0.001
            min_notional = float(limits.get("cost", {}).get("min") or 5)
            max_lev = float(market.get("leverage", {}).get("max", 10)) if isinstance(market.get("leverage"), dict) else 10.0

            corr_group = CORRELATION_GROUPS.get(base, f"other_{base[:3].lower()}")

            discovered.append(DiscoveredPair(
                symbol=symbol, base=base, quote=self.quote,
                volume_24h_usd=vol_usd, spread_bps=spread_bps, price=mid,
                tick_size=tick_size, lot_size=lot_size,
                min_notional=min_notional, max_leverage=max_lev,
                correlation_group=corr_group,
            ))

        # ── 4. Rank by volume, take top N ────────────
        discovered.sort(key=lambda d: d.volume_24h_usd, reverse=True)
        discovered = discovered[: self.max_pairs]

        # ── 5. Build PairUniverse ────────────────────
        universe = PairUniverse(max_correlated=self._settings.risk.max_correlated_pairs)
        universe.pairs.clear()
        for d in discovered:
            universe.add_pair(PairConfig(
                symbol=d.symbol, base=d.base, quote=d.quote,
                max_leverage=min(d.max_leverage, self._settings.risk.max_leverage),
                tick_size=d.tick_size, lot_size=d.lot_size,
                min_notional=d.min_notional,
                correlation_group=d.correlation_group,
            ))

        log.info(
            "autodiscovery.complete",
            pairs=len(discovered),
            top_10=[d.symbol for d in discovered[:10]],
            total_volume=f"${sum(d.volume_24h_usd for d in discovered):,.0f}",
        )
        return discovered, universe

    async def get_account_balance(self) -> Dict[str, float]:
        """Fetch real account balance. Returns {total, free, used} in USDT."""
        if self._exchange is None:
            return {"total": 0, "free": 0, "used": 0}
        try:
            bal = await self._exchange.fetch_balance()
            for ccy in ["USDT", "USDC", "USD"]:
                info = bal.get(ccy)
                if info and isinstance(info, dict):
                    t = float(info.get("total") or 0)
                    if t > 0:
                        return {
                            "total": t,
                            "free": float(info.get("free") or 0),
                            "used": float(info.get("used") or 0),
                        }
            # Fallback: parse the 'total' top-level dict
            total_dict = bal.get("total", {})
            if isinstance(total_dict, dict):
                for ccy in ["USDT", "USDC", "USD"]:
                    v = float(total_dict.get(ccy) or 0)
                    if v > 0:
                        free_dict = bal.get("free", {})
                        used_dict = bal.get("used", {})
                        return {
                            "total": v,
                            "free": float(free_dict.get(ccy) or 0) if isinstance(free_dict, dict) else 0,
                            "used": float(used_dict.get(ccy) or 0) if isinstance(used_dict, dict) else 0,
                        }
            return {"total": 0, "free": 0, "used": 0}
        except Exception as exc:
            log.error("autodiscovery.balance_error", error=str(exc))
            return {"total": 0, "free": 0, "used": 0}

    async def refresh(self) -> Tuple[List[DiscoveredPair], PairUniverse, Dict[str, float]]:
        """Full refresh: discover pairs + fetch balance."""
        discovered, universe = await self.discover()
        balance = await self.get_account_balance()
        log.info("autodiscovery.refresh", pairs=len(discovered),
                 balance_total=balance["total"], balance_free=balance["free"])
        return discovered, universe, balance
