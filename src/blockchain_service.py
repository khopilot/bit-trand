"""
Blockchain Data Service for BTC Elite Trader - FREE API Edition

Uses free public APIs for on-chain metrics (no Bitcoin Core node required):
- Mempool.space: Mempool stats, fee rates (unlimited, no auth)
- Blockchair: Large transactions, network stats (1,000 calls/day free)

Author: khopilot
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger("btc_trader.blockchain")


# Free API endpoints
MEMPOOL_API = "https://mempool.space/api"
BLOCKCHAIR_API = "https://api.blockchair.com/bitcoin"


@dataclass
class MempoolMetrics:
    """Bitcoin mempool state."""
    size: int  # Number of pending transactions
    bytes: int  # Total size in bytes (vbytes)
    avg_fee_rate: float  # Satoshis per vbyte
    fast_fee: float  # Fee for next block
    medium_fee: float  # Fee for ~30 min
    slow_fee: float  # Fee for ~1 hour
    congestion: str  # "low", "normal", "high", "extreme"


@dataclass
class LargeTransaction:
    """Large BTC transaction."""
    txid: str
    value_btc: float
    block_height: Optional[int]
    timestamp: datetime


@dataclass
class OnChainSignal:
    """On-chain signal for trading decision adjustment."""
    mempool: MempoolMetrics
    large_tx_count: int  # Recent large transactions
    large_tx_volume: float  # Total BTC in large txs
    recommendation: str  # "boost_buy", "reduce_buy", "boost_sell", "reduce_sell", "neutral"
    confidence_adjustment: float  # Multiply trading signal confidence by this
    reason: str


class FreeAPIBlockchainService:
    """
    Free API-based blockchain service (no node required).

    Uses:
    - Mempool.space: Unlimited requests, no auth (mempool, fees)
    - Blockchair: 1,000 calls/day free (large txs, stats)
    """

    def __init__(
        self,
        mempool_api: str = MEMPOOL_API,
        blockchair_api: str = BLOCKCHAIR_API,
        large_tx_threshold: float = 100.0,  # BTC threshold for "large" tx
        cache_ttl: int = 300,  # Cache for 5 minutes
        blockchair_daily_limit: int = 1000,
        enabled: bool = True,
    ):
        """
        Initialize FreeAPIBlockchainService.

        Args:
            mempool_api: Mempool.space API base URL
            blockchair_api: Blockchair API base URL
            large_tx_threshold: BTC threshold for large transaction alerts
            cache_ttl: Cache duration in seconds
            blockchair_daily_limit: Max Blockchair calls per day
            enabled: Enable service
        """
        self.mempool_api = mempool_api.rstrip("/")
        self.blockchair_api = blockchair_api.rstrip("/")
        self.large_tx_threshold = large_tx_threshold
        self.cache_ttl = cache_ttl
        self.blockchair_daily_limit = blockchair_daily_limit
        self.enabled = enabled

        # Rate limiting for Blockchair
        self._blockchair_calls_today: int = 0
        self._blockchair_reset_date: date = date.today()

        # Cache
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._session: Optional[aiohttp.ClientSession] = None

        if enabled:
            logger.info(
                "FreeAPIBlockchainService initialized: mempool=%s, blockchair=%s",
                mempool_api, blockchair_api
            )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "BTC-Elite-Trader/1.0"}
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_cache(self, key: str) -> Optional[any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
        cached_time, value = self._cache[key]
        if (datetime.now(timezone.utc) - cached_time).total_seconds() > self.cache_ttl:
            del self._cache[key]
            return None
        return value

    def _set_cache(self, key: str, value: any):
        """Set cache value."""
        self._cache[key] = (datetime.now(timezone.utc), value)

    def _check_blockchair_limit(self) -> bool:
        """Check if we can make a Blockchair call."""
        today = date.today()
        if today != self._blockchair_reset_date:
            self._blockchair_calls_today = 0
            self._blockchair_reset_date = today

        return self._blockchair_calls_today < self.blockchair_daily_limit

    def _increment_blockchair_calls(self):
        """Track Blockchair API usage."""
        self._blockchair_calls_today += 1

    # ========================
    # Mempool.space API (FREE)
    # ========================

    async def get_mempool_info(self) -> Optional[MempoolMetrics]:
        """
        Get mempool statistics from Mempool.space.

        Endpoints:
        - /mempool: Mempool stats
        - /v1/fees/recommended: Fee estimates
        """
        cached = self._get_cache("mempool_info")
        if cached:
            return cached

        if not self.enabled:
            return None

        session = await self._get_session()

        try:
            # Get mempool stats
            async with session.get(f"{self.mempool_api}/mempool") as resp:
                if resp.status != 200:
                    logger.warning("Mempool.space /mempool failed: %d", resp.status)
                    return None
                mempool_data = await resp.json()

            # Get fee estimates
            async with session.get(f"{self.mempool_api}/v1/fees/recommended") as resp:
                if resp.status != 200:
                    logger.warning("Mempool.space /fees failed: %d", resp.status)
                    fees_data = {"fastestFee": 10, "halfHourFee": 5, "hourFee": 2}
                else:
                    fees_data = await resp.json()

            # Parse mempool data
            count = mempool_data.get("count", 0)
            vsize = mempool_data.get("vsize", 0)
            total_fee = mempool_data.get("total_fee", 0)

            # Calculate average fee rate
            avg_fee = (total_fee / vsize) if vsize > 0 else 0.0

            # Determine congestion level
            if count > 100000:
                congestion = "extreme"
            elif count > 50000:
                congestion = "high"
            elif count > 10000:
                congestion = "normal"
            else:
                congestion = "low"

            metrics = MempoolMetrics(
                size=count,
                bytes=vsize,
                avg_fee_rate=avg_fee,
                fast_fee=fees_data.get("fastestFee", 10),
                medium_fee=fees_data.get("halfHourFee", 5),
                slow_fee=fees_data.get("hourFee", 2),
                congestion=congestion,
            )

            self._set_cache("mempool_info", metrics)
            logger.debug("Mempool: %d txs, %s congestion, fast=%d sat/vB",
                        count, congestion, metrics.fast_fee)
            return metrics

        except asyncio.TimeoutError:
            logger.error("Mempool.space timeout")
            return None
        except aiohttp.ClientError as e:
            logger.error("Mempool.space error: %s", e)
            return None
        except Exception as e:
            logger.error("Mempool.space failed: %s", e)
            return None

    async def get_block_height(self) -> Optional[int]:
        """Get current block height from Mempool.space."""
        if not self.enabled:
            return None

        session = await self._get_session()
        try:
            async with session.get(f"{self.mempool_api}/blocks/tip/height") as resp:
                if resp.status == 200:
                    return int(await resp.text())
                return None
        except Exception as e:
            logger.error("Failed to get block height: %s", e)
            return None

    # ========================
    # Blockchair API (1,000/day)
    # ========================

    async def get_large_transactions(self, min_btc: float = None) -> List[LargeTransaction]:
        """
        Get recent large transactions from Blockchair.

        Rate limited to 1,000 calls/day.
        """
        min_btc = min_btc or self.large_tx_threshold

        cached = self._get_cache(f"large_txs_{min_btc}")
        if cached:
            return cached

        if not self.enabled:
            return []

        if not self._check_blockchair_limit():
            logger.warning("Blockchair daily limit reached (%d)", self.blockchair_daily_limit)
            return []

        session = await self._get_session()

        try:
            # Convert BTC to satoshis
            min_sats = int(min_btc * 100_000_000)

            # Query large transactions
            url = f"{self.blockchair_api}/transactions"
            params = {
                "q": f"output_total({min_sats}..)",
                "s": "time(desc)",
                "limit": 10,
            }

            async with session.get(url, params=params) as resp:
                self._increment_blockchair_calls()

                if resp.status == 430:
                    logger.debug("Blockchair rate limited (free tier)")
                    return []
                if resp.status != 200:
                    logger.warning("Blockchair failed: %d", resp.status)
                    return []

                data = await resp.json()

            transactions = []
            for tx in data.get("data", []):
                try:
                    transactions.append(LargeTransaction(
                        txid=tx.get("hash", ""),
                        value_btc=tx.get("output_total", 0) / 100_000_000,
                        block_height=tx.get("block_id"),
                        timestamp=datetime.fromisoformat(
                            tx.get("time", "").replace("Z", "+00:00")
                        ) if tx.get("time") else datetime.now(timezone.utc),
                    ))
                except Exception:
                    continue

            self._set_cache(f"large_txs_{min_btc}", transactions)
            logger.debug("Found %d large txs (>%.0f BTC)", len(transactions), min_btc)
            return transactions

        except asyncio.TimeoutError:
            logger.error("Blockchair timeout")
            return []
        except aiohttp.ClientError as e:
            logger.error("Blockchair error: %s", e)
            return []
        except Exception as e:
            logger.error("Blockchair failed: %s", e)
            return []

    async def get_network_stats(self) -> Optional[dict]:
        """Get Bitcoin network stats from Blockchair."""
        cached = self._get_cache("network_stats")
        if cached:
            return cached

        if not self.enabled:
            return None

        if not self._check_blockchair_limit():
            return None

        session = await self._get_session()

        try:
            async with session.get(f"{self.blockchair_api}/stats") as resp:
                self._increment_blockchair_calls()

                if resp.status != 200:
                    return None

                data = await resp.json()
                stats = data.get("data", {})
                self._set_cache("network_stats", stats)
                return stats

        except Exception as e:
            logger.error("Blockchair stats failed: %s", e)
            return None

    # ========================
    # Combined Signal
    # ========================

    async def get_on_chain_signal(self) -> Optional[OnChainSignal]:
        """
        Get combined on-chain signal for trading decision adjustment.

        Uses:
        - Mempool congestion/fees (momentum indicator)
        - Large transaction activity (whale alert)
        """
        if not self.enabled:
            return None

        # Get mempool info (always available)
        mempool = await self.get_mempool_info()
        if mempool is None:
            return None

        # Get large transactions (rate limited)
        large_txs = await self.get_large_transactions()
        large_tx_count = len(large_txs)
        large_tx_volume = sum(tx.value_btc for tx in large_txs)

        # Determine signal based on conditions
        confidence_adjustment = 1.0
        reasons = []

        # High mempool congestion = market activity = momentum
        if mempool.congestion == "extreme":
            confidence_adjustment *= 1.15
            reasons.append("extreme_mempool")
        elif mempool.congestion == "high":
            confidence_adjustment *= 1.08
            reasons.append("high_mempool")
        elif mempool.congestion == "low":
            confidence_adjustment *= 0.95
            reasons.append("low_activity")

        # High fees = urgency = strong moves
        if mempool.fast_fee > 50:  # High fee environment
            confidence_adjustment *= 1.05
            reasons.append("high_fees")
        elif mempool.fast_fee < 5:  # Very low fees
            confidence_adjustment *= 0.98
            reasons.append("low_fees")

        # Large transaction activity = whale movement
        if large_tx_count >= 5:
            confidence_adjustment *= 1.10
            reasons.append(f"whale_activity({large_tx_count})")
        elif large_tx_count >= 3:
            confidence_adjustment *= 1.05
            reasons.append(f"large_txs({large_tx_count})")

        # Determine recommendation
        if confidence_adjustment > 1.10:
            recommendation = "boost_buy"  # High activity = momentum
        elif confidence_adjustment > 1.0:
            recommendation = "neutral"
        elif confidence_adjustment < 0.95:
            recommendation = "reduce_buy"  # Low activity = caution
        else:
            recommendation = "neutral"

        # Clamp adjustment
        confidence_adjustment = min(max(confidence_adjustment, 0.8), 1.25)

        return OnChainSignal(
            mempool=mempool,
            large_tx_count=large_tx_count,
            large_tx_volume=large_tx_volume,
            recommendation=recommendation,
            confidence_adjustment=confidence_adjustment,
            reason=", ".join(reasons) if reasons else "normal",
        )

    async def health_check(self) -> bool:
        """Check if APIs are accessible."""
        if not self.enabled:
            return False

        try:
            height = await self.get_block_height()
            if height:
                logger.info("Blockchain APIs connected: block=%d", height)
                return True
            return False
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    def get_status(self) -> dict:
        """Get service status for monitoring."""
        return {
            "enabled": self.enabled,
            "provider": "free_api",
            "mempool_api": self.mempool_api,
            "blockchair_api": self.blockchair_api,
            "blockchair_calls_today": self._blockchair_calls_today,
            "blockchair_daily_limit": self.blockchair_daily_limit,
            "large_tx_threshold": self.large_tx_threshold,
            "cache_ttl": self.cache_ttl,
        }


# Backwards compatibility alias
BlockchainDataService = FreeAPIBlockchainService
