"""
OKX Exchange Client

Full implementation for spot + perpetual swap trading on OKX.
Supports funding rate fetching, order placement, and position management.
"""

import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("btc_trader.exchanges.okx")


@dataclass
class OKXConfig:
    """OKX API configuration."""
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # OKX requires a passphrase
    testnet: bool = True
    timeout: int = 10


@dataclass
class OKXOrderResult:
    """Result of an OKX order."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    filled_qty: float
    status: str
    timestamp: datetime
    fee: float = 0.0
    fee_currency: str = ""


@dataclass
class OKXPosition:
    """OKX perpetual position."""
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    margin: float
    leverage: int
    liquidation_price: float


@dataclass
class OKXBalance:
    """OKX account balance."""
    currency: str
    available: float
    frozen: float
    total: float


class OKXClient:
    """
    OKX Exchange Client for spot and perpetual swap trading.

    Features:
    - Funding rate fetching
    - Spot trading (buy/sell)
    - Perpetual swap trading (long/short)
    - Position and balance queries
    - Proper authentication with passphrase

    Note: OKX uses different terminology:
    - "SWAP" = Perpetual futures
    - "SPOT" = Spot trading
    - Instrument ID format: "BTC-USDT" (spot), "BTC-USDT-SWAP" (perp)
    """

    # API URLs
    MAINNET_URL = "https://www.okx.com"
    TESTNET_URL = "https://www.okx.com"  # OKX uses same URL, different headers for demo

    def __init__(self, config: Optional[OKXConfig] = None):
        self.config = config or OKXConfig()
        self._session = requests.Session()

        # Set base URL
        self.base_url = self.TESTNET_URL if self.config.testnet else self.MAINNET_URL

        logger.info(
            "OKXClient initialized: testnet=%s",
            self.config.testnet,
        )

    def _get_timestamp(self) -> str:
        """Get ISO timestamp for OKX API."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Create OKX signature.

        Signature = Base64(HMAC-SHA256(timestamp + method + path + body, secret))
        """
        message = timestamp + method + path + body
        signature = hmac.new(
            self.config.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict:
        """Get authenticated headers for OKX API."""
        timestamp = self._get_timestamp()
        signature = self._sign(timestamp, method, path, body)

        headers = {
            "OK-ACCESS-KEY": self.config.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.config.passphrase,
            "Content-Type": "application/json",
        }

        if self.config.testnet:
            headers["x-simulated-trading"] = "1"

        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        authenticated: bool = False,
    ) -> Optional[Dict]:
        """Make an API request."""
        url = self.base_url + path

        try:
            body = ""
            if data:
                import json
                body = json.dumps(data)

            headers = {}
            if authenticated:
                headers = self._get_headers(method, path, body)

            if method == "GET":
                response = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout,
                )
            else:
                response = self._session.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=self.config.timeout,
                )

            response.raise_for_status()
            result = response.json()

            # OKX uses "code" field for errors (0 = success)
            if result.get("code") != "0":
                logger.error(
                    "OKX API error: code=%s, msg=%s",
                    result.get("code"),
                    result.get("msg"),
                )
                return None

            return result

        except requests.RequestException as e:
            logger.error("OKX request failed: %s", e)
            return None

    # =========================================================================
    # PUBLIC ENDPOINTS (No authentication required)
    # =========================================================================

    def get_funding_rate(self, symbol: str = "BTC-USDT-SWAP") -> Optional[float]:
        """
        Get current funding rate for a perpetual swap.

        Args:
            symbol: Instrument ID (e.g., "BTC-USDT-SWAP")

        Returns:
            Current funding rate as decimal (e.g., 0.0001 = 0.01%)
        """
        path = "/api/v5/public/funding-rate"
        params = {"instId": symbol}

        result = self._request("GET", path, params=params)

        if not result or not result.get("data"):
            logger.warning("Failed to get OKX funding rate")
            return None

        try:
            data = result["data"][0]
            rate = float(data["fundingRate"])
            next_rate = float(data.get("nextFundingRate", 0))

            logger.debug(
                "OKX funding rate: %.4f%% (next: %.4f%%)",
                rate * 100,
                next_rate * 100,
            )

            return rate

        except (KeyError, IndexError, ValueError) as e:
            logger.error("Failed to parse OKX funding rate: %s", e)
            return None

    def get_funding_rate_history(
        self,
        symbol: str = "BTC-USDT-SWAP",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get historical funding rates.

        Args:
            symbol: Instrument ID
            limit: Number of records (max 100)

        Returns:
            List of funding rate records
        """
        path = "/api/v5/public/funding-rate-history"
        params = {"instId": symbol, "limit": str(limit)}

        result = self._request("GET", path, params=params)

        if not result or not result.get("data"):
            return []

        return [
            {
                "timestamp": datetime.fromtimestamp(
                    int(r["fundingTime"]) / 1000, tz=timezone.utc
                ),
                "rate": float(r["fundingRate"]),
                "realized_rate": float(r.get("realizedRate", r["fundingRate"])),
            }
            for r in result["data"]
        ]

    def get_ticker(self, symbol: str = "BTC-USDT") -> Optional[Dict]:
        """Get current ticker data."""
        path = "/api/v5/market/ticker"
        params = {"instId": symbol}

        result = self._request("GET", path, params=params)

        if not result or not result.get("data"):
            return None

        try:
            data = result["data"][0]
            return {
                "symbol": symbol,
                "last_price": float(data["last"]),
                "bid": float(data["bidPx"]),
                "ask": float(data["askPx"]),
                "volume_24h": float(data["vol24h"]),
                "timestamp": datetime.fromtimestamp(
                    int(data["ts"]) / 1000, tz=timezone.utc
                ),
            }
        except (KeyError, IndexError, ValueError) as e:
            logger.error("Failed to parse ticker: %s", e)
            return None

    def get_mark_price(self, symbol: str = "BTC-USDT-SWAP") -> Optional[float]:
        """Get mark price for a swap."""
        path = "/api/v5/public/mark-price"
        params = {"instId": symbol}

        result = self._request("GET", path, params=params)

        if not result or not result.get("data"):
            return None

        try:
            return float(result["data"][0]["markPx"])
        except (KeyError, IndexError, ValueError):
            return None

    # =========================================================================
    # PRIVATE ENDPOINTS (Authentication required)
    # =========================================================================

    def get_account_balance(self, currency: str = "USDT") -> Optional[OKXBalance]:
        """Get account balance for a currency."""
        path = "/api/v5/account/balance"
        params = {"ccy": currency}

        result = self._request("GET", path, params=params, authenticated=True)

        if not result or not result.get("data"):
            return None

        try:
            details = result["data"][0]["details"]
            for d in details:
                if d["ccy"] == currency:
                    return OKXBalance(
                        currency=currency,
                        available=float(d["availBal"]),
                        frozen=float(d["frozenBal"]),
                        total=float(d["cashBal"]),
                    )
            return None
        except (KeyError, IndexError, ValueError) as e:
            logger.error("Failed to parse balance: %s", e)
            return None

    def get_positions(self, symbol: str = "BTC-USDT-SWAP") -> List[OKXPosition]:
        """Get open perpetual positions."""
        path = "/api/v5/account/positions"
        params = {"instId": symbol}

        result = self._request("GET", path, params=params, authenticated=True)

        if not result or not result.get("data"):
            return []

        positions = []
        for p in result["data"]:
            try:
                qty = float(p["pos"])
                if qty == 0:
                    continue

                positions.append(OKXPosition(
                    symbol=p["instId"],
                    side="long" if qty > 0 else "short",
                    quantity=abs(qty),
                    entry_price=float(p["avgPx"]) if p["avgPx"] else 0,
                    mark_price=float(p["markPx"]) if p["markPx"] else 0,
                    unrealized_pnl=float(p["upl"]) if p["upl"] else 0,
                    margin=float(p["margin"]) if p["margin"] else 0,
                    leverage=int(p["lever"]) if p["lever"] else 1,
                    liquidation_price=float(p["liqPx"]) if p["liqPx"] else 0,
                ))
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse position: %s", e)

        return positions

    def place_spot_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        order_type: str = "market",  # "market" or "limit"
        price: Optional[float] = None,
    ) -> Optional[OKXOrderResult]:
        """
        Place a spot order.

        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            side: "buy" or "sell"
            quantity: Amount in base currency
            order_type: "market" or "limit"
            price: Required for limit orders

        Returns:
            OKXOrderResult or None on failure
        """
        path = "/api/v5/trade/order"

        data = {
            "instId": symbol,
            "tdMode": "cash",  # Spot trading
            "side": side.lower(),
            "ordType": order_type.lower(),
            "sz": str(quantity),
        }

        if order_type.lower() == "limit" and price:
            data["px"] = str(price)

        result = self._request("POST", path, data=data, authenticated=True)

        if not result or not result.get("data"):
            return None

        try:
            order_data = result["data"][0]
            return OKXOrderResult(
                order_id=order_data["ordId"],
                client_order_id=order_data.get("clOrdId", ""),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(order_data.get("px", 0)) or 0,
                filled_qty=0,  # Need to query order for fill
                status="submitted",
                timestamp=datetime.now(timezone.utc),
            )
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse order result: %s", e)
            return None

    def place_swap_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[OKXOrderResult]:
        """
        Place a perpetual swap order.

        Args:
            symbol: Swap instrument (e.g., "BTC-USDT-SWAP")
            side: "buy" (long) or "sell" (short)
            quantity: Contract quantity
            order_type: "market" or "limit"
            price: Required for limit orders
            reduce_only: If True, only reduce existing position

        Returns:
            OKXOrderResult or None on failure
        """
        path = "/api/v5/trade/order"

        data = {
            "instId": symbol,
            "tdMode": "cross",  # Cross margin
            "side": side.lower(),
            "ordType": order_type.lower(),
            "sz": str(quantity),
        }

        if order_type.lower() == "limit" and price:
            data["px"] = str(price)

        if reduce_only:
            data["reduceOnly"] = True

        result = self._request("POST", path, data=data, authenticated=True)

        if not result or not result.get("data"):
            return None

        try:
            order_data = result["data"][0]
            return OKXOrderResult(
                order_id=order_data["ordId"],
                client_order_id=order_data.get("clOrdId", ""),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(order_data.get("px", 0)) or 0,
                filled_qty=0,
                status="submitted",
                timestamp=datetime.now(timezone.utc),
            )
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse order result: %s", e)
            return None

    def get_order_status(self, symbol: str, order_id: str) -> Optional[OKXOrderResult]:
        """Get status of an order."""
        path = "/api/v5/trade/order"
        params = {"instId": symbol, "ordId": order_id}

        result = self._request("GET", path, params=params, authenticated=True)

        if not result or not result.get("data"):
            return None

        try:
            data = result["data"][0]
            return OKXOrderResult(
                order_id=data["ordId"],
                client_order_id=data.get("clOrdId", ""),
                symbol=data["instId"],
                side=data["side"],
                quantity=float(data["sz"]),
                price=float(data["avgPx"]) if data["avgPx"] else 0,
                filled_qty=float(data["accFillSz"]),
                status=data["state"],  # live, canceled, partially_filled, filled
                timestamp=datetime.fromtimestamp(
                    int(data["uTime"]) / 1000, tz=timezone.utc
                ),
                fee=float(data["fee"]) if data["fee"] else 0,
                fee_currency=data.get("feeCcy", ""),
            )
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse order status: %s", e)
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a swap instrument."""
        path = "/api/v5/account/set-leverage"

        data = {
            "instId": symbol,
            "lever": str(leverage),
            "mgnMode": "cross",
        }

        result = self._request("POST", path, data=data, authenticated=True)

        return result is not None and result.get("code") == "0"

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        path = "/api/v5/trade/cancel-order"

        data = {
            "instId": symbol,
            "ordId": order_id,
        }

        result = self._request("POST", path, data=data, authenticated=True)

        return result is not None and result.get("code") == "0"


# Convenience function to compare funding rates across exchanges
def compare_funding_rates() -> Dict[str, float]:
    """
    Compare funding rates between Binance and OKX.

    Returns:
        Dict with exchange names and their current funding rates
    """
    rates = {}

    # OKX
    try:
        okx = OKXClient()
        okx_rate = okx.get_funding_rate()
        if okx_rate is not None:
            rates["okx"] = okx_rate
    except Exception as e:
        logger.warning("Failed to get OKX rate: %s", e)

    # Binance (import from existing client)
    try:
        from ..funding_arbitrage.exchange_client import BinanceClient
        binance = BinanceClient(testnet=True)
        # Would need to add get_funding_rate method to BinanceClient
        # For now, use rate_monitor
        from ..funding_arbitrage.rate_monitor import FundingRateMonitor
        monitor = FundingRateMonitor()
        binance_rate = monitor.get_binance_rate()
        if binance_rate:
            rates["binance"] = binance_rate.rate
    except Exception as e:
        logger.warning("Failed to get Binance rate: %s", e)

    return rates
