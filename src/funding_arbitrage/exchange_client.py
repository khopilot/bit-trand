"""
Exchange Client for Funding Arbitrage

Unified API wrapper for spot and perpetual futures trading
across multiple exchanges.
"""

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger("btc_trader.funding_arb.exchange_client")


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


@dataclass
class OrderResult:
    """Result of an order execution."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    filled_qty: float
    status: str
    timestamp: datetime
    commission: float = 0.0
    commission_asset: str = ""


@dataclass
class AccountBalance:
    """Account balance information."""

    asset: str
    free: float
    locked: float
    total: float


@dataclass
class FuturesPosition:
    """Futures position information."""

    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin: float
    liquidation_price: float


class ExchangeClient:
    """
    Base exchange client with common functionality.

    Subclasses implement exchange-specific API calls.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        timeout: int = 10,
    ):
        """
        Initialize exchange client.

        Args:
            api_key: API key for authenticated endpoints
            api_secret: API secret for signing requests
            testnet: Use testnet endpoints if True
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.timeout = timeout
        self._session = requests.Session()

    def _sign_request(self, params: Dict) -> str:
        """Sign request parameters with HMAC-SHA256."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)

    def get_spot_balance(self, asset: str = "BTC") -> Optional[AccountBalance]:
        """Get spot account balance for an asset."""
        raise NotImplementedError

    def get_futures_balance(self, asset: str = "USDT") -> Optional[AccountBalance]:
        """Get futures account balance for an asset."""
        raise NotImplementedError

    def get_futures_position(self, symbol: str) -> Optional[FuturesPosition]:
        """Get current futures position for a symbol."""
        raise NotImplementedError

    def place_spot_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Optional[OrderResult]:
        """Place a spot market/limit order."""
        raise NotImplementedError

    def place_futures_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[OrderResult]:
        """Place a futures market/limit order."""
        raise NotImplementedError

    def set_futures_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a futures symbol."""
        raise NotImplementedError

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price."""
        raise NotImplementedError

    def get_futures_price(self, symbol: str) -> Optional[float]:
        """Get current futures mark price."""
        raise NotImplementedError


class BinanceClient(ExchangeClient):
    """
    Binance exchange client for spot and USDT-M futures.

    Supports both mainnet and testnet endpoints.
    """

    SPOT_BASE_URL = "https://api.binance.com"
    SPOT_TESTNET_URL = "https://testnet.binance.vision"
    FUTURES_BASE_URL = "https://fapi.binance.com"
    FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        timeout: int = 10,
    ):
        super().__init__(api_key, api_secret, testnet, timeout)

        self.spot_url = self.SPOT_TESTNET_URL if testnet else self.SPOT_BASE_URL
        self.futures_url = self.FUTURES_TESTNET_URL if testnet else self.FUTURES_BASE_URL

        logger.info(
            "BinanceClient initialized (testnet=%s, spot=%s, futures=%s)",
            testnet,
            self.spot_url,
            self.futures_url,
        )

    def _headers(self) -> Dict:
        """Get headers for authenticated requests."""
        return {"X-MBX-APIKEY": self.api_key}

    def get_spot_price(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Get current spot price from Binance."""
        try:
            url = f"{self.spot_url}/api/v3/ticker/price"
            response = self._session.get(
                url, params={"symbol": symbol}, timeout=self.timeout
            )
            response.raise_for_status()
            return float(response.json()["price"])
        except requests.RequestException as e:
            logger.error("Failed to get spot price: %s", e)
            return None

    def get_futures_price(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Get current futures mark price from Binance."""
        try:
            url = f"{self.futures_url}/fapi/v1/premiumIndex"
            response = self._session.get(
                url, params={"symbol": symbol}, timeout=self.timeout
            )
            response.raise_for_status()
            return float(response.json()["markPrice"])
        except requests.RequestException as e:
            logger.error("Failed to get futures price: %s", e)
            return None

    def get_spot_balance(self, asset: str = "BTC") -> Optional[AccountBalance]:
        """Get spot account balance for an asset."""
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials required for balance check")
            return None

        try:
            params = {"timestamp": self._get_timestamp()}
            params["signature"] = self._sign_request(params)

            url = f"{self.spot_url}/api/v3/account"
            response = self._session.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            for balance in data.get("balances", []):
                if balance["asset"] == asset:
                    free = float(balance["free"])
                    locked = float(balance["locked"])
                    return AccountBalance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=free + locked,
                    )

            return AccountBalance(asset=asset, free=0.0, locked=0.0, total=0.0)

        except requests.RequestException as e:
            logger.error("Failed to get spot balance: %s", e)
            return None

    def get_futures_balance(self, asset: str = "USDT") -> Optional[AccountBalance]:
        """Get futures account balance for an asset."""
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials required for balance check")
            return None

        try:
            params = {"timestamp": self._get_timestamp()}
            params["signature"] = self._sign_request(params)

            url = f"{self.futures_url}/fapi/v2/balance"
            response = self._session.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            for balance in data:
                if balance["asset"] == asset:
                    wallet = float(balance["balance"])
                    available = float(balance["availableBalance"])
                    return AccountBalance(
                        asset=asset,
                        free=available,
                        locked=wallet - available,
                        total=wallet,
                    )

            return AccountBalance(asset=asset, free=0.0, locked=0.0, total=0.0)

        except requests.RequestException as e:
            logger.error("Failed to get futures balance: %s", e)
            return None

    def get_futures_position(self, symbol: str = "BTCUSDT") -> Optional[FuturesPosition]:
        """Get current futures position for a symbol."""
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials required for position check")
            return None

        try:
            params = {"symbol": symbol, "timestamp": self._get_timestamp()}
            params["signature"] = self._sign_request(params)

            url = f"{self.futures_url}/fapi/v2/positionRisk"
            response = self._session.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            for pos in data:
                if pos["symbol"] == symbol:
                    quantity = float(pos["positionAmt"])
                    if quantity == 0:
                        return None  # No position

                    return FuturesPosition(
                        symbol=symbol,
                        side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
                        quantity=abs(quantity),
                        entry_price=float(pos["entryPrice"]),
                        unrealized_pnl=float(pos["unRealizedProfit"]),
                        leverage=int(pos["leverage"]),
                        margin=float(pos["isolatedMargin"]),
                        liquidation_price=float(pos["liquidationPrice"]),
                    )

            return None

        except requests.RequestException as e:
            logger.error("Failed to get futures position: %s", e)
            return None

    def place_spot_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Optional[OrderResult]:
        """Place a spot market/limit order."""
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials required for trading")
            return None

        try:
            params = {
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": f"{quantity:.8f}",
                "timestamp": self._get_timestamp(),
            }

            if order_type == OrderType.LIMIT and price is not None:
                params["price"] = f"{price:.2f}"
                params["timeInForce"] = "GTC"

            params["signature"] = self._sign_request(params)

            url = f"{self.spot_url}/api/v3/order"
            response = self._session.post(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Calculate commission
            commission = 0.0
            commission_asset = ""
            for fill in data.get("fills", []):
                commission += float(fill.get("commission", 0))
                commission_asset = fill.get("commissionAsset", "")

            return OrderResult(
                order_id=str(data["orderId"]),
                symbol=data["symbol"],
                side=side,
                quantity=quantity,
                price=float(data.get("price", 0)) or float(data.get("fills", [{}])[0].get("price", 0)),
                filled_qty=float(data["executedQty"]),
                status=data["status"],
                timestamp=datetime.now(timezone.utc),
                commission=commission,
                commission_asset=commission_asset,
            )

        except requests.RequestException as e:
            logger.error("Failed to place spot order: %s", e)
            return None

    def place_futures_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[OrderResult]:
        """Place a futures market/limit order."""
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials required for trading")
            return None

        try:
            params = {
                "symbol": symbol,
                "side": side.value,
                "type": order_type.value,
                "quantity": f"{quantity:.3f}",
                "timestamp": self._get_timestamp(),
            }

            if reduce_only:
                params["reduceOnly"] = "true"

            if order_type == OrderType.LIMIT and price is not None:
                params["price"] = f"{price:.2f}"
                params["timeInForce"] = "GTC"

            params["signature"] = self._sign_request(params)

            url = f"{self.futures_url}/fapi/v1/order"
            response = self._session.post(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            return OrderResult(
                order_id=str(data["orderId"]),
                symbol=data["symbol"],
                side=side,
                quantity=quantity,
                price=float(data.get("avgPrice", 0)) or float(data.get("price", 0)),
                filled_qty=float(data["executedQty"]),
                status=data["status"],
                timestamp=datetime.now(timezone.utc),
            )

        except requests.RequestException as e:
            logger.error("Failed to place futures order: %s", e)
            return None

    def set_futures_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a futures symbol."""
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials required for leverage change")
            return False

        try:
            params = {
                "symbol": symbol,
                "leverage": leverage,
                "timestamp": self._get_timestamp(),
            }
            params["signature"] = self._sign_request(params)

            url = f"{self.futures_url}/fapi/v1/leverage"
            response = self._session.post(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()

            logger.info("Set leverage for %s to %dx", symbol, leverage)
            return True

        except requests.RequestException as e:
            logger.error("Failed to set leverage: %s", e)
            return False

    def get_exchange_info(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """Get exchange info for a symbol (min qty, price precision, etc)."""
        try:
            url = f"{self.futures_url}/fapi/v1/exchangeInfo"
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            for sym in data.get("symbols", []):
                if sym["symbol"] == symbol:
                    return {
                        "symbol": symbol,
                        "price_precision": sym["pricePrecision"],
                        "quantity_precision": sym["quantityPrecision"],
                        "min_qty": float(sym["filters"][1]["minQty"]),
                        "max_qty": float(sym["filters"][1]["maxQty"]),
                        "step_size": float(sym["filters"][1]["stepSize"]),
                    }

            return None

        except requests.RequestException as e:
            logger.error("Failed to get exchange info: %s", e)
            return None
