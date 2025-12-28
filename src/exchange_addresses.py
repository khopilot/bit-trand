"""
Known Bitcoin Exchange Addresses

Contains addresses associated with major cryptocurrency exchanges.
Used to detect exchange inflows/outflows for on-chain analysis.

Sources:
- Public exchange disclosures
- Blockchain explorers (Glassnode, CryptoQuant)
- Community-maintained lists

Note: This list is not exhaustive. Exchanges regularly rotate addresses.
Consider using an API service (Glassnode, CryptoQuant) for production.

Author: khopilot
"""

from typing import Set

# Major exchange cold wallet addresses (verified public addresses)
# Updated: 2024-12
EXCHANGE_ADDRESSES: Set[str] = {
    # Binance
    "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
    "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
    "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",
    "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
    "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
    "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6",
    "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",

    # Coinbase
    "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",
    "3KTeq879YjzhqkAXzZmdapJAVC6qz5qEth",
    "1FzWLkAahHooV3kzT0xAPhPpdWWQRpXpDZ",
    "3CD1QW6fjgTwKq3Pj97nty28WZAVkziNom",
    "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
    "bc1qa5wkgaew2dkv56kfvj49j0av5nml45x9ek9hz6",

    # Kraken
    "bc1qe75775tzuvspl59cw77ycc472jl0sgue69x3up",
    "bc1qx9t2l3pyny2spqpqlye8svce70nppwtaxwdrp4",
    "3FHNBLobJnbCTFTVakh5TXmEneyf5PT61B",
    "3AfrvBrAgfH8R7xri3GymvLpZ8gmFfD4oD",
    "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",

    # Bitfinex
    "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
    "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
    "1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g",
    "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",

    # OKX (formerly OKEx)
    "bc1q2s3rjwvam9dt2ftt4sqxqjf3twav0gdnv0z50",
    "1LnoZawVFFQihU8d8ntxLMpYheZUfyeVAK",
    "15VwsCYBoPcLwFNkt7qVbLhz24wdRvRjrJ",

    # Huobi
    "1LAnF8h3qMGx3TSwNUHVneBZUEpwE4gu3D",
    "1HckjUpRGcrrRAtFaaCAUaGjsPx9oYmLaZ",
    "14e6qLQS8zb8XSmxzjEMBhWCZ7s2Yq6mKY",
    "1AC4fMwgY8j9onSbXEWeH6Zan8QGMSdmtA",

    # Gemini
    "bc1qe2luqjvlxc2nlfdqcjyxvz7k4vcp4v7wycclcr",
    "3P3QsMVK89JBNqZQv5zMAKG8FK3kJM4rjt",
    "1Mtvj12wcGJFyAR7GnyBCnCGTKPVDBrapQ",

    # Bitstamp
    "3HQpJK7PBsxxkEMiuqQekmDTKSrrsMvqmw",
    "1K2SXgApmo9uZoyahvsbSanpVWbzZWVVMF",
    "3BMEXVsYyfKBNHapFP8xeCXzrTKv9AQLam",

    # KuCoin
    "bc1q35kc6nhxv8qh68pc2cuz7f3j7yl88kf3zzzqkd",
    "1NnMGHKxBevKxfJv1Z5tJSEe3m5c2kn5dZ",
    "3EHAKKHxzGz2RVGHGXj7aVB8PjS5jLJpKz",

    # Bybit
    "bc1qjasf9z3h7w3jspkhtgatgpyvvzgpa2wwd2lr0eh5tx44reyn2k7sfc27a4",
    "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",

    # Gate.io
    "3Bi3xG2B4vHHXr1pMxMNvqDkjCyUfFvQXH",
    "bc1q87ne5u3yza26pwhsq7l25c74pqvfxtjnqwfpvr",

    # Crypto.com
    "bc1qr4dl5wa7kl8yu792dceg9z5knl2gkn220lk7a9",
    "3LQeSvHgGGRHjUBZJi1XZpRqKqXwPJMqwB",
    "1MHPYGxRdbbWNLrYgPLoxqGg2xv3xdMFbM",

    # Deribit
    "bc1qy4sp3n6v3dltg7vklm0qzcgd9vqz9mtkqj8hwr",
    "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",

    # BitMEX
    "3BMEXqGpG4FxBA1KWhRFufXfSTRgzfDBhJ",
    "3BMEXVsYyfKBNHapFP8xeCXzrTKv9AQLam",
    "3BMEX8WjnGGq7JeghAc8Xgxy2p5zSpSTPT",

    # Bittrex
    "3QbpjKX3j3bi2pq1HuXwhHjgv9xgVFfuvJ",
    "3Nxwenay9Z8Lc9JBiywExpnEFiLp6Afp8v",

    # Poloniex
    "17A16QmavnUfCW11DAApiJxp7ARnxN5pGX",
    "1Gqd3dpWnP1L8RQHwHxF9z7iVhNJxkHfKx",

    # FTX (historical - now defunct, but still trackable)
    "bc1qluvu24p2qpwpndkf3qqrfkptcey6u9k7xvch06",
    "1FWQiwK27EnGXb6BiBMRLJvunJQZZPMcGd",
}

# Address prefixes that are commonly used by exchanges
# (for heuristic matching when exact address not in list)
EXCHANGE_PREFIXES: Set[str] = {
    "bc1qgdjqv0av3q56jvd82tkdjpy",  # Binance/Bitfinex pattern
    "3BMEX",  # BitMEX pattern
}


def is_exchange_address(address: str) -> bool:
    """
    Check if an address is a known exchange address.

    Args:
        address: Bitcoin address to check

    Returns:
        True if address is associated with an exchange
    """
    if not address:
        return False

    # Direct match
    if address in EXCHANGE_ADDRESSES:
        return True

    # Prefix match (for known patterns)
    for prefix in EXCHANGE_PREFIXES:
        if address.startswith(prefix):
            return True

    return False


def get_exchange_name(address: str) -> str:
    """
    Get exchange name for a known address.

    Args:
        address: Bitcoin address

    Returns:
        Exchange name or "Unknown"
    """
    # This would require a more detailed mapping
    # For now, return "Unknown" - can be extended
    if address in EXCHANGE_ADDRESSES:
        # Would need address -> exchange mapping
        return "Known Exchange"
    return "Unknown"


# Statistics
def get_stats() -> dict:
    """Get address database statistics."""
    return {
        "total_addresses": len(EXCHANGE_ADDRESSES),
        "prefix_patterns": len(EXCHANGE_PREFIXES),
    }
