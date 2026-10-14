"""Kraken WebSocket v2 book checksums (L2 `book` and L3 `level3` channels).

Both are CRC32 over the top 10 price levels: asks (low->high) then bids
(high->low). Each price/qty is printed at the pair's fixed precision, then the
decimal point and leading zeros are removed. L2 contributes one price+qty per
level; L3 contributes one price+qty per *order*, in the order Kraken lists them.

Messages send prices as JSON numbers, so trailing zeros are lost in transit
(0.10000000 arrives as 0.1). That's why formatting needs the pair's
price_precision / qty_precision from the instrument channel.
"""

import zlib
from decimal import Decimal


def fmt(value: Decimal, precision: int) -> str:
    return f"{value:.{precision}f}".replace(".", "").lstrip("0")


def crc32(s: str) -> int:
    return zlib.crc32(s.encode()) & 0xFFFFFFFF
