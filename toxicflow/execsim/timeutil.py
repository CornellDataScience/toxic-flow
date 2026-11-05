"""Exchange timestamps as integer nanoseconds, and tick helpers.

Kraken sends RFC 3339 strings with up to nine fractional digits
('2026-10-18T02:06:55.103101894Z'). Floats cannot hold that precision, so
parsing is done on the digits.
"""

import calendar
import re
from datetime import datetime, timezone
from decimal import Decimal

S = 1_000_000_000
MS = 1_000_000

_RFC3339 = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})")


def parse_rfc3339_ns(ts: str) -> int:
    """'2026-10-18T02:06:55.103101894Z' -> ns since the epoch, exact. Raises ValueError on anything else."""
    m = _RFC3339.fullmatch(ts) if isinstance(ts, str) else None
    if m is None:
        raise ValueError(f"not an RFC 3339 timestamp with at most 9 fractional digits: {ts!r}")
    year, month, day, hour, minute, second = (int(g) for g in m.groups()[:6])
    datetime(year, month, day, hour, minute, second)  # validates the calendar fields
    secs = calendar.timegm((year, month, day, hour, minute, second))
    offset = m.group(8)
    if offset != "Z":
        sign = 1 if offset[0] == "+" else -1
        secs -= sign * (int(offset[1:3]) * 3600 + int(offset[4:6]) * 60)
    return secs * S + int((m.group(7) or "").ljust(9, "0"))


def format_ns(ns: int) -> str:
    """Inverse of parse_rfc3339_ns for UTC: always nine fractional digits and 'Z'."""
    secs, frac = divmod(ns, S)
    return datetime.fromtimestamp(secs, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{frac:09d}Z"


def default_tick(price_precision: int) -> Decimal:
    """10 ** -price_precision. The recorder does not store tick_size (G5, OQ-6), so this is the fallback."""
    return Decimal(1).scaleb(-price_precision)
