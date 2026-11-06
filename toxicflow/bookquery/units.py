"""Time and price/quantity unit conversions.

Times are int64 nanoseconds since the Unix epoch (UTC) everywhere inside the
library. Prices and quantities are kept as exact integers: a price is a count
of ticks (price * 10**price_precision) and a quantity is a count of lots
(qty * 10**qty_precision). Converting to float only happens through the
explicit helpers here, so rounding can't creep in silently.
"""

import calendar
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from functools import lru_cache

import numpy as np
import pandas as pd

TimeLike = int | np.integer | str | datetime | pd.Timestamp | np.datetime64
TimesLike = TimeLike | Iterable[TimeLike] | np.ndarray | pd.Series | pd.DatetimeIndex
HorizonLike = int | str | pd.Timedelta


@lru_cache(maxsize=4096)
def _epoch_seconds(prefix: str) -> int:
    return calendar.timegm(datetime.fromisoformat(prefix).timetuple())


def parse_iso_ns(s: str) -> int:
    """Kraken timestamp ("2026-10-16T21:07:35.159647442Z") -> ns since epoch.

    Faster than pd.Timestamp for one string at a time and keeps all nine
    fractional digits (datetime only keeps six).
    """
    secs = _epoch_seconds(s[:19])
    frac = s[20:].rstrip("Z") if len(s) > 19 and s[19] == "." else ""
    return secs * 1_000_000_000 + int(frac.ljust(9, "0")[:9] or 0)


def to_ns(t: TimeLike) -> int:
    """One time -> int ns. Integers are taken as ns already; naive datetimes as UTC."""
    if isinstance(t, (int, np.integer)):
        return int(t)
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value)


def to_ns_array(t: TimesLike) -> np.ndarray:
    """Many times -> int64 ns array (see to_ns)."""
    if isinstance(t, (pd.Series, pd.DatetimeIndex)) and pd.api.types.is_datetime64_any_dtype(t):
        idx = pd.DatetimeIndex(t)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx.as_unit("ns").asi8.copy()
    arr = np.asarray(t)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]").astype(np.int64)
    return np.array([to_ns(x) for x in arr.ravel()], dtype=np.int64).reshape(arr.shape)


def horizon_ns(h: HorizonLike) -> int:
    """"100ms" / "1s" / pd.Timedelta / int ns -> int ns."""
    if isinstance(h, (int, np.integer)):
        return int(h)
    return int(pd.Timedelta(h).value)


def decimal_to_ticks(value: Decimal | str | float | int, precision: int) -> int:
    """Exact conversion to an integer count of 10**-precision units.

    Floats go through repr(), which is the shortest string that round-trips,
    so 81537.4 becomes Decimal("81537.4") rather than its binary expansion.
    Raises ValueError if the value has more decimals than `precision`.
    """
    d = value if isinstance(value, Decimal) else Decimal(repr(value) if isinstance(value, float) else str(value))
    scaled = d.scaleb(precision)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"{value} has more than {precision} decimal places")
    return int(scaled)


def ticks_to_decimal(ticks: int, precision: int) -> Decimal:
    return Decimal(int(ticks)).scaleb(-precision)


def ticks_to_float(ticks: int | np.ndarray, precision: int) -> float | np.ndarray:
    """The explicit float boundary. Arrays pass through with NaN preserved."""
    if isinstance(ticks, np.ndarray):
        return ticks.astype(np.float64) / 10.0**precision
    return int(ticks) / 10.0**precision
