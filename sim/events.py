"""Event types the implementation code consumes.

This is the DATA CONTRACT with the data extraction group: whatever they
collect from the exchange gets converted into a time-ordered stream of these
two event types. Agree on it with them and change it here in one place.

Conventions
- ts: float, seconds since epoch (exchange timestamp, not our receive time)
- side: "bid" or "ask" for resting orders
- aggressor: "buy" or "sell" for trades (the side that crossed the spread)
- prices/quantities: floats exactly as the exchange reports them
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BookEvent:
    """An L3 change to one resting order."""
    ts: float
    action: str      # "add" | "modify" | "delete"
    side: str        # "bid" | "ask"
    order_id: str
    price: float
    qty: float       # for "modify": the order's NEW remaining size


@dataclass(frozen=True, slots=True)
class Trade:
    """An execution. aggressor = the side that crossed the spread."""
    ts: float
    price: float
    qty: float
    aggressor: str   # "buy" (lifted an ask) | "sell" (hit a bid)
