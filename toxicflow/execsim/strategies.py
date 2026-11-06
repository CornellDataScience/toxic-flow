"""Placement strategies: pure functions of (request, view) -> OrderIntent | Skip (design §8.5).

A strategy decides side and price only. Size, latency, queue tracking, expiry,
data-quality gating and the crossing check at insertion belong to the engine
(P1, D3), so every strategy is judged under the same rules.

Prices are derived from the view's best prices by whole ticks, so they stay on
the tick grid as long as the book does. A best price that is not a multiple of
`view.tick` (a mistaken `tick_size` override) is reported as INVALID_PRICE rather
than rounded, and no strategy ever returns a price that locks or crosses the
opposite best in its own view.
"""

from decimal import Decimal
from typing import Protocol

from .models import BookView, OrderIntent, PlacementRequest, Side, Skip, SkipReason


class OrderPlacementStrategy(Protocol):
    name: str

    def decide(self, req: PlacementRequest, view: BookView) -> OrderIntent | Skip: ...


class JoinBestStrategy:
    """Join the back of the queue at our side's best price."""

    name = "join_best"

    def decide(self, req: PlacementRequest, view: BookView) -> OrderIntent | Skip:
        px = view.best(req.side)
        return Skip(SkipReason.NO_TOUCH) if px is None else OrderIntent(req.side, px)


class ImproveQuoteStrategy:
    """Improve our side's best price by one tick: best bid + tick for BUY, best ask - tick for SELL.

    Needs both sides of the book (ONE_SIDED_BOOK otherwise) and a spread of at
    least two ticks; a narrower, locked or crossed spread is SPREAD_TOO_NARROW.
    """

    name = "improve_quote"

    def decide(self, req: PlacementRequest, view: BookView) -> OrderIntent | Skip:
        bid, ask = view.best_bid, view.best_ask
        if bid is None or ask is None:
            return Skip(SkipReason.ONE_SIDED_BOOK)
        if not _on_grid(view.tick, bid, ask):
            return Skip(SkipReason.INVALID_PRICE)
        px = bid + view.tick if req.side is Side.BUY else ask - view.tick
        if _marketable(req.side, px, view):
            return Skip(SkipReason.SPREAD_TOO_NARROW)  # would lock or cross: spread < 2 ticks
        return OrderIntent(req.side, px)


class QuoteDeeperStrategy:
    """Quote `depth_ticks` ticks behind our side's best price: best bid - k*tick for BUY,
    best ask + k*tick for SELL.

    Target depth is a tick distance from the own-side touch (design §8.5), not a
    level index or a cumulative quantity: the price is placed whether or not a
    level exists there. Only our side is needed (NO_TOUCH if it is empty). Whether
    the price is inside the subscribed book depth is checked by the engine at
    insertion (OUT_OF_DEPTH), like every other strategy's price.

    `name` defaults to the design's 'quote_deeper'; give each instance its own
    name to run several depths side by side (the engine requires unique names).
    """

    def __init__(self, depth_ticks: int = 1, name: str = "quote_deeper"):
        if isinstance(depth_ticks, bool) or not isinstance(depth_ticks, int) or depth_ticks < 1:
            raise ValueError(f"depth_ticks must be an int >= 1, got {depth_ticks!r}")
        self.depth_ticks = depth_ticks
        self.name = name

    def decide(self, req: PlacementRequest, view: BookView) -> OrderIntent | Skip:
        best = view.best(req.side)
        if best is None:
            return Skip(SkipReason.NO_TOUCH)
        if not _on_grid(view.tick, best):
            return Skip(SkipReason.INVALID_PRICE)
        px = best - req.side.sign * self.depth_ticks * view.tick
        if px <= 0 or _marketable(req.side, px, view):  # the latter only on a crossed view
            return Skip(SkipReason.INVALID_PRICE)
        return OrderIntent(req.side, px)


def _on_grid(tick: Decimal, *prices: Decimal) -> bool:
    """True if `tick` is a usable tick and every price is positive and a whole number of ticks."""
    if not (tick.is_finite() and tick > 0):
        return False
    return all(p.is_finite() and p > 0 and p % tick == 0 for p in prices)


def _marketable(side: Side, px: Decimal, view: BookView) -> bool:
    """True if a `side` order at `px` would lock or cross the opposite best in `view`."""
    if side is Side.BUY:
        return view.best_ask is not None and px >= view.best_ask
    return view.best_bid is not None and px <= view.best_bid
