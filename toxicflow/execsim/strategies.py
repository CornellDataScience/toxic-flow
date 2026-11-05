"""Placement strategies: pure functions of (request, view) -> OrderIntent | Skip (design §8.5).

A strategy decides side and price only. Size, latency, queue tracking, expiry,
data-quality gating and the crossing check at insertion belong to the engine
(P1, D3), so every strategy is judged under the same rules.
"""

from typing import Protocol

from .models import BookView, OrderIntent, PlacementRequest, Skip, SkipReason


class OrderPlacementStrategy(Protocol):
    name: str

    def decide(self, req: PlacementRequest, view: BookView) -> OrderIntent | Skip: ...


class JoinBestStrategy:
    """Join the back of the queue at our side's best price."""

    name = "join_best"

    def decide(self, req: PlacementRequest, view: BookView) -> OrderIntent | Skip:
        px = view.best(req.side)
        return Skip(SkipReason.NO_TOUCH) if px is None else OrderIntent(req.side, px)
