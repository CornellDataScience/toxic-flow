"""Placement schedules, generated from the clock only (never from book state or outcomes, §11.4)."""

from collections.abc import Iterable

from .models import PlacementRequest, Side


class PlacementSchedule:
    @staticmethod
    def regular(
        symbols: Iterable[str],
        start_ns: int,
        end_ns: int,
        spacing_ns: int,
        sides: Iterable[Side] = (Side.BUY, Side.SELL),
    ) -> list[PlacementRequest]:
        """One request per (symbol, side) every `spacing_ns` in [start_ns, end_ns), sorted by time."""
        if spacing_ns <= 0 or end_ns < start_ns:
            raise ValueError(f"need spacing_ns > 0 and end_ns >= start_ns, got {spacing_ns}, {start_ns}..{end_ns}")
        symbols, sides = tuple(symbols), tuple(sides)
        return [
            PlacementRequest(f"{symbol}|{side.value}|{ts}", symbol, side, ts)
            for ts in range(start_ns, end_ns, spacing_ns)
            for symbol in symbols
            for side in sides
        ]
