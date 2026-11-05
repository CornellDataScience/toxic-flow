"""Simulator configuration (design §8.3), validated on construction.

Only the fields the replay engine uses so far are here. `policies` and
`trade_tol_ns` arrive with FillSimulator and TradeIndex; the schedule takes
its own seed.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from .queue_tracker import INCREASE_POLICIES
from .reader import DEFAULT_GAP_NS


def _decimal(value, what: str) -> Decimal:
    if isinstance(value, float):
        value = str(value)  # never carry binary float error into prices or sizes
    try:
        out = Decimal(value)
    except (ArithmeticError, TypeError, ValueError):
        raise ValueError(f"{what} is not a number: {value!r}") from None
    if not out.is_finite() or out <= 0:
        raise ValueError(f"{what} must be positive, got {value!r}")
    return out


@dataclass(frozen=True)
class SimConfig:
    symbols: tuple[str, ...]
    order_qty: dict[str, Decimal]  # fixed size per symbol, identical for all strategies (D2)
    horizon_ns: int  # cancellation horizon, shared by all strategies
    latency_ns: int = 0  # decision -> insertion delay (S6)
    tick_size: dict[str, Decimal] = field(default_factory=dict)  # override; default 10**-price_precision
    gap_ns: int = DEFAULT_GAP_NS  # receive gap that censors (same default as validate_recording)
    warmup_ns: int = 0  # synced time required after each (re)sync before deciding
    increase_policy: str = "keep"  # 'keep' mirrors L3Book | 'back' follows Kraken's amend rule
    view_depth: int = 10  # levels per side in the strategy BookView
    audit_invariants: bool = False  # check I3 against the book after every trusted item (slow)

    def __post_init__(self):
        symbols = (self.symbols,) if isinstance(self.symbols, str) else tuple(self.symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError(f"symbols must be non-empty and unique, got {symbols!r}")
        missing = [s for s in symbols if s not in self.order_qty]
        if missing:
            raise ValueError(f"order_qty missing for {missing}")
        unknown = sorted(set(self.order_qty) - set(symbols)) + sorted(set(self.tick_size) - set(symbols))
        if unknown:
            raise ValueError(f"order_qty/tick_size given for symbols not simulated: {unknown}")
        for name in ("horizon_ns", "gap_ns", "view_depth"):
            _check_int(self, name, minimum=1)
        for name in ("latency_ns", "warmup_ns"):
            _check_int(self, name, minimum=0)
        if self.increase_policy not in INCREASE_POLICIES:
            raise ValueError(f"increase_policy must be one of {INCREASE_POLICIES}, got {self.increase_policy!r}")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "order_qty", {s: _decimal(self.order_qty[s], f"order_qty[{s}]") for s in symbols})
        object.__setattr__(
            self, "tick_size", {s: _decimal(t, f"tick_size[{s}]") for s, t in self.tick_size.items()}
        )


def _check_int(cfg: SimConfig, name: str, minimum: int) -> None:
    value = getattr(cfg, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
