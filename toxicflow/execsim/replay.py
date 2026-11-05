"""HistoricalReplayEngine: drives recorded L3 items, placements and queue trackers in stream order (§7).

Per item of a simulated symbol:

    1. parse the item timestamp; guard it to be non-decreasing (R2, R3)
    2. a snapshot after a synced one is a RESNAPSHOT discontinuity, raised
       before any action fires: events may be missing before it
    3. fire every action due at or before the item's time: expiries, then
       decisions, then insertions (R4 to R6, strict-before)
    4. SimL3Book.apply(), the only writer of the historical book
    5. trust the item's events only if its checksum passed (R8); route them
       to every live order's QueueTracker and log the ReachEvents
    6. post-item: end orders whose level left the depth, track level rank

Discontinuities (meta line, receive gap, truncated file, checksum failure,
resnapshot) censor every live order and drop pending insertions. ReachEvents
are logged as they are, with no evidence and no settlement; `on_reach` is the
hook FillSimulator will use.
"""

import heapq
import itertools
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from toxicflow.kraken.book import CHECKSUM_LEVELS, PairInfo

from .config import SimConfig
from .models import BookView, EndReason, PlacementRequest, ReachEvent, Side, Skip, SkipReason
from .orders import SimulatedOrder
from .reader import KIND_GAP, KIND_META, KIND_TRUNCATED, RecordingReader, StreamRecord, book_items
from .results import QualityEvent, ReachRecord, ReplayResult, SkipRecord
from .sim_book import BookEvent, SimL3Book
from .strategies import OrderPlacementStrategy
from .timeutil import default_tick, parse_rfc3339_ns

EXPIRE, DECIDE, INSERT = 0, 1, 2  # also the priority among actions due at the same time

OnReach = Callable[[SimulatedOrder, ReachEvent, BookEvent], None]


@dataclass
class Action:
    kind: int
    due: int
    request: PlacementRequest | None = None
    order: SimulatedOrder | None = None


@dataclass
class SymbolState:
    symbol: str
    actions: list = field(default_factory=list)  # heap of (due, kind, n, Action)
    live: dict[str, SimulatedOrder] = field(default_factory=dict)
    book: SimL3Book | None = None
    synced: bool = False
    needs_validation: bool = False  # after RECV_GAP: wait for one checksum-valid item
    has_synced_before: bool = False
    synced_since_ts: int | None = None  # start of the current synced segment (warmup)
    last_ts_eff: int = 0
    last_good_ts: int | None = None  # time of the last item whose checksum passed
    last_discontinuity_ts: int | None = None
    view_cache: BookView | None = None  # the view built for the last decision time in this release
    incident: int = 0  # number of the latest data incident (see QualityEvent)
    incident_open: bool = False  # True from a discontinuity until the book is synced again
    counters: Counter = field(default_factory=Counter)


class HistoricalReplayEngine:
    def __init__(
        self,
        cfg: SimConfig,
        strategies: Sequence[OrderPlacementStrategy],
        schedule: Iterable[PlacementRequest],
        on_reach: OnReach | None = None,
    ):
        names = [s.name for s in strategies]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"need at least one strategy and unique names, got {names}")
        self.cfg = cfg
        self.strategies = tuple(strategies)
        self.on_reach = on_reach
        self.state = {sym: SymbolState(sym) for sym in cfg.symbols}
        self.meta: dict | None = None
        self._n = itertools.count()
        self._rec: StreamRecord | None = None
        self._finished = False
        self._orders: list[SimulatedOrder] = []
        self._skips: list[SkipRecord] = []
        self._reaches: list[ReachRecord] = []
        self._quality: list[QualityEvent] = []
        self._views = {}
        seen = set()
        for req in schedule:
            if req.symbol not in self.state:
                raise ValueError(f"request {req.request_id} is for {req.symbol}, which is not simulated")
            if req.request_id in seen:
                raise ValueError(f"duplicate request_id {req.request_id}")
            seen.add(req.request_id)
            self._push(self.state[req.symbol], Action(DECIDE, req.ts_ns, request=req))

    # --- driving ------------------------------------------------------------

    def run(self, files: Sequence[Path]) -> ReplayResult:
        return self.replay(RecordingReader(files=files, symbols=self.cfg.symbols, gap_ns=self.cfg.gap_ns))

    def replay(self, records: Iterable[StreamRecord]) -> ReplayResult:
        """Feed every record, then finish. Same result as process_record() per record plus finish()."""
        for rec in records:
            self.process_record(rec)
        return self.finish()

    def process_record(self, rec: StreamRecord) -> None:
        """Incremental core: feed records in reader order."""
        if self._finished:
            raise RuntimeError("process_record after finish")
        self._rec = rec
        if rec.kind == KIND_META:  # new file or writer session: books restart from a snapshot
            self.meta = rec.payload
            for s in self.state.values():
                self._discontinuity(s, "SESSION_RESTART", {"file": str(rec.file)})
                s.book = None
            return
        if rec.kind == KIND_GAP:
            for s in self.state.values():
                self._discontinuity(s, "RECV_GAP", dict(rec.payload), needs_validation=True)
            return
        if rec.kind == KIND_TRUNCATED:  # the rest of the file is unreadable
            for s in self.state.values():
                self._discontinuity(s, "TRUNCATED", dict(rec.payload))
                s.book = None
            return
        for msg_type, item in book_items(rec):
            s = self.state.get(item.get("symbol"))
            if s is not None:  # symbols not simulated are not applied
                self._process_item(s, msg_type, item, rec)

    def finish(self) -> ReplayResult:
        """End of data: censor live orders; requests the data never reached become NO_DATA skips."""
        if not self._finished:
            self._finished = True
            self._rec = None
            for s in self.state.values():
                for o in s.live.values():
                    o.finish(EndReason.CENSORED_END_OF_DATA, self._censor_ts(s, o))
                s.live.clear()
                for *_, a in sorted(s.actions, key=lambda e: e[:3]):
                    if a.kind == DECIDE:
                        self._skip_all(a.request, SkipReason.NO_DATA)
                    elif a.kind == INSERT:
                        a.order.skip(SkipReason.NO_DATA)
                s.actions = []
                self._log(s, "END_OF_DATA")
        return ReplayResult(
            orders=list(self._orders),
            skips=list(self._skips),
            reaches=list(self._reaches),
            quality=list(self._quality),
            views=dict(self._views),
            counters={sym: Counter(s.counters) for sym, s in self.state.items()},
        )

    # --- per item -------------------------------------------------------------

    def _process_item(self, s: SymbolState, msg_type: str, item: dict, rec: StreamRecord) -> None:
        c = s.counters
        c["items"] += 1
        ts = self._clock(s, item)
        if msg_type == "snapshot" and s.has_synced_before:
            self._discontinuity(s, "RESNAPSHOT")
        self._release_due(s, ts)

        if s.book is None:
            s.book = self._new_book(s.symbol)
            if s.book is None:
                c["no_meta"] += 1
                return
        ok = s.book.apply(msg_type, item, position=rec)  # the only write to the historical book

        if msg_type == "snapshot":
            c["snapshots"] += 1
            if ok:
                c["checksum_ok"] += 1
                if not s.has_synced_before:
                    self._log(s, "SNAPSHOT_FIRST")
                s.synced = s.has_synced_before = True
                s.needs_validation = s.incident_open = False
                s.synced_since_ts = s.last_good_ts = ts
            else:
                c["checksum_fail"] += 1
                s.synced = False
                self._log(s, "SNAPSHOT_CHECKSUM_FAIL")
            return
        c["updates"] += 1
        if ok is None:  # before the first snapshot of this book
            c["skipped_unsynced"] += 1
            return
        if ok is False:  # events discarded (R8); censored at the last good item
            c["checksum_fail"] += 1
            self._discontinuity(s, "DESYNC")
            return
        c["checksum_ok"] += 1
        if s.needs_validation:
            s.needs_validation = s.incident_open = False
            s.synced_since_ts = ts
        s.synced = True
        self._route(s, s.book, ts)
        s.last_good_ts = ts

    def _clock(self, s: SymbolState, item: dict) -> int:
        raw = item.get("timestamp")
        try:
            ts = parse_rfc3339_ns(raw)
        except ValueError:
            s.counters["ts_missing"] += 1
            self._log(s, "TS_MISSING", {"timestamp": raw})
            return s.last_ts_eff
        if ts < s.last_ts_eff:
            s.counters["ts_regressions"] += 1
            self._log(s, "TS_REGRESSION", {"ts_ns": ts})
            return s.last_ts_eff
        s.last_ts_eff = ts
        return ts

    def _route(self, s: SymbolState, book: SimL3Book, ts: int) -> None:
        events = book.last_events
        s.counters["events"] += len(events)
        if not s.live:
            return
        for ev in events:
            for o in list(s.live.values()):
                reach = o.tracker.on_event(ev, ts)
                if reach is not None:
                    # the book is in its post-item state: the one the checksum just validated
                    covered = book.level_rank(ev.side, reach.event_price) <= CHECKSUM_LEVELS
                    self._reaches.append(
                        ReachRecord(o.order_id, o.strategy, s.symbol, reach, o.tracker.ahead_qty, covered)
                    )
                    if self.on_reach is not None:
                        self.on_reach(o, reach, ev)
        truncated = book.last_info.truncated
        for o in list(s.live.values()):
            if truncated:
                o.tracker.on_truncation(truncated)
            if o.tracker.invalidated is not None or not book.level_visible(o.book_side, o.price):
                self._end(s, o, EndReason.OUT_OF_DEPTH, ts)
            else:
                o.max_level_rank = max(o.max_level_rank, book.level_rank(o.book_side, o.price))
        if self.cfg.audit_invariants:
            for o in s.live.values():
                problems = o.tracker.audit(book.level_orders(o.book_side, o.price))
                if problems:
                    s.counters["audit_fail"] += 1
                    self._log(s, "AUDIT_FAIL", {"order_id": o.order_id, "problems": problems})

    # --- actions ----------------------------------------------------------------

    def _push(self, s: SymbolState, a: Action) -> None:
        heapq.heappush(s.actions, (a.due, a.kind, next(self._n), a))

    def _release_due(self, s: SymbolState, boundary: int) -> None:
        s.view_cache = None  # the book only changes between releases
        while s.actions and s.actions[0][0] <= boundary:
            a = heapq.heappop(s.actions)[-1]
            if a.kind == EXPIRE:
                if s.live.get(a.order.order_id) is a.order:  # not already censored
                    self._end(s, a.order, EndReason.HORIZON, a.due)
            elif a.kind == DECIDE:
                self._decide(s, a.request)
            else:
                self._insert(s, a)

    def _decide(self, s: SymbolState, req: PlacementRequest) -> None:
        reason = self._gate(s, req)
        if reason is not None:
            self._skip_all(req, reason)
            return
        view = s.view_cache  # one view for all strategies, and for all requests due at the same T (R7)
        if view is None or view.ts_ns != req.ts_ns:
            view = s.view_cache = s.book.book_view(req.ts_ns, self._tick(s), self.cfg.view_depth)
        self._views[req.request_id] = view
        for strat in self.strategies:
            out = strat.decide(req, view)
            if isinstance(out, Skip):
                self._skips.append(SkipRecord(req, strat.name, out.reason))
                continue
            o = SimulatedOrder.new(req, strat.name, out, self.cfg.order_qty[s.symbol], decided_ts=req.ts_ns)
            self._orders.append(o)
            self._push(s, Action(INSERT, req.ts_ns + self.cfg.latency_ns, order=o))

    def _gate(self, s: SymbolState, req: PlacementRequest) -> SkipReason | None:
        if s.book is None or not s.synced:
            return SkipReason.UNSYNCED
        if s.needs_validation:
            return SkipReason.DATA_GAP
        if self.cfg.warmup_ns and req.ts_ns - s.synced_since_ts < self.cfg.warmup_ns:
            return SkipReason.WARMUP
        return None

    def _insert(self, s: SymbolState, a: Action) -> None:
        o, book = a.order, s.book
        if book is None or not s.synced or s.needs_validation:
            o.skip(SkipReason.DATA_GAP)
            return
        opposite = book.best_ask() if o.side is Side.BUY else book.best_bid()
        if opposite is not None and (o.price >= opposite if o.side is Side.BUY else o.price <= opposite):
            o.skip(SkipReason.CROSSED_AT_INSERT)
            return
        if not book.level_visible(o.book_side, o.price):
            o.skip(SkipReason.OUT_OF_DEPTH)
            return
        o.start(
            inserted_ts=a.due,
            ahead=book.level_orders(o.book_side, o.price),
            rank=book.level_rank(o.book_side, o.price),
            touch_qty=book.touch_qty(o.book_side),
            horizon_ns=self.cfg.horizon_ns,
            increase_policy=self.cfg.increase_policy,
        )
        s.live[o.order_id] = o
        self._push(s, Action(EXPIRE, o.expiry_ts, order=o))

    def _end(self, s: SymbolState, o: SimulatedOrder, reason: EndReason, end_ts: int) -> None:
        o.finish(reason, end_ts)
        del s.live[o.order_id]

    def _skip_all(self, req: PlacementRequest, reason: SkipReason) -> None:
        for strat in self.strategies:
            self._skips.append(SkipRecord(req, strat.name, reason))

    # --- discontinuities ---------------------------------------------------------

    def _discontinuity(self, s: SymbolState, reason: str, detail: dict | None = None,
                       needs_validation: bool = False) -> None:
        end = EndReason(f"CENSORED_{reason}")
        for o in s.live.values():
            o.finish(end, self._censor_ts(s, o))
        s.live.clear()
        if reason != "RECV_GAP":  # a gap keeps the book; the next checksum-valid item validates it
            s.synced = False
        s.needs_validation |= needs_validation
        s.last_discontinuity_ts = s.last_ts_eff
        kept = []
        for entry in s.actions:  # insertions decided before the hole would see a stale queue
            if entry[-1].kind == INSERT:
                entry[-1].order.skip(SkipReason.DATA_GAP)
            else:
                kept.append(entry)
        if len(kept) != len(s.actions):
            heapq.heapify(kept)
            s.actions = kept
        if not s.incident_open:  # otherwise this record describes the break already open
            s.incident += 1
            s.incident_open = True
        self._log(s, reason, detail, incident=s.incident)

    @staticmethod
    def _censor_ts(s: SymbolState, o: SimulatedOrder) -> int:
        # the last trusted item, but never before the order existed
        return max(s.last_good_ts if s.last_good_ts is not None else o.inserted_ts, o.inserted_ts)

    # --- helpers --------------------------------------------------------------------

    def _new_book(self, symbol: str) -> SimL3Book | None:
        if self.meta is None or symbol not in self.meta.get("pairs", {}):
            return None
        p = self.meta["pairs"][symbol]
        return SimL3Book(PairInfo(symbol, p["price_precision"], p["qty_precision"]), self.meta["depth"])

    def _tick(self, s: SymbolState) -> Decimal:
        return self.cfg.tick_size.get(s.symbol) or default_tick(s.book.pair.price_precision)

    def _log(self, s: SymbolState, kind: str, detail: dict | None = None, incident: int | None = None) -> None:
        rec = self._rec
        self._quality.append(QualityEvent(
            s.last_ts_eff, s.symbol, kind, rec.seq if rec else None, rec.recv_ns if rec else None, detail or {},
            incident,
        ))
