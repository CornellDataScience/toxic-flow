"""ImproveQuote and QuoteDeeper strategies (design §8.5) and their replay through the shared engine
(§10.2 scenarios 8 to 12, §10.3 strategy independence and determinism)."""

import random
from decimal import Decimal

import pytest
from execsim_helpers import order
from test_execsim_replay import ASKS, BIDS, SYM, Tape, at, engine, req

from toxicflow.execsim.models import (BookView, EndReason, LevelSummary, Mechanism, OrderIntent, PlacementRequest,
                                      Side, Skip, SkipReason)
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy, QuoteDeeperStrategy

D = Decimal
BUY, SELL = Side.BUY, Side.SELL
IMPROVE, DEEPER = ImproveQuoteStrategy(), QuoteDeeperStrategy()


def view(bids=(), asks=(), tick="0.1"):
    """A BookView from price strings (or (price, qty) pairs), best first."""
    def levels(prices):
        return tuple(LevelSummary(D(p), D("1"), 1) if isinstance(p, str) else LevelSummary(D(p[0]), D(p[1]), 1)
                     for p in prices)
    return BookView(SYM, 0, D(tick), levels(bids), levels(asks))


def decide(strategy, side, v):
    return strategy.decide(PlacementRequest("r", SYM, side, 0), v)


def placed(price, side):
    return OrderIntent(side, D(price))


# --- ImproveQuote --------------------------------------------------------------------------------


def test_improve_quote_steps_one_tick_inside_the_spread():
    v = view(bids=["100.0", "99.9"], asks=["100.5", "100.6"])
    assert decide(IMPROVE, BUY, v) == placed("100.1", BUY)
    assert decide(IMPROVE, SELL, v) == placed("100.4", SELL)


def test_improve_quote_two_tick_spread_places_both_sides_at_the_middle():
    v = view(bids=["100.0"], asks=["100.2"])
    assert decide(IMPROVE, BUY, v) == placed("100.1", BUY)
    assert decide(IMPROVE, SELL, v) == placed("100.1", SELL)


@pytest.mark.parametrize("bid, ask", [("100.0", "100.1"),  # one tick: scenario 11
                                      ("100.0", "100.0"),  # locked
                                      ("100.2", "100.0")])  # crossed
def test_improve_quote_skips_when_spread_too_narrow(bid, ask):
    v = view(bids=[bid], asks=[ask])
    assert decide(IMPROVE, BUY, v) == Skip(SkipReason.SPREAD_TOO_NARROW)
    assert decide(IMPROVE, SELL, v) == Skip(SkipReason.SPREAD_TOO_NARROW)
    assert decide(JoinBestStrategy(), BUY, view(bids=["100.0"], asks=["100.1"])) == placed("100.0", BUY)


@pytest.mark.parametrize("bids, asks", [((), ()), (("100.0",), ()), ((), ("100.2",))])
def test_improve_quote_needs_both_sides(bids, asks):
    v = view(bids=bids, asks=asks)
    assert decide(IMPROVE, BUY, v) == Skip(SkipReason.ONE_SIDED_BOOK)
    assert decide(IMPROVE, SELL, v) == Skip(SkipReason.ONE_SIDED_BOOK)


@pytest.mark.parametrize("bid, ask, tick", [("100.05", "100.5", "0.1"),  # bid off the tick grid
                                            ("100.0", "100.55", "0.1"),  # ask off the grid
                                            ("100.0", "101.0", "0"),  # unusable tick
                                            ("100.0", "101.0", "-0.1")])
def test_improve_quote_rejects_off_grid_prices(bid, ask, tick):
    v = view(bids=[bid], asks=[ask], tick=tick)
    assert decide(IMPROVE, BUY, v) == Skip(SkipReason.INVALID_PRICE)
    assert decide(IMPROVE, SELL, v) == Skip(SkipReason.INVALID_PRICE)


def test_improve_quote_is_decimal_exact():
    # ETH-like 0.01 tick; small-price 1e-8 tick; a tick override coarser than the price precision
    v = view(bids=["2500.01"], asks=["2500.04"], tick="0.01")
    assert decide(IMPROVE, BUY, v) == placed("2500.02", BUY) and decide(IMPROVE, SELL, v) == placed("2500.03", SELL)
    v = view(bids=["0.00001234"], asks=["0.00001240"], tick="0.00000001")
    out = decide(IMPROVE, BUY, v)
    assert out.price == D("0.00001235") and str(out.price) == "0.00001235"
    assert decide(IMPROVE, SELL, v).price == D("0.00001239")
    v = view(bids=["100.5"], asks=["101.5"], tick="0.5")
    assert decide(IMPROVE, BUY, v) == placed("101.0", BUY) and decide(IMPROVE, SELL, v) == placed("101.0", SELL)
    assert decide(IMPROVE, BUY, view(bids=["100.5"], asks=["101.0"], tick="0.5")) == Skip(SkipReason.SPREAD_TOO_NARROW)


# --- QuoteDeeper ---------------------------------------------------------------------------------


def test_quote_deeper_steps_back_from_own_touch():
    v = view(bids=["100.0", "99.9"], asks=["100.2", "100.5"])
    assert decide(DEEPER, BUY, v) == placed("99.9", BUY)  # joins an existing level
    assert decide(DEEPER, SELL, v) == placed("100.3", SELL)  # no level there: still placed
    assert DEEPER.name == "quote_deeper" and DEEPER.depth_ticks == 1


def test_quote_deeper_target_depth_is_a_tick_distance():
    v = view(bids=["100.0", "99.0"], asks=["100.1"])  # a sparse book: levels are not ticks
    s = QuoteDeeperStrategy(depth_ticks=5, name="deeper5")
    assert decide(s, BUY, v) == placed("99.5", BUY)
    assert decide(s, SELL, v) == placed("100.6", SELL)
    assert decide(QuoteDeeperStrategy(depth_ticks=3), BUY, view(bids=["2500.01"], tick="0.01")) == placed(
        "2499.98", BUY)


def test_quote_deeper_needs_only_own_side():
    assert decide(DEEPER, BUY, view(bids=["100.0"])) == placed("99.9", BUY)
    assert decide(DEEPER, SELL, view(asks=["100.2"])) == placed("100.3", SELL)
    assert decide(DEEPER, BUY, view(asks=["100.2"])) == Skip(SkipReason.NO_TOUCH)
    assert decide(DEEPER, SELL, view(bids=["100.0"])) == Skip(SkipReason.NO_TOUCH)
    assert decide(DEEPER, BUY, view()) == Skip(SkipReason.NO_TOUCH)
    assert decide(DEEPER, SELL, view()) == Skip(SkipReason.NO_TOUCH)


def test_quote_deeper_one_tick_spread_still_places():
    v = view(bids=["100.0"], asks=["100.1"])
    assert decide(DEEPER, BUY, v) == placed("99.9", BUY) and decide(DEEPER, SELL, v) == placed("100.2", SELL)


def test_quote_deeper_rejects_invalid_prices():
    assert decide(QuoteDeeperStrategy(2), BUY, view(bids=["0.2"])) == Skip(SkipReason.INVALID_PRICE)  # 0.0
    assert decide(QuoteDeeperStrategy(3), BUY, view(bids=["0.2"])) == Skip(SkipReason.INVALID_PRICE)  # negative
    assert decide(DEEPER, BUY, view(bids=["0.2"])) == placed("0.1", BUY)
    assert decide(DEEPER, BUY, view(bids=["100.05"])) == Skip(SkipReason.INVALID_PRICE)  # off grid
    assert decide(DEEPER, SELL, view(asks=["100.2"], tick="0")) == Skip(SkipReason.INVALID_PRICE)
    # a view crossed by more than the depth: the deeper price would still be marketable
    crossed = view(bids=["100.5"], asks=["100.0"])
    assert decide(DEEPER, BUY, crossed) == Skip(SkipReason.INVALID_PRICE)
    assert decide(DEEPER, SELL, crossed) == Skip(SkipReason.INVALID_PRICE)
    assert decide(QuoteDeeperStrategy(6), BUY, crossed) == placed("99.9", BUY)


@pytest.mark.parametrize("bad", [0, -1, 1.0, "1", True, None])
def test_quote_deeper_validates_depth(bad):
    with pytest.raises(ValueError):
        QuoteDeeperStrategy(depth_ticks=bad)


# --- properties over random views -------------------------------------------------------------------


def _random_view(rng):
    tick = rng.choice([D("0.1"), D("0.01"), D("0.5"), D("0.00000001")])
    mid = rng.randint(1, 50_000)
    spread = rng.randint(-2, 6)  # includes locked and crossed views
    bid, ask = mid * tick, (mid + spread) * tick
    bids = [str(bid - i * tick) for i in range(rng.randint(0, 3)) if bid - i * tick > 0]
    asks = [str(ask + i * tick) for i in range(rng.randint(0, 3)) if ask + i * tick > 0]
    return view(bids=bids, asks=asks, tick=str(tick))


def test_no_strategy_ever_proposes_a_marketable_or_off_grid_price():
    rng = random.Random(7)
    strategies = [JoinBestStrategy(), IMPROVE, DEEPER, QuoteDeeperStrategy(4)]
    counts = {s.name: 0 for s in strategies}
    for _ in range(3_000):
        v = _random_view(rng)
        crossed = v.best_bid is not None and v.best_ask is not None and v.best_bid >= v.best_ask
        for s in strategies:
            for side in (BUY, SELL):
                out = decide(s, side, v)
                assert isinstance(out, (OrderIntent, Skip))
                if isinstance(out, Skip):
                    continue
                counts[s.name] += 1
                assert out.side is side and isinstance(out.price, Decimal)
                assert out.price > 0 and out.price % v.tick == 0
                if not (crossed and isinstance(s, JoinBestStrategy)):
                    # join_best copies the touch, so a crossed view makes it marketable (the engine's
                    # insertion check catches that); the new strategies must stay passive regardless
                    opposite = v.best_ask if side is BUY else v.best_bid
                    assert opposite is None or (out.price < opposite if side is BUY else out.price > opposite)
    assert all(counts.values())  # every strategy placed somewhere


# --- replay integration ---------------------------------------------------------------------------------


ALL = (JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy())


def test_all_strategies_share_the_order_life_cycle(tmp_path):
    tape = Tape()
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "0.7")], asks=ASKS)  # bid 100.0 / ask 100.2
    tape.update(10)
    result = engine([req(5), req(5, SELL)], strategies=ALL).run(tape.write(tmp_path))
    orders = by_id_strategy(result)
    expect = {  # (price, ahead qty, ahead orders, level rank, touch qty)
        ("buy@5", "join_best"): (D("100.0"), D("3.0"), 2, 1, D("3.0")),
        ("buy@5", "improve_quote"): (D("100.1"), 0, 0, 1, D("3.0")),  # new best level: empty queue
        ("buy@5", "quote_deeper"): (D("99.9"), D("0.7"), 1, 2, D("3.0")),  # behind C1
        ("sell@5", "join_best"): (D("100.2"), D("3.0"), 1, 1, D("3.0")),
        ("sell@5", "improve_quote"): (D("100.1"), 0, 0, 1, D("3.0")),
        ("sell@5", "quote_deeper"): (D("100.3"), 0, 0, 2, D("3.0")),  # no level there
    }
    assert {k: (o.price, o.queue_ahead_qty0, o.queue_ahead_orders0, o.level_rank0, o.touch_qty0)
            for k, o in orders.items()} == expect
    for o in result.orders:  # identical size, timestamps and horizon (D2)
        assert (o.qty, o.decided_ts, o.inserted_ts, o.expiry_ts) == (D("0.5"), at(5), at(5), at(1_005))
        assert o.end_reason is EndReason.CENSORED_END_OF_DATA and o.order_id.endswith("/" + o.strategy)
    assert result.skips == [] and result.counters[SYM]["audit_fail"] == 0


def by_id_strategy(result):
    return {(o.request.request_id, o.strategy): o for o in result.orders}


def test_spread_too_narrow_skips_only_improve_quote(tmp_path):  # scenario 11
    tape = Tape()
    tape.snapshot(0, asks=[order("A1", "100.1", "1.0")])
    tape.update(10)
    result = engine([req(5), req(5, SELL)], strategies=ALL).run(tape.write(tmp_path))
    assert {(s.request.request_id, s.strategy, s.reason) for s in result.skips} == {
        ("buy@5", "improve_quote", SkipReason.SPREAD_TOO_NARROW),
        ("sell@5", "improve_quote", SkipReason.SPREAD_TOO_NARROW)}
    assert {(o.order_id, o.price) for o in result.orders} == {
        ("buy@5/join_best", D("100.0")), ("buy@5/quote_deeper", D("99.9")),
        ("sell@5/join_best", D("100.1")), ("sell@5/quote_deeper", D("100.2"))}


def test_one_sided_book_in_replay(tmp_path):
    tape = Tape()
    tape.snapshot(0, bids=BIDS, asks=[])
    tape.update(10)
    result = engine([req(5), req(5, SELL)], strategies=ALL).run(tape.write(tmp_path))
    assert {(s.request.request_id, s.strategy, s.reason) for s in result.skips} == {
        ("buy@5", "improve_quote", SkipReason.ONE_SIDED_BOOK), ("sell@5", "improve_quote", SkipReason.ONE_SIDED_BOOK),
        ("sell@5", "join_best", SkipReason.NO_TOUCH), ("sell@5", "quote_deeper", SkipReason.NO_TOUCH)}
    assert {(o.order_id, o.price) for o in result.orders} == {("buy@5/join_best", D("100.0")),
                                                              ("buy@5/quote_deeper", D("99.9"))}


def test_improve_quote_intercept(tmp_path):  # scenario 10
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B1", "100.0", "0.7", "modify")])  # touch below us reduced by 0.3
    result = engine([req(5)], strategies=[ImproveQuoteStrategy()]).run(tape.write(tmp_path))
    (o,) = result.orders
    assert o.price == D("100.1") and o.tracker.at_front
    assert [(r.reach.volume, r.reach.mechanism, r.reach.event_price) for r in result.reaches] == [
        (D("0.3"), Mechanism.INTERCEPT, D("100.0"))]


def test_improve_quote_cross_arrival(tmp_path):  # scenario 12
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, asks=[order("A2", "100.1", "0.2", "add")])  # an ask arrives on our price
    result = engine([req(5)], strategies=[ImproveQuoteStrategy()]).run(tape.write(tmp_path))
    assert [(r.reach.volume, r.reach.mechanism) for r in result.reaches] == [(D("0.2"), Mechanism.CROSS_ARRIVAL)]


def test_quote_deeper_non_touch_reduction_is_not_a_candidate(tmp_path):  # scenario 8
    tape = Tape()
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")], asks=ASKS)
    tape.update(10, bids=[order("C2", "99.9", "1.0", "add")])  # behind us
    tape.update(20, bids=[order("C2", "99.9", "0.4", "modify")])  # reduced, but 100.0 is still the touch
    result = engine([req(5)], strategies=[QuoteDeeperStrategy()]).run(tape.write(tmp_path))
    (o,) = result.orders
    assert o.price == D("99.9") and list(o.tracker.ahead_ids) == ["C1"]
    assert result.reaches == []


def test_quote_deeper_touch_swept_then_reduced(tmp_path):  # scenario 9
    tape = Tape()
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")], asks=ASKS)
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete"),
                          order("C1", "99.9", "0.6", "modify")])
    result = engine([req(5)], strategies=[QuoteDeeperStrategy()]).run(tape.write(tmp_path))
    (o,) = result.orders
    # C1 is ahead of us, so its reduction advances us rather than filling us; the tracker sees it at the touch
    assert o.tracker.ahead_qty == D("0.6") and o.tracker.advanced_qty == D("0.4")
    assert result.reaches == [] and result.counters[SYM]["audit_fail"] == 0


def test_quote_deeper_new_level_fills_only_once_it_is_the_touch(tmp_path):  # scenarios 8 and 9, empty queue
    tape = Tape()
    tape.snapshot(0)  # bids only at 100.0: 99.9 is a new level for us
    tape.update(10, bids=[order("C2", "99.9", "1.0", "add")])  # behind us
    tape.update(20, bids=[order("C2", "99.9", "0.8", "modify")])  # not the touch yet: no candidate
    tape.update(30, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete"),
                          order("C2", "99.9", "0.3", "modify")])  # touch swept, then reduced in the same item
    result = engine([req(5)], strategies=[QuoteDeeperStrategy()]).run(tape.write(tmp_path))
    (o,) = result.orders
    assert (o.price, o.queue_ahead_qty0, o.level_rank0, o.max_level_rank) == (D("99.9"), 0, 2, 2)
    assert [(r.reach.volume, r.reach.mechanism, r.reach.item_ts_ns) for r in result.reaches] == [
        (D("0.5"), Mechanism.LEVEL, at(30))]


def test_quote_deeper_insufficient_depth_is_out_of_depth(tmp_path):
    tape = Tape(depth=2)
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "1.0")])
    tape.update(10)
    strategies = [QuoteDeeperStrategy(1, "deeper1"), QuoteDeeperStrategy(2, "deeper2")]
    result = engine([req(5)], strategies=strategies).run(tape.write(tmp_path))
    orders = {o.strategy: o for o in result.orders}
    assert orders["deeper1"].price == D("99.9") and orders["deeper1"].inserted_ts == at(5)
    # 99.8 is worse than the worst level retained by a full depth-2 book: the engine refuses it
    assert orders["deeper2"].price == D("99.8") and orders["deeper2"].skip_reason is SkipReason.OUT_OF_DEPTH
    assert orders["deeper2"].tracker is None


def test_improve_quote_crossed_at_insert_with_latency(tmp_path):  # scenario 18 for a new level
    tape = Tape()
    tape.snapshot(0)
    tape.update(20, asks=[order("A2", "100.1", "1.0", "add")])  # the ask comes down onto our price
    tape.update(100)
    result = engine([req(5)], latency_ms=50, strategies=ALL).run(tape.write(tmp_path))
    orders = {o.strategy: o for o in result.orders}
    assert orders["improve_quote"].price == D("100.1")  # priced from the decision view
    assert orders["improve_quote"].skip_reason is SkipReason.CROSSED_AT_INSERT
    assert orders["join_best"].inserted_ts == orders["quote_deeper"].inserted_ts == at(55)


def test_duplicate_strategy_names_are_rejected():
    with pytest.raises(ValueError):
        engine([], strategies=[QuoteDeeperStrategy(1), QuoteDeeperStrategy(2)])


def _busy_tape():
    tape = Tape()
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")], asks=ASKS + [order("D1", "100.3", "1.0")])
    for i in range(1, 40):
        bids = [order(f"N{i}", ("100.0", "100.1", "99.9")[i % 3], "1.0", "add")]
        if i % 4 == 0:
            bids.append(order(f"N{i - 1}", ("100.0", "100.1", "99.9")[(i - 1) % 3], "0.4", "modify"))
        if i % 6 == 0:
            bids.append(order(f"N{i - 2}", ("100.0", "100.1", "99.9")[(i - 2) % 3], "1.0", "delete"))
        asks = [order("A1", "100.2", str(1 + i % 5), "modify")]
        if i % 7 == 0:
            asks.append(order(f"S{i}", "100.3", "0.5", "add"))
        tape.update(i * 10, bids=bids, asks=asks)
    return tape


def _summary(result, strategy=None):
    return ([(o.order_id, o.price, o.skip_reason, o.inserted_ts, o.end_reason, o.end_ts, o.queue_ahead_qty0,
              o.level_rank0, o.max_level_rank, dict(o.tracker.ahead_ids) if o.tracker else None,
              o.tracker.advanced_qty if o.tracker else None, dict(o.diag))
             for o in result.orders if strategy in (None, o.strategy)],
            [(r.order_id, r.reach, r.ahead_qty, r.checksum_covered) for r in result.reaches
             if strategy in (None, r.strategy)],
            sorted((s.request.request_id, s.strategy, s.reason) for s in result.skips
                   if strategy in (None, s.strategy)))


def test_strategies_are_independent_and_replay_is_deterministic(tmp_path):  # §10.3
    files = _busy_tape().write(tmp_path)
    schedule = [req(ms, side) for ms in range(5, 400, 15) for side in (BUY, SELL)]

    def run(strategies):
        return engine(schedule, horizon_ms=60, strategies=strategies).run(files)

    together = run([JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()])
    again = run([JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()])
    assert _summary(together) == _summary(again)
    assert [(q.kind, q.ts_ns, q.seq) for q in together.quality] == [(q.kind, q.ts_ns, q.seq) for q in again.quality]
    for alone in ([JoinBestStrategy()], [ImproveQuoteStrategy()], [QuoteDeeperStrategy()]):
        name = alone[0].name
        assert _summary(run(alone), name) == _summary(together, name)  # join_best unchanged by the others
    by_strategy = {name: [o for o in together.orders if o.strategy == name]
                   for name in ("join_best", "improve_quote", "quote_deeper")}
    assert all(by_strategy.values()) and together.counters[SYM]["audit_fail"] == 0
    assert any(r.strategy == "improve_quote" for r in together.reaches)


def test_view_prices_are_passive_for_every_placed_order(tmp_path):
    files = _busy_tape().write(tmp_path)
    schedule = [req(ms, side) for ms in range(5, 400, 15) for side in (BUY, SELL)]
    result = engine(schedule, horizon_ms=60, strategies=ALL).run(files)
    for o in result.orders:
        v = result.views[o.request.request_id]
        opposite = v.best_ask if o.side is BUY else v.best_bid
        assert o.price < opposite if o.side is BUY else o.price > opposite
        assert o.price % v.tick == 0
    assert {o.strategy for o in result.orders} == {"join_best", "improve_quote", "quote_deeper"}
