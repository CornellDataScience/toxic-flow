"""Kraken WebSocket v2 connection handling.

BookStream owns one websocket connection and keeps a set of local books in
sync: it subscribes, routes snapshot/update messages to the books, resubscribes
a symbol whenever its checksum fails, and reconnects (with a fresh auth token
for L3) when the connection drops or goes silent.
"""

import asyncio
import json
import logging
import ssl
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import certifi
import requests
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from .book import L2Book, L3Book, PairInfo

PUBLIC_WS = "wss://ws.kraken.com/v2"
L3_WS = "wss://ws-l3.kraken.com/v2"

L2_DEPTHS = (10, 25, 100, 500, 1000)
L3_DEPTHS = (10, 100, 1000)

# Kraken sends a heartbeat roughly every second; silence this long means the
# connection is dead even if TCP hasn't noticed.
STALE_AFTER_S = 30

log = logging.getLogger(__name__)

# Use certifi's CA bundle: python.org macOS builds ship without system certs,
# which makes websocket TLS fail with CERTIFICATE_VERIFY_FAILED.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def ws_connect(url: str):
    return connect(url, ssl=_SSL_CONTEXT, max_size=None, close_timeout=2)


def loads(raw: str | bytes) -> dict:
    """Parse with Decimal so prices/quantities keep full precision."""
    return json.loads(raw, parse_float=Decimal)


class SubscriptionError(RuntimeError):
    pass


async def fetch_pair_info(symbols: list[str], timeout: float = 15) -> dict[str, PairInfo]:
    """Price/qty precision per symbol from the public instrument channel."""
    async with asyncio.timeout(timeout), ws_connect(PUBLIC_WS) as ws:
        await ws.send(json.dumps({"method": "subscribe", "params": {"channel": "instrument", "snapshot": True}}))
        async for raw in ws:
            msg = loads(raw)
            if msg.get("channel") == "instrument" and msg.get("type") == "snapshot":
                pairs = {p["symbol"]: p for p in msg["data"]["pairs"]}
                break
    missing = [s for s in symbols if s not in pairs]
    if missing:
        raise ValueError(f"unknown symbols {missing}; use Kraken v2 names like 'BTC/USD'")
    return {
        s: PairInfo(s, int(pairs[s]["price_precision"]), int(pairs[s]["qty_precision"]))
        for s in symbols
    }


@dataclass
class StreamStats:
    messages: int = 0
    snapshots: int = 0
    updates: int = 0
    checksum_ok: int = 0
    checksum_fail: int = 0
    resyncs: int = 0
    reconnects: int = 0
    started: float = field(default_factory=time.time)

    def summary(self) -> str:
        checked = self.checksum_ok + self.checksum_fail
        fail_pct = 100 * self.checksum_fail / checked if checked else 0.0
        return (
            f"{time.time() - self.started:.0f}s msgs={self.messages} snapshots={self.snapshots} "
            f"updates={self.updates} checksum_ok={self.checksum_ok} checksum_fail={self.checksum_fail} "
            f"({fail_pct:.3f}%) resyncs={self.resyncs} reconnects={self.reconnects}"
        )


OnRaw = Callable[[int, str, dict], None]
OnBook = Callable[[L2Book | L3Book, str, dict, bool | None], None]
OnMessage = Callable[[dict], None]


class BookStream:
    """
    url/channel: PUBLIC_WS + "book" for L2, L3_WS + "level3" for L3.
    books: symbol -> L2Book/L3Book, all with the same depth.
    token_provider: required for level3; called (in a thread) on every connect.
    extra_subs: additional subscription params to send on connect, e.g.
        {"channel": "trade", "symbol": [...]}. Their messages go to on_message.
    on_raw(recv_ns, raw, msg): every non-ack message, for recording.
    on_book(book, msg_type, item, checksum_ok): after each book item is applied.
    """

    def __init__(
        self,
        url: str,
        channel: str,
        books: dict,
        *,
        token_provider: Callable[[], str] | None = None,
        extra_subs: Sequence[dict] = (),
        on_raw: OnRaw | None = None,
        on_book: OnBook | None = None,
        on_message: OnMessage | None = None,
    ):
        self.url = url
        self.channel = channel
        self.books = books
        self.depth = next(iter(books.values())).depth
        self.token_provider = token_provider
        self.extra_subs = list(extra_subs)
        self.on_raw = on_raw
        self.on_book = on_book
        self.on_message = on_message
        self.stats = StreamStats()
        self._ws = None
        self._token: str | None = None

    async def run(self, duration: float | None = None) -> None:
        try:
            async with asyncio.timeout(duration):
                await self._run_forever()
        except TimeoutError:
            pass

    async def resync(self, symbols: list[str] | None = None) -> None:
        """Drop local state and resubscribe to get fresh snapshots."""
        symbols = symbols or list(self.books)
        for s in symbols:
            self.books[s].reset()
        self.stats.resyncs += len(symbols)
        if self._ws is not None:
            await self._send("unsubscribe", self._book_params(symbols))
            await self._send("subscribe", self._book_params(symbols))

    async def _run_forever(self) -> None:
        backoff = 1
        while True:
            try:
                await self._session()
            except (ConnectionClosed, InvalidHandshake, OSError, TimeoutError, requests.RequestException) as e:
                self.stats.reconnects += 1
                log.warning("%s connection lost (%r); reconnecting in %ss", self.channel, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            else:
                backoff = 1

    async def _session(self) -> None:
        if self.token_provider:
            self._token = await asyncio.to_thread(self.token_provider)
        async with ws_connect(self.url) as ws:
            self._ws = ws
            try:
                for book in self.books.values():
                    book.reset()
                await self._send("subscribe", self._book_params(list(self.books)))
                for params in self.extra_subs:
                    await self._send("subscribe", params)
                while True:
                    raw = await asyncio.wait_for(ws.recv(), STALE_AFTER_S)
                    await self._handle(time.time_ns(), raw)
            finally:
                self._ws = None

    async def _handle(self, recv_ns: int, raw: str) -> None:
        msg = loads(raw)
        if "method" in msg:  # subscribe/unsubscribe ack
            if not msg.get("success", True):
                err = f"{msg['method']} failed: {msg.get('error')}"
                if msg["method"] == "subscribe":
                    raise SubscriptionError(err)
                log.warning(err)
            return
        self.stats.messages += 1
        if self.on_raw:
            self.on_raw(recv_ns, raw, msg)
        if msg.get("channel") == self.channel and msg.get("type") in ("snapshot", "update"):
            for item in msg["data"]:
                book = self.books.get(item["symbol"])
                if book is None:
                    continue
                ok = book.apply(msg["type"], item)
                if msg["type"] == "snapshot":
                    self.stats.snapshots += 1
                else:
                    self.stats.updates += 1
                if ok is True:
                    self.stats.checksum_ok += 1
                elif ok is False:
                    self.stats.checksum_fail += 1
                    log.warning("%s %s checksum mismatch; resyncing", self.channel, book.symbol)
                if self.on_book:
                    self.on_book(book, msg["type"], item, ok)
                if ok is False:
                    await self.resync([book.symbol])
        elif self.on_message:
            self.on_message(msg)

    def _book_params(self, symbols: list[str]) -> dict:
        params = {"channel": self.channel, "symbol": symbols, "depth": self.depth}
        if self._token:
            params["token"] = self._token
        return params

    async def _send(self, method: str, params: dict) -> None:
        if method == "subscribe" and params["channel"] == self.channel:
            params = {**params, "snapshot": True}
        await self._ws.send(json.dumps({"method": method, "params": params}))
