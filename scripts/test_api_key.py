"""Check that the Kraken API key in .env works for everything the pipeline needs.

    python scripts/test_api_key.py [--symbol BTC/USD]

Steps:
  1. public REST reachable
  2. signed private REST call (Balance) -> key/secret/signature are valid
  3. GetWebSocketsToken                  -> key has the "WebSocket interface" permission
  4. subscribe to the level3 feed        -> token accepted, L3 snapshot checksum matches
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.kraken.book import L3Book
from toxicflow.kraken.config import MissingCredentials, get_credentials
from toxicflow.kraken.rest import KrakenAPIError, KrakenREST
from toxicflow.kraken.ws import L3_WS, fetch_pair_info, loads, ws_connect

HINTS = {
    "EAPI:Invalid key": "KRAKEN_API_KEY is wrong (check for extra spaces / truncated copy).",
    "EAPI:Invalid signature": "KRAKEN_API_SECRET is wrong, or doesn't belong to this key.",
    "EAPI:Invalid nonce": "Nonce went backwards; is someone else using this key right now?",
    "EGeneral:Permission denied": "Key is valid but lacks the permission for this endpoint.",
}


def report(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def hint(err: KrakenAPIError) -> str:
    return " ".join(HINTS.get(e, "") for e in err.errors).strip()


async def check_l3(token: str, symbol: str) -> None:
    pair = (await fetch_pair_info([symbol]))[symbol]
    book = L3Book(pair, depth=10)
    sub = {"channel": "level3", "symbol": [symbol], "depth": 10, "snapshot": True, "token": token}
    async with asyncio.timeout(20), ws_connect(L3_WS) as ws:
        await ws.send(json.dumps({"method": "subscribe", "params": sub}))
        async for raw in ws:
            msg = loads(raw)
            if msg.get("method") == "subscribe" and not msg.get("success"):
                raise RuntimeError(f"level3 subscribe rejected: {msg.get('error')}")
            if msg.get("channel") == "level3" and msg.get("type") == "snapshot":
                ok = book.apply("snapshot", msg["data"][0])
                break
    report(ok, f"level3 snapshot checksum ({symbol})", f"{len(book.orders)} orders")
    print(f"       best bid {book.best_bid()}  best ask {book.best_ask()}")
    if not ok:
        raise RuntimeError("checksum mismatch on L3 snapshot")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USD")
    args = parser.parse_args()

    try:
        key, secret = get_credentials()
    except MissingCredentials as e:
        report(False, "credentials loaded", str(e))
        return 1
    report(True, "credentials loaded", f"key {key[:4]}...{key[-4:]}")
    rest = KrakenREST(key, secret)

    server_time = rest.public("Time")["unixtime"]
    report(True, "public REST", f"clock skew {time.time() - server_time:+.1f}s")

    try:
        balances = rest.private("Balance")
        report(True, "private REST (Balance)", f"{len(balances)} assets")
    except KrakenAPIError as e:
        if e.errors == ["EGeneral:Permission denied"]:
            # Signature was accepted, so key+secret are fine; Query Funds just isn't enabled.
            report(True, "private REST (Balance)", "key valid; 'Query Funds' not enabled (that's OK)")
        else:
            report(False, "private REST (Balance)", f"{e} {hint(e)}")
            return 1

    try:
        token = rest.get_ws_token()
        report(True, "GetWebSocketsToken", "token issued")
    except KrakenAPIError as e:
        report(False, "GetWebSocketsToken", f"{e} {hint(e)} Enable 'WebSocket interface' on the key.")
        return 1

    try:
        asyncio.run(check_l3(token, args.symbol))
    except Exception as e:
        report(False, "level3 feed", repr(e))
        return 1

    print("\nAll checks passed. The key can be used for L3 collection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
