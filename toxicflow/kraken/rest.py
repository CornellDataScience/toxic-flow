"""Minimal Kraken Spot REST client (public + signed private endpoints).

We only need REST for one thing in the data pipeline: minting a WebSocket
auth token (GetWebSocketsToken), which the L3 feed requires.
"""

import base64
import hashlib
import hmac
import time
import urllib.parse

import requests

API_URL = "https://api.kraken.com"


class KrakenAPIError(RuntimeError):
    def __init__(self, method: str, errors: list[str]):
        self.method = method
        self.errors = errors
        super().__init__(f"{method}: {', '.join(errors)}")


def sign(url_path: str, data: dict, secret_b64: str) -> str:
    """API-Sign = b64(HMAC-SHA512(b64decode(secret), path + SHA256(nonce + postdata)))."""
    postdata = urllib.parse.urlencode(data)
    digest = hashlib.sha256((str(data["nonce"]) + postdata).encode()).digest()
    mac = hmac.new(base64.b64decode(secret_b64), url_path.encode() + digest, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


class KrakenREST:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None, timeout: float = 10):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.session = requests.Session()

    def public(self, method: str, **params) -> dict:
        resp = self.session.get(f"{API_URL}/0/public/{method}", params=params, timeout=self.timeout)
        return self._unwrap(method, resp)

    def private(self, method: str, **params) -> dict:
        if not (self.api_key and self.api_secret):
            raise RuntimeError("private endpoint called without API credentials")
        url_path = f"/0/private/{method}"
        # Nonces must strictly increase per key. If several people use the same
        # key at once you may see EAPI:Invalid nonce; separate keys avoid that.
        data = {"nonce": time.time_ns() // 1000, **params}
        headers = {
            "API-Key": self.api_key,
            "API-Sign": sign(url_path, data, self.api_secret),
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        resp = self.session.post(f"{API_URL}{url_path}", data=data, headers=headers, timeout=self.timeout)
        return self._unwrap(method, resp)

    def get_ws_token(self) -> str:
        """Token for authenticated WebSocket channels (e.g. level3).

        Must be used to connect within 15 minutes; stays valid for the life of
        the connection after that.
        """
        return self.private("GetWebSocketsToken")["token"]

    @staticmethod
    def _unwrap(method: str, resp: requests.Response) -> dict:
        resp.raise_for_status()
        body = resp.json()
        if body.get("error"):
            raise KrakenAPIError(method, body["error"])
        return body["result"]
