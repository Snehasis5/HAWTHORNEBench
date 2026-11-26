"""Minimal async HTTP: urllib in a worker thread on CPython, pyfetch under Pyodide (browser)."""
from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request

IN_BROWSER = sys.platform == "emscripten"


class HTTPError(Exception):
    def __init__(self, status: int, body: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body, self.retry_after = status, body, retry_after


def _sync_post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        ra = e.headers.get("retry-after") if e.headers else None
        try:
            ra = float(ra) if ra else None
        except ValueError:
            ra = None
        raise HTTPError(e.code, e.read().decode("utf-8", "replace"), ra) from None


async def post_json(url: str, headers: dict, body: dict, timeout: float = 60.0) -> dict:
    if IN_BROWSER:  # pragma: no cover - exercised in the browser build
        from pyodide.http import pyfetch
        r = await pyfetch(url, method="POST", headers=headers, body=json.dumps(body))
        text = await r.string()
        if not r.ok:
            ra = r.headers.get("retry-after") if hasattr(r, "headers") else None
            raise HTTPError(r.status, text, float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        return json.loads(text)
    return await asyncio.wait_for(asyncio.to_thread(_sync_post, url, headers, body, timeout), timeout + 5)
