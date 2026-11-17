"""Web search. Providers: Tavily (official API, free tier), keyless DuckDuckGo (HTML page, rate-limits servers) and
Ollama (port of WebSearch.cpp, needs paid credits)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from .config import config

log = logging.getLogger("WebSearch")
URL = "https://ollama.com/api/web_search"
TAVILY_URL = "https://api.tavily.com/search"
DDG_URL = "https://html.duckduckgo.com/html/"
DDG_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
               "Accept-Language": "ru,en;q=0.8"}
DDG_MIN_GAP = 1.5  # seconds between requests: be a polite client, the service is free
CACHE_SECONDS, CACHE_MAX = 600, 50
_last_ddg = 0.0
_cache: dict[str, tuple[float, list]] = {}


class WebSearchError(RuntimeError):
    """A readable failure for the model; it never contains the API key or a response body dump."""


@dataclass
class Result:
    title: str
    url: str
    content: str


def configured() -> bool:
    """True when the search tool can work: enabled, and either the keyless provider or an Ollama key."""
    c = config()
    key = {"duckduckgo": "free", "tavily": c.webSearchTavilyKey, "ollama": c.webSearchOllamaKey}[c.webSearchProvider]
    return c.capabilityWebSearch and bool(key)


async def search(query: str, max_results: int = 0, http: httpx.AsyncClient | None = None) -> list[Result]:
    provider = config().webSearchProvider
    if provider == "duckduckgo":
        return await _search_duckduckgo(query, max_results, http)
    if provider == "tavily":
        return await _search_tavily(query, max_results, http)
    return await _search_ollama(query, max_results, http)


async def _search_tavily(query: str, max_results: int, http: httpx.AsyncClient | None) -> list[Result]:
    key = config().webSearchTavilyKey
    if not key:
        raise WebSearchError("web search is not configured (no Tavily key)")
    body = {"query": query, "max_results": max_results if max_results > 0 else 5, "search_depth": "basic"}
    client = http or httpx.AsyncClient(timeout=20.0)
    try:
        try:
            resp = await client.post(TAVILY_URL, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as exc:
            raise WebSearchError(f"the search service is unreachable ({type(exc).__name__})") from None
    finally:
        if http is None:
            await client.aclose()
    if resp.status_code in (401, 403):
        raise WebSearchError("the search service rejected the key")
    if resp.status_code in (429, 432, 433):  # 432/433: the plan's monthly credits are used up
        raise WebSearchError("the search service limit was reached (the free monthly credits may be used up)")
    try:
        data = resp.json()
    except ValueError:
        raise WebSearchError(f"the search service returned an unexpected answer (HTTP {resp.status_code})") from None
    if not 200 <= resp.status_code < 300 or not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise WebSearchError(f"the search service returned an unexpected answer (HTTP {resp.status_code})")
    return [Result(str(r.get("title", "")), str(r.get("url", "")), str(r.get("content", "")))
            for r in data["results"] if isinstance(r, dict)]


class _DuckParser(HTMLParser):
    """Reads result links and snippets from html.duckduckgo.com; anything else on the page is ignored."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[Result] = []
        self._part: str | None = None  # "title" | "snippet" while inside such an element
        self._buffer: list[str] = []
        self.no_results = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append(Result("", _real_url(attributes.get("href") or ""), ""))
            self._part, self._buffer = "title", []
        elif tag in ("a", "div") and "result__snippet" in classes and self.results:
            self._part, self._buffer = "snippet", []
        elif "no-results" in classes:
            self.no_results = True

    def handle_endtag(self, tag):
        if self._part and tag in ("a", "div"):
            text = " ".join("".join(self._buffer).split())
            if self._part == "title":
                self.results[-1].title = text
            else:
                self.results[-1].content = text
            self._part = None

    def handle_data(self, data):
        if self._part:
            self._buffer.append(data)
        if "No results" in data:
            self.no_results = True


def _real_url(href: str) -> str:
    """DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<encoded url>; ads go through /y.js and are dropped."""
    parts = urlsplit(href)
    target = parse_qs(parts.query).get("uddg", [None])[0]
    url = unquote(target) if target else href
    return "" if "duckduckgo.com/y.js" in url or not url.startswith(("http://", "https://")) else url


async def _search_duckduckgo(query: str, max_results: int, http: httpx.AsyncClient | None) -> list[Result]:
    global _last_ddg
    limit = max_results if max_results > 0 else 5
    cached = _cache.get(query.lower())
    if cached and cached[0] > time.monotonic():
        return cached[1][:limit]
    wait = _last_ddg + DDG_MIN_GAP - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_ddg = time.monotonic()
    client = http or httpx.AsyncClient(timeout=20.0)
    try:
        try:
            resp = await client.post(DDG_URL, data={"q": query}, headers=DDG_HEADERS)
        except httpx.HTTPError as exc:
            raise WebSearchError(f"the search service is unreachable ({type(exc).__name__})") from None
    finally:
        if http is None:
            await client.aclose()
    if resp.status_code in (202, 403, 429):
        raise WebSearchError("the free search service is rate limiting this server; try again in a few minutes")
    if resp.status_code != 200:
        raise WebSearchError(f"the search service returned an unexpected answer (HTTP {resp.status_code})")
    parser = _DuckParser()
    parser.feed(resp.text[:600_000])
    results = [r for r in parser.results if r.url and r.title]
    if not results and not parser.no_results:
        raise WebSearchError("the search page was not recognised (the service may have changed)")
    if len(_cache) >= CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[query.lower()] = (time.monotonic() + CACHE_SECONDS, results)
    return results[:limit]


async def _search_ollama(query: str, max_results: int, http: httpx.AsyncClient | None) -> list[Result]:
    key = config().webSearchOllamaKey
    if not key:
        raise WebSearchError("web search is not configured (no Ollama key)")
    body: dict = {"query": query}
    if max_results > 0:
        body["max_results"] = max_results
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    client = http or httpx.AsyncClient(timeout=min(20.0, float(config().requestTimeoutSecs)))
    try:
        try:
            resp = await client.post(URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise WebSearchError(f"the search service is unreachable ({type(exc).__name__})") from None
    finally:
        if http is None:
            await client.aclose()
    if resp.status_code in (401, 403):
        raise WebSearchError("the search service rejected the key")
    if resp.status_code == 429:
        raise WebSearchError("the search service rate limit was reached; try again later")
    try:
        data = resp.json()
    except ValueError:
        raise WebSearchError(f"the search service returned an unexpected answer (HTTP {resp.status_code})") from None
    if isinstance(data, dict) and "error" in data:
        raise WebSearchError(f"search error: {str(data['error'])[:200]}")
    if not 200 <= resp.status_code < 300 or not isinstance(data, dict) or not isinstance(data.get("results", []), list):
        raise WebSearchError(f"the search service returned an unexpected answer (HTTP {resp.status_code})")
    return [Result(str(r.get("title", "")), str(r.get("url", "")), str(r.get("content", "")))
            for r in data.get("results", []) if isinstance(r, dict)]
