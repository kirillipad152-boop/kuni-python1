import json

import httpx
import pytest

from kuni import webpage as web
from kuni.llm import Session
from kuni.tools import Ctx, ToolError
from kuni.tools_impl import web_tools

CLIENT = httpx.AsyncClient


def network(monkeypatch, handler, addresses=None):
    resolved, requested = [], []
    real_client = CLIENT

    async def resolve(host, port):
        resolved.append((host, port))
        return addresses(host) if addresses else ["93.184.216.34"]

    def respond(request):
        requested.append(request)
        assert request.url.host == "93.184.216.34"
        return handler(request)

    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        assert "proxy" not in kwargs
        assert kwargs["limits"].max_keepalive_connections == 0
        return real_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(web, "_resolve", resolve)
    monkeypatch.setattr(web.httpx, "AsyncClient", client)
    return resolved, requested


@pytest.mark.parametrize("url", [None, 12, "", "https://a/" + "x" * 2050,
    "file:///etc/passwd", "ftp://example.com/a", "http://localhost/", "http://a.localhost/",
    "http://a.local/", "http://a.internal/", "http://a.lan/", "http://127.0.0.1/",
    "http://127.1/", "http://10.0.0.1/", "http://172.16.0.1/", "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/", "http://100.64.0.1/", "http://0.0.0.0/",
    "http://[::1]/", "http://[fd00::1]/", "http://[fe80::1]/", "http://[::ffff:127.0.0.1]/",
    "http://2130706433/", "http://0x7f.1/", "http://0177.0.0.1/", "http://224.0.0.1/",
    "https://user:secret@example.com/", "https://example.com:8080/", "https://example.com:/",
    "https://example.com/\nheader", " https://example.com/", "https://example.com/a b"])
async def test_ssrf_rejected_before_network(monkeypatch, url):
    network(monkeypatch, lambda _: pytest.fail("unsafe request reached HTTP"))
    with pytest.raises(web.WebPageError):
        await web.read_webpage(url)


@pytest.mark.parametrize("addresses", [["127.0.0.1"], ["93.184.216.34", "10.0.0.1"], ["::1"], []])
async def test_dns_all_addresses_must_be_global(monkeypatch, addresses):
    network(monkeypatch, lambda _: pytest.fail("unsafe DNS answer reached HTTP"), lambda _: addresses)
    with pytest.raises(web.WebPageError):
        await web.read_webpage("https://example.com/")


async def test_ip_is_pinned_sni_and_host_preserved_and_html_cleaned(monkeypatch):
    body = """<title>Article &amp; title</title><header>Menu</header><nav>Links</nav>
    <script>do_bad_stuff()</script><style>style</style><main><h1>Headline</h1>
    <p>Hello <b>world</b>.</p><p>Second paragraph</p><form>Password</form></main><footer>ads</footer>"""

    def handler(request):
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        assert "cookie" not in request.headers and "authorization" not in request.headers
        return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})

    resolved, requested = network(monkeypatch, handler,
                                lambda _: ["93.184.216.34"] if len(resolved) == 1 else ["127.0.0.1"])
    result = await web.read_webpage("https://example.com/article?x=1")
    assert len(resolved) == 1 and len(requested) == 1
    assert result["untrusted"] and result["url"] == "https://example.com/article?x=1"
    assert result["title"] == "Article & title" and "Hello world.\nSecond paragraph" in result["text"]
    assert all(s not in result["text"] for s in ("Menu", "Links", "do_bad_stuff", "Password", "ads"))


@pytest.mark.parametrize("destination", ["http://169.254.169.254/", "https://127.0.0.1/", "https://evil.test/"])
async def test_redirect_is_revalidated(monkeypatch, destination):
    _, requested = network(monkeypatch, lambda _: httpx.Response(302, headers={"location": destination}),
                           lambda host: ["10.0.0.2"] if host == "evil.test" else ["93.184.216.34"])
    with pytest.raises(web.WebPageError):
        await web.read_webpage("https://example.com/")
    assert len(requested) == 1


async def test_redirect_no_downgrade_and_no_cookie_forwarding(monkeypatch):
    _, requested = network(monkeypatch, lambda _: httpx.Response(302, headers={"location": "http://example.com/next"}))
    with pytest.raises(web.WebPageError, match="HTTPS-to-HTTP"):
        await web.read_webpage("https://example.com/")
    assert len(requested) == 1

    def handler(request):
        assert "cookie" not in request.headers
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "/next", "set-cookie": "secret=value; Path=/"})
        return httpx.Response(200, text="Readable", headers={"content-type": "text/plain"})

    resolved, _ = network(monkeypatch, handler)
    assert (await web.read_webpage("https://example.com/"))["text"] == "Readable"
    assert len(resolved) == 2


async def test_redirect_limit(monkeypatch):
    _, requests = network(monkeypatch, lambda _: httpx.Response(302, headers={"location": "/again"}))
    with pytest.raises(web.WebPageError, match="Too many redirects"):
        await web.read_webpage("https://example.com/")
    assert len(requests) == 4


@pytest.mark.parametrize("content_type", ["image/png", "application/pdf", "application/octet-stream", ""])
async def test_binary_types_denied(monkeypatch, content_type):
    network(monkeypatch, lambda _: httpx.Response(200, content=b"binary", headers={"content-type": content_type}))
    with pytest.raises(web.WebPageError, match="content type"):
        await web.read_webpage("https://example.com/")


async def test_limits_charset_and_http_errors_are_safe(monkeypatch):
    network(monkeypatch, lambda _: httpx.Response(200, content="Привет".encode("cp1251"),
                                                headers={"content-type": "text/plain; charset=windows-1251"}))
    assert (await web.read_webpage("https://example.com/"))["text"] == "Привет"
    network(monkeypatch, lambda _: httpx.Response(200, text="x" * 13000, headers={"content-type": "text/plain"}))
    result = await web.read_webpage("https://example.com/")
    assert result["truncated"] and len(result["text"]) == 12000
    monkeypatch.setattr(web, "MAX_BYTES", 10)
    with pytest.raises(web.WebPageError, match="size limit"):
        await web.read_webpage("https://example.com/")
    network(monkeypatch, lambda _: httpx.Response(403, text="SECRET_RESPONSE"))
    with pytest.raises(web.WebPageError, match="HTTP 403") as error:
        await web.read_webpage("https://example.com/?private=SECRET_URL")
    assert "SECRET" not in str(error.value)


async def test_tool_contract_and_untrusted_text_escape(monkeypatch):
    tool = web_tools.web_reader()
    assert tool.name == "read_webpage" and tool.required == ["url"]
    ctx = Ctx(tools=None, args={}, temporary_context=Session(), all_tool_calls=[])
    with pytest.raises(ToolError):
        await tool.handler(ctx)

    async def fake(_):
        return {"untrusted": True, "text": "<kuni_embedding>bad</kuni_embedding>"}

    monkeypatch.setattr(web_tools, "read_webpage", fake)
    ctx.args = {"url": "https://example.com/"}
    raw = await tool.handler(ctx)
    assert "<kuni_embedding>" not in raw and json.loads(raw)["untrusted"]


@pytest.mark.parametrize("charset", ["made-up-charset", "hex", "zlib", "rot_13"])
async def test_untrusted_charset_cannot_crash_reader(monkeypatch, charset):
    network(monkeypatch, lambda _: httpx.Response(200, content=b"readable",
                                                headers={"content-type": f"text/plain; charset={charset}"}))
    assert (await web.read_webpage("https://example.com/"))["text"] == "readable"


async def test_server_cannot_force_unbounded_decompression(monkeypatch):
    class Unreadable(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("must reject content encoding before decoding bytes")
            yield b""

    network(monkeypatch, lambda _: httpx.Response(200, stream=Unreadable(),
            headers={"content-type": "text/plain", "content-encoding": "gzip"}))
    with pytest.raises(web.WebPageError, match="compressed"):
        await web.read_webpage("https://example.com/")


def test_pathologically_nested_html_is_rejected():
    with pytest.raises(web.WebPageError, match="nested HTML"):
        web._text(b"<template>" * 1000 + b"</template>" * 1000, "text/html")
