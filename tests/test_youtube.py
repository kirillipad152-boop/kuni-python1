from __future__ import annotations

import asyncio
import json
import sys
import tomllib

import httpx
import pytest

from kuni import config as cfgmod, youtube as yt
from kuni.llm import Session
from kuni.tools import Ctx, ToolError
from kuni.tools_impl import youtube_tools as tools

VIDEO_ID = "abcdefghijk"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
CAPTION_URL = f"https://www.youtube.com/api/timedtext?v={VIDEO_ID}&lang=en&fmt=json3"


def track(url=CAPTION_URL):
    return [{"ext": "json3", "url": url}]


def info(**extra):
    return {"id": VIDEO_ID, "title": "A video", "channel": "A channel", "description": "A description", **extra}


def captions(text="Hello &amp; welcome", ms=1500):
    return {"events": [{"tStartMs": 0, "segs": [{"utf8": "\n"}]},
                       {"tStartMs": ms, "segs": [{"utf8": text}]}]}


def metadata(monkeypatch, data):
    calls = []

    async def extract(target, **kwargs):
        calls.append((target, kwargs))
        return data

    monkeypatch.setattr(yt, "_extract", extract)
    return calls


def backend_output(monkeypatch, data, warning="", returncode=0):
    real_start = asyncio.create_subprocess_exec
    calls = []

    async def start(*args, **kwargs):
        calls.append(args)
        code = "import sys; print(sys.argv[1]); sys.stderr.write(sys.argv[2]); sys.exit(int(sys.argv[3]))"
        return await real_start(sys.executable, "-c", code, json.dumps(data), warning, str(returncode), **kwargs)

    monkeypatch.setattr(yt.asyncio, "create_subprocess_exec", start)
    return calls


def caption_http(monkeypatch, handler):
    client_class = httpx.AsyncClient
    calls = []

    def client(**kwargs):
        calls.append(kwargs.copy())
        assert kwargs["follow_redirects"] is False and kwargs["trust_env"] is False
        assert "auth" not in kwargs and "cookies" not in kwargs
        kwargs.pop("proxy", None)  # The proxy is checked by tests; requests still use the offline transport.
        return client_class(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(yt.httpx, "AsyncClient", client)
    return calls


def test_youtube_proxy_config_defaults_and_roundtrips():
    assert cfgmod.Config().youtubeProxy == ""
    proxy = "http://127.0.0.1:11990"
    cfg = cfgmod.parse({"capabilities": {"youtube": {"proxy_url": proxy}}})
    assert cfg.youtubeProxy == proxy
    assert tomllib.loads(cfgmod.render(cfg))["capabilities"]["youtube"]["proxy_url"] == proxy


@pytest.mark.parametrize("proxy", ["", "http://127.0.0.1:11990", "https://127.0.0.1:11990",
                                  "http://localhost", "http://[::1]:11990/"])
async def test_same_explicit_proxy_for_metadata_search_and_captions(monkeypatch, proxy):
    cfgmod.config().youtubeProxy = proxy
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://environment.invalid:9999")
    monkeypatch.setenv("NO_PROXY", "*")
    calls = backend_output(monkeypatch, info(subtitles={"en": track()}))
    clients = caption_http(monkeypatch, lambda r: httpx.Response(200, json=captions()))
    assert (await yt.read_youtube_video(URL))["transcript"]["text"] == "[0:01] Hello & welcome"
    await yt.search_youtube("test")
    assert len(calls) == 2 and len(clients) == 1
    assert all(args[args.index("--proxy") + 1] == proxy for args in calls)
    assert "--flat-playlist" in calls[1] and "--no-warnings" not in calls[0]
    assert clients[0].get("proxy", "") == proxy
    if not proxy:
        assert "proxy" not in clients[0]


@pytest.mark.parametrize("proxy", [None, 42, "socks5://127.0.0.1:11990", "vless://invalid",
                                  "localhost:11990", "http://", "http://:11990", "http://[::1",
                                  "http://localhost:", "http://localhost:0", "http://localhost:65536",
                                  "http://localhost:bad", "http://localhost/path", "http://localhost?",
                                  "http://localhost#fragment", "http://bad%20host:11990",
                                  "http://bad\\host:11990", " http://localhost:11990", "http://localhost\n"])
async def test_invalid_proxy_rejected_before_any_backend_io(monkeypatch, proxy):
    cfgmod.config().youtubeProxy = proxy

    def forbidden(*args, **kwargs):
        pytest.fail("invalid proxy reached subprocess or HTTP client")

    monkeypatch.setattr(yt.asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(yt.httpx, "AsyncClient", forbidden)
    for request in (yt.search_youtube("test"), yt.read_youtube_video(URL), yt._fetch_captions(CAPTION_URL)):
        with pytest.raises(yt.YouTubeError, match=r"capabilities.youtube.proxy_url.*HTTP/HTTPS"):
            await request


async def test_video_read_keeps_proxy_across_config_reload(monkeypatch):
    proxy = "http://127.0.0.1:11990"
    cfgmod.config().youtubeProxy = proxy

    async def extract(target, **kwargs):
        assert kwargs == {"proxy": proxy}
        cfgmod.config().youtubeProxy = "http://different.invalid:11990"
        return info(subtitles={"en": track()})

    monkeypatch.setattr(yt, "_extract", extract)
    clients = caption_http(monkeypatch, lambda r: httpx.Response(200, json=captions()))
    assert (await yt.read_youtube_video(URL))["transcript"]
    assert clients[0]["proxy"] == proxy


@pytest.mark.parametrize("warning,blocked", [
    ("WARNING: Sign in to confirm you are not a bot", True),
    ("WARNING: Sign in to confirm you’re not a bot", True),
    ("WARNING: All player responses are invalid. Your IP is likely being blocked by Youtube", True),
    ("WARNING: Some web client subtitles require a PO Token which was not provided", True),
    ("WARNING: There are missing subtitles languages because a PO token was not provided", True),
    ("WARNING: HTTP Error 429: Too Many Requests", True),
    ("WARNING: No supported JavaScript runtime could be found", False),
    ("", False),
])
async def test_exit_zero_access_warnings_report_unknown_not_absent_captions(monkeypatch, warning, blocked):
    backend_output(monkeypatch, info(), warning)
    result = await yt.read_youtube_video(URL)
    assert result["title"] == "A video" and result["description"] == "A description"
    assert result["transcript"] is None and result["available_languages"] == []
    reason = result["transcript_unavailable_reason"]
    if blocked:
        assert "blocked this server" in reason and "caption accessibility is unknown" in reason
    else:
        assert reason == ("No accessible subtitles were provided by YouTube. Use the description only. "
                          "The video was not watched or listened to.")


async def test_usable_captions_win_over_another_client_access_warning(monkeypatch):
    backend_output(monkeypatch, info(subtitles={"en": track()}), "WARNING: Other client: not a bot")
    caption_http(monkeypatch, lambda r: httpx.Response(200, json=captions()))
    result = await yt.read_youtube_video(URL)
    assert result["transcript"]["text"] == "[0:01] Hello & welcome"
    assert "transcript_unavailable_reason" not in result and "_caption_access_error" not in result


async def test_empty_http_200_captions_remain_a_distinct_retrieval_failure(monkeypatch):
    backend_output(monkeypatch, info(subtitles={"en": track()}), "WARNING: Other client: not a bot")
    caption_http(monkeypatch, lambda r: httpx.Response(200, content=b""))
    result = await yt.read_youtube_video(URL)
    assert result["transcript"] is None
    assert "empty or invalid captions" in result["transcript_unavailable_reason"]
    assert "accessibility is unknown" not in result["transcript_unavailable_reason"]


async def test_proxy_credentials_never_appear_in_reported_errors(monkeypatch):
    secret = "private-proxy-password"
    proxy = f"http://user:{secret}@127.0.0.1:11990"
    cfgmod.config().youtubeProxy = proxy
    with monkeypatch.context() as patch:
        backend_output(patch, info(), "WARNING: not a bot " + proxy)
        assert secret not in json.dumps(await yt.read_youtube_video(URL))
    backend_output(monkeypatch, {}, "ERROR: " + proxy, returncode=1)
    with pytest.raises(yt.YouTubeError, match="anonymous access") as error:
        await yt._extract(URL)
    assert secret not in str(error.value)

    def denied(**kwargs):
        raise httpx.ProxyError(proxy)

    monkeypatch.setattr(yt.httpx, "AsyncClient", denied)
    with pytest.raises(yt.YouTubeError, match="caption request") as error:
        await yt._fetch_captions(CAPTION_URL)
    assert secret not in str(error.value)
    cfgmod.config().youtubeProxy = proxy + "?invalid"
    with pytest.raises(yt.YouTubeError, match="proxy_url") as error:
        await yt._extract(URL)
    assert secret not in str(error.value)


@pytest.mark.parametrize("url", [URL, URL + "&list=xyz&t=30", f"https://youtu.be/{VIDEO_ID}?si=xyz",
                                  f"http://m.youtube.com/shorts/{VIDEO_ID}",
                                  f"https://youtube.com/live/{VIDEO_ID}",
                                  f"https://www.youtube.com/embed/{VIDEO_ID}/"])
def test_video_urls_are_canonicalized(url):
    assert yt._video_url(url) == URL


@pytest.mark.parametrize("url", [None, 42, [], VIDEO_ID, "https://example.com/" + VIDEO_ID,
                                  URL.replace("youtube.com", "youtube.com.evil.test"),
                                  URL.replace("www.youtube.com", "evil.youtube.com"),
                                  URL.replace("www.youtube.com", "youtube.com@127.0.0.1"),
                                  URL.replace("www.youtube.com", "user:pass@www.youtube.com"),
                                  URL.replace("www.youtube.com", "www.youtube.com:8080"),
                                  URL.replace("https:", "file:"), URL + "&v=xxxxxxxxxxx",
                                  URL.replace(VIDEO_ID, "short"), URL.replace(VIDEO_ID, VIDEO_ID + "X"),
                                  URL + "\n--exec=touch /tmp/nope", URL + "\n",
                                  f"https://youtu.be/{VIDEO_ID}/other",
                                  f"https://youtube.com/watch?v={VIDEO_ID}%0a--exec=bad"])
async def test_invalid_urls_never_reach_backend(monkeypatch, url):
    calls = metadata(monkeypatch, {})
    with pytest.raises(ValueError, match="url must"):
        await yt.read_youtube_video(url)
    assert not calls


@pytest.mark.parametrize("query", [None, False, 1, [], "", "  ", "x" * 501, "hello\x00", "hello\nworld"])
async def test_invalid_search_args(monkeypatch, query):
    calls = metadata(monkeypatch, {})
    with pytest.raises(ValueError, match="query must"):
        await yt.search_youtube(query)
    assert not calls


@pytest.mark.parametrize("language", [True, [], 1, "", " ", "en.*", "--all", "english", "en\x00"])
async def test_invalid_language_args(monkeypatch, language):
    calls = metadata(monkeypatch, {})
    with pytest.raises(ValueError, match="language must"):
        await yt.read_youtube_video(URL, language)
    assert not calls


async def test_search_parses_flat_entries_limits_and_canonicalizes_links(monkeypatch):
    entries = [None, {"id": "bad"}, {"id": VIDEO_ID, "title": "Title", "uploader": "Uploader",
                                             "description": "a" * 800, "url": "http://127.0.0.1"},
               {"id": VIDEO_ID, "title": "duplicate"}]
    entries += [{"id": f"{i:011d}", "title": "t" * 400, "channel": "c" * 300} for i in range(10)]
    calls = metadata(monkeypatch, {"entries": entries})
    result = await yt.search_youtube(" --exec=bad && query ")
    assert calls == [("ytsearch5:--exec=bad && query", {"search": True})]
    assert result["untrusted"] and len(result["results"]) == 5
    first = result["results"][0]
    assert first == {"title": "Title", "channel": "Uploader", "url": URL, "description_snippet": "a" * 600}
    assert len(result["results"][1]["title"]) == 300
    assert len(result["results"][1]["channel"]) == 200
    assert "not watched" in result["notice"]


async def test_no_search_results_and_invalid_shape(monkeypatch):
    metadata(monkeypatch, {"entries": []})
    assert (await yt.search_youtube("nothing"))["results"] == []
    metadata(monkeypatch, {"entries": {"malformed": True}})
    with pytest.raises(yt.YouTubeError, match="invalid search"):
        await yt.search_youtube("test")


@pytest.mark.parametrize("kind", ["manual", "automatic"])
async def test_read_manual_and_automatic_captions_with_timestamps(monkeypatch, kind):
    field = "subtitles" if kind == "manual" else "automatic_captions"
    calls = metadata(monkeypatch, info(**{field: {"en": track()}}))

    def respond(request):
        assert request.url == CAPTION_URL
        assert "cookie" not in request.headers and "authorization" not in request.headers
        return httpx.Response(200, json=captions())

    caption_http(monkeypatch, respond)
    result = await yt.read_youtube_video(f"https://youtu.be/{VIDEO_ID}", " EN_us ")
    assert calls == [(URL, {"proxy": ""})]
    assert result["description"] == "A description"
    assert result["transcript"] == {"language": "en", "kind": kind, "translated": False,
                                     "text": "[0:01] Hello & welcome", "truncated": False}


async def test_manual_captions_preferred_and_failed_track_can_fall_back_to_automatic(monkeypatch):
    auto_url = CAPTION_URL + "&kind=asr"
    metadata(monkeypatch, info(subtitles={"en": track()}, automatic_captions={"en-orig": track(auto_url)}))
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(403) if len(requests) == 1 else httpx.Response(200, json=captions("automatic"))

    caption_http(monkeypatch, respond)
    result = await yt.read_youtube_video(URL)
    assert requests == [CAPTION_URL, auto_url]
    assert result["transcript"]["kind"] == "automatic"


async def test_original_auto_language_preferred_and_translation_is_labelled(monkeypatch):
    translated = CAPTION_URL + "&tlang=ru"
    metadata(monkeypatch, info(automatic_captions={"ru": track(translated), "en-orig": track(), "en": track()}))
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(200, json=captions())

    caption_http(monkeypatch, respond)
    assert (await yt.read_youtube_video(URL))["transcript"]["language"] == "en"
    result = await yt.read_youtube_video(URL, "RU")
    assert requests == [CAPTION_URL, translated]
    assert result["transcript"]["translated"] is True


@pytest.mark.parametrize("extra,language", [({}, None), ({"subtitles": {"en": track()}}, "ru")])
async def test_unavailable_captions_preserve_description_and_do_not_claim_watching(monkeypatch, extra, language):
    metadata(monkeypatch, info(**extra))
    result = await yt.read_youtube_video(URL, language)
    assert result["description"] == "A description" and result["transcript"] is None
    assert "not watched" in result["transcript_unavailable_reason"]
    assert result["available_languages"] == (["en"] if extra else [])


@pytest.mark.parametrize("url", ["http://www.youtube.com/api/timedtext?v=" + VIDEO_ID,
                                  "https://www.youtube.com.evil.test/api/timedtext?v=" + VIDEO_ID,
                                  "https://127.0.0.1/api/timedtext?v=" + VIDEO_ID,
                                  CAPTION_URL.replace("www.youtube.com", "user:pass@www.youtube.com"),
                                  CAPTION_URL.replace("www.youtube.com", "www.youtube.com:8443"),
                                  CAPTION_URL.replace("api/timedtext", "redirect"),
                                  CAPTION_URL.replace(VIDEO_ID, "xxxxxxxxxxx"), CAPTION_URL + "\n"])
async def test_unsafe_caption_urls_are_not_fetched(monkeypatch, url):
    metadata(monkeypatch, info(subtitles={"en": track(url)}))

    async def forbidden(_):
        pytest.fail("unsafe caption URL was fetched")

    monkeypatch.setattr(yt, "_fetch_captions", forbidden)
    assert (await yt.read_youtube_video(URL))["transcript"] is None


@pytest.mark.parametrize("status,body", [(302, None), (429, None), (200, b""), (200, b"not json"),
                                        (200, b"[]"), (200, b'{"events":{}}'),
                                        (200, b'{"events":[]}')])
async def test_caption_errors_are_actionable_not_fatal_and_redirects_not_followed(monkeypatch, status, body):
    metadata(monkeypatch, info(subtitles={"en": track()}))
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, content=body, headers={"location": "http://127.0.0.1/private"})

    caption_http(monkeypatch, respond)
    result = await yt.read_youtube_video(URL)
    assert len(requests) == 1 and result["transcript"] is None
    assert result["description"] and result["transcript_unavailable_reason"]


async def test_caption_timeout(monkeypatch):
    metadata(monkeypatch, info(subtitles={"en": track()}))
    monkeypatch.setattr(yt, "_CAPTION_TIMEOUT", 0.01)

    async def blocked(request):
        await asyncio.Future()

    caption_http(monkeypatch, blocked)
    assert "timed out" in (await yt.read_youtube_video(URL))["transcript_unavailable_reason"]


async def test_caption_response_limit(monkeypatch):
    metadata(monkeypatch, info(subtitles={"en": track()}))
    monkeypatch.setattr(yt, "_MAX_JSON_BYTES", 10)
    caption_http(monkeypatch, lambda r: httpx.Response(200, json=captions()))
    assert "size limit" in (await yt.read_youtube_video(URL))["transcript_unavailable_reason"]


async def test_description_and_transcript_truncation(monkeypatch):
    metadata(monkeypatch, info(description="d" * 6000, subtitles={"en": track()}))
    caption_http(monkeypatch, lambda r: httpx.Response(200, json=captions("a" * 20_000)))
    result = await yt.read_youtube_video(URL)
    assert len(result["description"]) == 5000 and result["description_truncated"]
    assert len(result["transcript"]["text"]) == 16_000 and result["transcript"]["truncated"]
    assert yt._transcript({"events": [None, {"tStartMs": -1, "segs": []},
                                       {"tStartMs": 0, "segs": [{"utf8": 3}]}]}) == ("", False)


@pytest.mark.parametrize("bad_info", [{}, {"id": VIDEO_ID, "title": f"youtube video #{VIDEO_ID}"},
                                      {"id": "xxxxxxxxxxx", "title": "Wrong video"}])
async def test_missing_or_wrong_metadata_is_not_a_success(monkeypatch, bad_info):
    metadata(monkeypatch, bad_info)
    with pytest.raises(yt.YouTubeError, match="usable metadata"):
        await yt.read_youtube_video(URL)


async def test_subprocess_argv_is_anonymous_and_shell_free(monkeypatch):
    real_start = asyncio.create_subprocess_exec
    seen = []

    async def start(*args, **kwargs):
        seen.append(args)
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL
        return await real_start(sys.executable, "-c", "print('{\"entries\": []}')", **kwargs)

    monkeypatch.setattr(yt.asyncio, "create_subprocess_exec", start)
    assert await yt._extract("ytsearch5:--exec=bad; echo unsafe", search=True) == {"entries": []}
    args = seen[0]
    assert args[:3] == (sys.executable, "-m", "yt_dlp")
    assert args[-2:] == ("--", "ytsearch5:--exec=bad; echo unsafe")
    for flag in ("--ignore-config", "--no-plugin-dirs", "--no-cache-dir", "--skip-download", "--flat-playlist"):
        assert flag in args
    assert "--cookies" not in args and "--cookies-from-browser" not in args


@pytest.mark.parametrize("mode", ["timeout", "oversize", "cancel"])
async def test_subprocess_is_killed_and_reaped_on_timeout_oversize_or_cancellation(monkeypatch, mode):
    real_start = asyncio.create_subprocess_exec
    processes = []
    started = asyncio.Event()
    code = "import time; time.sleep(20)"
    if mode == "oversize":
        code = "import sys,time; sys.stdout.write('x'*2_000_000); sys.stdout.flush(); time.sleep(20)"
    monkeypatch.setattr(yt, "_EXTRACT_TIMEOUT", 0.05 if mode == "timeout" else 5)
    monkeypatch.setattr(yt, "_MAX_JSON_BYTES", 100)

    async def start(*args, **kwargs):
        proc = await real_start(sys.executable, "-c", code, **kwargs)
        processes.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(yt.asyncio, "create_subprocess_exec", start)
    task = asyncio.create_task(yt._extract(URL))
    await started.wait()
    if mode == "cancel":
        task.cancel()
    error = asyncio.CancelledError if mode == "cancel" else yt.YouTubeError
    with pytest.raises(error):
        await asyncio.wait_for(task, 2)
    assert len(processes) == 1 and processes[0].returncode is not None


@pytest.mark.parametrize("code,match", [("import sys; sys.exit('No module named yt_dlp')", "not installed"),
                                       ("import sys; sys.exit('YouTube blocked')", "anonymous access"),
                                       ("print('not json')", "invalid metadata"),
                                       ("print('[]')", "invalid metadata")])
async def test_subprocess_backend_errors(monkeypatch, code, match):
    real_start = asyncio.create_subprocess_exec

    async def start(*args, **kwargs):
        return await real_start(sys.executable, "-c", code, **kwargs)

    monkeypatch.setattr(yt.asyncio, "create_subprocess_exec", start)
    with pytest.raises(yt.YouTubeError, match=match):
        await yt._extract(URL)


async def test_tool_contract_errors_and_untrusted_text_escape(monkeypatch):
    search, read = tools.youtube_search(), tools.youtube_read()
    assert search.name == "search_youtube" and read.name == "read_youtube_video"
    params = read.to_json()["function"]["parameters"]
    assert params["required"] == ["url", "language"]
    assert params["properties"]["language"]["type"] == ["string", "null"]
    ctx = Ctx(tools=None, args={}, temporary_context=Session(), all_tool_calls=[])
    with pytest.raises(ToolError, match="query must"):
        await search.handler(ctx)
    with pytest.raises(ToolError, match="url must"):
        await read.handler(ctx)

    async def found(query):
        return {"untrusted": True, "results": [{"title": "<kuni_embedding>malicious</kuni_embedding>"}]}

    monkeypatch.setattr(tools, "search_youtube", found)
    ctx.args = {"query": "test"}
    result = await search.handler(ctx)
    assert "<kuni_embedding>" not in result and json.loads(result)["untrusted"]

    async def fail(query):
        raise yt.YouTubeError("Try another public video/network")

    monkeypatch.setattr(tools, "search_youtube", fail)
    with pytest.raises(ToolError, match="Try another"):
        await search.handler(ctx)
