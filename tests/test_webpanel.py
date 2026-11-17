from __future__ import annotations

import logging
import os
import signal
import tomllib
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from conftest import FakeLLM, assistant
from kuni import config as cfgmod
from kuni.config import config
from kuni.notifications import Notification
from kuni.telegram.fake import FakeTelegram
from kuni.telegram_app import TelegramApp
from kuni.webpanel import LogBuffer, WebPanel, load_or_create_token

H = {"X-Kuni-Panel": "1"}


@pytest.fixture
async def panel(tmp_path):
    cfgmod.CONFIG_TOML.write_text(cfgmod.render(config()))  # cwd is the test's tmp dir
    config().papikChatId = 42
    tg = FakeTelegram()
    tg.add_user(42, "Owner")
    tg.add_user(43, "Friend")
    app = TelegramApp(tg, FakeLLM(default=lambda p, m: assistant("", [("wait", {})])))
    clock = [1000.0]
    web = WebPanel(app, port=0, token_path=tmp_path / "token", clock=lambda: clock[0])
    web.clock = clock
    for worker in app.workers:  # nothing may consume the queue behind the test's back
        worker.close()
    client = TestClient(TestServer(web.make_app()))
    await client.start_server()
    web.client, web.tg = client, tg
    yield web
    await client.close()
    await app.close()


async def sign_in(web: WebPanel):
    response = await web.client.post("/api/login", json={"token": web._token}, headers=H)
    assert response.status == 200
    return response


async def act(web, **body):
    return await web.client.post("/api/action", json=body, headers=H)


async def test_token_is_private_and_stable(tmp_path):
    path = tmp_path / "t"
    token = load_or_create_token(path)
    assert len(token) >= 40 and path.stat().st_mode & 0o777 == 0o600
    assert load_or_create_token(path) == token


async def test_api_needs_a_session_but_the_page_does_not(panel):
    for path in ("/api/state", "/api/logs"):
        assert (await panel.client.get(path)).status == 401
    assert (await act(panel, action="pause")).status == 401
    assert (await panel.client.post("/api/logout", json={}, headers=H)).status == 401
    page = await panel.client.get("/")
    assert page.status == 200 and "Куни" in await page.text()
    assert "default-src 'none'" in page.headers["Content-Security-Policy"] and page.headers["Cache-Control"] == "no-store"
    assert page.headers["X-Frame-Options"] == "DENY"


async def test_host_origin_and_csrf_checks(panel):
    assert (await panel.client.get("/", headers={"Host": "evil.example"})).status == 403  # DNS rebinding
    assert (await panel.client.get("/", headers={"Host": "127.0.0.1:18765"})).status == 200  # tunnel port is fine
    assert (await panel.client.post("/api/login", json={"token": panel._token},
                                    headers={**H, "Origin": "https://evil.example"})).status == 403
    assert (await panel.client.post("/api/login", json={"token": panel._token},
                                    headers={**H, "Origin": "null"})).status == 403
    assert (await panel.client.post("/api/login", json={"token": panel._token})).status == 403  # no custom header
    assert (await panel.client.post("/api/login", data="token=x", headers={**H, "Content-Type": "text/plain"})).status == 403
    await sign_in(panel)
    assert (await panel.client.post("/api/action", json={"action": "wake"})).status == 403  # session alone is not enough
    assert (await act(panel, action="wake")).status == 200


async def test_login_lockout_and_expiry(panel):
    for _ in range(5):
        assert (await panel.client.post("/api/login", json={"token": "wrong"}, headers=H)).status == 401
    locked = await panel.client.post("/api/login", json={"token": panel._token}, headers=H)  # even the right one
    assert locked.status == 429 and locked.headers["Retry-After"]
    panel.clock[0] += 121
    await sign_in(panel)
    assert (await panel.client.get("/api/state")).status == 200
    panel.clock[0] += 13 * 3600
    assert (await panel.client.get("/api/state")).status == 401


async def test_logout_ends_the_session(panel):
    await sign_in(panel)
    assert (await panel.client.post("/api/logout", json={}, headers=H)).status == 200
    assert (await panel.client.get("/api/state")).status == 401


async def test_non_object_and_bad_json_are_rejected(panel):
    await sign_in(panel)
    for body in ("[]", "not json", '"x"'):
        response = await panel.client.post("/api/action", data=body, headers={**H, "Content-Type": "application/json"})
        assert response.status == 400
    assert (await act(panel, action="nope")).status == 400


async def test_state_describes_the_bot_without_secrets(panel):
    config().llm.endpoint.bearerKey = "sk-SECRETSECRETSECRET"
    config().llm.endpoint.extraBearerKeys = ("sk-OTHERSECRETKEY123",)
    panel.app.reminders.add(43, 4_000_000_000 - 1, "buy milk", now=4_000_000_000 - 3600)
    panel.app.notification_manager.pass_notification_to_ai(Notification("hello", chat_id=43, priority=5))
    await sign_in(panel)
    raw = await (await panel.client.get("/api/state")).text()
    assert "SECRET" not in raw and panel._token not in raw and "OTHERSECRET" not in raw
    state = await (await panel.client.get("/api/state")).json()
    assert state["paused"] is False and state["llm"]["keys"] == 2 and state["queue"][0]["title"] == "Friend"
    assert state["reminders"][0]["note"] == "buy milk" and state["memory"]["nodes"] >= 0
    assert {s["key"] for s in state["settings"]} == {k for k, v in cfgmod.EDITABLE_SETTINGS.items() if v[0] != "ids"}
    assert not any("key" in s["key"].lower() and "Bearer" in s["key"] for s in state["settings"])


async def test_pause_and_settings_are_persisted_and_validated(panel):
    await sign_in(panel)
    assert (await act(panel, action="pause")).status == 200
    assert tomllib.loads(Path("config.toml").read_text())["general"]["paused"] is True and config().paused
    assert (await act(panel, action="resume")).status == 200 and not config().paused
    assert (await act(panel, action="set", key="channelNotifications", value=False)).status == 200
    assert tomllib.loads(Path("config.toml").read_text())["misc"]["channel_notifications"] is False
    assert (await act(panel, action="set", key="lockdown", value="contacts_only")).status == 200
    assert config().lockdown.value == "contacts_only"
    before = Path("config.toml").read_text()
    for key, value in (("llm", {}), ("telegramApiHash", "x"), ("papikChatId", 1), ("bindHost", "0.0.0.0"),
                       ("lockdown", "everyone"), ("workerCount", 99), ("workerCount", True), ("suggestIgnoreChance", float("nan")),
                       ("canJoinChats", "yes"), ("webPanelPort", 1)):
        assert (await act(panel, action="set", key=key, value=value)).status == 400, key
    assert (await act(panel, action="set", value=True)).status == 400
    assert Path("config.toml").read_text() == before  # rejected changes touch nothing


async def test_credentials_in_the_config_survive_panel_edits(panel):
    config().llm.endpoint.bearerKey = "sk-keepme"
    config().llm.endpoint.extraBearerKeys = ("sk-keepme-too",)
    cfgmod.CONFIG_TOML.write_text(cfgmod.render(config()))
    await sign_in(panel)
    assert (await act(panel, action="set", key="canJoinChats", value=True)).status == 200
    endpoint = tomllib.loads(Path("config.toml").read_text())["general"]["llm"]["endpoint"]
    assert endpoint["bearerKey"] == "sk-keepme" and endpoint["extraBearerKeys"] == ["sk-keepme-too"]
    assert Path("config.toml").stat().st_mode & 0o777 == 0o600


async def test_queue_reminders_and_wake(panel):
    app = panel.app
    app.notification_manager.pass_notification_to_ai(Notification("a", chat_id=43))
    app.notification_manager.pass_notification_to_ai(Notification("b", chat_id=43))
    item = app.reminders.add(43, 4_000_000_000, "later", now=4_000_000_000 - 100)
    await sign_in(panel)
    assert (await act(panel, action="cancel_reminder", id="r000000")).status == 400
    assert (await act(panel, action="cancel_reminder", id=item.id)).status == 200 and not app.reminders.all()
    response = await act(panel, action="clear_queue")
    assert (await response.json())["message"].endswith("2.") and not app.notification_manager._notifications


async def test_owner_can_message_any_chat_but_input_is_checked(panel):
    await sign_in(panel)
    assert (await act(panel, action="send_message", chat_id=43, text="  hi there  ")).status == 200
    sent = [a for a in panel.tg.actions if a[0] == "send_message" and a[1] == 43]
    assert sent[0][2] == "hi there"
    for chat, text in ((0, "x"), (True, "x"), ("43", "x"), (43, ""), (43, "  "), (43, "x" * 4001), (43, 5)):
        assert (await act(panel, action="send_message", chat_id=chat, text=text)).status == 400
    assert len([a for a in panel.tg.actions if a[0] == "send_message" and a[1] == 43]) == 1


async def test_proactive_respects_pause_and_restart_needs_systemd(panel, monkeypatch):
    await sign_in(panel)
    assert (await act(panel, action="proactive")).status == 200 and len(panel.app.notification_manager._notifications) <= 1
    config().paused = True
    assert (await act(panel, action="proactive")).status == 400
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    killed = []
    monkeypatch.setattr(os, "kill", lambda *a: killed.append(a))
    assert (await act(panel, action="restart")).status == 400 and not killed
    monkeypatch.setenv("INVOCATION_ID", "x")
    assert (await act(panel, action="restart")).status == 200
    import asyncio

    await asyncio.sleep(1.2)
    assert killed == [(os.getpid(), signal.SIGTERM)]


async def test_logs_mask_secrets_and_filter_by_kind(panel):
    await sign_in(panel)
    logging.getLogger().addHandler(panel._logs)
    try:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("MessageFlow").info("event=queued chat_id=1 Authorization: Bearer abcdef123456")
        logging.getLogger("x").warning("key sk-ABCDEFGH12345678 leaked")
        logging.getLogger("x").info("private text")
    finally:
        logging.getLogger().removeHandler(panel._logs)
    flow = (await (await panel.client.get("/api/logs?kind=flow")).json())["entries"]
    warn = (await (await panel.client.get("/api/logs?kind=warn")).json())["entries"]
    everything = (await (await panel.client.get("/api/logs?kind=all")).json())["entries"]
    assert [e["name"] for e in flow] == ["MessageFlow"] and "abcdef" not in str(everything) and "ABCDEFGH" not in str(everything)
    assert [e["level"] for e in warn] == ["WARNING"] and len(everything) == 3
    after = everything[-1]["seq"]
    assert (await (await panel.client.get(f"/api/logs?kind=all&after={after}")).json())["entries"] == []
    assert (await panel.client.get("/api/logs?kind=secret")).status == 400
    assert (await panel.client.get("/api/logs?after=x")).status == 400


def test_log_buffer_is_bounded_and_truncates():
    buffer = LogBuffer(size=3)
    for i in range(10):
        buffer.emit(logging.LogRecord("n", logging.INFO, "", 0, "x" * 5000 + str(i), None, None))
    assert len(buffer.entries) == 3 and all(len(e["text"]) <= 600 for e in buffer.entries)


def test_frontend_never_inserts_server_data_as_markup():
    source = (Path(__file__).resolve().parents[1] / "kuni/webui/app.js").read_text()
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "Function("):
        assert forbidden not in source


# ---------------------------------------------------------------------------------------------------------------------
# allow / block lists
# ---------------------------------------------------------------------------------------------------------------------
async def test_allow_and_block_lists_change_access_and_persist(panel):
    from kuni.lockdown import is_accessible_from_lockdown

    tg = panel.tg
    config().lockdown = cfgmod.LockdownMode.PAPIK_ONLY
    await sign_in(panel)
    await tg.incoming(43, "let me in")  # turned away by the lockdown and remembered for the panel
    state = await (await panel.client.get("/api/state")).json()
    assert [d["chat_id"] for d in state["denied"]] == [43] and state["denied"][0]["title"] == "Friend"
    assert not await is_accessible_from_lockdown(tg, 43)

    assert (await act(panel, action="list_add", list="allowed", chat_id=43)).status == 200
    assert tomllib.loads(Path("config.toml").read_text())["general"]["allowed_chat_ids"] == [43]
    assert await is_accessible_from_lockdown(tg, 43)
    state = await (await panel.client.get("/api/state")).json()
    assert state["denied"] == [] and state["lists"]["allowed"][0]["title"] == "Friend"

    config().lockdown = cfgmod.LockdownMode.NONE
    assert (await act(panel, action="list_add", list="blocked", chat_id=43)).status == 200  # moves it, never both
    assert config().allowedChatIds == [] and config().blockedChatIds == [43]
    assert not await is_accessible_from_lockdown(tg, 43)  # blocked wins even in the open mode
    assert (await act(panel, action="list_remove", list="blocked", chat_id=43)).status == 200
    assert await is_accessible_from_lockdown(tg, 43)


async def test_the_owner_cannot_be_blocked_or_locked_out_and_input_is_checked(panel):
    from kuni.lockdown import is_accessible_from_lockdown

    await sign_in(panel)
    assert (await act(panel, action="list_add", list="blocked", chat_id=42)).status == 400
    config().blockedChatIds = [42]
    assert await is_accessible_from_lockdown(panel.tg, 42)  # even a hand-edited file cannot lock the owner out
    for body in ({"list": "x", "chat_id": 1}, {"list": "allowed", "chat_id": 0}, {"list": "allowed", "chat_id": True},
                 {"list": "allowed", "chat_id": "5"}, {"list": "allowed", "chat_id": 2 ** 70}):
        assert (await act(panel, action="list_add", **body)).status == 400


def test_chat_id_lists_roundtrip_through_the_config_file():
    cfg = cfgmod.parse({"general": {"allowed_chat_ids": [5, -100, 5], "blocked_chat_ids": [7]}})
    assert cfg.allowedChatIds == [5, -100] and cfg.blockedChatIds == [7]
    assert cfgmod.parse(tomllib.loads(cfgmod.render(cfg))).allowedChatIds == [5, -100]
    for bad in ("5", [True], ["1"], [1.5]):
        with pytest.raises(ValueError):
            cfgmod.parse({"general": {"allowed_chat_ids": bad}})
    assert cfgmod.parse({}).allowedChatIds == [] and cfgmod.Config().allowedChatIds is not cfgmod.Config().allowedChatIds


# ---------------------------------------------------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------------------------------------------------
async def test_keys_can_be_added_tested_and_removed_without_ever_being_shown(panel):
    import httpx

    config().llm.endpoint.bearerKey = "sk-primary-key-1234"
    cfgmod.CONFIG_TOML.write_text(cfgmod.render(config()))
    await sign_in(panel)
    new = "sk-brand-new-key-ABCDEF"
    assert (await act(panel, action="key_add", key=new)).status == 200
    assert tomllib.loads(Path("config.toml").read_text())["general"]["llm"]["endpoint"]["extraBearerKeys"] == [new]
    assert Path("config.toml").stat().st_mode & 0o777 == 0o600
    raw = await (await panel.client.get("/api/state")).text()
    assert new not in raw and "primary-key" not in raw and "…CDEF" in raw and "…1234" in raw
    state = await (await panel.client.get("/api/state")).json()
    assert [k["primary"] for k in state["keys"]] == [True, False]

    for key in (new, "sk-primary-key-1234", "short", "has space in it", "x" * 401, "", None, 5):
        assert (await act(panel, action="key_add", key=key)).status == 400
    answer = await act(panel, action="key_add", key="sk-second-extra-key")
    assert answer.status == 200 and "second-extra" not in await answer.text()  # the reply never echoes a key

    seen = []
    panel.app.openai._http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: seen.append(r.headers["authorization"]) or httpx.Response(429, text="Bearer secret-in-body")))
    fingerprint = state["keys"][1]["id"]
    result = await (await act(panel, action="key_test", id=fingerprint)).json()
    assert "HTTP 429" in result["message"] and new not in str(result) and "secret-in-body" not in str(result)
    assert seen == ["Bearer " + new]
    assert (await act(panel, action="key_test", id="nope")).status == 400

    primary = state["keys"][0]["id"]
    assert (await act(panel, action="key_remove", id=primary)).status == 400  # the primary key stays
    assert (await act(panel, action="key_remove", id=fingerprint)).status == 200
    assert tomllib.loads(Path("config.toml").read_text())["general"]["llm"]["endpoint"]["extraBearerKeys"] == ["sk-second-extra-key"]
    await panel.app.openai._http.aclose()


async def test_per_key_counters_and_request_events():
    import httpx

    from kuni.llm import LLMError, OpenAIChat

    endpoint = cfgmod.config().llm.endpoint
    endpoint.bearerKey, endpoint.extraBearerKeys = "first-key-aaaa", ("second-key-bbbb",)
    events = []

    def handler(request):
        if request.headers["authorization"].endswith("aaaa"):
            return httpx.Response(429, text="limit", headers={"retry-after": "1"})
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        llm = OpenAIChat(http)
        llm.request_events.append(lambda status, seconds: events.append((status, seconds >= 0)))
        await llm.post_json(cfgmod.config().resolved_llm().endpoint, "chat/completions", {})
        await llm.post_json(cfgmod.config().resolved_llm().endpoint, "chat/completions", {})
    first, second = cfgmod.key_fingerprint("first-key-aaaa"), cfgmod.key_fingerprint("second-key-bbbb")
    assert llm.key_stats[first]["rate_limited"] == 1 and llm.key_stats[first]["ok"] == 0
    assert llm.key_stats[second]["ok"] == 2 and llm.key_stats[second]["requests"] == 2
    assert events[0] == (429, True) and {e[0] for e in events} == {429, 200}
    assert "first-key" not in str(llm.key_stats)


# ---------------------------------------------------------------------------------------------------------------------
# usage history
# ---------------------------------------------------------------------------------------------------------------------
def test_usage_history_aggregates_persists_and_forgets_old_hours(tmp_path):
    from kuni.llm import Usage
    from kuni.usage import KEEP_HOURS, UsageStats

    now = [1_800_000_000.0]
    stats = UsageStats(tmp_path / "usage.json", clock=lambda: now[0])
    stats.request(200, 2.0)
    stats.request(200, 4.0)
    stats.request(429, 0.1)
    stats.request("error", 1)
    stats.tokens("m", Usage(prompt_tokens=100, completion_tokens=7))
    now[0] += 3600
    stats.request(500, 1)
    stats.flush()
    assert (tmp_path / "usage.json").stat().st_mode & 0o777 == 0o600
    again = UsageStats(tmp_path / "usage.json", clock=lambda: now[0])
    first, last = again.series(2)
    assert (first["req"], first["ok"], first["r429"], first["err"], first["avg_ms"]) == (4, 2, 1, 1, 3000)
    assert (first["prompt"], first["completion"]) == (100, 7) and (last["req"], last["err"], last["avg_ms"]) == (1, 1, None)
    now[0] += (KEEP_HOURS + 5) * 3600
    again.request(200, 1)
    again.flush()
    assert len(UsageStats(tmp_path / "usage.json", clock=lambda: now[0])._hours) == 1
    (tmp_path / "usage.json").write_text("not json")
    assert UsageStats(tmp_path / "usage.json")._hours == {}


async def test_usage_endpoint_clamps_and_validates(panel):
    await sign_in(panel)
    panel._usage.request(200, 1.0)
    data = await (await panel.client.get("/api/usage?hours=100000")).json()
    assert len(data["series"]) == 336 and data["totals"]["req"] == 1
    assert len((await (await panel.client.get("/api/usage?hours=0")).json())["series"]) == 1
    assert (await panel.client.get("/api/usage?hours=abc")).status == 400


# ---------------------------------------------------------------------------------------------------------------------
# turn inspector
# ---------------------------------------------------------------------------------------------------------------------
async def test_turn_inspector_explains_silence_and_masks_secrets(tmp_path):
    from conftest import wait_until

    config().papikChatId = 42
    tg = FakeTelegram()
    tg.add_user(42, "Owner")
    script = [assistant("visible words", [("wait", {})])]
    script[0].choices[0].message.reasoning_content = "I think sk-LEAKEDKEY123456 is silly, stay quiet"
    app = TelegramApp(tg, FakeLLM(script, default=lambda p, m: assistant("", [("wait", {})])))
    try:
        await tg.incoming(42, "hello?")
        await wait_until(lambda: app.traces.summary() and app.traces.summary()[0]["outcome"] != "идёт", what="the turn to end")
        summary = app.traces.summary()
        assert summary[0]["chat_id"] == 42 and summary[0]["outcome"] == "wait, без ответа"
        turn = app.traces.turns(42)[0]
        steps = [s for s in turn["steps"] if s["calls"]]
        assert steps[0]["calls"][0]["name"] == "wait" and "visible words" in steps[0]["said"]
        assert "LEAKEDKEY" not in str(turn) and "[hidden]" in steps[0]["thought"]
    finally:
        await app.close()


async def test_turn_inspector_records_replies_errors_and_bounds(panel, monkeypatch):
    from kuni.llm import LLMError
    from kuni.trace import TurnTraces, clip

    traces = TurnTraces(per_chat=2, max_chats=2)
    for chat in (1, 2, 3):
        traces.begin(chat, None, "чат")
    assert [c["chat_id"] for c in traces.summary()] == [3, 2]  # the oldest chat was evicted
    for i in range(5):
        traces.begin(3, i, "чат")
    assert len(traces.turns(3)) == 2 and len(clip("x" * 5000)) < 1600

    async def broken(*args):
        raise LLMError("upstream failed")

    monkeypatch.setattr(panel.app.openai, "chat", broken)
    await panel.app.workers[0].handle_notification(Notification("hi", chat_id=43, message_id=9))
    assert panel.app.traces.summary()[0]["outcome"].startswith("ошибка LLMError")
    await sign_in(panel)
    listing = await (await panel.client.get("/api/traces")).json()
    assert listing["chats"][0]["chat_id"] == 43 and listing["chats"][0]["title"] == "Friend"
    detail = await (await panel.client.get("/api/traces?chat_id=43")).json()
    assert detail["turns"][0]["message_id"] == 9 and detail["turns"][0]["outcome"].endswith("без ответа")
    assert (await panel.client.get("/api/traces?chat_id=abc")).status == 400
    assert (await panel.client.get("/api/traces?chat_id=999")).status == 200


# ---------------------------------------------------------------------------------------------------------------------
# channels, memory, backups, alerts
# ---------------------------------------------------------------------------------------------------------------------
async def test_channel_manager_reads_blocks_and_leaves(panel):
    from kuni.telegram.types import ChatType

    tg = panel.tg
    tg.add_group(-200, "News", ChatType.CHANNEL)
    tg.add_group(-300, "Devs")
    tg.chats[-200].unread_count = 4
    got = []
    panel.app.notification_manager.pass_notification_to_ai = lambda n: got.append(n) or None
    await sign_in(panel)
    channels = (await (await panel.client.get("/api/channels")).json())["channels"]
    assert channels == [{"chat_id": -200, "title": "News", "unread": 4, "blocked": False}]
    assert (await act(panel, action="channel_read", chat_id=-200)).status == 200 and [n.chat_id for n in got] == [-200]
    assert (await act(panel, action="channel_read", chat_id=-300)).status == 400  # a group, not a channel
    assert (await act(panel, action="channel_read", chat_id=43)).status == 400
    assert (await act(panel, action="list_add", list="blocked", chat_id=-200)).status == 200
    assert (await act(panel, action="channel_read", chat_id=-200)).status == 400 and len(got) == 1
    assert (await act(panel, action="channel_leave", chat_id=-300)).status == 400
    assert (await act(panel, action="channel_leave", chat_id=-200)).status == 200 and ("leave_chat", -200) in tg.actions


async def test_memory_browser_working_memory_and_jobs(panel):
    from conftest import wait_until

    memory = panel.app.memory
    node, _ = memory.upsert_entity("Alice", "person", summary="likes jazz", confidence=0.7, source_chat_id=43)
    other, _ = memory.upsert_entity("Jazz", "topic", summary="music", confidence=0.5, source_chat_id=43)
    memory.link(node.id, other.id, "likes", fact="Alice likes jazz", confidence=0.6, source_chat_id=43)
    await sign_in(panel)
    found = await (await panel.client.get("/api/memory?q=alice")).json()
    assert [n["name"] for n in found["nodes"]] == ["Alice"] and found["total"] >= 2
    detail = await (await panel.client.get(f"/api/memory?id={node.id}")).json()
    assert detail["edges"][0]["other"] == "Jazz" and detail["summary"] == "likes jazz"
    assert (await panel.client.get("/api/memory?id=nope")).status == 404
    assert (await act(panel, action="memory_forget", id="nope")).status == 400
    assert (await act(panel, action="memory_forget", id=other.id)).status == 200 and other.id not in memory.nodes

    assert (await act(panel, action="save_working_memory", chat_id=43, text="## state\nsleepy")).status == 200
    assert (await (await panel.client.get("/api/working_memory?chat_id=43")).json())["text"] == "## state\nsleepy"
    chats = (await (await panel.client.get("/api/working_memory")).json())["chats"]
    assert [c["chat_id"] for c in chats] == [42, 43] and chats[1]["size"] > 0
    for chat, text in ((0, "x"), (43, "x" * 20001), (43, 5), ("43", "x")):
        assert (await act(panel, action="save_working_memory", chat_id=chat, text=text)).status == 400

    assert (await act(panel, action="memory_dump")).status == 200
    await wait_until(lambda: panel._jobs["memory_dump"].done(), what="the memory dump")
    state = await (await panel.client.get("/api/state")).json()
    assert state["jobs"]["running"] == [] and state["jobs"]["last"]["memory_dump"] == "память сохранена"


async def test_backup_now_needs_a_directory_and_refuses_a_click_spree(panel, tmp_path):
    Path("prompts").mkdir(exist_ok=True)
    await sign_in(panel)
    assert (await act(panel, action="backup_now")).status == 400
    assert (await (await panel.client.get("/api/backups")).json()) == {"configured": False, "archives": []}
    config().backupDir = str(tmp_path.with_name(tmp_path.name + "-backups"))
    made = await act(panel, action="backup_now")
    assert made.status == 200, await made.text()
    archives = (await (await panel.client.get("/api/backups")).json())["archives"]
    assert len(archives) == 1 and archives[0]["name"].startswith("kuni-backup-") and archives[0]["age"] < 60
    assert (await act(panel, action="backup_now")).status == 400  # too soon: real history must not be pushed out


async def test_owner_is_alerted_in_telegram_about_panel_logins_and_lockout(panel):
    import asyncio

    def alerts():
        return [a[2] for a in panel.tg.actions if a[0] == "send_message" and a[1] == 42]

    for _ in range(5):
        await panel.client.post("/api/login", json={"token": "wrong"}, headers=H)
    await sign_in_after_lockout(panel)
    await asyncio.sleep(0.05)
    texts = alerts()
    assert any("неверный токен" in t for t in texts) and any("Вход в веб-панель" in t for t in texts)
    assert panel._token not in str(texts) and sum("неверный токен" in t for t in texts) == 1


async def sign_in_after_lockout(web):
    web.clock[0] += 121
    await sign_in(web)


# ---------------------------------------------------------------------------------------------------------------------
# phone access through Tailscale (Tailscale Serve forwards to the panel's 127.0.0.1 port)
# ---------------------------------------------------------------------------------------------------------------------
async def test_extra_hosts_are_accepted_only_when_configured_and_cookie_is_secure_behind_https(panel):
    host = {"Host": "kuni.tail1234.ts.net"}
    assert (await panel.client.get("/", headers=host)).status == 403  # unknown names stay forbidden
    config().webPanelExtraHosts = ["Kuni.Tail1234.ts.net"]
    assert (await panel.client.get("/", headers=host)).status == 200
    assert (await panel.client.get("/", headers={"Host": "evil.example"})).status == 403
    login = await panel.client.post("/api/login", json={"token": panel._token},
                                    headers={**H, **host, "Origin": "https://kuni.tail1234.ts.net", "X-Forwarded-Proto": "https"})
    assert login.status == 200 and "Secure" in login.headers["Set-Cookie"] and "HttpOnly" in login.headers["Set-Cookie"]
    assert "SameSite=Strict" in login.headers["Set-Cookie"]
    plain = await panel.client.post("/api/login", json={"token": panel._token}, headers={**H, **host})
    assert "Secure" not in plain.headers["Set-Cookie"]  # plain http inside the tunnel: a Secure cookie would never come back
    assert (await panel.client.post("/api/login", json={"token": panel._token},
                                    headers={**H, **host, "Origin": "https://evil.example"})).status == 403


async def test_tailscale_identity_can_narrow_the_tailnet_to_one_person(panel):
    config().webPanelExtraHosts = ["kuni.tail1234.ts.net"]
    config().webPanelTailscaleLogin = "me@example.com"
    host = {"Host": "kuni.tail1234.ts.net"}
    assert (await panel.client.get("/", headers=host)).status == 403
    assert (await panel.client.get("/", headers={**host, "Tailscale-User-Login": "other@example.com"})).status == 403
    assert (await panel.client.get("/", headers={**host, "Tailscale-User-Login": "ME@example.com"})).status == 200
    assert (await panel.client.get("/")).status == 200  # the local tunnel (scripts/kuni-web) is not affected


def test_extra_hosts_config_is_validated_normalised_and_roundtrips():
    cfg = cfgmod.parse({"misc": {"web_panel_extra_hosts": ["Kuni.Tail1.ts.net", "kuni.tail1.ts.net"],
                                 "web_panel_tailscale_login": "me@example.com"}})
    assert cfg.webPanelExtraHosts == ["kuni.tail1.ts.net"] and cfg.webPanelTailscaleLogin == "me@example.com"
    again = cfgmod.parse(tomllib.loads(cfgmod.render(cfg)))
    assert again.webPanelExtraHosts == ["kuni.tail1.ts.net"] and again.webPanelTailscaleLogin == "me@example.com"
    for bad in ("kuni.ts.net", ["has space"], ["a/b"], ["-x"], [5], ["x" * 300], ["h"] * 21, ["evil.com:80"], [""]):
        with pytest.raises(ValueError):
            cfgmod.parse({"misc": {"web_panel_extra_hosts": bad}})
    assert cfgmod.parse({}).webPanelExtraHosts == [] and cfgmod.Config().webPanelExtraHosts is not cfgmod.Config().webPanelExtraHosts
    assert "webPanelExtraHosts" not in cfgmod.EDITABLE_SETTINGS  # the panel cannot widen its own access
