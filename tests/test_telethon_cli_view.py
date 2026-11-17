from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime, timezone

import pytest
from telethon.tl import types

from kuni import cli
from kuni import config as cfgmod
from kuni.config import config
from kuni.graph_view import graph_payload, render_html
from kuni.memory.graph import MENTIONS, GraphMemory
from kuni.telegram.telethon_client import TelethonClient
from kuni.telegram.types import ChatType

DT = datetime(2026, 5, 21, 6, 38, 8, tzinfo=timezone.utc)


@pytest.fixture
def tl():
    """A TelethonClient that never connects: only the pure conversion code is exercised."""
    cfg = config()
    cfg.telegramApiId, cfg.telegramApiHash = 12345, "abcdef"
    client = TelethonClient()
    client._my_id = 1000
    yield client


def msg(**kw) -> types.Message:
    base = dict(id=5, peer_id=types.PeerUser(7), date=DT, message="hi", out=False, from_id=types.PeerUser(7))
    base.update(kw)
    return types.Message(**base)


def doc(*attrs, mime="application/octet-stream") -> types.MessageMediaDocument:
    return types.MessageMediaDocument(document=types.Document(id=99, access_hash=1, file_reference=b"", date=DT, mime_type=mime,
                                                              size=10, dc_id=2, attributes=list(attrs)))


def test_basic_fields_ids_and_dates(tl):
    m = tl.convert(msg())
    assert (m.id, m.chat_id, m.sender_id, m.text, m.kind, m.is_outgoing) == (5, 7, 7, "hi", "text", False)
    assert m.date == int(DT.timestamp()) and not m.sender_is_chat and m.raw is not None


def test_group_ids_are_marked_like_tdlib_chat_ids(tl):
    m = tl.convert(msg(peer_id=types.PeerChat(123), from_id=types.PeerUser(7)))
    assert m.chat_id == -123 and m.sender_id == 7
    m = tl.convert(msg(peer_id=types.PeerChannel(456), from_id=None, post_author="Editor"))
    assert m.chat_id == -1000000000456 and m.sender_id == m.chat_id and m.sender_is_chat  # a channel post: the sender is the channel
    assert m.author_signature == "Editor"


def test_outgoing_and_private_messages_without_from_id(tl):
    mine = tl.convert(msg(out=True, from_id=None))
    assert mine.sender_id == 1000 and mine.is_outgoing
    theirs = tl.convert(msg(from_id=None))
    assert theirs.sender_id == 7 and not theirs.sender_is_chat


def test_member_tag_reply_forward_and_reactions(tl):
    m = tl.convert(msg(
        from_rank="wizard",
        reply_to=types.MessageReplyHeader(reply_to_msg_id=3),
        fwd_from=types.MessageFwdHeader(date=DT, from_id=types.PeerChannel(555), from_name=None),
        reactions=types.MessageReactions(
            results=[types.ReactionCount(reaction=types.ReactionEmoji("👍"), count=2, chosen_order=None),
                     types.ReactionCount(reaction=types.ReactionCustomEmoji(5), count=9)],
            recent_reactions=[types.MessagePeerReaction(reaction=types.ReactionEmoji("👍"), peer_id=types.PeerUser(9), date=DT)])))
    assert m.sender_tag == "wizard" and m.reply_to_id == 3
    assert m.forward.origin_id == -1000000000555
    assert [(r.emoji, r.total_count, r.recent_sender_ids) for r in m.reactions] == [("👍", 2, [9])]  # custom emoji: don't care


@pytest.mark.parametrize("media,kind,extra", [
    (types.MessageMediaPhoto(photo=types.PhotoEmpty(id=1)), "photo", {}),
    (doc(types.DocumentAttributeSticker(alt="😀", stickerset=types.InputStickerSetEmpty()), mime="image/webp"), "sticker", {"sticker_id": 99, "emoji": "😀"}),
    (doc(types.DocumentAttributeAudio(duration=3, voice=True)), "voice", {}),
    (doc(types.DocumentAttributeAudio(duration=3, title="Song")), "audio", {"title": "Song"}),
    (doc(types.DocumentAttributeVideo(duration=3, w=10, h=10, round_message=True)), "video_note", {}),
    (doc(types.DocumentAttributeVideo(duration=3, w=10, h=10)), "video", {}),
    (doc(types.DocumentAttributeVideo(duration=3, w=10, h=10), types.DocumentAttributeAnimated()), "animation", {}),
    (doc(types.DocumentAttributeFilename(file_name="a.pdf")), "document", {"file_name": "a.pdf"}),
    (types.MessageMediaGeo(geo=types.GeoPoint(long=2.5, lat=1.5, access_hash=0)), "location", {"lat": 1.5, "lon": 2.5}),
    (types.MessageMediaContact(phone_number="+1", first_name="A", last_name="B", vcard="", user_id=0), "contact", {"first_name": "A", "last_name": "B", "phone": "+1"}),
    (types.MessageMediaDice(value=6, emoticon="🎲"), "dice", {"emoji": "🎲", "value": 6}),
    (types.MessageMediaUnsupported(), "unsupported", {}),
])
def test_media_kinds(tl, media, kind, extra):
    m = tl.convert(msg(media=media, message="caption"))
    assert m.kind == kind and m.text == "caption"
    for k, v in extra.items():
        assert m.extra[k] == v
    if kind == "sticker":
        assert 99 in tl._stickers  # remembered so that it can be saved/sent later


def test_poll(tl):
    q = types.TextWithEntities("Cats or dogs?", [])
    poll = types.Poll(id=1, question=q, answers=[types.PollAnswer(types.TextWithEntities("cats", []), b"1"),
                                                  types.PollAnswer(types.TextWithEntities("dogs", []), b"2")], hash=0)
    m = tl.convert(msg(media=types.MessageMediaPoll(poll=poll, results=types.PollResults())))
    assert m.kind == "poll" and m.extra == {"question": "Cats or dogs?", "options": ["cats", "dogs"]}


def test_web_page_becomes_a_link_preview(tl):
    wp = types.WebPage(id=1, url="https://x.test", display_url="x.test", hash=0, site_name="X", title="Title", description="Desc", author="Me")
    m = tl.convert(msg(media=types.MessageMediaWebPage(webpage=wp)))
    assert m.kind == "text" and m.link_preview == {"url": "https://x.test", "site_name": "X", "title": "Title", "author": "Me", "description": "Desc"}


@pytest.mark.parametrize("action,expected", [
    (types.MessageActionChatAddUser(users=[1, 2]), "[members added] 2 member(s)"),
    (types.MessageActionChatDeleteUser(user_id=7), "[member removed] user_id=7"),
    (types.MessageActionChatEditTitle(title="New"), "[title changed] New"),
    (types.MessageActionChatCreate(title="G", users=[]), "[group created] G"),
    (types.MessageActionChannelCreate(title="C"), "[supergroup created] C"),
    (types.MessageActionChatJoinedByLink(inviter_id=1), "[joined via link]"),
    (types.MessageActionScreenshotTaken(), "[screenshot taken]"),
    (types.MessageActionPinMessage(), "[message pinned]"),
    (types.MessageActionContactSignUp(), "[contact registered]"),
])
def test_service_messages(tl, action, expected):
    m = tl.convert(types.MessageService(id=5, peer_id=types.PeerChat(1), date=DT, action=action, from_id=types.PeerUser(7)))
    assert m.kind == "service" and m.text == expected


def test_calls_and_gifts_have_their_own_markup(tl):
    call = tl.convert(types.MessageService(id=1, peer_id=types.PeerUser(7), date=DT, from_id=types.PeerUser(7),
                                           action=types.MessageActionPhoneCall(call_id=1, video=True, duration=42)))
    assert call.kind == "call" and call.extra == {"is_video": True, "duration": 42}
    unknown = tl.convert(types.MessageService(id=1, peer_id=types.PeerUser(7), date=DT, from_id=types.PeerUser(7),
                                              action=types.MessageActionGiveawayResults(winners_count=1, unclaimed_count=0)))
    assert unknown.kind == "service" and unknown.text.startswith("[")  # an unmapped service message degrades gracefully


def test_chat_types(tl):
    user = types.User(id=1, first_name="A")
    assert tl._chat_type(user) == ChatType.PRIVATE
    assert tl._chat_type(types.Chat(id=1, title="g", photo=types.ChatPhotoEmpty(), participants_count=2, date=DT, version=1)) == ChatType.BASIC_GROUP
    mk = lambda **kw: types.Channel(id=1, title="c", photo=types.ChatPhotoEmpty(), date=DT, **kw)  # noqa: E731
    assert tl._chat_type(mk(megagroup=True)) == ChatType.SUPERGROUP and tl._chat_type(mk(broadcast=True)) == ChatType.CHANNEL


def test_throttle_kicks_in_after_20_calls_per_second(tl):
    import asyncio

    sleeps = []

    async def run():
        real = asyncio.sleep

        async def rec(s):
            sleeps.append(s)

        asyncio.sleep = rec  # type: ignore[assignment]
        try:
            for _ in range(21):
                await tl._throttle()
        finally:
            asyncio.sleep = real  # type: ignore[assignment]

    asyncio.run(run())
    assert sleeps == [1.0]  # "Telegram is strict about using 3rdparty clients"


# ---------------------------------------------------------------------------------------------------------------------
# graph viewer
# ---------------------------------------------------------------------------------------------------------------------
def graph_with_content() -> GraphMemory:
    g = GraphMemory()
    a, _ = g.upsert_entity("Alex </script>", "person", summary="owner")
    j, _ = g.upsert_entity("jazz", "topic")
    g.link(a.id, j.id, "likes", "Alex likes jazz", 0.3)
    ep = g.add_episode("a diary page <b>html</b>")
    g.link(ep.id, a.id, MENTIONS)
    return g


def test_graph_payload_and_html_are_self_contained_and_safe():
    g = graph_with_content()
    payload = graph_payload(g)
    assert payload["stats"]["nodes"] == 3 and len(payload["edges"]) == 2
    alex = next(n for n in payload["nodes"] if n["type"] == "person")
    assert alex["facts"] == ["Alex </script> —[likes]→ jazz: Alex likes jazz (confidence 0.30, seen 1x)"] and alex["pages"] == ["a diary page <b>html</b>"]
    html = render_html(g)
    data = re.search(r"const D=(.*?);const COL", html, re.S).group(1)
    assert json.loads(data.replace("<\\/", "</"))["stats"]["nodes"] == 3
    assert html.count("</script>") == 1  # the entity name cannot break out of the script tag
    assert "http://" not in html and "https://" not in html  # no CDN: works offline


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------
def test_init_command_writes_a_private_config_with_the_given_credentials(tmp_path, capsys):
    assert cli.main(["init", "--api-key", "sk-test", "--base-url", "https://x.test/v1", "--model", "m1", "--reasoning-effort", "high"]) == 0
    p = tmp_path / "config.toml"
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    cfg = cfgmod.load(p)
    assert (cfg.llm.endpoint.bearerKey, cfg.llm.endpoint.baseUrl, cfg.llm.model, cfg.llm.reasoningEffort) == ("sk-test", "https://x.test/v1", "m1", "high")
    assert cli.main(["init"]) == 1 and "already exists" in capsys.readouterr().out  # never silently overwrites secrets
    assert cli.main(["init", "--force", "--api-key", "sk-2"]) == 0 and cfgmod.load(p).llm.endpoint.bearerKey == "sk-2"


def test_graph_commands(tmp_path, capsys):
    g = GraphMemory(tmp_path / "data" / "memory" / "graph.sqlite")
    a, _ = g.upsert_entity("Alex2772", "person", aliases=["Алекс"], summary="owner")
    j, _ = g.upsert_entity("jazz", "topic")
    g.link(a.id, j.id, "likes", "Alex likes jazz", 0.3)
    ep = g.add_episode("Алекс любит джаз и играет на пианино.")
    g.link(ep.id, a.id, MENTIONS)
    g.close()

    assert cli.main(["graph", "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == 3
    assert cli.main(["graph", "query", "что", "любит", "Алекса?"]) == 0
    out = capsys.readouterr().out
    assert "likes" in out and "<memory_piece" in out
    assert cli.main(["graph", "show", "Алекс"]) == 0 and "Alex likes jazz" in capsys.readouterr().out
    assert cli.main(["graph", "export", "--html", "g.html", "--json", "g.json", "--md", "pages"]) == 0
    assert (tmp_path / "g.html").read_text().startswith("<!doctype html>")
    assert len(json.loads((tmp_path / "g.json").read_text())["nodes"]) == 3
    assert [p.read_text() for p in (tmp_path / "pages").glob("*.md")] == ["Алекс любит джаз и играет на пианино.\n"]


def test_the_original_debug_flag_opens_the_graph_viewer(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    assert cli.main(["--debug"]) == 0
    assert opened and opened[0].startswith("file://") and opened[0].endswith(".html")


def test_parser_covers_every_command():
    p = cli.build_parser()
    for argv in (["run"], ["chat"], ["sleep", "--max-minutes", "5"], ["graph", "view"], ["init"], []):
        p.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------------
# login: tell the user where the code went (Telethon's start() doesn't)
# ---------------------------------------------------------------------------------------------------------------------
from telethon import errors  # noqa: E402
from telethon.tl.types import auth as tl_auth  # noqa: E402


class LoginTL:
    """A stand-in for telethon.TelegramClient that only knows about signing in."""

    def __init__(self, sent_type, next_type=None, send_error=None, sign_in_errors=()):
        self.sent_type, self.next_type, self.send_error = sent_type, next_type, send_error
        self.sign_in_errors = list(sign_in_errors)
        self.sent_to, self.signed_in, self.resent = [], [], 0
        self.connected = False

    async def connect(self):
        self.connected = True

    async def is_user_authorized(self):
        return False

    async def send_code_request(self, phone):
        if self.send_error:
            raise self.send_error
        self.sent_to.append(phone)
        return tl_auth.SentCode(type=self.sent_type, phone_code_hash="hash1", next_type=self.next_type)

    async def sign_in(self, phone=None, code=None, password=None, phone_code_hash=None):
        self.signed_in.append((phone, code, password, phone_code_hash))
        if self.sign_in_errors:
            raise self.sign_in_errors.pop(0)

    async def __call__(self, request):  # auth.ResendCodeRequest
        self.resent += 1
        return tl_auth.SentCode(type=tl_auth.SentCodeTypeSms(length=5), phone_code_hash="hash2")


def scripted(tl, *answers):
    answers = list(answers)
    prompts = []

    async def ask(prompt):
        prompts.append(prompt)
        return answers.pop(0)

    tl._ask = ask
    return prompts


async def test_login_tells_the_user_the_code_went_to_the_app_not_sms(tl, capsys):
    tl.client = LoginTL(tl_auth.SentCodeTypeApp(length=5), next_type=tl_auth.CodeTypeSms())
    prompts = scripted(tl, "+7 (999) 123-45-67", "12345")
    await tl._login()
    out = capsys.readouterr().out
    assert "Telegram APP on another device" in out and "NOT an SMS" in out
    assert tl.client.sent_to == ["79991234567"]  # + ( ) - and spaces are stripped
    assert tl.client.signed_in == [("79991234567", "12345", None, "hash1")]
    assert "'r' = resend" in prompts[1]  # another method is available


async def test_login_resend_switches_to_the_next_delivery_method(tl, capsys):
    tl.client = LoginTL(tl_auth.SentCodeTypeApp(length=5), next_type=tl_auth.CodeTypeSms())
    scripted(tl, "79991234567", "r", "54 321")
    await tl._login()
    out = capsys.readouterr().out
    assert tl.client.resent == 1 and "via an SMS" in out
    assert tl.client.signed_in[-1][3] == "hash2" and tl.client.signed_in[-1][1] == "54321"  # the NEW hash, digits only


async def test_login_resend_without_alternatives_does_not_crash(tl, capsys):
    tl.client = LoginTL(tl_auth.SentCodeTypeSms(length=5))
    scripted(tl, "79991234567", "r", "11111")
    await tl._login()
    assert tl.client.resent == 0 and "no other delivery method" in capsys.readouterr().out


async def test_login_wrong_code_then_right_code_and_2fa(tl, capsys):
    tl.client = LoginTL(tl_auth.SentCodeTypeSms(length=5), sign_in_errors=[
        errors.PhoneCodeInvalidError(None), errors.SessionPasswordNeededError(None), errors.PasswordHashInvalidError(None)])
    scripted(tl, "79991234567", "00000", "11111", "wrong-pw", "right-pw")
    await tl._login()
    out = capsys.readouterr().out
    assert "wrong code, try again" in out and "wrong password, try again" in out
    assert [c[2] for c in tl.client.signed_in if c[2]] == ["wrong-pw", "right-pw"]


@pytest.mark.parametrize("error,fragment", [
    (errors.PhoneNumberInvalidError(None), "international format"),
    (errors.ApiIdInvalidError(None), "my.telegram.org"),
    (errors.PhoneNumberBannedError(None), "banned"),
    (errors.FloodWaitError(None, capture=3600), "wait 3600 s"),
])
async def test_login_explains_typical_failures(tl, error, fragment):
    from kuni.telegram.types import TelegramError

    tl.client = LoginTL(tl_auth.SentCodeTypeSms(length=5), send_error=error)
    scripted(tl, "79991234567")
    with pytest.raises(TelegramError, match=fragment):
        await tl._login()


def test_every_delivery_type_has_a_human_description():
    for name in ("App", "Sms", "Call", "FlashCall", "MissedCall", "EmailCode", "FragmentSms", "FirebaseSms"):
        assert hasattr(tl_auth, f"SentCodeType{name}")  # the names we describe really exist in the installed layer
        fake = type(f"SentCodeType{name}", (), {})()
        assert TelethonClient.describe_delivery(fake) != f"SentCodeType{name}"
    assert TelethonClient.describe_delivery(None) == "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# QR login: the way in when the code never shows up
# ---------------------------------------------------------------------------------------------------------------------
class FakeQR:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # "ok" | "timeout" | "2fa"
        self.url = "tg://login?token=AAAA"
        self.recreated = 0

    async def wait(self, timeout=None):
        outcome = self.outcomes.pop(0)
        if outcome == "timeout":
            import asyncio

            raise asyncio.TimeoutError
        if outcome == "2fa":
            raise errors.SessionPasswordNeededError(None)
        return object()

    async def recreate(self):
        self.recreated += 1
        self.url = f"tg://login?token=BBBB{self.recreated}"


class QRClient(LoginTL):
    def __init__(self, qr, **kw):
        super().__init__(tl_auth.SentCodeTypeApp(length=5), **kw)
        self.qr = qr

    async def qr_login(self):
        return self.qr


async def test_qr_login_draws_a_scannable_code_and_refreshes_it_when_it_expires(tl, capsys):
    qr = FakeQR(["timeout", "timeout", "ok"])
    tl.client = QRClient(qr)
    scripted(tl)
    await tl._login_qr()
    out = capsys.readouterr().out
    assert qr.recreated == 2 and out.count("Link Desktop Device") == 3  # a fresh QR code each time
    assert "█" in out or "▀" in out or "▄" in out  # an actual picture, not just the link
    assert "tg://login?token=AAAA" in out and "tg://login?token=BBBB2" in out


async def test_qr_login_with_two_factor_password(tl, capsys):
    tl.client = QRClient(FakeQR(["2fa"]), sign_in_errors=[errors.PasswordHashInvalidError(None)])
    scripted(tl, "wrong", "right")
    await tl._login_qr()
    assert [c[2] for c in tl.client.signed_in] == ["wrong", "right"]
    assert "wrong password" in capsys.readouterr().out


async def test_typing_qr_at_the_code_prompt_switches_to_qr_login(tl):
    tl.client = QRClient(FakeQR(["ok"]))
    scripted(tl, "79991234567", "qr")
    await tl._login()
    assert tl.client.sent_to == ["79991234567"] and tl.client.signed_in == []  # no code was ever typed


async def test_start_can_prefer_qr_and_skips_login_when_the_session_exists(tl):
    class Authorized(QRClient):
        async def is_user_authorized(self):
            return True

        async def get_me(self):
            return types.User(id=777, first_name="Kuni")

        async def get_dialogs(self, limit=None):
            return []

        def add_event_handler(self, *a, **k):
            pass

    tl.client = Authorized(FakeQR([]))
    scripted(tl)  # any question would crash the test: an existing session must not ask anything
    await tl.start()
    assert tl.my_id == 777 and tl.client.connected

    tl.client = QRClient(FakeQR(["ok"]))
    tl.client.get_me = Authorized.get_me.__get__(tl.client)
    tl.client.get_dialogs = Authorized.get_dialogs.__get__(tl.client)
    tl.client.add_event_handler = lambda *a, **k: None
    scripted(tl)
    await tl.start(prefer_qr=True)  # no phone number needed at all
    assert tl.client.sent_to == []


def test_login_command_is_registered():
    args = cli.build_parser().parse_args(["login", "--qr"])
    assert args.cmd == "login" and args.qr is True


async def test_contacts_are_refreshed_so_new_contacts_get_in_without_a_restart(tl, monkeypatch):
    from kuni.telegram import telethon_client as mod

    answers = [[1, 2], [1, 2, 3]]
    calls = []

    class Res:
        def __init__(self, ids):
            self.contacts = [types.Contact(user_id=i, mutual=True) for i in ids]

    async def fake_call(aw):
        calls.append(1)
        return Res(answers[len(calls) - 1])

    async def fake_client(request):
        return None

    tl._call = fake_call
    tl.client = lambda request: None  # the request object is irrelevant for this test
    now = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: now[0])
    assert await tl.get_contact_ids() == [1, 2] and len(calls) == 1
    now[0] += 60
    assert await tl.get_contact_ids() == [1, 2] and len(calls) == 1  # cached for a few minutes
    now[0] += mod.CONTACTS_TTL
    assert await tl.get_contact_ids() == [1, 2, 3] and len(calls) == 2  # then refreshed
