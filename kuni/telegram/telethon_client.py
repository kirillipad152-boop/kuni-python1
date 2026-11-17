"""Telethon (MTProto user client) implementation of TelegramClient.

The C++ kernel talks to Telegram through tdlib. tdlib has no maintained pure-Python binding, so the port uses Telethon:
the same user account / api_id / api_hash, no native library to build. Behaviour that tdlib did implicitly (the
online status, the read marks, chat actions) is done explicitly here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from telethon import TelegramClient as TL
from telethon import errors, events, functions, types, utils

from ..config import config
from .base import TelegramClient
from .types import (Chat, ChatType, Forward, MemberStatus, Message, NotFoundError, Reaction, TelegramError, User)

log = logging.getLogger("TelegramClient")

SESSION_DIR = Path("tg_session")
MEDIA_DIR = Path("cache") / "telegram_media"
CHAT_CACHE_TTL = 3.0  # seconds; unread counters must be fresh when a chat is opened
REACTIONS_CACHE_TTL = 600.0
CONTACTS_TTL = 300.0

_SERVICE_TEXT = {
    "MessageActionChatJoinedByLink": "[joined via link]",
    "MessageActionChatJoinedByRequest": "[joined by request]",
    "MessageActionChatEditPhoto": "[chat photo changed]",
    "MessageActionChatDeletePhoto": "[chat photo deleted]",
    "MessageActionScreenshotTaken": "[screenshot taken]",
    "MessageActionSetChatTheme": "[chat theme set] ",
    "MessageActionSetChatWallPaper": "[chat background set]",
    "MessageActionContactSignUp": "[contact registered]",
    "MessageActionChatMigrateTo": "[upgraded to supergroup]",
    "MessageActionChannelMigrateFrom": "[upgraded from group]",
    "MessageActionSetMessagesTTL": "[auto-delete time set]",
    "MessageActionBoostApply": "[chat boost]",
    "MessageActionTopicCreate": "[forum topic created]",
    "MessageActionTopicEdit": "[forum topic edited]",
    "MessageActionGroupCall": "[group call]",
    "MessageActionGroupCallScheduled": "[video chat scheduled]",
    "MessageActionInviteToGroupCall": "[invited to video chat]",
    "MessageActionGiftCode": "[premium gift code]",
    "MessageActionGiveawayLaunch": "[giveaway created]",
    "MessageActionPaymentRefunded": "[payment refunded]",
    "MessageActionBotAllowed": "[bot write access allowed]",
    "MessageActionWebViewDataSent": "[web app data sent]",
    "MessageActionWebViewDataSentMe": "[web app data received]",
    "MessageActionSecureValuesSent": "[passport data sent]",
    "MessageActionSecureValuesSentMe": "[passport data received]",
    "MessageActionSuggestProfilePhoto": "[suggest profile photo]",
    "MessageActionSuggestBirthday": "[suggest birthdate]",
    "MessageActionStarGiftUnique": "[upgraded gift]",
    "MessageActionHistoryClear": "[history cleared]",
}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name.replace("MessageAction", "")).lower()


class TelethonClient(TelegramClient):
    def __init__(self) -> None:
        super().__init__()
        c = config()
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = TL(str(SESSION_DIR / "kuni"), c.telegramApiId, c.telegramApiHash,
                         device_model="Desktop", system_lang_code="en", app_version="0.0.2")
        self._my_id = 0
        self._connected = asyncio.Event()
        self._chat_cache: dict[int, tuple[float, Chat]] = {}
        self._reactions_cache: dict[int, tuple[float, list[str] | None]] = {}
        self._user_cache: dict[int, User] = {}
        self._stickers: dict[int, types.Document] = {}
        self._contacts: list[int] | None = None
        self._contacts_at = 0.0
        self._calls = 0
        self._window = 0.0

    # ----------------------------------------------------------------------------------------------------------------
    # plumbing
    # ----------------------------------------------------------------------------------------------------------------
    async def _throttle(self) -> None:
        # Telegram is strict about using 3rdparty clients. We have to ensure that we wouldn't trigger their
        # security, leading to a ban of the account.
        now = time.monotonic()
        if now - self._window >= 1.0:
            self._window, self._calls = now, 0
        self._calls += 1
        if self._calls > 20:
            log.info("Too many calls to Telegram! Throttling...")
            await asyncio.sleep(1.0)
            self._window, self._calls = time.monotonic(), 1

    async def _call(self, aw):
        await self._throttle()
        try:
            return await aw
        except errors.FrozenMethodInvalidError:
            # the userbot was reported and therefore suspended (banned). Crash immediately to prevent failed API calls.
            log.error("The userbot's account was suspended - can't operate")
            os._exit(1)
        except errors.RPCError as e:
            raise TelegramError(str(e)) from e
        except ValueError as e:
            if "find any entity" in str(e) or "Could not find" in str(e):
                raise NotFoundError(str(e)) from e
            raise

    async def _input(self, chat_id: int):
        try:
            return await self.client.get_input_entity(chat_id)
        except ValueError:
            await self._call(self.client.get_dialogs(limit=None))  # fill the entity cache
            try:
                return await self.client.get_input_entity(chat_id)
            except ValueError as e:
                raise NotFoundError(f"Chat {chat_id} not found") from e

    # ----------------------------------------------------------------------------------------------------------------
    # lifecycle
    # ----------------------------------------------------------------------------------------------------------------
    # --- login -------------------------------------------------------------------------------------------------------
    _DELIVERY = {
        "SentCodeTypeApp": "the Telegram APP on another device where you are already logged in (chat «Telegram», "
                           "NOT an SMS)",
        "SentCodeTypeSms": "an SMS",
        "SentCodeTypeCall": "a phone call (a robot reads the code)",
        "SentCodeTypeFlashCall": "a flash call (the code is the last digits of the calling number)",
        "SentCodeTypeMissedCall": "a missed call (the code is the last digits of the calling number)",
        "SentCodeTypeEmailCode": "your e-mail",
        "SentCodeTypeFragmentSms": "Fragment (anonymous number)",
        "SentCodeTypeFirebaseSms": "an SMS",
    }

    @classmethod
    def describe_delivery(cls, kind) -> str:
        return "unknown" if kind is None else cls._DELIVERY.get(type(kind).__name__, type(kind).__name__)

    @staticmethod
    async def _ask(prompt: str) -> str:
        return (await asyncio.get_running_loop().run_in_executor(None, input, prompt)).strip()

    async def _login(self) -> None:
        """Explicit sign-in: Telethon's `start()` never says WHERE the code went, which is the #1 login problem."""
        phone = re.sub(r"[+()\s-]", "", await self._ask(
            "[Authentication] phone number, international format with the country code (e.g. 79991234567): "))
        try:
            sent = await self.client.send_code_request(phone)
        except errors.PhoneNumberInvalidError as e:
            raise TelegramError("Telegram says the phone number is invalid. Use the international format with the "
                                "country code, e.g. 79991234567 (Russia: 7…, not 8…).") from e
        except errors.ApiIdInvalidError as e:
            raise TelegramError("Telegram rejected general.telegram_api_id / telegram_api_hash. Check them at "
                                "https://my.telegram.org/apps") from e
        except errors.PhoneNumberBannedError as e:
            raise TelegramError("This phone number is banned by Telegram.") from e
        except (errors.FloodWaitError, errors.PhoneNumberFloodError) as e:
            wait = getattr(e, "seconds", None)
            raise TelegramError("Too many code requests for this number" + (f": Telegram asks to wait {wait} s "
                                f"(~{wait // 60} min)" if wait else " (try again in a few hours)") + ".") from e
        print(f"[Authentication] Telegram sent the code via {self.describe_delivery(sent.type)}.", flush=True)
        while True:
            hint = " ('r' = resend via the next method)" if getattr(sent, "next_type", None) else ""
            code = await self._ask(f"[Authentication] verification code{hint} ('qr' = log in by scanning a QR code instead): ")
            if code.lower() == "qr":
                await self._login_qr()
                return
            if code.lower() == "r":
                if not getattr(sent, "next_type", None):
                    print("[Authentication] no other delivery method is available; wait for the timeout and retry.", flush=True)
                    continue
                try:
                    sent = await self.client(functions.auth.ResendCodeRequest(phone, sent.phone_code_hash))
                except errors.RPCError as e:
                    print(f"[Authentication] can't resend: {e}", flush=True)
                    continue
                print(f"[Authentication] Telegram sent the code via {self.describe_delivery(sent.type)}.", flush=True)
                continue
            try:
                await self.client.sign_in(phone, re.sub(r"\D", "", code), phone_code_hash=sent.phone_code_hash)
                return
            except errors.SessionPasswordNeededError:
                while True:
                    try:
                        await self.client.sign_in(password=await self._ask("[Authentication] cloud (2FA) password: "))
                        return
                    except errors.PasswordHashInvalidError:
                        print("[Authentication] wrong password, try again.", flush=True)
            except errors.PhoneCodeInvalidError:
                print("[Authentication] wrong code, try again.", flush=True)
            except errors.PhoneCodeExpiredError:
                print("[Authentication] the code has expired: type 'r' to get a new one.", flush=True)
                sent = await self.client(functions.auth.ResendCodeRequest(phone, sent.phone_code_hash))
                print(f"[Authentication] Telegram sent a new code via {self.describe_delivery(sent.type)}.", flush=True)

    async def _login_qr(self) -> None:
        """Log in WITHOUT a code: scan a QR code with the Telegram app (Settings -> Devices -> Link Desktop Device).
        Works whenever the code doesn't arrive, e.g. when the "Telegram" service chat can't be seen."""
        import qrcode

        qr = await self.client.qr_login()
        while True:
            print("\n[Authentication] On your phone open Telegram -> Settings -> Devices -> Link Desktop Device and scan "
                  "this QR code (it refreshes every ~30 s):\n", flush=True)
            code = qrcode.QRCode(border=1)
            code.add_data(qr.url)
            code.make(fit=True)
            code.print_ascii(invert=True)
            print(f"\n(if the picture is garbled, enlarge the terminal or open this link on the phone: {qr.url})", flush=True)
            try:
                await qr.wait(timeout=30)
                return
            except asyncio.TimeoutError:
                await qr.recreate()  # the token expired: show a fresh QR code
            except errors.SessionPasswordNeededError:
                while True:
                    try:
                        await self.client.sign_in(password=await self._ask("[Authentication] cloud (2FA) password: "))
                        return
                    except errors.PasswordHashInvalidError:
                        print("[Authentication] wrong password, try again.", flush=True)

    async def start(self, prefer_qr: bool = False) -> None:
        log.info("[Authentication] connecting...")
        await self.client.connect()
        if not await self.client.is_user_authorized():
            await (self._login_qr() if prefer_qr else self._login())
        me = await self.client.get_me()
        self._my_id = me.id
        await self._call(self.client.get_dialogs(limit=200))  # warm the entity cache
        self.client.add_event_handler(self._on_new_message, events.NewMessage(incoming=True))
        log.info("[Authentication] logged in as %s (id=%d)", utils.get_display_name(me), me.id)
        self._connected.set()
        for h in self.logged_in_handlers:
            h()

    async def stop(self) -> None:
        await self.client.disconnect()

    async def wait_for_connection(self) -> None:
        await self._connected.wait()

    @property
    def my_id(self) -> int:
        return self._my_id

    async def _on_new_message(self, event) -> None:
        try:
            msg = self.convert(event.message)
            self._chat_cache.pop(msg.chat_id, None)  # unread counters changed
            for h in list(self.new_message_handlers):
                await h(msg)
        except Exception:  # noqa: BLE001
            log.exception("failed to handle an incoming message")

    # ----------------------------------------------------------------------------------------------------------------
    # conversion (Telethon objects -> kuni.telegram.types)
    # ----------------------------------------------------------------------------------------------------------------
    def convert(self, m) -> Message:
        chat_id = utils.get_peer_id(m.peer_id)
        if getattr(m, "from_id", None) is not None:
            sender_id = utils.get_peer_id(m.from_id)
            sender_is_chat = not isinstance(m.from_id, types.PeerUser)
        elif m.out:
            sender_id, sender_is_chat = self._my_id, False
        else:
            sender_id = chat_id
            sender_is_chat = not isinstance(m.peer_id, types.PeerUser)
        date = m.date
        if isinstance(date, datetime):
            date = int(date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp())

        out = Message(id=m.id, chat_id=chat_id, date=int(date or 0), sender_id=sender_id, sender_is_chat=sender_is_chat,
                      text=getattr(m, "message", None) or "", is_outgoing=bool(m.out), raw=m,
                      author_signature=getattr(m, "post_author", None) or "",
                      sender_tag=getattr(m, "from_rank", None) or "")
        reply = getattr(m, "reply_to", None)
        if reply is not None and getattr(reply, "reply_to_msg_id", None):
            out.reply_to_id = reply.reply_to_msg_id
        fwd = getattr(m, "fwd_from", None)
        if fwd is not None:
            origin = 0
            if getattr(fwd, "from_id", None) is not None:
                origin = utils.get_peer_id(fwd.from_id)
            out.forward = Forward(origin_id=origin, origin_name=getattr(fwd, "from_name", "") or "")
        if getattr(m, "reactions", None) is not None:
            recent: dict[str, list[int]] = {}
            for r in getattr(m.reactions, "recent_reactions", None) or []:
                if isinstance(r.reaction, types.ReactionEmoji):
                    recent.setdefault(r.reaction.emoticon, []).append(utils.get_peer_id(r.peer_id))
            for rc in m.reactions.results or []:
                if isinstance(rc.reaction, types.ReactionEmoji):  # custom / paid reactions: don't care
                    out.reactions.append(Reaction(rc.reaction.emoticon, rc.count, recent.get(rc.reaction.emoticon, [])))
        self._fill_kind(m, out)
        return out

    def _fill_kind(self, m, out: Message) -> None:
        if getattr(m, "action", None) is not None and not isinstance(m.action, types.MessageActionEmpty):
            out.kind = "service"
            out.text = self._service_text(m.action)
            if isinstance(m.action, types.MessageActionPhoneCall):
                out.kind = "call"
                out.extra = {"is_video": bool(m.action.video), "duration": m.action.duration or 0}
            elif isinstance(m.action, types.MessageActionGiftStars):
                out.kind = "gifted_stars"
                out.extra = {"count": m.action.stars}
            return
        media = getattr(m, "media", None)
        if media is None or isinstance(media, types.MessageMediaEmpty):
            return
        if isinstance(media, types.MessageMediaPhoto):
            out.kind = "photo"
        elif isinstance(media, types.MessageMediaDocument) and media.document is not None:
            self._fill_document(media, media.document, out)
        elif isinstance(media, (types.MessageMediaGeo, types.MessageMediaGeoLive)):
            out.kind, out.extra = "location", {"lat": media.geo.lat, "lon": media.geo.long}
        elif isinstance(media, types.MessageMediaVenue):
            out.kind, out.extra = "venue", {"title": media.title, "address": media.address}
        elif isinstance(media, types.MessageMediaContact):
            out.kind = "contact"
            out.extra = {"first_name": media.first_name, "last_name": media.last_name, "phone": media.phone_number}
        elif isinstance(media, types.MessageMediaPoll):
            q = media.poll.question
            out.kind = "poll"
            out.extra = {"question": getattr(q, "text", q), "options": [getattr(a.text, "text", a.text) for a in media.poll.answers]}
        elif isinstance(media, types.MessageMediaDice):
            out.kind, out.extra = "dice", {"emoji": media.emoticon, "value": media.value}
        elif isinstance(media, types.MessageMediaGame):
            out.kind, out.extra = "game", {"title": media.game.title, "description": media.game.description}
        elif isinstance(media, types.MessageMediaInvoice):
            out.kind = "invoice"
        elif isinstance(media, types.MessageMediaWebPage):
            wp = media.webpage
            if isinstance(wp, types.WebPage):
                out.link_preview = {"url": wp.url or "", "site_name": wp.site_name or "", "title": wp.title or "",
                                    "author": wp.author or "", "description": wp.description or ""}
        elif isinstance(media, types.MessageMediaGiveaway):
            out.kind = "giveaway"
        elif isinstance(media, types.MessageMediaGiveawayResults):
            out.kind, out.extra = "giveaway_winners", {"count": media.winners_count, "unclaimed": media.unclaimed_count or 0}
        elif isinstance(media, types.MessageMediaPaidMedia):
            out.kind = "paid_media"
        elif isinstance(media, types.MessageMediaStory):
            out.kind = "story"
        elif isinstance(media, types.MessageMediaUnsupported):
            out.kind = "unsupported"
        else:
            out.kind = "unsupported"

    def _fill_document(self, media, doc: types.Document, out: Message) -> None:
        attrs = {type(a): a for a in doc.attributes}
        if types.DocumentAttributeSticker in attrs:
            a = attrs[types.DocumentAttributeSticker]
            self._stickers[doc.id] = doc
            out.kind, out.extra = "sticker", {"sticker_id": doc.id, "emoji": a.alt or ""}
        elif types.DocumentAttributeAudio in attrs and attrs[types.DocumentAttributeAudio].voice:
            out.kind = "voice"
        elif types.DocumentAttributeVideo in attrs and attrs[types.DocumentAttributeVideo].round_message:
            out.kind = "video_note"
        elif types.DocumentAttributeAnimated in attrs:
            out.kind = "animation"
        elif types.DocumentAttributeVideo in attrs:
            out.kind = "video"
        elif types.DocumentAttributeAudio in attrs:
            out.kind, out.extra = "audio", {"title": attrs[types.DocumentAttributeAudio].title or ""}
        else:
            fn = attrs.get(types.DocumentAttributeFilename)
            out.kind, out.extra = "document", {"file_name": fn.file_name if fn else ""}

    @staticmethod
    def _service_text(action) -> str:
        name = type(action).__name__
        if name == "MessageActionChatAddUser":
            return f"[members added] {len(action.users)} member(s)"
        if name == "MessageActionChatDeleteUser":
            return f"[member removed] user_id={action.user_id}"
        if name == "MessageActionChatCreate":
            return f"[group created] {action.title}"
        if name == "MessageActionChannelCreate":
            return f"[supergroup created] {action.title}"
        if name == "MessageActionChatEditTitle":
            return f"[title changed] {action.title}"
        if name == "MessageActionPinMessage":
            return "[message pinned]"
        if name in _SERVICE_TEXT:
            return _SERVICE_TEXT[name]
        return f"[{_snake(name)}]"

    def _chat_type(self, entity) -> ChatType:
        if isinstance(entity, (types.User, types.UserEmpty)):
            return ChatType.PRIVATE
        if isinstance(entity, (types.Chat, types.ChatForbidden, types.ChatEmpty)):
            return ChatType.BASIC_GROUP
        if isinstance(entity, (types.Channel, types.ChannelForbidden)):
            return ChatType.CHANNEL if not getattr(entity, "megagroup", False) else ChatType.SUPERGROUP
        return ChatType.PRIVATE

    # ----------------------------------------------------------------------------------------------------------------
    # reading
    # ----------------------------------------------------------------------------------------------------------------
    async def _fetch_chat(self, chat_id: int) -> Chat:
        peer = await self._input(chat_id)
        res = await self._call(self.client(functions.messages.GetPeerDialogsRequest([types.InputDialogPeer(peer)])))
        entity = await self._call(self.client.get_entity(peer))
        title = utils.get_display_name(entity) or str(chat_id)
        chat = Chat(id=chat_id, title=title, type=self._chat_type(entity), raw=entity,
                    has_photo=bool(getattr(entity, "photo", None)) and not isinstance(
                        entity.photo, (types.ChatPhotoEmpty, types.UserProfilePhotoEmpty)))
        if res.dialogs:
            d = res.dialogs[0]
            chat.unread_count = d.unread_count
            chat.last_read_inbox_message_id = d.read_inbox_max_id
            chat.pinned = bool(d.pinned)
            mu = getattr(d.notify_settings, "mute_until", None)
            chat.muted = bool(mu and (mu.timestamp() if isinstance(mu, datetime) else mu) > time.time())
            top = next((m for m in res.messages if m.id == d.top_message), None)
            if top is not None:
                chat.last_message = self.convert(top)
        if chat.type.is_group or chat.type == ChatType.CHANNEL:
            chat.available_reactions = await self._available_reactions(chat_id, entity)
        return chat

    async def _available_reactions(self, chat_id: int, entity) -> list[str] | None:
        cached = self._reactions_cache.get(chat_id)
        if cached and time.monotonic() - cached[0] < REACTIONS_CACHE_TTL:
            return cached[1]
        result: list[str] | None = None
        try:
            if isinstance(entity, types.Channel):
                full = (await self._call(self.client(functions.channels.GetFullChannelRequest(entity)))).full_chat
            else:
                full = (await self._call(self.client(functions.messages.GetFullChatRequest(entity.id)))).full_chat
            ar = getattr(full, "available_reactions", None)
            if isinstance(ar, types.ChatReactionsNone):
                result = []
            elif isinstance(ar, types.ChatReactionsSome):
                result = [r.emoticon for r in ar.reactions if isinstance(r, types.ReactionEmoji)]
        except TelegramError:
            result = None
        self._reactions_cache[chat_id] = (time.monotonic(), result)
        return result

    async def get_chat(self, chat_id: int) -> Chat:
        cached = self._chat_cache.get(chat_id)
        if cached and time.monotonic() - cached[0] < CHAT_CACHE_TTL:
            return cached[1]
        chat = await self._fetch_chat(chat_id)
        self._chat_cache[chat_id] = (time.monotonic(), chat)
        return chat

    async def get_user(self, user_id: int) -> User:
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        e = await self._call(self.client.get_entity(user_id))
        if not isinstance(e, types.User):
            raise NotFoundError(f"{user_id} is not a user")
        names = [u.username for u in (e.usernames or []) if u.active] or ([e.username] if e.username else [])
        user = User(e.id, e.first_name or "", e.last_name or "", names, bool(e.premium))
        self._user_cache[user_id] = user
        return user

    async def get_message(self, chat_id: int, message_id: int) -> Message:
        peer = await self._input(chat_id)
        m = await self._call(self.client.get_messages(peer, ids=message_id))
        if m is None:
            raise NotFoundError("Not Found")
        return self.convert(m)

    async def get_chats(self, limit: int = 50) -> list[Chat]:
        dialogs = await self._call(self.client.get_dialogs(limit=limit))
        chats = []
        for d in dialogs:
            if d.archived:
                continue
            chat_id = utils.get_peer_id(d.entity)
            chat = Chat(id=chat_id, title=d.name or str(chat_id), type=self._chat_type(d.entity),
                        unread_count=d.unread_count, pinned=bool(d.pinned), raw=d.entity,
                        last_read_inbox_message_id=d.dialog.read_inbox_max_id,
                        has_photo=bool(getattr(d.entity, "photo", None)))
            mu = getattr(d.dialog.notify_settings, "mute_until", None)
            chat.muted = bool(mu and (mu.timestamp() if isinstance(mu, datetime) else mu) > time.time())
            if d.message is not None:
                chat.last_message = self.convert(d.message)
            self._chat_cache[chat_id] = (time.monotonic(), chat)
            chats.append(chat)
        return chats

    async def get_chat_history(self, chat_id: int, from_message_id: int = 0, limit: int = 30) -> list[Message]:
        peer = await self._input(chat_id)
        msgs = await self._call(self.client.get_messages(peer, limit=limit, offset_id=from_message_id))
        return [self.convert(m) for m in msgs]

    async def get_messages_around(self, chat_id: int, message_id: int, before: int, after: int) -> list[Message]:
        peer = await self._input(chat_id)
        target = await self._call(self.client.get_messages(peer, ids=message_id))
        if target is None:
            return []
        newer = await self._call(self.client.get_messages(peer, limit=after, min_id=message_id, reverse=True)) if after else []
        older = await self._call(self.client.get_messages(peer, limit=before, offset_id=message_id)) if before else []
        return [self.convert(m) for m in [*reversed(list(newer)), target, *older]]

    async def search_chats_on_server(self, query: str, limit: int = 50) -> list[Chat]:
        res = await self._call(self.client(functions.contacts.SearchRequest(q=query, limit=limit)))
        out = []
        for peer in res.my_results:
            try:
                out.append(await self.get_chat(utils.get_peer_id(peer)))
            except TelegramError:
                continue
        return out

    async def search_public_chat(self, username: str) -> Chat | None:
        try:
            entity = await self._call(self.client.get_entity(username))
        except (TelegramError, NotFoundError):  # also USERNAME_NOT_OCCUPIED (RPC errors are wrapped by _call)
            return None
        return await self.get_chat(utils.get_peer_id(entity))

    async def search_chat_messages(self, chat_id: int, query: str, sender_id: int | None, limit: int
                                   ) -> tuple[int, list[Message]]:
        peer = await self._input(chat_id)
        kwargs = {"from_user": await self._input(sender_id)} if sender_id else {}
        msgs = await self._call(self.client.get_messages(peer, search=query, limit=limit, **kwargs))
        return int(getattr(msgs, "total", len(msgs))), [self.convert(m) for m in msgs]

    async def search_messages_global(self, query: str, limit: int, min_date: int, max_date: int
                                     ) -> tuple[int, list[Message]]:
        def dt(ts: int):
            return datetime.fromtimestamp(ts, timezone.utc) if ts else None

        res = await self._call(self.client(functions.messages.SearchGlobalRequest(
            q=query, filter=types.InputMessagesFilterEmpty(), min_date=dt(min_date), max_date=dt(max_date),
            offset_rate=0, offset_peer=types.InputPeerEmpty(), offset_id=0, limit=limit)))
        msgs = [self.convert(m) for m in res.messages if isinstance(m, types.Message)]
        return int(getattr(res, "count", len(msgs))), msgs

    async def get_contact_ids(self) -> list[int]:
        # refreshed every few minutes: someone you add to the contacts of this account is let in without a restart
        if self._contacts is None or time.monotonic() - self._contacts_at > CONTACTS_TTL:
            res = await self._call(self.client(functions.contacts.GetContactsRequest(hash=0)))
            self._contacts = [c.user_id for c in getattr(res, "contacts", [])]
            self._contacts_at = time.monotonic()
        return self._contacts

    async def get_member_status(self, chat_id: int, user_id: int) -> MemberStatus:
        peer = await self._input(chat_id)
        entity = await self._call(self.client.get_entity(peer))
        try:
            if isinstance(entity, types.Channel):
                p = (await self._call(self.client(functions.channels.GetParticipantRequest(
                    entity, await self._input(user_id))))).participant
                if isinstance(p, types.ChannelParticipantCreator):
                    return MemberStatus.CREATOR
                if isinstance(p, types.ChannelParticipantAdmin):
                    return MemberStatus.ADMIN
                if isinstance(p, types.ChannelParticipantBanned):
                    return MemberStatus.BANNED if p.banned_rights.view_messages else MemberStatus.MEMBER
                if isinstance(p, types.ChannelParticipantLeft):
                    return MemberStatus.LEFT
                return MemberStatus.MEMBER
            perms = await self._call(self.client.get_permissions(entity, await self._input(user_id)))
            if perms.is_creator:
                return MemberStatus.CREATOR
            return MemberStatus.ADMIN if perms.is_admin else MemberStatus.MEMBER
        except TelegramError as e:
            if "USER_NOT_PARTICIPANT" in str(e):
                return MemberStatus.LEFT
            raise

    async def download_media(self, message: Message, what: str = "media") -> Path:
        raw = message.raw
        if raw is None:
            raise TelegramError("message has no media")
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        kwargs = {"thumb": -1} if what == "thumbnail" else {}
        path = await self._call(self.client.download_media(raw, file=str(MEDIA_DIR) + "/", **kwargs))
        if not path:
            raise TelegramError("failed to download media")
        return Path(path)

    async def download_chat_photo(self, chat: Chat) -> Path:
        path = await self._call(self.client.download_profile_photo(chat.raw, file=str(MEDIA_DIR / f"chat_{chat.id}.jpg")))
        if not path:
            raise TelegramError("chat has no photo")
        return Path(path)

    # ----------------------------------------------------------------------------------------------------------------
    # writing
    # ----------------------------------------------------------------------------------------------------------------
    async def send_message(self, chat_id: int, text: str = "", photo_path: Path | None = None,
                           audio_path: Path | None = None, reply_to: int = 0) -> Message:
        peer = await self._input(chat_id)
        reply = reply_to or None
        if photo_path:
            m = await self._call(self.client.send_file(peer, str(photo_path), caption=text, reply_to=reply))
        elif audio_path:
            m = await self._call(self.client.send_file(peer, str(audio_path), voice_note=True, caption=text or None, reply_to=reply))
        else:
            m = await self._call(self.client.send_message(peer, text, reply_to=reply))
        return self.convert(m)

    async def send_sticker(self, chat_id: int, sticker_id: int, reply_to: int = 0) -> None:
        doc = self._stickers.get(sticker_id)
        if doc is None:
            await self.get_saved_stickers()
            doc = self._stickers.get(sticker_id)
        if doc is None:
            raise NotFoundError("sticker not found")
        await self._call(self.client.send_file(await self._input(chat_id), doc, reply_to=reply_to or None))

    async def get_saved_stickers(self) -> list[Message]:
        # a combination of favorite + recent stickers gives total storage for 30-40 stickers.
        fav = await self._call(self.client(functions.messages.GetFavedStickersRequest(hash=0)))
        rec = await self._call(self.client(functions.messages.GetRecentStickersRequest(hash=0)))
        out = []
        for doc in [*getattr(fav, "stickers", []), *getattr(rec, "stickers", [])]:
            alt = next((a.alt for a in doc.attributes if isinstance(a, types.DocumentAttributeSticker)), "")
            self._stickers[doc.id] = doc
            out.append(Message(id=0, chat_id=0, kind="sticker", extra={"sticker_id": doc.id, "emoji": alt or ""}, raw=doc))
        return out

    async def add_favorite_sticker(self, sticker_id: int) -> None:
        doc = self._stickers.get(sticker_id)
        if doc is None:
            raise NotFoundError("unknown sticker; you can only save stickers you have seen")
        await self._call(self.client(functions.messages.FaveStickerRequest(id=utils.get_input_document(doc), unfave=False)))

    async def react(self, chat_id: int, message_id: int, emoji: str) -> None:
        await self._call(self.client(functions.messages.SendReactionRequest(
            peer=await self._input(chat_id), msg_id=message_id, reaction=[types.ReactionEmoji(emoticon=emoji)])))

    async def edit_message(self, chat_id: int, message_id: int, text: str, is_caption: bool) -> None:
        # Telethon edits the text of a text message and the caption of a media message with the same call
        await self._call(self.client.edit_message(await self._input(chat_id), message_id, text))

    async def delete_messages(self, chat_id: int, message_ids: list[int]) -> None:
        await self._call(self.client.delete_messages(await self._input(chat_id), message_ids, revoke=True))

    async def forward_messages(self, to_chat_id: int, from_chat_id: int, message_ids: list[int]) -> None:
        await self._call(self.client.forward_messages(await self._input(to_chat_id), message_ids,
                                                      from_peer=await self._input(from_chat_id)))

    async def join_chat_by_invite_link(self, link: str) -> Chat:
        m = re.search(r"(?:joinchat/|\+)([\w-]+)", link)
        if m:
            res = await self._call(self.client(functions.messages.ImportChatInviteRequest(m.group(1))))
            chats = getattr(res, "chats", None) or []
            if not chats:
                raise TelegramError("joined, but the chat is not available yet")
            return await self.get_chat(utils.get_peer_id(chats[0]))
        username = link.rstrip("/").rsplit("/", 1)[-1].lstrip("@")
        entity = await self._call(self.client.get_entity(username))
        await self._call(self.client(functions.channels.JoinChannelRequest(entity)))
        return await self.get_chat(utils.get_peer_id(entity))

    async def join_chat(self, chat_id: int) -> None:
        await self._call(self.client(functions.channels.JoinChannelRequest(await self._input(chat_id))))
        self._chat_cache.pop(chat_id, None)

    async def leave_chat(self, chat_id: int) -> None:
        peer = await self._input(chat_id)
        entity = await self._call(self.client.get_entity(peer))
        if isinstance(entity, types.Channel):
            await self._call(self.client(functions.channels.LeaveChannelRequest(entity)))
        else:
            await self._call(self.client.delete_dialog(entity))
        self._chat_cache.pop(chat_id, None)

    async def ban_member(self, chat_id: int, user_id: int) -> None:
        await self._call(self.client.edit_permissions(await self._input(chat_id), await self._input(user_id),
                                                      view_messages=False))

    async def set_member_tag(self, chat_id: int, user_id: int, tag: str) -> None:
        await self._call(self.client(functions.messages.EditChatParticipantRankRequest(
            peer=await self._input(chat_id), participant=await self._input(user_id), rank=tag)))

    async def block_user(self, user_id: int) -> None:
        await self._call(self.client(functions.contacts.BlockRequest(await self._input(user_id))))

    async def delete_chat_history(self, chat_id: int) -> None:
        await self._call(self.client(functions.messages.DeleteHistoryRequest(
            peer=await self._input(chat_id), max_id=0, revoke=True)))
        self._chat_cache.pop(chat_id, None)

    # ----------------------------------------------------------------------------------------------------------------
    # presence
    # ----------------------------------------------------------------------------------------------------------------
    async def set_online(self, online: bool) -> None:
        await self._call(self.client(functions.account.UpdateStatusRequest(offline=not online)))

    async def open_chat(self, chat_id: int) -> None:
        pass  # MTProto has no "open chat" call: the online status + read marks + chat actions do the job

    async def close_chat(self, chat_id: int) -> None:
        pass

    async def view_messages(self, chat_id: int, message_ids: list[int]) -> None:
        if message_ids:
            await self._call(self.client.send_read_acknowledge(await self._input(chat_id), max_id=max(message_ids)))
            self._chat_cache.pop(chat_id, None)

    async def send_chat_action(self, chat_id: int, action: str | None) -> None:
        mapping = {
            "typing": types.SendMessageTypingAction,
            "choosing_sticker": types.SendMessageChooseStickerAction,
            "recording_voice": types.SendMessageRecordAudioAction,
            "uploading_photo": lambda: types.SendMessageUploadPhotoAction(progress=0),
            None: types.SendMessageCancelAction,
        }
        await self._call(self.client(functions.messages.SetTypingRequest(
            peer=await self._input(chat_id), action=mapping[action]())))

    # ----------------------------------------------------------------------------------------------------------------
    async def recognize_speech(self, message: Message) -> str | None:
        try:
            peer = await self._input(message.chat_id)
            for _ in range(20):
                res = await self._call(self.client(functions.messages.TranscribeAudioRequest(peer, message.id)))
                if not res.pending:
                    return res.text
                await asyncio.sleep(1.5)
        except (TelegramError, errors.PremiumAccountRequiredError):
            return None
        return None
