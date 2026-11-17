"""Telegram -> text representation for the LLM. Port of llmui/telegram.cpp."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Sequence

from ..config import config
from ..lockdown import is_accessible_from_lockdown
from ..llm import Message as LlmMessage
from ..llm import OpenAIChat
from ..telegram.base import TelegramClient
from ..telegram.types import Chat, ChatType, Message, NotFoundError, TelegramError
from ..util import fmt_ts, time_ago
from . import media
from .media import check_for_malicious_payloads

log = logging.getLogger("llmui")


async def extract_sender_name(telegram: TelegramClient, sender_id: int) -> str:
    name = ""
    if sender_id == telegram.my_id:
        name = f"You ({config().characterName})"
    elif sender_id != 0:
        try:
            u = await telegram.get_user(sender_id)
            name = f"{u.first_name} {u.last_name}".strip()
            if u.usernames:
                name += f" (@{u.usernames[0]})"
        except TelegramError:
            pass
        if not name.strip():
            try:
                name = (await telegram.get_chat(sender_id)).title
            except TelegramError:
                name = ""
    return check_for_malicious_payloads(name)


def format_link_preview(p: dict) -> str:
    out = "<link_preview"
    if p.get("url"):
        out += f' url="{p["url"]}"'
    if p.get("site_name"):
        out += f' site="{p["site_name"]}"'
    if p.get("title"):
        out += f' title="{p["title"]}"'
    if p.get("author"):
        out += f' author="{p["author"]}"'
    if p.get("description"):
        out += f'>\n{p["description"]}\n</link_preview>'
    else:
        out += " />"
    return out + "\n"


def extract_message_type_and_text(msg: Message) -> str:
    """`[photo]`, `[voice message]`, ... + the text/caption."""
    k, x = msg.kind, msg.extra
    caption = check_for_malicious_payloads(msg.text) if msg.text else ""

    def cap() -> str:
        return ("\n" + caption) if caption else ""

    if k == "text":
        out = caption
        if msg.link_preview:
            out += "\n" + format_link_preview(msg.link_preview)
        return out
    simple = {
        "photo": "[photo]", "animation": "[animation]", "video": "[video]", "voice": "[voice message]",
    }
    if k in simple:
        return simple[k] + cap()
    if k == "audio":
        return "[audio] " + x.get("title", "") + cap()
    if k == "document":
        return "[document] " + (x.get("file_name") or "<unnamed>") + cap()
    if k == "video_note":
        return "[video note]"
    if k == "sticker":
        return "[sticker]"
    if k == "location":
        return f"[location] lat={x.get('lat')} lon={x.get('lon')}"
    if k == "venue":
        return f"[venue] {x.get('title', '')} — {x.get('address', '')}"
    if k == "contact":
        return f"[contact] {x.get('first_name', '')} {x.get('last_name', '')} ({x.get('phone', '')})"
    if k == "poll":
        return f"[poll] {x.get('question', '')}\n" + "".join(f"- {o}\n" for o in x.get("options", []))
    if k == "invoice":
        return "[invoice]"
    if k == "game":
        return f"[game] {x.get('title', '')} — {x.get('description', '')}"
    if k == "dice":
        return f"[dice] {x.get('emoji', '')} = {x.get('value', '')}"
    if k == "call":
        return f'<call is_video="{str(bool(x.get("is_video"))).lower()}" duration="{x.get("duration", 0)}" />'
    if k == "service":
        return msg.text
    if k == "gifted_stars":
        return f'<gifted_stars count="{x.get("count", 0)}" />'
    if k == "giveaway":
        return "[giveaway]"
    if k == "giveaway_winners":
        return f'<giveaway_winners count="{x.get("count", 0)}" unclaimed="{x.get("unclaimed", 0)}" additional_chats="0" />'
    if k == "paid_media":
        return "[paid media]"
    if k == "story":
        return "[story]"
    if k == "gift":
        return "[gift]"
    if k == "unsupported":
        return "[unsupported message]"
    return f"[{k}]" + cap()


async def format_chat_single(telegram: TelegramClient, chat: Chat) -> str:
    if not await is_accessible_from_lockdown(telegram, chat.id):
        return ""
    type_ = {ChatType.PRIVATE: "direct messages", ChatType.SECRET: "direct messages", ChatType.BASIC_GROUP: "group chat",
             ChatType.SUPERGROUP: "group chat", ChatType.CHANNEL: "channel"}.get(chat.type, "unknown")
    preview = ""
    if chat.last_message:
        preview = await extract_sender_name(telegram, chat.last_message.sender_id)
        preview += ": " + extract_message_type_and_text(chat.last_message)
        preview = preview.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:30] + "..." + preview[-30:]
    last_ago = ""
    if chat.last_message:
        last_ago = time_ago(datetime.fromtimestamp(chat.last_message.date, timezone.utc))
    result = f'<chat chat_id="{chat.id}" title="{chat.title}" preview="{preview}" type="{type_}" last_message="{last_ago}"'
    if chat.unread_count > 0:
        result += f' unread_count="{chat.unread_count}"'
    return result + " />\n"


async def format_chat_list(telegram: TelegramClient, chats: Sequence[Chat]) -> str:
    out = ""
    for chat in chats:
        out += await format_chat_single(telegram, chat)
    return out


async def format_chat_history_message(telegram: TelegramClient, msg: Message, chat: Chat, llm: OpenAIChat,
                                      temporary_context: Sequence[LlmMessage], xml_tag: str = "message") -> str:
    sender_name = await extract_sender_name(telegram, msg.sender_id)
    tag = f'{xml_tag} message_id="{msg.id}" date="{fmt_ts(msg.date)}"'
    if msg.sender_tag:
        tag += f' chat_tag="{msg.sender_tag}"'
    if msg.author_signature:
        tag += f' author_signature="{msg.author_signature}"'
    if msg.sender_id != telegram.my_id and msg.id > chat.last_read_inbox_message_id:
        tag += " unread"
    if msg.forward is not None:
        # explanation: from the perspective of telegram, `sender` is the one who shared a message to you.
        #   <message sender="John" forwarded_from="Fox News">btc is 100k$</message>
        # but LLM doesn't care about `forwarded_from` and responds as if the text was authored by John. So we swap:
        #   <message sender="Fox News" forwarded_by="John">btc is 100k$</message>
        origin = await extract_sender_name(telegram, msg.forward.origin_id) if msg.forward.origin_id else \
            check_for_malicious_payloads(msg.forward.origin_name)
        tag += f' sender="{origin}"'
        if sender_name:
            tag += f' forwarded_by="{sender_name}"'
    else:
        if sender_name:
            tag += f' sender="{sender_name}"'
        if chat.type.is_group:
            tag += f' sender_id="{msg.sender_id}"'  # for bans.
    if msg.reactions:
        attr = ""
        for r in msg.reactions:
            if not r.emoji:
                continue
            if attr:
                attr += ";"
            attr += f"({r.emoji} "
            if r.total_count > 3:
                # if reactions above 3, format as emoji + react counts, just like regular telegram clients do.
                attr += f"{r.total_count}"
            else:
                names = [await extract_sender_name(telegram, sid) for sid in r.recent_sender_ids]
                attr += "by " + ", ".join(names)
            attr += ")"
        if attr:
            tag += f' reactions="{attr}"'

    result = f"<{tag}>\n"
    if xml_tag != "reply_to":
        if msg.reply_to_id:
            try:
                reply = await telegram.get_message(msg.chat_id, msg.reply_to_id)
                result += await format_chat_history_message(telegram, reply, chat, llm, temporary_context, "reply_to")
            except NotFoundError:
                result += "<reply_to>Deleted Message</reply_to>"
            except TelegramError as e:
                if "Not Found" in str(e):
                    result += "<reply_to>Deleted Message</reply_to>"
                else:
                    log.error("formatChatHistoryMessage: %s", e)

        k = msg.kind
        if k == "photo":
            result += await media.image(temporary_context, llm, await telegram.download_media(msg))
        elif k == "sticker":
            st = "sticker"
            if msg.extra.get("emoji"):
                st += f' emoji="{check_for_malicious_payloads(msg.extra["emoji"])}"'
            if config().capabilityUseStickers:
                st += f' sticker_id="{msg.extra.get("sticker_id", 0)}"'
            result += await media.image(temporary_context, llm, await telegram.download_media(msg), st)
        elif k == "gift":
            st = f'gift cost="{msg.extra.get("star_count", 0)} stars"'
            if msg.extra.get("text"):
                st += f' text="{check_for_malicious_payloads(msg.extra["text"])}"'
            if msg.extra.get("emoji"):
                st += f' emoji="{check_for_malicious_payloads(msg.extra["emoji"])}"'
            if msg.raw is not None:
                result += await media.image(temporary_context, llm, await telegram.download_media(msg), st)
            else:
                result += f"<{st} />"
        elif k == "animation":
            result += await media.image(temporary_context, llm, await telegram.download_media(msg, "thumbnail"), "animation")
        elif k == "video":
            result += await media.video(temporary_context, llm, await telegram.download_media(msg), "video")
        elif k == "voice":
            result += await media.voice_message_transcription(telegram, msg, llm)
        elif k == "video_note":
            result += await media.video_note_transcription(telegram, llm, msg)

    result += extract_message_type_and_text(msg)
    result += f"\n</{tag}>\n"
    return result
