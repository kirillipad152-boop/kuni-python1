"""Telegram data model, independent of any MTProto library (the C++ kernel used tdlib's td_api objects)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ChatType(enum.Enum):
    PRIVATE = "private"
    SECRET = "secret"
    BASIC_GROUP = "basic_group"
    SUPERGROUP = "supergroup"  # megagroup
    CHANNEL = "channel"  # broadcast channel (tdlib: supergroup with is_channel)

    @property
    def is_group(self) -> bool:
        return self in (ChatType.BASIC_GROUP, ChatType.SUPERGROUP)


@dataclass
class User:
    id: int
    first_name: str = ""
    last_name: str = ""
    usernames: list[str] = field(default_factory=list)
    is_premium: bool = False


@dataclass
class Reaction:
    emoji: str
    total_count: int = 1
    recent_sender_ids: list[int] = field(default_factory=list)


@dataclass
class Forward:
    origin_id: int = 0  # user / chat / channel id of the original author
    origin_name: str = ""  # fallback when the author is hidden


@dataclass
class Message:
    """`kind` decides how the message is rendered for the LLM, see llmui.format.extract_message_type_and_text."""

    id: int
    chat_id: int
    date: int = 0  # unix seconds (UTC)
    sender_id: int = 0  # user id, or chat id for channel posts / anonymous admins
    sender_is_chat: bool = False
    kind: str = "text"
    text: str = ""  # text, or caption of media
    reply_to_id: int = 0
    forward: Forward | None = None
    reactions: list[Reaction] = field(default_factory=list)
    sender_tag: str = ""
    author_signature: str = ""
    is_outgoing: bool = False
    link_preview: dict | None = None  # url, site_name, title, author, description
    # kind-specific details: sticker(emoji, sticker_id), audio(title), document(file_name), location(lat, lon),
    # venue(title, address), contact(first_name, last_name, phone), poll(question, options), dice(emoji, value),
    # call(is_video, duration), service(text), gift(star_count, text, emoji), ...
    extra: dict[str, Any] = field(default_factory=dict)
    # opaque handle for the library-specific object to download media from
    raw: Any = None


@dataclass
class Chat:
    id: int
    title: str = ""
    type: ChatType = ChatType.PRIVATE
    unread_count: int = 0
    last_message: Message | None = None
    last_read_inbox_message_id: int = 0
    muted: bool = False
    pinned: bool = False
    has_photo: bool = False
    # None = every reaction is allowed; [] = reactions are disabled; otherwise the allowed emoji list
    available_reactions: list[str] | None = None
    raw: Any = None


class MemberStatus(enum.Enum):
    CREATOR = "creator"
    ADMIN = "admin"
    MEMBER = "member"
    LEFT = "left"
    BANNED = "banned"


class TelegramError(Exception):
    """RPC failure (the message is reported to the LLM)."""


class NotFoundError(TelegramError):
    pass
