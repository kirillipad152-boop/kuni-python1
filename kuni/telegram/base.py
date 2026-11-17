"""Abstract Telegram client. Port of ITelegramClient (tdlib calls became intent-level methods)."""

from __future__ import annotations

import abc
import asyncio
from pathlib import Path
from typing import Awaitable, Callable

from .types import Chat, MemberStatus, Message, User

NewMessageHandler = Callable[[Message], Awaitable[None]]


class TelegramClient(abc.ABC):
    def __init__(self) -> None:
        self.logged_in_handlers: list[Callable[[], None]] = []
        self.new_message_handlers: list[NewMessageHandler] = []

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    @abc.abstractmethod
    async def start(self) -> None:
        """Connects and logs in (phone / code / 2FA password are read from stdin)."""

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def wait_for_connection(self) -> None: ...

    @property
    @abc.abstractmethod
    def my_id(self) -> int: ...

    # ---- reading -----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    async def get_chat(self, chat_id: int) -> Chat: ...

    @abc.abstractmethod
    async def get_user(self, user_id: int) -> User: ...

    @abc.abstractmethod
    async def get_message(self, chat_id: int, message_id: int) -> Message:
        """Raises NotFoundError if the message does not exist."""

    @abc.abstractmethod
    async def get_chats(self, limit: int = 50) -> list[Chat]:
        """Main chat list, newest first (as in the Telegram app)."""

    @abc.abstractmethod
    async def get_chat_history(self, chat_id: int, from_message_id: int = 0, limit: int = 30) -> list[Message]:
        """Messages older than `from_message_id` (0 = from the latest), newest first."""

    @abc.abstractmethod
    async def get_messages_around(self, chat_id: int, message_id: int, before: int, after: int) -> list[Message]:
        """[after newer, target, before older], newest first."""

    @abc.abstractmethod
    async def search_chats_on_server(self, query: str, limit: int = 50) -> list[Chat]: ...

    @abc.abstractmethod
    async def search_public_chat(self, username: str) -> Chat | None: ...

    @abc.abstractmethod
    async def search_chat_messages(self, chat_id: int, query: str, sender_id: int | None, limit: int
                                   ) -> tuple[int, list[Message]]:
        """Returns (total_count, messages)."""

    @abc.abstractmethod
    async def search_messages_global(self, query: str, limit: int, min_date: int, max_date: int
                                     ) -> tuple[int, list[Message]]: ...

    @abc.abstractmethod
    async def get_contact_ids(self) -> list[int]: ...

    @abc.abstractmethod
    async def get_member_status(self, chat_id: int, user_id: int) -> MemberStatus:
        """Raises TelegramError if the member list is not accessible."""

    @abc.abstractmethod
    async def download_media(self, message: Message, what: str = "media") -> Path:
        """Downloads media of `message` (what = media | thumbnail | voice) and returns the local path."""

    @abc.abstractmethod
    async def download_chat_photo(self, chat: Chat) -> Path: ...

    # ---- writing -----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    async def send_message(self, chat_id: int, text: str = "", photo_path: Path | None = None,
                           audio_path: Path | None = None, reply_to: int = 0) -> Message: ...

    @abc.abstractmethod
    async def send_sticker(self, chat_id: int, sticker_id: int, reply_to: int = 0) -> None: ...

    @abc.abstractmethod
    async def get_saved_stickers(self) -> list[Message]:
        """favorite + recent stickers as pseudo messages (kind=sticker, extra: sticker_id, emoji)."""

    @abc.abstractmethod
    async def add_favorite_sticker(self, sticker_id: int) -> None: ...

    @abc.abstractmethod
    async def react(self, chat_id: int, message_id: int, emoji: str) -> None: ...

    @abc.abstractmethod
    async def edit_message(self, chat_id: int, message_id: int, text: str, is_caption: bool) -> None: ...

    @abc.abstractmethod
    async def delete_messages(self, chat_id: int, message_ids: list[int]) -> None: ...

    @abc.abstractmethod
    async def forward_messages(self, to_chat_id: int, from_chat_id: int, message_ids: list[int]) -> None: ...

    @abc.abstractmethod
    async def join_chat_by_invite_link(self, link: str) -> Chat: ...

    @abc.abstractmethod
    async def join_chat(self, chat_id: int) -> None: ...

    @abc.abstractmethod
    async def leave_chat(self, chat_id: int) -> None: ...

    @abc.abstractmethod
    async def ban_member(self, chat_id: int, user_id: int) -> None: ...

    @abc.abstractmethod
    async def set_member_tag(self, chat_id: int, user_id: int, tag: str) -> None: ...

    @abc.abstractmethod
    async def block_user(self, user_id: int) -> None: ...

    @abc.abstractmethod
    async def delete_chat_history(self, chat_id: int) -> None: ...

    # ---- presence ----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    async def set_online(self, online: bool) -> None: ...

    @abc.abstractmethod
    async def open_chat(self, chat_id: int) -> None: ...

    @abc.abstractmethod
    async def close_chat(self, chat_id: int) -> None: ...

    @abc.abstractmethod
    async def view_messages(self, chat_id: int, message_ids: list[int]) -> None:
        """Marks messages as read (the two checkmarks)."""

    @abc.abstractmethod
    async def send_chat_action(self, chat_id: int, action: str | None) -> None:
        """action: typing | choosing_sticker | recording_voice | uploading_photo | None (cancel)."""

    # ---- premium -----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    async def recognize_speech(self, message: Message) -> str | None:
        """Telegram Premium transcription of a voice/video note. None if unavailable."""

    # ---- helpers -----------------------------------------------------------------------------------------------
    async def get_chats_by_ids(self, ids: list[int]) -> list[Chat]:
        return list(await asyncio.gather(*(self.get_chat(i) for i in ids)))
