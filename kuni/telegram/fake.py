"""In-memory Telegram, used by the tests and by `python -m kuni chat` (talk to Kuni without a Telegram account)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from .base import TelegramClient
from .types import Chat, ChatType, MemberStatus, Message, NotFoundError, TelegramError, User


class FakeTelegram(TelegramClient):
    def __init__(self, my_id: int = 1000) -> None:
        super().__init__()
        self._my_id = my_id
        self.users: dict[int, User] = {my_id: User(my_id, "Kuni", "", ["kunii_chan"], is_premium=False)}
        self.chats: dict[int, Chat] = {}
        self.messages: dict[int, list[Message]] = {}
        self.actions: list[tuple] = []  # every write call, for assertions
        self.contacts: list[int] = []
        self.statuses: dict[tuple[int, int], MemberStatus] = {}
        self.saved_stickers: list[Message] = []
        self._connected = asyncio.Event()
        self._connected.set()
        self._next_id = 1
        self.printer = None  # optional callable(chat, message) used by the REPL to display outgoing messages

    # ---- fixtures ----------------------------------------------------------------------------------------------
    def add_user(self, user_id: int, first_name: str, last_name: str = "", username: str | None = None,
                 contact: bool = True) -> Chat:
        self.users[user_id] = User(user_id, first_name, last_name, [username] if username else [])
        if contact:
            self.contacts.append(user_id)
        chat = Chat(user_id, f"{first_name} {last_name}".strip(), ChatType.PRIVATE)
        self.chats[user_id] = chat
        self.messages.setdefault(user_id, [])
        return chat

    def add_group(self, chat_id: int, title: str, kind: ChatType = ChatType.SUPERGROUP) -> Chat:
        chat = Chat(chat_id, title, kind)
        self.chats[chat_id] = chat
        self.messages.setdefault(chat_id, [])
        return chat

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _store(self, m: Message) -> Message:
        self.messages.setdefault(m.chat_id, []).insert(0, m)  # newest first
        chat = self.chats.get(m.chat_id)
        if chat:
            chat.last_message = m
        return m

    async def incoming(self, chat_id: int, text: str, sender_id: int | None = None, **kw) -> Message:
        """Simulates a new incoming message and fires the handlers (like Telegram would)."""
        sender_id = sender_id if sender_id is not None else chat_id
        m = Message(id=self._new_id(), chat_id=chat_id, date=int(time.time()), sender_id=sender_id, text=text, **kw)
        self._store(m)
        chat = self.chats[chat_id]
        chat.unread_count += 1
        for h in list(self.new_message_handlers):
            await h(m)
        return m

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    async def start(self) -> None:
        for h in self.logged_in_handlers:
            h()

    async def stop(self) -> None:
        pass

    async def wait_for_connection(self) -> None:
        await self._connected.wait()

    @property
    def my_id(self) -> int:
        return self._my_id

    # ---- reading -----------------------------------------------------------------------------------------------
    async def get_chat(self, chat_id: int) -> Chat:
        if chat_id not in self.chats:
            raise NotFoundError(f"Chat {chat_id} not found")
        return self.chats[chat_id]

    async def get_user(self, user_id: int) -> User:
        if user_id not in self.users:
            raise NotFoundError(f"User {user_id} not found")
        return self.users[user_id]

    async def get_message(self, chat_id: int, message_id: int) -> Message:
        for m in self.messages.get(chat_id, []):
            if m.id == message_id:
                return m
        raise NotFoundError("Not Found")

    async def get_chats(self, limit: int = 50) -> list[Chat]:
        def key(c: Chat):
            return c.last_message.date if c.last_message else 0

        return sorted(self.chats.values(), key=key, reverse=True)[:limit]

    async def get_chat_history(self, chat_id: int, from_message_id: int = 0, limit: int = 30) -> list[Message]:
        msgs = self.messages.get(chat_id, [])
        if from_message_id:
            msgs = [m for m in msgs if m.id < from_message_id]
        return msgs[:limit]

    async def get_messages_around(self, chat_id: int, message_id: int, before: int, after: int) -> list[Message]:
        msgs = self.messages.get(chat_id, [])  # newest first
        idx = next((i for i, m in enumerate(msgs) if m.id == message_id), None)
        if idx is None:
            return []
        return msgs[max(0, idx - after) : idx + 1 + before]

    async def search_chats_on_server(self, query: str, limit: int = 50) -> list[Chat]:
        q = query.lower()
        return [c for c in self.chats.values() if q in c.title.lower()][:limit]

    async def search_public_chat(self, username: str) -> Chat | None:
        for uid, u in self.users.items():
            if username.lower() in [x.lower() for x in u.usernames] and uid in self.chats:
                return self.chats[uid]
        return None

    async def search_chat_messages(self, chat_id, query, sender_id, limit):
        found = [m for m in self.messages.get(chat_id, [])
                 if query.lower() in m.text.lower() and (sender_id is None or m.sender_id == sender_id)]
        return len(found), found[:limit]

    async def search_messages_global(self, query, limit, min_date, max_date):
        found = []
        for msgs in self.messages.values():
            for m in msgs:
                if query.lower() not in m.text.lower():
                    continue
                if m.date < min_date or (max_date and m.date > max_date):
                    continue
                found.append(m)
        found.sort(key=lambda m: m.date, reverse=True)
        return len(found), found[:limit]

    async def get_contact_ids(self) -> list[int]:
        return list(self.contacts)

    async def get_member_status(self, chat_id: int, user_id: int) -> MemberStatus:
        return self.statuses.get((chat_id, user_id), MemberStatus.MEMBER)

    async def download_media(self, message: Message, what: str = "media") -> Path:
        if isinstance(message.raw, (str, Path)):
            return Path(message.raw)
        raise TelegramError("no media")

    async def download_chat_photo(self, chat: Chat) -> Path:
        if isinstance(chat.raw, (str, Path)):
            return Path(chat.raw)
        raise TelegramError("no photo")

    # ---- writing -----------------------------------------------------------------------------------------------
    async def send_message(self, chat_id, text="", photo_path=None, audio_path=None, reply_to=0) -> Message:
        kind = "photo" if photo_path else ("voice" if audio_path else "text")
        m = Message(id=self._new_id(), chat_id=chat_id, date=int(time.time()), sender_id=self._my_id, kind=kind,
                    text=text, reply_to_id=reply_to, is_outgoing=True)
        self._store(m)
        self.actions.append(("send_message", chat_id, text, str(photo_path or ""), str(audio_path or ""), reply_to))
        if self.printer:
            self.printer(self.chats.get(chat_id), m)
        return m

    async def send_sticker(self, chat_id, sticker_id, reply_to=0) -> None:
        self.actions.append(("send_sticker", chat_id, sticker_id, reply_to))

    async def get_saved_stickers(self) -> list[Message]:
        return list(self.saved_stickers)

    async def add_favorite_sticker(self, sticker_id: int) -> None:
        self.actions.append(("add_favorite_sticker", sticker_id))

    async def react(self, chat_id, message_id, emoji) -> None:
        self.actions.append(("react", chat_id, message_id, emoji))

    async def edit_message(self, chat_id, message_id, text, is_caption) -> None:
        self.actions.append(("edit_message", chat_id, message_id, text, is_caption))
        (await self.get_message(chat_id, message_id)).text = text

    async def delete_messages(self, chat_id, message_ids) -> None:
        self.actions.append(("delete_messages", chat_id, list(message_ids)))
        self.messages[chat_id] = [m for m in self.messages.get(chat_id, []) if m.id not in message_ids]

    async def forward_messages(self, to_chat_id, from_chat_id, message_ids) -> None:
        self.actions.append(("forward_messages", to_chat_id, from_chat_id, list(message_ids)))

    async def join_chat_by_invite_link(self, link: str) -> Chat:
        self.actions.append(("join_chat_by_invite_link", link))
        chat = self.add_group(-1001234, "Joined chat")
        return chat

    async def join_chat(self, chat_id: int) -> None:
        self.actions.append(("join_chat", chat_id))

    async def leave_chat(self, chat_id: int) -> None:
        self.actions.append(("leave_chat", chat_id))

    async def ban_member(self, chat_id: int, user_id: int) -> None:
        self.actions.append(("ban_member", chat_id, user_id))

    async def set_member_tag(self, chat_id: int, user_id: int, tag: str) -> None:
        self.actions.append(("set_member_tag", chat_id, user_id, tag))

    async def block_user(self, user_id: int) -> None:
        self.actions.append(("block_user", user_id))

    async def delete_chat_history(self, chat_id: int) -> None:
        self.actions.append(("delete_chat_history", chat_id))

    # ---- presence ----------------------------------------------------------------------------------------------
    async def set_online(self, online: bool) -> None:
        self.actions.append(("set_online", online))

    async def open_chat(self, chat_id: int) -> None:
        self.actions.append(("open_chat", chat_id))

    async def close_chat(self, chat_id: int) -> None:
        self.actions.append(("close_chat", chat_id))

    async def view_messages(self, chat_id: int, message_ids: list[int]) -> None:
        self.actions.append(("view_messages", chat_id, list(message_ids)))
        if chat_id in self.chats:
            self.chats[chat_id].unread_count = 0
            if message_ids:
                self.chats[chat_id].last_read_inbox_message_id = max(message_ids)

    async def send_chat_action(self, chat_id: int, action: str | None) -> None:
        self.actions.append(("chat_action", chat_id, action))

    async def recognize_speech(self, message: Message) -> str | None:
        return None
