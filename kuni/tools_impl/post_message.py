"""Port of util/post_message.cpp."""

from __future__ import annotations

from pathlib import Path

from ..config import config
from ..lockdown import is_accessible_from_lockdown
from ..telegram.base import TelegramClient
from ..telegram.types import Message
from ..tools import ToolError


async def telegram_post_message(telegram: TelegramClient, chat_id: int, text: str, photo_path: Path | None = None,
                                audio_path: Path | None = None, reply_to: int = 0) -> Message:
    # Check lockdown mode - only allow PAPIK_CHAT_ID if lockdown is enabled
    if not await is_accessible_from_lockdown(telegram, chat_id):
        raise ToolError(f"Lockdown mode is enabled. You can only send messages to chat with ID {config().papikChatId} "
                        "(PAPIK_CHAT_ID).")
    return await telegram.send_message(chat_id, text, photo_path, audio_path, reply_to)
