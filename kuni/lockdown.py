"""Lockdown mode. Port of util/is_accessible_from_lockdown.cpp."""

from __future__ import annotations

from .config import LockdownMode, config
from .telegram.base import TelegramClient


async def is_accessible_from_lockdown(telegram: TelegramClient, chat_id: int, lockdown: LockdownMode | None = None) -> bool:
    cfg = config()
    lockdown = lockdown if lockdown is not None else cfg.lockdown
    if chat_id == cfg.papikChatId:
        return True  # the owner can neither be locked out nor blocked
    if chat_id in cfg.blockedChatIds:
        return False
    if chat_id in cfg.allowedChatIds:
        return True  # an explicit exception to contacts_only / papik_only
    if lockdown == LockdownMode.NONE:
        return True
    if lockdown == LockdownMode.CONTACTS_ONLY:
        if chat_id < 0:
            return True  # a group chat that kuni participates in, ok.
        if chat_id in await telegram.get_contact_ids():
            return True
        # fall through to papik
    return chat_id == config().papikChatId
