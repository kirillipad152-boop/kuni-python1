"""Remembers hidden messages (the proxy's own tool calls) and re-inserts them into later client requests. Port of
proxy_server/message_injector.cpp + json_fingerprint.h."""

from __future__ import annotations

import json


def fingerprint(obj) -> str:
    """JSON text without brackets: a fingerprint of an array of messages is a prefix of the fingerprint of the same
    array with more messages appended (used to tell a continuation of a dialog from a new dialog)."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return "".join(c for c in s if c not in "(){}[]")


class MessageInjector:
    def __init__(self) -> None:
        self._after: dict[str, list[dict]] = {}

    def after(self, msg: dict) -> list[dict]:
        return self._after.setdefault(fingerprint(msg), [])

    def merge(self, messages: list[dict]) -> list[dict]:
        if not self._after:
            return messages
        out: list[dict] = []
        for m in messages:
            out.append(m)
            out.extend(self._after.get(fingerprint(m), []))
        return out
