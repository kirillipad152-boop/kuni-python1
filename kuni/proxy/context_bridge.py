"""Bridges proxy sessions to the memory. Port of proxy_server/context_bridge.cpp.

Dumps a session to the memory (graph) once it's no longer active for a while, or when papik writes to Kuni (this
object is a chat-history message processor).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass

from ..config import Endpoint, config
from ..llm import ROLE_USER, Message, OpenAIChat, Response
from ..memory.extract import save_pages
from ..memory.graph import GraphMemory
from ..prompts import prompts
from .message_injector import fingerprint

log = logging.getLogger("ContextBridge")


@dataclass
class PendingSession:
    salt: str
    request: dict
    update_time: float


class ContextBridge:
    def __init__(self, endpoint: Endpoint, memory: GraphMemory, llm: OpenAIChat, timeout: float = 3600.0) -> None:
        self.endpoint, self.memory, self.llm, self.timeout = endpoint, memory, llm, timeout
        self.pending: list[PendingSession] = []
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            try:
                await self.collect_and_save_sessions_not_newer_than(time.time() - self.timeout)
            except Exception:  # noqa: BLE001
                log.exception("context bridge update failed")

    async def process_chat_history_message(self, chat, msg, formatted: str) -> str:
        try:
            if chat.id != config().papikChatId:
                return formatted  # early return - this feature is locked down to the papik only.
            work_items = ""
            for item in await self.collect_and_save_sessions_not_newer_than(float(msg.date)):
                work_items += (f"<work_item>\n{item}\n</work_item instructions=\"This describes a project or a task you were "
                               f"working together with {chat.title}\">\n")
            return work_items + formatted
        except Exception as e:  # noqa: BLE001
            log.error("processChatHistoryMessage: %s", e)
            return formatted

    async def collect_and_save_sessions_not_newer_than(self, tp: float) -> list[str]:
        sessions = [s.request for s in self.pending if s.update_time <= tp]
        self.pending = [s for s in self.pending if s.update_time > tp]
        result: list[str] = []
        for session in sessions:
            session["stream"] = False
            session.pop("tools", None)
            session.setdefault("messages", []).append(Message(role=ROLE_USER, content=prompts().diary_save).to_json())
            session_id = session.get("session_id") or "context_save"
            session["session_id"] = session_id
            for _attempt in range(3):
                data = await self.llm.post_json(self.endpoint, "chat/completions", session, session_id)
                response = Response.from_json(data)
                message = response.choices[0].message
                saved = await save_pages(self.memory, self.llm, message.content, source_chat_id=config().papikChatId)
                if saved:
                    result.append("\n---\n".join(n.summary for n in saved))
                    break
                log.warning("LLM returned empty response")
                session["messages"].append(message.to_json())
                session["messages"].append(
                    Message(role=ROLE_USER, content="You should provide a response with long entries, no tool calls.").to_json())
        return result

    def collect_request_to_llm(self, request: dict) -> None:
        """Handles ProxyServer.sent_request_to_llm. Identifies the session by the request["messages"] JSON array. We use
        dicts instead of Session to preserve the client's original arguments (like top_k) and avoid cache misses."""
        incoming = PendingSession(fingerprint(request.get("messages", [])), copy.deepcopy(request), time.time())
        for i, s in enumerate(self.pending):
            if incoming.salt.startswith(s.salt):
                self.pending[i] = incoming  # an existing session was updated.
                return
        self.pending.append(incoming)  # a new session.
