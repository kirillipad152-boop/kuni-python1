"""SSE filter that hides Kuni's own (injected) tool calls from the client. Port of proxy_server/streaming_filter.cpp.

Lines pass through untouched until a tool call shows up. Chunks of a tool call are buffered until its name is known;
if the call is one of ours, everything is held until `finish_reason` arrives, then our tool calls are dispatched and the
client's own tool calls (if any) are re-emitted as one clean synthetic chunk.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Awaitable, Callable

from ..llm import Choice, Response, ToolCall, collect_chunk

log = logging.getLogger("streaming_filter")


async def _call(cb, *args):
    r = cb(*args)
    if inspect.isawaitable(r):
        await r


class StreamingFilter:
    def __init__(self, filtered_tool_names: set[str]) -> None:
        self.filtered = set(filtered_tool_names)
        self.response = Response()
        self.pending: list[str] = []
        self.dispatched = False
        self.has_passed_through_content = False

    @property
    def choices(self) -> list[Choice]:
        return self.response.choices

    async def process_line(self, line: str, pass_through: Callable[[str], object],
                           handle_tool_call: Callable[[ToolCall], object]) -> None:
        delim = line.find(": ")
        if delim == -1:
            await _call(pass_through, line)  # not a key-value SSE line (e.g. ": OPENROUTER PROCESSING")
            return
        command, value = line[:delim], line[delim + 2 :]
        if command != "data" or value == "[DONE]" or not value or value[0] != "{":
            await _call(pass_through, line)
            return
        try:
            chunk = json.loads(value)
        except json.JSONDecodeError:
            log.debug("JSON parse error, passing through")
            await _call(pass_through, line)
            return

        collect_chunk(self.response, chunk)
        indices = [int(c.get("index") or 0) for c in chunk.get("choices") or []]
        states = [self.choices[i] for i in indices if i < len(self.choices)]
        any_unresolved = any(not tc.function.name for s in states for tc in s.message.tool_calls)
        has_tool_calls = any(s.message.tool_calls for s in states)

        async def flush() -> None:
            for p in self.pending:
                await _call(pass_through, p)
            self.pending.clear()

        if not has_tool_calls:
            # no tool calls involved — flush any pending lines and pass this one through.
            await flush()
            self.has_passed_through_content = True
            await _call(pass_through, line)
            return
        if any_unresolved:
            self.pending.append(line)  # buffer until we know the tool name
            return

        any_injected = any(tc.function.name in self.filtered for s in states for tc in s.message.tool_calls)
        if not any_injected:
            await flush()
            await _call(pass_through, line)
            return

        if not any(c.finish_reason for c in self.choices):
            self.pending.append(line)  # keep buffering until finish_reason arrives
            return
        if self.dispatched:
            return  # guard against duplicate finish_reason chunks from upstream
        self.dispatched = True

        non_injected: list[ToolCall] = []
        for choice in self.choices:
            for tc in choice.message.tool_calls:
                if tc.function.name in self.filtered:
                    await _call(handle_tool_call, tc)
                else:
                    re = ToolCall(tc.id, len(non_injected), tc.type, tc.function)
                    non_injected.append(re)
        self.pending.clear()  # they contain injected tool call chunks

        # Always emit a synthetic SSE chunk so the client sees a proper finish. If there are non-injected tool calls,
        # emit them with finish_reason="tool_calls"; if all were injected, an empty delta so the stream ends cleanly.
        delta: dict = {"role": "assistant"}
        if non_injected:
            delta["tool_calls"] = [{"index": tc.index, "id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in non_injected]
        choice_json: dict = {"index": 0, "delta": delta}
        if non_injected:
            choice_json["finish_reason"] = "tool_calls"
        response_json = {"id": chunk.get("id"), "object": "chat.completion.chunk", "created": chunk.get("created"),
                         "model": chunk.get("model"), "choices": [choice_json]}
        await _call(pass_through, "data: " + json.dumps(response_json, ensure_ascii=False))
