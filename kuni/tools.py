"""Tool registry (OpenAI function calling). Port of OpenAITools."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .llm import EMBEDDING_TAG, Message, Session, ToolCall, ROLE_TOOL  # noqa: F401
from .util import remove_control_characters

log = logging.getLogger("OpenAITools")


@dataclass
class Property:
    type: str = "string"
    description: str = ""
    # vibecode.moe (and OpenAI strict mode in general) requires every property to be listed in `required`;
    # genuinely optional properties must instead be made nullable.
    nullable: bool = False
    # the LLM may pass either a single value of `type` or an array of `items`
    or_array: bool = False
    items: "Property | None" = None

    def to_json(self) -> dict:
        types = [self.type]
        if self.or_array:
            types.append("array")
        if self.nullable:
            types.append("null")
        d: dict[str, Any] = {"description": self.description, "type": types[0] if len(types) == 1 else types}
        if self.type == "array" or self.or_array:
            assert self.items is not None, "array-typed property must specify items"
            d["items"] = self.items.to_json()
        return d


@dataclass
class Ctx:
    tools: "ToolSet"
    args: dict
    temporary_context: Session
    all_tool_calls: list[ToolCall]
    logger: logging.Logger = log


Handler = Callable[[Ctx], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    handler: Handler
    properties: dict[str, Property] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    strict: bool = True
    type: str = "function"

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {k: v.to_json() for k, v in self.properties.items()},
                    "required": list(self.required),
                    "additionalProperties": False,
                },
                "strict": self.strict,
            },
        }


class ToolSet:
    def __init__(self, tools: "list[Tool] | tuple[Tool, ...]" = ()) -> None:
        self._handlers: dict[str, Tool] = {}
        # Hook fired after each tool call handler completes successfully (not called if the handler throws).
        self.on_after_tool_call: list[Callable[[str], None]] = []
        for t in tools:
            self.insert(t)

    def insert(self, tool: Tool) -> None:
        assert tool.handler is not None
        self._handlers[tool.name] = tool

    def replace_with(self, other: "ToolSet") -> None:
        """`tools = OpenAITools{...}` of the C++ kernel: swaps the whole set, hooks included."""
        self._handlers = dict(other._handlers)
        self.on_after_tool_call = list(other.on_after_tool_call)

    def handlers(self) -> dict[str, Tool]:
        return dict(self._handlers)

    def __len__(self) -> int:
        return len(self._handlers)

    def as_json(self) -> list[dict]:
        return [t.to_json() for t in self._handlers.values()]

    async def handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        metrics=None,
        temporary_context: Session | None = None,
        logger: logging.Logger = log,
    ) -> Session:
        logger.debug("handleToolCalls")
        result = Session()
        temporary_context = temporary_context if temporary_context is not None else Session()
        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                tool = self._handlers.get(name)
                if tool is None:
                    content = f'tool "{name}" is not currently available. Please use another tool instead.'
                else:
                    args = json.loads(tool_call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    ctx = Ctx(tools=self, args=args, temporary_context=temporary_context,
                              all_tool_calls=tool_calls, logger=logger)
                    if metrics is not None:
                        with metrics.point("function", name):
                            content = await tool.handler(ctx)
                    else:
                        content = await tool.handler(ctx)
                    for hook in list(self.on_after_tool_call):
                        hook(name)
            except Exception as e:  # noqa: BLE001  (AException + whatever a handler may raise)
                logger.error('error while executing "%s" tool: %s', name, e, exc_info=not isinstance(e, ToolError))
                content = f'error while executing "{name}" tool: {e}'
            result.append(
                Message(role=ROLE_TOOL, content=remove_control_characters(str(content)), tool_call_id=tool_call.id)
            )
        return result


class ToolError(Exception):
    """Technical error that is reported to the LLM as the tool result (AException of the C++ kernel)."""
