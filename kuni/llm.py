"""OpenAI-compatible chat client (streaming + tool calls). Port of IOpenAIChat / OpenAIChatImpl / OpenAIChatMeasurable."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from .config import Endpoint, EndpointAndModel, config, key_fingerprint

log = logging.getLogger("OpenAIChat")

EMBEDDING_TAG = "kuni_embedding"  # marks base64 images inside text content

ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_TOOL = "tool"


class LLMError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Function:
    name: str = ""
    arguments: str = ""


@dataclass
class ToolCall:
    id: str = ""
    index: int = 0
    type: str = ""
    function: Function = field(default_factory=Function)

    def merge(self, other: "ToolCall") -> None:
        """Streaming: a tool call arrives in pieces."""
        if other.id and other.id != self.id:
            self.id += other.id
        self.index = other.index
        if other.type and other.type != self.type:
            self.type += other.type
        if other.function.name and other.function.name != self.function.name:
            self.function.name += other.function.name
        self.function.arguments += other.function.arguments

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "type": self.type or "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments or "{}"},
        }

    @staticmethod
    def from_json(d: dict) -> "ToolCall":
        f = d.get("function") or {}
        return ToolCall(
            id=d.get("id") or "",
            index=int(d.get("index") or 0),
            type=d.get("type") or "",
            function=Function(f.get("name") or "", f.get("arguments") or ""),
        )


@dataclass
class Message:
    role: str = ROLE_USER
    content: str = ""
    tool_call_id: str = ""
    reasoning: str = ""
    reasoning_content: str = ""  # deepseek-like providers require it to be passed back
    tool_calls: list[ToolCall] = field(default_factory=list)

    def merge(self, other: "Message") -> None:
        """For streaming."""
        self.role = other.role or self.role
        self.content += other.content
        self.tool_call_id = other.tool_call_id or self.tool_call_id
        self.reasoning += other.reasoning
        self.reasoning_content += other.reasoning_content
        for tc in other.tool_calls:
            while len(self.tool_calls) <= tc.index:
                self.tool_calls.append(ToolCall(index=len(self.tool_calls)))
            self.tool_calls[tc.index].merge(tc)

    def to_json(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.tool_calls:
            d["tool_calls"] = [tc.to_json() for tc in self.tool_calls]
        return d

    @staticmethod
    def from_json(d: dict) -> "Message":
        content = d.get("content")
        if isinstance(content, list):  # multimodal parts -> text only
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return Message(
            role=d.get("role") or ROLE_USER,
            content=content or "",
            tool_call_id=d.get("tool_call_id") or "",
            reasoning=d.get("reasoning") or "",
            reasoning_content=d.get("reasoning_content") or "",
            tool_calls=[ToolCall.from_json(t) for t in (d.get("tool_calls") or [])],
        )


class Session(list):
    """A conversation (list of messages) with a stable session id (sent as the `x-session-id` header)."""

    def __init__(self, messages=(), session_id: str | None = None, *, chat_id: int | None = None):
        super().__init__(messages)
        self.session_id = session_id or str(uuid.uuid4())
        self.chat_id = chat_id  # None is reserved for trusted CLI/proxy clients; Telegram always supplies a scope.

    def last(self) -> Message:
        return self[-1]

    def copy(self) -> "Session":  # type: ignore[override]
        return Session(self, self.session_id, chat_id=self.chat_id)

    def __lshift__(self, other):  # session << message / session << [messages]
        if isinstance(other, Message):
            self.append(other)
        else:
            self.extend(other)
        return self

    def to_json(self) -> list[dict]:
        result: list[dict] = []
        for message in self:
            if f"</{EMBEDDING_TAG}>" not in message.content:
                result.append(message.to_json())
                continue
            # reverse engineered from vscode copilot plugin: images go as separate content parts
            content = message.content

            def append(text: str) -> None:
                if text:
                    result.append(Message(role=message.role, content=text).to_json())

            while True:
                tag_pos = content.find(f"<{EMBEDDING_TAG}>")
                append(content[: tag_pos if tag_pos != -1 else len(content)])
                if tag_pos == -1:
                    break
                content = content[tag_pos:]
                content = content[content.find(">") + 1 :]
                body = content[: content.find(f"</{EMBEDDING_TAG}>")]
                append("<attachments>")
                result.append(
                    {
                        "role": message.role,
                        "content": [{"type": "image_url", "image_url": {"url": body}}],
                    }
                )
                append("</attachments>")
                content = content[len(body) :]
                content = content[content.find(">") + 1 :]
        return result

    @staticmethod
    def from_json(arr: list[dict], session_id: str | None = None, *, chat_id: int | None = None) -> "Session":
        return Session([Message.from_json(m) for m in arr], session_id, chat_id=chat_id)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    @staticmethod
    def from_json(v: Any) -> "Usage":
        u = Usage()
        if not isinstance(v, dict):
            # OpenAI-compatible endpoints send `"usage": null` in every streaming chunk but the last one.
            return u
        u.prompt_tokens = int(v.get("prompt_tokens") or 0)
        u.completion_tokens = int(v.get("completion_tokens") or 0)
        u.total_tokens = int(v.get("total_tokens") or (u.prompt_tokens + u.completion_tokens))
        hit = v.get("prompt_cache_hit_tokens")
        if hit is None:
            details = v.get("prompt_tokens_details")
            hit = details.get("cached_tokens") if isinstance(details, dict) else None  # openrouter
        if hit is None or hit < 0:
            u.prompt_cache_hit_tokens = 0
        else:
            u.prompt_cache_hit_tokens = int(hit)
            miss = v.get("prompt_cache_miss_tokens")  # deepseek
            u.prompt_cache_miss_tokens = int(miss if miss is not None else u.prompt_tokens - u.prompt_cache_hit_tokens)
        return u


@dataclass
class Choice:
    index: int = 0
    message: Message = field(default_factory=lambda: Message(role=ROLE_ASSISTANT))
    finish_reason: str = ""


@dataclass
class Response:
    id: str = ""
    object: str = ""
    created: int = 0
    model: str = ""
    provider: str = ""
    choices: list[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "created": self.created,
            "choices": [
                {"index": c.index, "message": c.message.to_json(), "finish_reason": c.finish_reason}
                for c in self.choices
            ],
            "usage": self.usage.__dict__,
        }

    @staticmethod
    def from_json(d: dict) -> "Response":
        r = Response(id=d.get("id") or "", model=d.get("model") or "", created=int(d.get("created") or 0))
        for c in d.get("choices") or []:
            r.choices.append(
                Choice(
                    index=int(c.get("index") or 0),
                    message=Message.from_json({**(c.get("message") or {}), "role": ROLE_ASSISTANT}),
                    finish_reason=c.get("finish_reason") or "",
                )
            )
        r.usage = Usage.from_json(d.get("usage"))
        return r


@dataclass
class Params:
    system_prompt: str = ""
    max_output_tokens: int = 8192
    config: EndpointAndModel | None = None  # None = config().llm
    seed: int | None = None
    tools: list = field(default_factory=list)


@dataclass
class Transcription:
    text: str = ""
    language: str = ""
    duration: float = 0.0
    segments: list[dict] = field(default_factory=list)  # {start, end, text}


def embed_image_bytes(jpeg: bytes) -> str:
    return f"<{EMBEDDING_TAG}>data:image/jpg;base64,{base64.b64encode(jpeg).decode()}</{EMBEDDING_TAG}>"


def embed_image(image) -> str:
    """image: PIL.Image. Downscaled to fit 672x672 (aspect ratio is preserved), like IOpenAIChat::embedImage."""
    from PIL import Image

    img = image.convert("RGB")
    img.thumbnail((672, 672), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return embed_image_bytes(buf.getvalue())


def collect_chunk(r: Response, chunk: dict) -> None:
    """Merges one streaming chunk (`chat.completion.chunk`) into the response being assembled."""
    r.id = chunk.get("id") or r.id
    r.model = chunk.get("model") or r.model
    r.created = int(chunk.get("created") or r.created)
    if chunk.get("usage"):
        r.usage = Usage.from_json(chunk["usage"])
    for choice in chunk.get("choices") or []:
        idx = int(choice.get("index") or 0)
        while len(r.choices) <= idx:
            r.choices.append(Choice(index=len(r.choices)))
        delta = choice.get("delta") or {}
        msg = Message(
            role=delta.get("role") or ROLE_ASSISTANT,
            content=delta.get("content") or "",
            reasoning=delta.get("reasoning") or "",
            reasoning_content=delta.get("reasoning_content") or "",
            tool_calls=[ToolCall.from_json(t) for t in (delta.get("tool_calls") or [])],
        )
        r.choices[idx].message.merge(msg)
        r.choices[idx].message.role = ROLE_ASSISTANT
        if choice.get("finish_reason"):
            r.choices[idx].finish_reason = choice["finish_reason"]


# ---------------------------------------------------------------------------------------------------------------------
# streaming response
# ---------------------------------------------------------------------------------------------------------------------
class StreamingResponse:
    def __init__(self) -> None:
        self.response = Response()
        self.completed: asyncio.Task | None = None
        self._listeners: list[Callable[[Response], None]] = []

    def on_changed(self, cb: Callable[[Response], None]) -> None:
        self._listeners.append(cb)

    def _changed(self) -> None:
        for cb in list(self._listeners):
            try:
                cb(self.response)
            except Exception:  # noqa: BLE001
                log.exception("response listener failed")

    def __await__(self):
        assert self.completed is not None
        return self.completed.__await__()


class TuiStreamingPrinter:
    """Prints the response incrementally to the terminal (reasoning is dimmed)."""

    def __init__(self, out=None) -> None:
        self.out = out or sys.stderr
        self.printed_reasoning = 0
        self.printed_content = 0
        self.printed_args: dict[int, int] = {}
        self.mode = ""

    def _w(self, s: str) -> None:
        self.out.write(s)
        self.out.flush()

    def update(self, r: Response) -> None:
        if not r.choices:
            return
        m = r.choices[0].message
        reasoning = m.reasoning_content or m.reasoning
        if len(reasoning) > self.printed_reasoning:
            if self.mode != "r":
                self._w("\n\x1b[2m[thinking] ")
                self.mode = "r"
            self._w(reasoning[self.printed_reasoning :])
            self.printed_reasoning = len(reasoning)
        if len(m.content) > self.printed_content:
            if self.mode != "c":
                self._w("\x1b[0m\n[reply] ")
                self.mode = "c"
            self._w(m.content[self.printed_content :])
            self.printed_content = len(m.content)
        for i, tc in enumerate(m.tool_calls):
            if self.mode != f"t{i}":
                self._w(f"\x1b[0m\n[tool] {tc.function.name}(")
                self.mode = f"t{i}"
            done = self.printed_args.get(i, 0)
            if len(tc.function.arguments) > done:
                self._w(tc.function.arguments[done:])
                self.printed_args[i] = len(tc.function.arguments)

    def finish(self) -> None:
        if self.mode.startswith("t"):
            self._w(")")
        self._w("\x1b[0m\n")


# ---------------------------------------------------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------------------------------------------------
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MODEL_LOAD_ERROR = "model failed to load, this may be due to resource limitations or an internal error"


def max_retries(status: int) -> int:
    return 5 if status == 429 else 3  # rate limits (e.g. free tiers) deserve more patience


def backoff(status: int, attempt: int, headers) -> float:
    """Seconds to wait before retry #attempt. Honors Retry-After; 429: 5,10,20,40,60s; others: 2,4,8s."""
    ra = headers.get("retry-after")
    if ra:
        try:
            delay = float(ra)
        except ValueError:
            try:
                delay = parsedate_to_datetime(ra).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                delay = float("nan")
        if math.isfinite(delay):
            return max(0.0, delay)
    if status == 429:
        return min(60.0, 5.0 * 2 ** (attempt - 1))
    return min(30.0, 2.0 ** attempt)


def _headers(endpoint: Endpoint, session_id: str | None = None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if session_id:
        h["x-session-id"] = session_id
    if endpoint.bearerKey:
        h["Authorization"] = f"Bearer {endpoint.bearerKey}"
    return h


def _timeout() -> httpx.Timeout:
    t = float(config().requestTimeoutSecs)
    return httpx.Timeout(connect=min(t, 30.0), read=t, write=t, pool=t)


class OpenAIChat:
    """Stateless client: every call gets its endpoint/model from `Params.config` (default: config().llm)."""

    def __init__(self, http: httpx.AsyncClient | None = None) -> None:
        self._http = http
        self.response_metrics: list[Callable[[str, Usage], None]] = []
        self.rate_limit_events: list[Callable[[], None]] = []
        self.success_events: list[Callable[[], None]] = []
        self._key_positions: dict[tuple, int] = {}
        self._key_ready_at: dict[tuple[str, str], float] = {}
        # Observers for the owner's panel: (status or "error", seconds) per HTTP attempt, and per-key counters that
        # never contain the key itself (only its fingerprint).
        self.request_events: list[Callable[[object, float], None]] = []
        self.key_stats: dict[str, dict] = {}

    def _note_key(self, headers: dict, field: str) -> None:
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            stat = self.key_stats.setdefault(key_fingerprint(auth[7:]), {"requests": 0, "ok": 0, "rate_limited": 0,
                                                                         "errors": 0, "last_used": None})
            stat[field] += 1
            if field == "requests":
                stat["last_used"] = time.time()

    def _emit_request(self, status, started: float, headers: dict | None = None) -> None:
        if headers is not None and status != 200 and status != 429:
            self._note_key(headers, "errors")
        for callback in list(self.request_events):
            try:
                callback(status, time.monotonic() - started)
            except Exception:  # noqa: BLE001
                log.warning("LLM observer callback failed")

    @staticmethod
    def _emit(callbacks) -> None:
        for callback in list(callbacks):
            try:
                callback()
            except Exception:
                log.warning("LLM observer callback failed")

    async def _request_headers(self, endpoint: Endpoint, session_id: str | None) -> dict[str, str]:
        headers = _headers(endpoint, session_id)
        keys = endpoint.bearer_keys()
        if not keys:
            return headers
        # (stats are counted below, once a key has been chosen)
        pool = (endpoint.baseUrl, tuple(keys))
        while True:
            position = self._key_positions.get(pool, 0)
            ordered = keys[position:] + keys[:position]
            now = time.monotonic()
            for key in ordered:
                if self._key_ready_at.get((endpoint.baseUrl, key), 0) <= now:
                    self._key_positions[pool] = (keys.index(key) + 1) % len(keys)
                    headers["Authorization"] = "Bearer " + key
                    self._note_key(headers, "requests")
                    return headers
            await asyncio.sleep(min(self._key_ready_at[(endpoint.baseUrl, key)] for key in keys) - now)

    def _record_rate_limit(self, endpoint: Endpoint, headers: dict, response: httpx.Response, attempt: int) -> None:
        self._emit(self.rate_limit_events)
        self._note_key(headers, "rate_limited")
        if len(endpoint.bearer_keys()) > 1:
            key = headers["Authorization"][len("Bearer "):]
            self._key_ready_at[(endpoint.baseUrl, key)] = time.monotonic() + backoff(429, attempt, response.headers)

    async def _retry_delay(self, endpoint: Endpoint, response: httpx.Response, attempt: int) -> None:
        if response.status_code != 429 or len(endpoint.bearer_keys()) <= 1:
            await asyncio.sleep(backoff(response.status_code, attempt, response.headers))

    # --- plumbing ---------------------------------------------------------------------------------------------------
    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=_timeout())
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    @staticmethod
    def _model_cfg(params: Params) -> EndpointAndModel:
        return config().resolve(params.config or config().llm)

    def make_query(self, params: Params, messages: Session, stream: bool) -> dict:
        mc = self._model_cfg(params)
        msgs = list(messages)
        msgs.insert(0, Message(role=ROLE_SYSTEM, content=params.system_prompt))
        c = config()
        body: dict[str, Any] = {
            "messages": Session(msgs).to_json(),
            "max_tokens": params.max_output_tokens,  # hopefully helps with stuck prediction (infinite reasoning)
            "stream": stream,
            "model": mc.model,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if params.tools:
            body["tools"] = params.tools
        if mc.reasoningEffort:
            body["reasoning_effort"] = mc.reasoningEffort
        for key, value in (
            ("temperature", c.llmTemperature),
            ("top_p", c.llmTopP),
            ("top_k", c.llmTopK),
            ("min_p", c.llmMinP),
            ("presence_penalty", c.llmPresencePenalty),
            ("repeat_penalty", c.llmRepetitionPenalty),
        ):
            if value is not None:
                body[key] = value
        if params.seed is not None:
            body["seed"] = params.seed
        return body

    @staticmethod
    def _dump(name: str, text: str) -> None:
        if not config().logLlmRequests:
            return
        try:
            Path("logs").mkdir(exist_ok=True)
            (Path("logs") / name).write_text(text, encoding="utf-8")
        except OSError:
            pass

    # --- chat -------------------------------------------------------------------------------------------------------
    async def chat(self, params: Params, messages: Session) -> Response:
        s = self.chat_streaming(params, messages)
        await s.completed
        return s.response

    def chat_streaming(self, params: Params, messages: Session) -> StreamingResponse:
        result = StreamingResponse()
        printer = TuiStreamingPrinter()
        result._listeners.append(printer.update)
        result.completed = asyncio.create_task(self._run_streaming(result, params, messages, printer))
        return result

    async def _run_streaming(self, result: StreamingResponse, params: Params, messages: Session,
                             printer: TuiStreamingPrinter) -> None:
        mc = self._model_cfg(params)
        body = self.make_query(params, messages, stream=True)
        query = json.dumps(body, ensure_ascii=False)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%f")
        self._dump(f"{stamp}.0query.json", query)
        if config().logLlmRequests:
            try:
                Path("last_query.json").write_text(query, encoding="utf-8")
            except OSError:
                pass
        url = mc.endpoint.baseUrl + "chat/completions"

        attempt = 0
        while True:
            attempt += 1
            got_data = False
            headers = await self._request_headers(mc.endpoint, messages.session_id)
            started = time.monotonic()
            try:
                async with self.http.stream("POST", url, content=query.encode("utf-8"), headers=headers) as resp:
                    if resp.status_code != 200:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        self._emit_request(resp.status_code, started, headers)
                        if resp.status_code == 429:
                            self._record_rate_limit(mc.endpoint, headers, resp, attempt)
                        if MODEL_LOAD_ERROR in text:
                            log.warning("Ollama model failed to load, wait and retry...")
                            await asyncio.sleep(30)
                            continue
                        if resp.status_code in RETRY_STATUSES and attempt <= max_retries(resp.status_code):
                            log.warning("chatStreaming: HTTP %s, retry #%d: %s", resp.status_code, attempt, text[:200])
                            await self._retry_delay(mc.endpoint, resp, attempt)
                            continue
                        raise LLMError(f"HTTP {resp.status_code}: {text[:2000]}")
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                        elif line.startswith("{"):
                            payload = line  # some servers don't use SSE framing for errors
                        else:
                            continue  # SSE comments / keep-alive
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue  # partial; not expected with line-based SSE
                        if "error" in chunk and "choices" not in chunk:
                            raise LLMError(f"LLM error: {json.dumps(chunk['error'], ensure_ascii=False)}")
                        got_data = True
                        self._apply_chunk(result, chunk)
                break
            except (httpx.TransportError, httpx.TimeoutException) as e:
                self._emit_request("error", started, headers)
                if got_data or attempt > 3:
                    raise LLMError(f"connection failed: {e!r}") from e
                log.warning("chatStreaming: %r, retry #%d", e, attempt)
                await asyncio.sleep(2 ** attempt)

        if not result.response.choices:
            raise LLMError("LLM returned an empty stream")
        printer.finish()
        self._emit_request(200, started)
        self._note_key(headers, "ok")
        self._emit(self.success_events)
        self._dump(f"{stamp}.1response.json", json.dumps(result.response.to_json(), ensure_ascii=False))
        for cb in self.response_metrics:
            try:
                cb(mc.model, result.response.usage)
            except Exception:  # noqa: BLE001
                log.exception("metrics callback failed")

    @staticmethod
    def _apply_chunk(result: StreamingResponse, chunk: dict) -> None:
        collect_chunk(result.response, chunk)
        result._changed()

    # --- raw JSON request (used by the proxy's context bridge) ------------------------------------------------------
    async def post_json(self, endpoint: Endpoint, path: str, body: dict, session_id: str | None = None) -> dict:
        url = endpoint.baseUrl + path
        for attempt in range(1, 7):
            headers = await self._request_headers(endpoint, session_id)
            started = time.monotonic()
            try:
                resp = await self.http.post(url, content=json.dumps(body, ensure_ascii=False).encode(),
                                             headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                self._emit_request("error", started, headers)
                if attempt >= 4:
                    raise LLMError(f"connection failed: {e!r}") from e
                await asyncio.sleep(2 ** attempt)
                continue
            self._emit_request(resp.status_code, started, headers)
            if resp.status_code == 429:
                self._record_rate_limit(endpoint, headers, resp, attempt)
            if resp.status_code in RETRY_STATUSES and attempt <= max_retries(resp.status_code):
                await self._retry_delay(endpoint, resp, attempt)
                continue
            if not 200 <= resp.status_code < 300:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:2000]}")
            try:
                data = resp.json()
            except ValueError:
                raise LLMError("LLM returned invalid JSON") from None
            if not isinstance(data, dict):
                raise LLMError("LLM returned a non-object JSON response")
            if isinstance(data, dict) and "error" in data:
                msg = json.dumps(data["error"], ensure_ascii=False)
                if MODEL_LOAD_ERROR in msg and attempt < 6:
                    await asyncio.sleep(30)
                    continue
                raise LLMError(f"LLM error: {msg}")
            self._note_key(headers, "ok")
            self._emit(self.success_events)
            return data
        raise LLMError("unreachable")

    # --- embeddings (optional; only the anti-repeat check may use them) --------------------------------------------
    async def embedding(self, mc: EndpointAndModel, text: str) -> list[float]:
        mc = config().resolve(mc)
        data = await self.post_json(mc.endpoint, "embeddings", {"model": mc.model, "input": text or " "})
        return [float(x) for x in data["data"][0]["embedding"]]

    # --- speech to text ---------------------------------------------------------------------------------------------
    async def transcribe_audio(self, audio: bytes, fmt: str) -> Transcription:
        mc = config().llmAudioToText
        headers = {}
        if mc.endpoint.bearerKey:
            headers["Authorization"] = f"Bearer {mc.endpoint.bearerKey}"
        resp = await self.http.post(
            mc.endpoint.baseUrl + "audio/transcriptions",
            data={"model": mc.model, "response_format": "verbose_json"},
            files={"file": (f"audio.{fmt}", audio, f"audio/{fmt}")},
            headers=headers,
        )
        if resp.status_code != 200:
            raise LLMError(f"transcribeAudio: HTTP {resp.status_code}: {resp.text[:500]}")
        d = resp.json()
        return Transcription(
            text=d.get("text", ""),
            language=d.get("language", ""),
            duration=float(d.get("duration") or 0),
            segments=d.get("segments") or [],
        )
