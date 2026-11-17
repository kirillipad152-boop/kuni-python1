"""Metrics breadcrumbs + Prometheus exporter. Port of MetricsBreadcumbs / Prometheus.cpp.

Breadcrumbs live in a ContextVar, so concurrent workers don't clobber each other's labels.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar

log = logging.getLogger("Prometheus")

_crumbs: ContextVar[dict] = ContextVar("kuni_breadcrumbs", default={})

LABEL_KEYS = ("function", "chat", "scenario")


class MetricsBreadcrumbs:
    def value(self) -> dict[str, str]:
        return dict(_crumbs.get())

    @contextmanager
    def point(self, key: str, value: str):
        token = _crumbs.set({**_crumbs.get(), key: value})
        try:
            yield
        finally:
            _crumbs.reset(token)

    def set(self, key: str, value: str) -> None:
        """Sticky breadcrumb: lives until `clear` (or until the current asyncio task ends)."""
        _crumbs.set({**_crumbs.get(), key: value})

    def clear(self, *keys: str) -> None:
        _crumbs.set({k: v for k, v in _crumbs.get().items() if k not in keys})


# kept for compatibility with the C++ naming
BUCKETS = (5, 10, 15, 20, 30, 40, 50, 60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400, 259200, 604800,
           1209600, 2592000)


class Exporter:
    """Starts a /metrics HTTP server and wires LLM usage + tool call metrics."""

    def __init__(self, breadcrumbs: MetricsBreadcrumbs, host: str = "127.0.0.1", port: int = 9464) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

        self.breadcrumbs = breadcrumbs
        self.registry = CollectorRegistry()
        labels = ["model", *LABEL_KEYS]
        self._in = Counter("llm_usage_input", "OpenAI endpoint token usage (input tokens)", labels, registry=self.registry)
        self._hit = Counter("llm_usage_input_cache_hit", "input tokens that were cached; cheap", labels, registry=self.registry)
        self._miss = Counter("llm_usage_input_cache_miss", "input tokens that weren't; expensive", labels, registry=self.registry)
        self._out = Counter("llm_usage_output", "output tokens; expensive", labels, registry=self.registry)
        g = ["model"]
        self._g_in = Gauge("llm_usage_input_gauge", "last query input tokens", g, registry=self.registry)
        self._g_hit = Gauge("llm_usage_input_cache_hit_gauge", "last query cached input tokens", g, registry=self.registry)
        self._g_miss = Gauge("llm_usage_input_cache_miss_gauge", "last query uncached input tokens", g, registry=self.registry)
        self._g_out = Gauge("llm_usage_output_gauge", "last query output tokens", g, registry=self.registry)
        tl = [*LABEL_KEYS, "tool"]
        self._tool = Counter("tool_call_total", "Number of tool calls dispatched by the LLM", tl, registry=self.registry)
        self._rt_h = Histogram("tool_call_response_time_seconds",
                               "Time between the last message in the chat and the bot's tool call dispatch", tl,
                               buckets=BUCKETS, registry=self.registry)
        self._rt_g = Gauge("tool_call_response_time_seconds_gauge", "same, as a gauge", tl, registry=self.registry)
        self._server = start_http_server(port, addr=host, registry=self.registry)
        log.info("Prometheus metrics on http://%s:%d/metrics", host, port)

    def _labels(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        crumbs = self.breadcrumbs.value()
        d = {k: str(crumbs.get(k, "")) for k in LABEL_KEYS}
        if extra:
            d.update(extra)
        return d

    def register_openai(self, chat) -> None:
        def on_usage(model: str, u) -> None:
            base = self._labels({"model": model})
            self._in.labels(**base).inc(u.prompt_tokens)
            self._hit.labels(**base).inc(u.prompt_cache_hit_tokens)
            self._miss.labels(**base).inc(u.prompt_cache_miss_tokens)
            self._out.labels(**base).inc(u.completion_tokens)
            self._g_in.labels(model).set(u.prompt_tokens)
            self._g_hit.labels(model).set(u.prompt_cache_hit_tokens)
            self._g_miss.labels(model).set(u.prompt_cache_miss_tokens)
            self._g_out.labels(model).set(u.completion_tokens)

        chat.response_metrics.append(on_usage)

    def register_app(self, app) -> None:
        def on_tool_call(ev) -> None:
            labels = {k: str(ev.breadcrumb_labels.get(k, "")) for k in LABEL_KEYS}
            labels["tool"] = ev.tool_name
            self._tool.labels(**labels).inc()
            if ev.last_opened_chat_last_message_time is not None:
                v = float(ev.last_opened_chat_last_message_time)
                self._rt_h.labels(**labels).observe(v)
                self._rt_g.labels(**labels).set(v)

        app.tool_call_fired.append(on_tool_call)

    def close(self) -> None:
        try:
            self._server[0].shutdown()
        except Exception:  # noqa: BLE001
            pass
