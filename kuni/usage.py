"""Hourly LLM usage history (requests, 429s, errors, latency, tokens) kept on disk for 14 days, for the panel charts."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .util import atomic_write_text

log = logging.getLogger("usage")
KEEP_HOURS = 14 * 24


class UsageStats:
    def __init__(self, path: Path, clock=time.time) -> None:
        self.path, self._clock = Path(path), clock
        self._hours: dict[str, dict] = {}
        self._dirty = False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._hours = {k: v for k, v in data.items() if isinstance(v, dict)}
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            log.warning("usage history is unreadable; starting empty")

    def _bucket(self) -> dict:
        key = time.strftime("%Y-%m-%dT%H", time.gmtime(self._clock()))
        self._dirty = True
        return self._hours.setdefault(key, {"req": 0, "ok": 0, "r429": 0, "err": 0, "ms": 0, "prompt": 0, "completion": 0})

    def request(self, status, seconds: float) -> None:
        b = self._bucket()
        b["req"] += 1
        if status == 200:
            b["ok"] += 1
            b["ms"] += int(seconds * 1000)
        elif status == 429:
            b["r429"] += 1
        else:
            b["err"] += 1

    def tokens(self, _model: str, usage) -> None:
        b = self._bucket()
        b["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        b["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)

    def series(self, hours: int) -> list[dict]:
        now = self._clock()
        out = []
        for i in range(hours - 1, -1, -1):
            key = time.strftime("%Y-%m-%dT%H", time.gmtime(now - i * 3600))
            b = self._hours.get(key, {})
            ok = b.get("ok", 0)
            out.append({"hour": key, "req": b.get("req", 0), "ok": ok, "r429": b.get("r429", 0), "err": b.get("err", 0),
                        "avg_ms": b.get("ms", 0) // ok if ok else None, "prompt": b.get("prompt", 0),
                        "completion": b.get("completion", 0)})
        return out

    def flush(self) -> None:
        if not self._dirty:
            return
        cutoff = time.strftime("%Y-%m-%dT%H", time.gmtime(self._clock() - KEEP_HOURS * 3600))
        self._hours = {k: v for k, v in self._hours.items() if k >= cutoff}
        try:
            atomic_write_text(self.path, json.dumps(self._hours))
            self._dirty = False
        except OSError:
            log.warning("could not save the usage history")
