"""Owner-only control panel. Binds to 127.0.0.1 only: reach it through an SSH tunnel, never expose the port.

Defence in depth on top of that: a random access token, a session cookie (HttpOnly, SameSite=Strict), Host/Origin
checks against DNS rebinding and cross-site requests, a custom header on every state-changing call, a strict CSP,
and a lockout after repeated bad tokens. The panel only changes whitelisted, non-secret settings and never prints keys.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import shutil
import signal
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from aiohttp import web

from . import KERNEL_NAME, __version__
from . import config as cfgmod
from .config import config, key_fingerprint
from .telegram.types import ChatType
from .trace import SECRETS, TurnTraces, clip  # noqa: F401
from .usage import UsageStats
from .util import atomic_write_text

log = logging.getLogger("WebPanel")

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SESSION_SECONDS = 12 * 3600
MAX_SESSIONS = 20
LOGIN_WINDOW = 120.0
LOGIN_MAX_FAILURES = 5
MAX_MESSAGE = 4000
WEBUI = Path(__file__).with_name("webui")
STATIC = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/app.css": ("app.css", "text/css")}
MAX_WORKING_MEMORY = 20_000
BACKUP_MIN_GAP = 600

LABELS = {
    "paused": "Пауза (ответы и инициатива остановлены)",
    "lockdown": "Кто может с ней общаться",
    "chatNotificationFilter": "Фильтр уведомлений",
    "channelNotifications": "Посты каналов будят её",
    "capabilityWebReader": "Чтение веб-страниц",
    "capabilityWebSearch": "Поиск в интернете",
    "capabilityReminders": "Напоминания",
    "capabilityYouTube": "YouTube",
    "canJoinChats": "Может вступать в каналы и группы",
    "canLeaveChats": "Может выходить из групп",
    "canWriteToANewPerson": "Может писать новым людям",
    "wakeUpOnPinnedChat": "Закреплённые чаты будят её",
    "randomlyGoSleep": "Случайный «сон»",
    "checkChatsOnStartup": "Проверять непрочитанные при старте",
    "autoSleepConsolidation": "Ночная консолидация памяти",
    "remindUseAsk": "Напоминать про #ask",
    "watchdogEnabled": "Сторож (уведомления о проблемах)",
    "logLlmRequests": "Дамп запросов к LLM в logs/ (с перепиской!)",
    "workerCount": "Число воркеров (1–4)",
    "suggestIgnoreChance": "Шанс «лени» (0–0.98)",
}


class LogBuffer(logging.Handler):
    """Bounded in-memory tail of the log for the panel; keys/bearer tokens are masked, lines are truncated."""

    def __init__(self, size: int = 1500) -> None:
        super().__init__(logging.INFO)
        self.entries: deque[dict] = deque(maxlen=size)
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = SECRETS.sub("[hidden]", record.getMessage())[:600]
        except Exception:  # noqa: BLE001
            return
        self._seq += 1
        self.entries.append({"seq": self._seq, "t": time.strftime("%H:%M:%S", time.gmtime(record.created)),
                             "level": record.levelname, "name": record.name, "text": text})

    def since(self, after: int, kind: str, limit: int = 300) -> list[dict]:
        def wanted(e: dict) -> bool:
            if kind == "flow":
                return e["name"] == "MessageFlow" or e["name"] == "WebPanel"
            if kind == "warn":
                return e["level"] in ("WARNING", "ERROR", "CRITICAL")
            return True

        return [e for e in self.entries if e["seq"] > after and wanted(e)][-limit:]


def load_or_create_token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
        if len(token) >= 32:
            return token
    except FileNotFoundError:
        pass
    token = secrets.token_urlsafe(32)
    atomic_write_text(path, token + "\n")  # mkstemp creates it 0600
    path.chmod(0o600)
    return token


def _fail(cls, message: str, **kw) -> web.HTTPException:
    return cls(text=json.dumps({"error": message}, ensure_ascii=False), content_type="application/json", **kw)


def _json(data: dict, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def _is_chat_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value != 0 and abs(value) < 2 ** 63


class WebPanel:
    def __init__(self, app, *, port: int | None = None, token_path: Path | None = None, clock=time.monotonic) -> None:
        self.app = app
        self.port = port if port is not None else config().webPanelPort
        self._token = load_or_create_token(token_path or app.working_dir / "web_panel_token")
        self._clock = clock
        self._sessions: dict[str, float] = {}
        self._failures: deque[float] = deque()
        self._logs = LogBuffer()
        self._titles: dict[int, tuple[str, float]] = {}
        self._rate_limits: deque[float] = deque()
        self._last_success: float | None = None
        self._runner: web.AppRunner | None = None
        self._usage = UsageStats(app.working_dir / "usage.json")
        self._tasks: set[asyncio.Task] = set()
        self._jobs: dict[str, asyncio.Task] = {}
        self._job_result: dict[str, str] = {}
        self._channels_cache: tuple[float, list] | None = None
        self._static = {path: ((WEBUI / name).read_bytes(), ctype) for path, (name, ctype) in STATIC.items()}

    # ---------------------------------------------------------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        logging.getLogger().addHandler(self._logs)
        self.app.openai.rate_limit_events.append(self._on_rate_limit)
        self.app.openai.success_events.append(self._on_success)
        self.app.openai.request_events.append(self._usage.request)
        self.app.openai.response_metrics.append(self._usage.tokens)
        self._tasks.add(asyncio.create_task(self._flush_loop()))
        self._runner = web.AppRunner(self.make_app(), access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()
        log.info("web panel listens on 127.0.0.1:%d (local only)", self.port)

    async def stop(self) -> None:
        logging.getLogger().removeHandler(self._logs)
        for events, cb in ((self.app.openai.rate_limit_events, self._on_rate_limit),
                           (self.app.openai.success_events, self._on_success),
                           (self.app.openai.request_events, self._usage.request),
                           (self.app.openai.response_metrics, self._usage.tokens)):
            if cb in events:
                events.remove(cb)
        for task in [*self._tasks, *self._jobs.values()]:
            task.cancel()
        await asyncio.gather(*self._tasks, *self._jobs.values(), return_exceptions=True)
        self._usage.flush()
        if self._runner:
            await self._runner.cleanup()

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._usage.flush()

    def _alert(self, text: str) -> None:
        """Tell the owner in Telegram (no LLM); a panel that nobody is watching must not be silently used."""
        async def send() -> None:
            try:
                await self.app._notify_owner(text)
            except Exception:  # noqa: BLE001
                log.warning("could not send the panel alert")

        task = asyncio.ensure_future(send())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_rate_limit(self) -> None:
        self._rate_limits.append(time.time())

    def _on_success(self) -> None:
        self._last_success = time.time()

    def make_app(self) -> web.Application:
        application = web.Application(middlewares=[self._guard], client_max_size=64 * 1024)
        for path in STATIC:
            application.router.add_get(path, self._serve)
        application.router.add_post("/api/login", self._login)
        application.router.add_post("/api/logout", self._logout)
        application.router.add_get("/api/state", self._state)
        application.router.add_get("/api/logs", self._logs_handler)
        application.router.add_post("/api/action", self._action)
        application.router.add_get("/api/traces", self._traces)
        application.router.add_get("/api/usage", self._usage_handler)
        application.router.add_get("/api/channels", self._channels)
        application.router.add_get("/api/memory", self._memory)
        application.router.add_get("/api/working_memory", self._working_memory)
        application.router.add_get("/api/backups", self._backups)
        return application

    # ---------------------------------------------------------------------------------------------------------
    # security
    # ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _hostname(value: str | None) -> str | None:
        try:
            return urlsplit("//" + value).hostname if value is not None else None
        except ValueError:
            return None

    def _check_request(self, request: web.Request) -> None:
        extra = {h.lower() for h in config().webPanelExtraHosts}
        allowed = LOCAL_HOSTS | extra
        host = self._hostname(request.headers.get("Host"))
        if host not in allowed:  # DNS rebinding
            raise _fail(web.HTTPForbidden, "forbidden host")
        login = config().webPanelTailscaleLogin
        if host in extra and login and request.headers.get("Tailscale-User-Login", "").lower() != login.lower():
            # Tailscale Serve sets this header for tailnet users; it narrows the tailnet to one person.
            raise _fail(web.HTTPForbidden, "forbidden identity")
        origin = request.headers.get("Origin")
        if origin is not None:
            try:
                origin_host = urlsplit(origin).hostname
            except ValueError:
                origin_host = None
            if origin_host not in allowed:
                raise _fail(web.HTTPForbidden, "forbidden origin")
        if request.method not in ("GET", "HEAD"):
            # A custom header and JSON body cannot be sent by a cross-site form, and CORS preflight is never granted.
            if request.headers.get("X-Kuni-Panel") != "1" or request.content_type != "application/json":
                raise _fail(web.HTTPForbidden, "forbidden request")

    def _session_ok(self, request: web.Request) -> bool:
        sid = request.cookies.get("kuni_session")
        expires = self._sessions.get(sid) if sid else None
        if expires is None or expires < self._clock():
            self._sessions.pop(sid, None)
            return False
        return True

    @web.middleware
    async def _guard(self, request: web.Request, handler):
        try:
            self._check_request(request)
            if request.path.startswith("/api/") and request.path != "/api/login" and not self._session_ok(request):
                raise _fail(web.HTTPUnauthorized, "sign in required")
            response = await handler(request)
        except web.HTTPException as exc:
            self._harden(exc)
            raise
        self._harden(response)
        return response

    @staticmethod
    def _harden(response) -> None:
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                                       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"})

    async def _serve(self, request: web.Request) -> web.Response:
        body, ctype = self._static[request.path]
        return web.Response(body=body, content_type=ctype, charset="utf-8")

    async def _body(self, request: web.Request) -> dict:
        try:
            data = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise _fail(web.HTTPBadRequest, "invalid JSON") from None
        if not isinstance(data, dict):
            raise _fail(web.HTTPBadRequest, "JSON object expected")
        return data

    async def _login(self, request: web.Request) -> web.Response:
        now = self._clock()
        while self._failures and now - self._failures[0] > LOGIN_WINDOW:
            self._failures.popleft()
        if len(self._failures) >= LOGIN_MAX_FAILURES:
            raise _fail(web.HTTPTooManyRequests, "too many wrong tokens; wait a couple of minutes",
                        headers={"Retry-After": str(int(LOGIN_WINDOW))})
        token = (await self._body(request)).get("token")
        if not isinstance(token, str) or not hmac.compare_digest(token.encode(), self._token.encode()):
            self._failures.append(now)
            log.warning("panel login failed")
            if len(self._failures) == LOGIN_MAX_FAILURES:
                self._alert(f"В веб-панель {LOGIN_MAX_FAILURES} раз подряд вводили неверный токен; вход заблокирован на пару минут.")
            raise _fail(web.HTTPUnauthorized, "wrong token")
        self._sessions = {s: e for s, e in self._sessions.items() if e > now}
        while len(self._sessions) >= MAX_SESSIONS:
            self._sessions.pop(next(iter(self._sessions)))
        sid = secrets.token_urlsafe(32)
        self._sessions[sid] = now + SESSION_SECONDS
        log.info("panel login ok")
        self._alert("Вход в веб-панель. Если это были не вы: удалите data/web_panel_token на сервере и перезапустите Куни.")
        response = _json({"ok": True})
        response.set_cookie("kuni_session", sid, max_age=SESSION_SECONDS, httponly=True, samesite="Strict", path="/",
                            secure=request.headers.get("X-Forwarded-Proto") == "https")
        return response

    async def _logout(self, request: web.Request) -> web.Response:
        self._sessions.pop(request.cookies.get("kuni_session", ""), None)
        response = _json({"ok": True})
        response.del_cookie("kuni_session", path="/")
        return response

    # ---------------------------------------------------------------------------------------------------------
    # read
    # ---------------------------------------------------------------------------------------------------------
    async def _title(self, chat_id: int | None) -> str:
        if not chat_id:
            return ""
        cached = self._titles.get(chat_id)
        if cached and cached[1] > time.monotonic():
            return cached[0]
        try:
            title = (await asyncio.wait_for(self.app.telegram.get_chat(chat_id), 3)).title[:80]
        except Exception:  # noqa: BLE001
            title = ""
        self._titles[chat_id] = (title, time.monotonic() + 600)
        return title

    @staticmethod
    def _rss_mb() -> int | None:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    async def _state(self, request: web.Request) -> web.Response:
        app, c, now = self.app, config(), time.monotonic()
        wall = time.time()
        while self._rate_limits and wall - self._rate_limits[0] > 600:
            self._rate_limits.popleft()
        queue = []
        for handle in list(app.notification_manager._notifications)[:40]:
            n = handle.notification
            queue.append({"chat_id": n.chat_id, "title": await self._title(n.chat_id), "priority": n.priority,
                          "age": int(now - n.queued_at),
                          "kind": "инициатива" if n.pin == "<act_proactively />" else "чат"})
        workers = []
        for w in app.workers:
            n = w.current_notification
            workers.append({"id": w.name, "busy": w._active_task is not None, "sleeping": w.sleeping,
                            "chat_id": n.chat_id if n else None, "title": await self._title(n.chat_id if n else None),
                            "age": int(now - n.queued_at) if n else None})
        reminders = [{"id": r.id, "chat_id": r.chat_id, "title": await self._title(r.chat_id),
                      "due": datetime.fromtimestamp(r.due, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                      "note": r.note[:200], "attempts": r.attempts} for r in app.reminders.all()[:60]]
        endpoint = c.resolved_llm().endpoint
        cooling = sum(1 for (base, _), ready in app.openai._key_ready_at.items() if base == endpoint.baseUrl and ready > now)
        disk = shutil.disk_usage(app.working_dir)
        settings = []
        for attr, spec in cfgmod.EDITABLE_SETTINGS.items():
            if spec[0] == "ids":
                continue  # chat lists have their own section
            value = getattr(c, attr)
            item = {"key": attr, "label": LABELS.get(attr, attr), "type": spec[0],
                    "value": value.value if hasattr(value, "value") else value}
            if spec[0] == "enum":
                item["options"] = spec[1]
            elif spec[0] in ("int", "float"):
                item["min"], item["max"] = spec[1], spec[2]
            settings.append(item)
        lists = {name: [{"chat_id": i, "title": await self._title(i)} for i in getattr(c, attr)]
                 for name, attr in (("allowed", "allowedChatIds"), ("blocked", "blockedChatIds"))}
        known = {c.papikChatId, *c.allowedChatIds, *c.blockedChatIds}
        denied = [{"chat_id": i, "title": await self._title(i), "count": e["count"], "ago": int(wall - e["last"])}
                  for i, e in reversed(list(getattr(app, "denied_chats", {}).items())) if i not in known][:10]
        return _json({
            "keys": self._keys_view(), "env_key_override": bool(os.environ.get("KUNI_API_KEY") or os.environ.get("KUNI_BASE_URL")),
            "lists": lists, "denied": denied, "backup_dir_set": bool(c.backupDir),
            "jobs": {"running": sorted(n for n, t in self._jobs.items() if not t.done()), "last": self._job_result},
            "version": f"{__version__} ({KERNEL_NAME})", "paused": c.paused, "lockdown": c.lockdown.value,
            "model": c.llm.model, "uptime": int(now - app._started_at), "memory": app.memory.stats(),
            "contexts": len(app.contexts), "stopping": app._stopping, "can_restart": "INVOCATION_ID" in os.environ,
            "llm": {"rate_limits_10min": len(self._rate_limits), "keys": len(endpoint.bearer_keys()), "keys_cooling": cooling,
                    "last_success_ago": int(wall - self._last_success) if self._last_success else None},
            "alerts": sorted(app.watchdog._active), "disk_free_percent": round(disk.free / disk.total * 100),
            "rss_mb": self._rss_mb(), "workers": workers, "queue": queue, "reminders": reminders, "settings": settings})

    def _keys_view(self) -> list[dict]:
        endpoint, now, wall = config().resolved_llm().endpoint, time.monotonic(), time.time()
        out = []
        for key in endpoint.bearer_keys():
            stat = self.app.openai.key_stats.get(key_fingerprint(key), {})
            ready = self.app.openai._key_ready_at.get((endpoint.baseUrl, key), 0)
            out.append({"id": key_fingerprint(key), "hint": "…" + key[-4:], "primary": key == endpoint.bearerKey,
                        "requests": stat.get("requests", 0), "ok": stat.get("ok", 0),
                        "rate_limited": stat.get("rate_limited", 0), "errors": stat.get("errors", 0),
                        "last_used_ago": int(wall - stat["last_used"]) if stat.get("last_used") else None,
                        "cooling": max(0, int(ready - now))})
        return out

    @staticmethod
    def _int_query(request: web.Request, name: str, default: int | None = None) -> int:
        try:
            return int(request.query[name]) if name in request.query else default
        except (ValueError, TypeError):
            raise _fail(web.HTTPBadRequest, f"{name} must be an integer") from None

    async def _traces(self, request: web.Request) -> web.Response:
        traces: TurnTraces = self.app.traces
        if "chat_id" in request.query:
            chat_id = self._int_query(request, "chat_id")
            return _json({"chat_id": chat_id, "title": await self._title(chat_id), "turns": traces.turns(chat_id)})
        chats = traces.summary()
        for item in chats:
            item["title"] = await self._title(item["chat_id"])
        return _json({"chats": chats})

    async def _usage_handler(self, request: web.Request) -> web.Response:
        hours = min(max(self._int_query(request, "hours", 48), 1), 336)
        series = self._usage.series(hours)
        totals = {k: sum(h[k] for h in series) for k in ("req", "ok", "r429", "err", "prompt", "completion")}
        return _json({"series": series, "totals": totals})

    async def _channels(self, request: web.Request) -> web.Response:
        now = time.monotonic()
        if self._channels_cache is None or self._channels_cache[0] < now:
            try:
                chats = await asyncio.wait_for(self.app.telegram.get_chats(100), 20)
            except Exception as exc:  # noqa: BLE001
                raise _fail(web.HTTPBadGateway, f"Telegram: {type(exc).__name__}") from None
            self._channels_cache = (now + 20, [c for c in chats if c.type == ChatType.CHANNEL])
        blocked = set(config().blockedChatIds)
        return _json({"channels": [{"chat_id": c.id, "title": c.title[:80], "unread": c.unread_count,
                                    "blocked": c.id in blocked} for c in self._channels_cache[1]]})

    async def _memory(self, request: web.Request) -> web.Response:
        memory = self.app.memory
        if "id" in request.query:
            node = memory.nodes.get(request.query["id"])
            if node is None:
                raise _fail(web.HTTPNotFound, "node not found")
            edges = []
            for e in memory.edges.values():
                if node.id in (e.src, e.dst) and len(edges) < 60:
                    other = memory.nodes.get(e.dst if e.src == node.id else e.src)
                    edges.append({"id": e.id, "relation": e.relation, "outgoing": e.src == node.id,
                                  "other": other.name if other else "?", "fact": clip(e.fact, 300),
                                  "confidence": round(e.confidence, 2)})
            return _json({"id": node.id, "type": node.type, "name": node.name, "aliases": node.aliases[:10],
                          "summary": clip(node.summary, 4000), "confidence": round(node.confidence, 2),
                          "visibility": node.visibility, "source_chat_id": node.source_chat_id, "edges": edges})
        query = request.query.get("q", "").strip().lower()[:100]
        nodes = [n for n in memory.nodes.values()
                 if not query or query in n.name.lower() or any(query in a.lower() for a in n.aliases)
                 or query in n.summary.lower()]
        nodes.sort(key=lambda n: n.updated, reverse=True)
        return _json({"total": len(memory.nodes), "found": len(nodes), "nodes": [
            {"id": n.id, "type": n.type, "name": clip(n.name, 100), "summary": clip(n.summary, 240),
             "confidence": round(n.confidence, 2), "visibility": n.visibility, "source_chat_id": n.source_chat_id}
            for n in nodes[:40]]})

    async def _working_memory(self, request: web.Request) -> web.Response:
        app, owner = self.app, config().papikChatId
        if "chat_id" in request.query:
            chat_id = self._int_query(request, "chat_id")
            path = app.working_memory_path(chat_id)
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            return _json({"chat_id": chat_id, "title": await self._title(chat_id), "text": text})
        chat_ids = [owner]
        for path in sorted((app.working_dir / "working_memory").glob("*.md")):
            try:
                chat_ids.append(int(path.stem))
            except ValueError:
                continue
        return _json({"chats": [{"chat_id": i, "title": await self._title(i),
                                 "size": app.working_memory_path(i).stat().st_size if app.working_memory_path(i).is_file() else 0}
                                for i in chat_ids]})

    async def _backups(self, request: web.Request) -> web.Response:
        from .backup import _archives

        directory = config().backupDir
        if not directory:
            return _json({"configured": False, "archives": []})
        path, now = Path(directory), time.time()
        try:
            archives = _archives(path) if path.is_dir() else []
        except OSError:
            archives = []
        return _json({"configured": True, "archives": [
            {"name": a.name, "size": a.stat().st_size, "age": int(now - a.stat().st_mtime)} for a in reversed(archives[-20:])]})

    async def _test_key(self, fingerprint) -> web.Response:
        mc = config().resolved_llm()
        key = next((k for k in mc.endpoint.bearer_keys() if key_fingerprint(k) == fingerprint), None)
        if key is None:
            raise ValueError("ключ не найден")
        started = time.monotonic()
        try:
            response = await self.app.openai.http.post(
                mc.endpoint.baseUrl + "chat/completions", timeout=40,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json={"model": mc.model, "max_tokens": 8, "messages": [{"role": "user", "content": "ok"}]})
        except httpx.HTTPError as exc:
            raise ValueError(f"нет ответа от API: {type(exc).__name__}") from None
        return _json({"message": f"Ключ …{key[-4:]}: HTTP {response.status_code} за {time.monotonic() - started:.1f} с"})

    def _start_job(self, name: str) -> web.Response:
        if name in self._jobs and not self._jobs[name].done():
            raise ValueError("это уже выполняется")

        async def run() -> None:
            try:
                if name == "memory_dump":
                    await self.app.diary_dump_messages()
                    self._job_result[name] = "память сохранена"
                else:
                    from .memory.consolidate import sleeping_consolidation

                    report = await sleeping_consolidation(self.app.memory, self.app.openai, cfgmod.SLEEP_MAX_TIME_SECS)
                    self._job_result[name] = "сон завершён: " + clip(report, 200)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("panel job %s failed: %s", name, type(exc).__name__)
                self._job_result[name] = f"ошибка: {type(exc).__name__}"

        self._jobs[name] = asyncio.create_task(run())
        return _json({"message": "Запущено в фоне; результат появится в статусе."})

    async def _backup_now(self) -> web.Response:
        from .backup import _archives, create_backup

        directory = config().backupDir
        if not directory:
            raise ValueError("misc.backup_dir не задан")
        path = Path(directory)
        newest = max((a.stat().st_mtime for a in _archives(path)), default=0) if path.is_dir() else 0
        if time.time() - newest < BACKUP_MIN_GAP:
            raise ValueError("последняя копия моложе 10 минут")  # a click spree must not push real history out
        try:
            archive = await asyncio.get_running_loop().run_in_executor(None, create_backup, Path.cwd(), path, True)
        except BlockingIOError:
            raise ValueError("копия уже создаётся") from None
        return _json({"message": f"Копия создана: {archive.name} ({archive.stat().st_size // 1024} КБ). "
                                 "На компьютер она попадёт при ближайшей выгрузке."})

    async def _logs_handler(self, request: web.Request) -> web.Response:
        kind = request.query.get("kind", "flow")
        try:
            after = int(request.query.get("after", "0"))
        except ValueError:
            raise _fail(web.HTTPBadRequest, "after must be an integer") from None
        if kind not in ("flow", "warn", "all"):
            raise _fail(web.HTTPBadRequest, "unknown log kind")
        return _json({"entries": self._logs.since(after, kind)})

    # ---------------------------------------------------------------------------------------------------------
    # control
    # ---------------------------------------------------------------------------------------------------------
    async def _action(self, request: web.Request) -> web.Response:
        body = await self._body(request)
        name = body.get("action")
        app = self.app
        log.info("panel action=%s", name if isinstance(name, str) else "?")
        try:
            if name in ("pause", "resume"):
                cfgmod.update_settings({"paused": name == "pause"})
                return _json({"message": "Пауза включена." if name == "pause" else "Работа продолжена."})
            if name == "set":
                key = body.get("key")
                if not isinstance(key, str):
                    raise ValueError("key is required")
                cfgmod.update_settings({key: body.get("value")})
                return _json({"message": "Сохранено, перезапуск не нужен."})
            if name == "wake":
                app.wake_up_if_sleeping()
                return _json({"message": "Разбудила воркеров."})
            if name == "proactive":
                if config().paused or app._stopping:
                    raise ValueError("Куни на паузе или останавливается")
                app.act_proactively()
                return _json({"message": "Запущено самостоятельное размышление."})
            if name == "clear_queue":
                return _json({"message": f"Очередь очищена: {app.notification_manager.clear()}."})
            if name == "cancel_reminder":
                rid = body.get("id")
                if not isinstance(rid, str) or rid not in {r.id for r in app.reminders.all()}:
                    raise ValueError("напоминание не найдено")
                app.reminders.drop(rid)
                return _json({"message": "Напоминание отменено."})
            if name == "send_message":
                chat_id, text = body.get("chat_id"), body.get("text")
                if (not isinstance(chat_id, int) or isinstance(chat_id, bool) or chat_id == 0
                        or not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_MESSAGE):
                    raise ValueError(f"нужны chat_id и текст до {MAX_MESSAGE} символов")
                await asyncio.wait_for(app.telegram.send_message(chat_id, text.strip()), 30)
                log.info("panel sent a message to chat_id=%d", chat_id)
                return _json({"message": "Отправлено."})
            if name in ("list_add", "list_remove"):
                which, chat_id = body.get("list"), body.get("chat_id")
                if which not in ("allowed", "blocked") or not _is_chat_id(chat_id):
                    raise ValueError("нужны list (allowed или blocked) и chat_id")
                c = config()
                if which == "blocked" and name == "list_add" and chat_id == c.papikChatId:
                    raise ValueError("владельца заблокировать нельзя")
                allowed, blocked = list(c.allowedChatIds), list(c.blockedChatIds)
                target, other = (allowed, blocked) if which == "allowed" else (blocked, allowed)
                if name == "list_add":
                    if chat_id not in target:
                        target.append(chat_id)
                    if chat_id in other:
                        other.remove(chat_id)  # a chat is either allowed or blocked, never both
                elif chat_id in target:
                    target.remove(chat_id)
                cfgmod.update_settings({"allowedChatIds": allowed, "blockedChatIds": blocked})
                getattr(app, "denied_chats", {}).pop(chat_id, None)
                return _json({"message": "Список обновлён, перезапуск не нужен."})
            if name == "key_add":
                fingerprint = cfgmod.add_extra_key(body.get("key"))
                note = " Внимание: задан KUNI_API_KEY/KUNI_BASE_URL — ключи из файла сейчас не используются." \
                    if os.environ.get("KUNI_API_KEY") or os.environ.get("KUNI_BASE_URL") else ""
                return _json({"message": f"Ключ добавлен (id {fingerprint[:6]}).{note}"})
            if name == "key_remove":
                if not isinstance(body.get("id"), str):
                    raise ValueError("нужен id ключа")
                cfgmod.remove_extra_key(body["id"])
                return _json({"message": "Ключ удалён из пула."})
            if name == "key_test":
                return await self._test_key(body.get("id"))
            if name in ("channel_read", "channel_leave"):
                chat_id = body.get("chat_id")
                if not _is_chat_id(chat_id):
                    raise ValueError("нужен chat_id")
                chat = await app.telegram.get_chat(chat_id)
                if chat.type != ChatType.CHANNEL:
                    raise ValueError("это не канал")
                if name == "channel_leave":
                    await asyncio.wait_for(app.telegram.leave_chat(chat_id), 30)
                    self._channels_cache = None
                    return _json({"message": "Вышла из канала."})
                if chat_id in config().blockedChatIds:
                    raise ValueError("канал в чёрном списке")
                app._queue_chat(chat_id)
                return _json({"message": "Канал поставлен в очередь на чтение."})
            if name in ("memory_dump", "memory_sleep"):
                return self._start_job(name)
            if name == "memory_forget":
                node_id = body.get("id")
                if not isinstance(node_id, str) or node_id not in app.memory.nodes:
                    raise ValueError("узел не найден")
                app.memory.remove_node(node_id)
                return _json({"message": "Узел забыт вместе со связями."})
            if name == "save_working_memory":
                chat_id, text = body.get("chat_id"), body.get("text")
                if not _is_chat_id(chat_id) or not isinstance(text, str) or len(text) > MAX_WORKING_MEMORY:
                    raise ValueError(f"нужны chat_id и текст до {MAX_WORKING_MEMORY} символов")
                async with app._working_memory_lock:
                    atomic_write_text(app.working_memory_path(chat_id), text)
                return _json({"message": "Рабочая память сохранена."})
            if name == "backup_now":
                return await self._backup_now()
            if name == "restart":
                if "INVOCATION_ID" not in os.environ:
                    raise ValueError("перезапуск доступен только под systemd (иначе она просто остановится)")
                asyncio.get_running_loop().call_later(1.0, os.kill, os.getpid(), signal.SIGTERM)
                return _json({"message": "Перезапуск: память сохраняется, это займёт до пары минут."})
        except (ValueError, OSError, asyncio.TimeoutError, KeyError) as exc:
            raise _fail(web.HTTPBadRequest, str(exc) or type(exc).__name__) from None
        except Exception as exc:  # noqa: BLE001  (Telegram errors etc.: report the kind, never a traceback)
            log.warning("panel action %s failed: %s", name, type(exc).__name__)
            raise _fail(web.HTTPBadGateway, f"{type(exc).__name__}: {str(exc)[:200]}") from None
        raise _fail(web.HTTPBadRequest, "unknown action")
