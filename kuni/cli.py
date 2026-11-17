"""Entry point: `python -m kuni [run|chat|graph|sleep|init]`. Port of main.cpp."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import tempfile
import webbrowser
from pathlib import Path

from . import __version__
from . import config as cfgmod
from . import prompts as promptsmod
from .config import config

LOG_TAG = "App"
log = logging.getLogger(LOG_TAG)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "telethon", "asyncio", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def open_memory():
    from .memory.graph import GraphMemory

    return GraphMemory(Path("data") / "memory" / "graph.sqlite")


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------
async def wait_for_shutdown() -> None:
    """Waits for Ctrl-C / SIGTERM / Enter (the C++ kernel: "Press enter to shutdown gracefully")."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # windows
            pass

    def reader() -> None:
        try:
            sys.stdin.readline()
        except Exception:  # noqa: BLE001
            return
        loop.call_soon_threadsafe(stop.set)

    if sys.stdin and sys.stdin.isatty():
        loop.run_in_executor(None, reader)
    log.info("Bot is up and running. Press enter (or Ctrl-C) to shutdown gracefully.")
    await stop.wait()
    log.info("Bot is shutting down. Please give some time to dump remaining context")


async def run_async() -> int:
    from .llm import OpenAIChat
    from .metrics import Exporter, MetricsBreadcrumbs
    from .proxy.context_bridge import ContextBridge
    from .proxy.server import ProxyServer
    from .tools import ToolSet
    from .tools_impl.ask import ask
    from .watchdog import heartbeat, mark_clean_shutdown, notify_systemd

    c = config()
    cfgmod.die_if_invalid(c)
    if not c.llm.endpoint.bearerKey and "localhost" not in c.llm.endpoint.baseUrl and "127.0.0.1" not in c.llm.endpoint.baseUrl:
        log.warning("general.llm has no bearerKey: put your API key to config.toml or set KUNI_API_KEY")
    background = [asyncio.create_task(cfgmod.watch()), asyncio.create_task(promptsmod.watch())]
    promptsmod.prompts()

    llm = OpenAIChat()
    app = None
    telegram = None
    bridge = None
    proxy = None
    exporter = None
    panel = None
    clean_shutdown = False
    metrics_breadcrumbs = MetricsBreadcrumbs()

    def start_proxy(memory) -> None:
        nonlocal proxy, bridge
        endpoint = c.resolved_llm().endpoint
        proxy = ProxyServer(endpoint, lambda ctx: ToolSet([ask(lambda: ctx[-1].content if ctx else "", llm, memory)]),
                            port=10434, host=c.bindHost)
        bridge = ContextBridge(endpoint, memory, llm)
        proxy.sent_request_to_llm.append(bridge.collect_request_to_llm)
        bridge.start()

    try:
        if c.telegramEnabled:
            from .telegram.telethon_client import TelethonClient
            from .telegram_app import TelegramApp

            telegram = TelethonClient()
            log.info("Waiting for Telegram network...")
            await telegram.start()
            if c.lockdown == cfgmod.LockdownMode.CONTACTS_ONLY:
                log.info("Lockdown mode is enabled (config.toml lockdown). Kuni can only chat with her contacts.")
            elif c.lockdown == cfgmod.LockdownMode.PAPIK_ONLY:
                log.info("Lockdown mode is enabled (config.toml lockdown). Kuni can only open chat with ID %s "
                         "(PAPIK_CHAT_ID).", c.papikChatId)
            app = TelegramApp(telegram, llm)
            app.start_timers()
            app.spawn(app.send_notifications_on_init())
            metrics_breadcrumbs = app.metric_breadcrumbs
            if c.proxyEnabled:
                start_proxy(app.memory)
                app.chat_history_message_processors.append(bridge)
        elif c.proxyEnabled:
            start_proxy(open_memory())
            log.info("Proxy server started standalone (No Telegram)!")
        try:
            exporter = Exporter(metrics_breadcrumbs, host=c.bindHost)
            exporter.register_openai(llm)
            if app:
                exporter.register_app(app)
        except OSError as e:
            log.warning("Prometheus exporter is disabled: %s", e)
        if proxy:
            await proxy.start()
        if app and c.webPanelEnabled:
            from .webpanel import WebPanel

            try:
                panel = WebPanel(app)
                await panel.start()
            except OSError as e:
                panel = None
                log.warning("Web panel is disabled: %s", e)

        background.append(asyncio.create_task(heartbeat()))
        await wait_for_shutdown()
        clean_shutdown = True
    finally:
        notify_systemd("STOPPING=1")
        if panel:
            await panel.stop()
        for t in background:
            t.cancel()
        if app:
            await app.close()  # stop workers before taking their final memory snapshot
        try:
            if app:
                await asyncio.wait_for(app.diary_dump_messages(), timeout=600)
            if bridge:
                bridge.stop()
                await asyncio.wait_for(bridge.collect_and_save_sessions_not_newer_than(float("inf")), timeout=600)
        except Exception as e:  # noqa: BLE001
            log.error("failed to dump context on shutdown: %r", e)
        if proxy:
            await proxy.stop()
        if telegram:
            await telegram.stop()
        if exporter:
            exporter.close()
        await llm.aclose()
        if clean_shutdown and app:
            mark_clean_shutdown(app._run_marker)
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# chat: talk to Kuni without Telegram
# ---------------------------------------------------------------------------------------------------------------------
async def chat_async(args) -> int:
    from .llm import OpenAIChat
    from .telegram.fake import FakeTelegram
    from .telegram_app import TelegramApp

    c = config()
    c.randomlyGoSleep = False  # it would hang the REPL for hours
    c.checkChatsOnStartup = False
    if not c.papikChatId:
        c.papikChatId = 1  # offline REPL: a demo owner id, nothing is sent to Telegram
    fake = FakeTelegram(my_id=1000)
    fake.add_user(c.papikChatId, c.papikName)
    llm = OpenAIChat()
    app = TelegramApp(fake, llm)
    fake.printer = lambda chat, m: print(f"\x1b[35m{c.characterName}:\x1b[0m {m.text}", flush=True)
    background = [asyncio.create_task(cfgmod.watch()), asyncio.create_task(promptsmod.watch())]
    print(f"Chatting with {c.characterName} as {c.papikName} (chat_id={c.papikChatId}) without Telegram. "
          "Her replies appear when she decides to send them. /quit saves her memory and exits.")
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, lambda: input("> "))
            if line.strip() in ("/quit", "/exit"):
                break
            if line.strip():
                await fake.incoming(c.papikChatId, line)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        print("Saving memory (this takes a few LLM calls)...")
        for t in background:
            t.cancel()
        await app.close()
        try:
            await app.diary_dump_messages()
        except Exception as e:  # noqa: BLE001
            print("failed:", e)
        await llm.aclose()
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# graph tools
# ---------------------------------------------------------------------------------------------------------------------
def graph_command(args) -> int:
    from .graph_view import render_html
    from .tools_impl.ask import Seen, explore_entity, query_memory

    memory = open_memory()
    cmd = args.graph_cmd
    if cmd == "stats":
        print(json.dumps(memory.stats(), indent=2, ensure_ascii=False))
    elif cmd == "query":
        print(query_memory(memory, Seen(), " ".join(args.text), args.entity or []))
    elif cmd == "show":
        print(explore_entity(memory, Seen(), " ".join(args.name), None))
    elif cmd == "export":
        if args.json:
            Path(args.json).write_text(json.dumps(memory.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
            print("written", args.json)
        if args.md:
            out = Path(args.md)
            out.mkdir(parents=True, exist_ok=True)
            for ep in memory.episodes():
                (out / f"{ep.id}.md").write_text(ep.summary + "\n", encoding="utf-8")
            print("written", len(memory.episodes()), "diary pages to", out)
        if args.html:
            Path(args.html).write_text(render_html(memory), encoding="utf-8")
            print("written", args.html)
    elif cmd == "view":
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(render_html(memory))
        print("opening", f.name)
        webbrowser.open("file://" + f.name)
    return 0


async def login_async(args) -> int:
    from .telegram.telethon_client import TelethonClient

    errors = cfgmod.validate(config())
    if errors:
        for e in errors:
            print("config.toml:", e)
        return 1
    tg = TelethonClient()
    try:
        await tg.start(prefer_qr=args.qr)
        print(f"Logged in as id={tg.my_id}. The session is saved in tg_session/ - `python -m kuni run` won't ask again.")
    finally:
        await tg.stop()
    return 0


async def sleep_async(args) -> int:
    from .llm import OpenAIChat
    from .memory.consolidate import sleeping_consolidation

    memory = open_memory()
    llm = OpenAIChat()
    try:
        report = await sleeping_consolidation(memory, llm, max_secs=args.max_minutes * 60)
    finally:
        await llm.aclose()
    print(report)
    print(json.dumps(memory.stats(), ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kuni", description="Kuni (くに): LLM character that lives in Telegram.")
    p.add_argument("--version", action="version", version=f"kuni {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--debug", action="store_true", help="open the memory graph viewer (same as `graph view`)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("run", help="run the bot (default)")
    sub.add_parser("chat", help="talk to Kuni in the terminal, without Telegram")
    ini = sub.add_parser("init", help="create config.toml")
    ini.add_argument("--api-key")
    ini.add_argument("--base-url")
    ini.add_argument("--model")
    ini.add_argument("--reasoning-effort")
    ini.add_argument("--force", action="store_true")
    lg = sub.add_parser("login", help="log in to Telegram and exit (the session is saved)")
    lg.add_argument("--qr", action="store_true", help="log in by scanning a QR code instead of typing a code")
    s = sub.add_parser("sleep", help="run the graph memory consolidation (\"sleep\") now")
    s.add_argument("--max-minutes", type=float, default=cfgmod.SLEEP_MAX_TIME_SECS / 60)
    g = sub.add_parser("graph", help="inspect the memory graph")
    gs = g.add_subparsers(dest="graph_cmd", required=True)
    gs.add_parser("stats")
    q = gs.add_parser("query")
    q.add_argument("text", nargs="+")
    q.add_argument("-e", "--entity", action="append", help="entity name to start from (repeatable)")
    sh = gs.add_parser("show")
    sh.add_argument("name", nargs="+")
    ex = gs.add_parser("export")
    ex.add_argument("--html")
    ex.add_argument("--json")
    ex.add_argument("--md", help="directory to write the diary pages to as markdown files")
    gs.add_parser("view")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if args.cmd == "init":
        path = cfgmod.CONFIG_TOML
        if path.exists() and not args.force:
            print(f"{path} already exists (use --force to overwrite)")
            return 1
        cfgmod.init_config_file(path, base_url=args.base_url, api_key=args.api_key, model=args.model,
                                effort=args.reasoning_effort)
        print(f"created {path}")
        return 0
    if args.debug:
        args.cmd, args.graph_cmd = "graph", "view"
    config()  # load config (creates config.toml on the first run)
    if args.cmd == "graph":
        return graph_command(args)
    if args.cmd == "login":
        return asyncio.run(login_async(args))
    if args.cmd == "sleep":
        return asyncio.run(sleep_async(args))
    if args.cmd == "chat":
        return asyncio.run(chat_async(args))
    return asyncio.run(run_async())


if __name__ == "__main__":
    sys.exit(main())
