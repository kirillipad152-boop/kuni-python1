"""The Telegram-facing application. Port of class App in main.cpp."""

from __future__ import annotations

import asyncio
import logging
import platform
import random
import shutil
import sys
import time

import httpx

from . import KERNEL_NAME, __version__
from . import config as cfgmod
from .app import AppBase, ChatHistoryMessageProcessor
from .config import LockdownMode, config
from .imagegen import StableDiffusionClient
from .llm import OpenAIChat, Response
from .llmui.format import format_chat_history_message
from .lockdown import is_accessible_from_lockdown
from .memory.graph import GraphMemory
from .notifications import Notification
from .prompts import prompts
from .reminders import Reminder, run_scheduler
from .telegram.base import TelegramClient
from .telegram.types import Chat, ChatType, MemberStatus, Message, TelegramError
from .tools import Ctx, Property, Tool, ToolError, ToolSet
from .tools_impl import media_tools, telegram_tools
from .tools_impl.ask import ask  # noqa: F401  (re-export for tests)
from .tools_impl.post_message import telegram_post_message
from .tools_impl.reminder_tools import reminder_tools
from .tools_impl.send_telegram_message import send_telegram_message
from .util import json_as_int
from .watchdog import Probe, Watchdog, mark_running, unclean_restart_message

log = logging.getLogger("App")
flow_log = logging.getLogger("MessageFlow")

rng = random.Random()
JOINS_PER_DAY = 5
HISTORY_ENTRY_OVERHEAD = 100  # tdlib's debug dump of a message content is ~100 chars longer than its text


class TelegramApp(AppBase):
    def __init__(self, telegram: TelegramClient, openai: OpenAIChat, working_dir="data",
                 memory: GraphMemory | None = None) -> None:
        super().__init__(openai, working_dir, memory)
        self.telegram = telegram
        self.chat_history_message_processors: list[ChatHistoryMessageProcessor] = []
        self._opened_chat: Chat | None = None
        self._last_typing = 0.0
        self._started_at = time.monotonic()
        self.denied_chats: dict[int, dict] = {}  # who the lockdown turned away lately, for the panel's "allow" button
        self._joins: list[float] = []  # shortcut: in memory only, the budget resets on restart
        self.watchdog = Watchdog(self._notify_owner)
        openai.rate_limit_events.append(self.watchdog.record_rate_limit)
        openai.success_events.append(self.watchdog.record_llm_success)
        self._run_marker = self.working_dir / ".running"
        self._restart_pending = False
        self._last_proxy_check = 0.0
        self._last_proxy_url = None
        self._proxy_ok: bool | None = None
        telegram.new_message_handlers.append(self.handle_new_message)

    def start_timers(self) -> None:
        super().start_timers()
        self._restart_pending = mark_running(self._run_marker)
        self.spawn(self.watchdog.run(self._watchdog_probe, active=lambda: config().watchdogEnabled))
        self.spawn(run_scheduler(self.reminders, self._deliver_reminder,
                                 active=lambda: config().capabilityReminders and not config().paused and not self._stopping))

    async def _notify_owner(self, text: str) -> None:
        await asyncio.wait_for(self.telegram.send_message(config().papikChatId, "⚠ Служебно: " + text), timeout=15)

    async def _proxy_health(self) -> bool | None:
        c = config()
        proxy = c.youtubeProxy if c.capabilityYouTube else ""
        if not proxy:
            return None
        if self._last_proxy_url == proxy and self._proxy_ok is True and time.monotonic() - self._last_proxy_check < 300:
            return self._proxy_ok
        self._last_proxy_url, self._last_proxy_check = proxy, time.monotonic()
        try:
            async with httpx.AsyncClient(proxy=proxy, timeout=5, trust_env=False, follow_redirects=False) as client:
                response = await client.head("https://www.youtube.com/robots.txt")
                self._proxy_ok = 200 <= response.status_code < 400
        except (ValueError, httpx.HTTPError, httpx.InvalidURL, ImportError):
            self._proxy_ok = False
        return self._proxy_ok

    async def _watchdog_probe(self) -> Probe:
        if not config().watchdogEnabled:
            return Probe(None, False, False, True, 1.0, None)
        if self._restart_pending:
            await self._notify_owner(unclean_restart_message())
            self._restart_pending = False
        ages = [self.notification_manager.oldest_age()]
        ages.extend(max(0, time.monotonic() - w.current_notification.queued_at)
                    for w in self.workers if w.current_notification is not None)
        disk = shutil.disk_usage(self.working_dir)
        return Probe(max((age for age in ages if age is not None), default=None),
                     any(w._active_task is not None for w in self.workers),
                     any(w.sleeping for w in self.workers), config().paused,
                     disk.free / disk.total, await self._proxy_health())

    async def _validate_reminder_chat(self, chat_id: int) -> None:
        if chat_id == 0 or not await telegram_tools._lockdown_accessible(self.telegram, chat_id):
            raise ToolError("This chat is not accessible; reminders cannot be sent here")
        chat = await self.telegram.get_chat(chat_id)
        if chat.type == ChatType.CHANNEL:
            raise ToolError("Reminders cannot be sent to read-only channels")
        if chat.type.is_group:
            status = await self.telegram.get_member_status(chat_id, self.telegram.my_id)
            if status in (MemberStatus.LEFT, MemberStatus.BANNED):
                raise ToolError("Join this group before setting reminders")

    async def _deliver_reminder(self, reminder: Reminder, late_seconds: float) -> None:
        await self._validate_reminder_chat(reminder.chat_id)
        if config().paused or self._stopping or not config().capabilityReminders:
            raise ToolError("Reminder delivery is paused")
        # shortcut: at-least-once delivery; a crash between Telegram's ACK and the disk write can duplicate a reminder.
        await asyncio.wait_for(self.telegram.send_message(reminder.chat_id, "⏰ Напоминание:\n" + reminder.note), timeout=20)
        self.wake_up_if_sleeping()
        flow_log.info("event=reminder_sent chat_id=%d reminder_id=%s late_seconds=%d",
                      reminder.chat_id, reminder.id, late_seconds)

    # ------------------------------------------------------------------------------------------------------------
    # presence
    # ------------------------------------------------------------------------------------------------------------
    def _release_opened_chat(self) -> None:
        chat, self._opened_chat = self._opened_chat, None
        if chat is not None:
            self.spawn(self._safe(self.telegram.send_chat_action(chat.id, None)))
            self.spawn(self._safe(self.telegram.close_chat(chat.id)))

    @staticmethod
    async def _safe(aw) -> None:
        try:
            await aw
        except Exception as e:  # noqa: BLE001
            log.warning("telegram call failed: %s", e)

    def set_online(self, online: bool = True) -> None:
        self.spawn(self._safe(self.telegram.set_online(online)))

    def on_offline(self) -> None:
        self._release_opened_chat()
        self.set_online(False)

    def on_response_assembling(self, response: Response) -> None:
        chat = self._opened_chat
        if chat is None or chat.type == ChatType.CHANNEL:
            return  # nobody types into a broadcast channel (Telegram answers CHAT_ADMIN_REQUIRED)
        now = time.monotonic()
        if now - self._last_typing < 1.0:
            return  # no need to spam.
        self._last_typing = now
        if response.choices:
            for tc in response.choices[0].message.tool_calls:
                action = {"sticker_send": "choosing_sticker", "record_audio": "recording_voice",
                          "take_photo": "uploading_photo"}.get(tc.function.name)
                if action:
                    self.spawn(self._safe(self.telegram.send_chat_action(chat.id, action)))
                    break
        self.spawn(self._safe(self.telegram.send_chat_action(chat.id, "typing")))

    # ------------------------------------------------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------------------------------------------------
    def update_tools(self, actions: ToolSet, temporary_context) -> None:
        super().update_tools(actions, temporary_context)
        c = config()
        if c.capabilityReminders and temporary_context.chat_id not in (None, 0):
            for tool in reminder_tools(self.reminders, validate=self._validate_reminder_chat):
                actions.insert(tool)
        if c.capabilityTakePhoto:
            actions.insert(media_tools.take_photo(StableDiffusionClient(), self.openai))
        if c.capabilityRecordVoice:
            actions.insert(media_tools.record_audio())
        if c.capabilityYouTube:
            from .tools_impl.youtube_tools import youtube_read, youtube_search

            actions.insert(youtube_search())
            actions.insert(youtube_read())
        actions.insert(telegram_tools.get_telegram_chats(self.telegram, self.openai, self.is_acting_proactively))
        actions.insert(telegram_tools.search_chats(self.telegram))
        actions.insert(telegram_tools.search_messages(self.telegram, self.openai, temporary_context))
        actions.insert(telegram_tools.view_messages_around(self.telegram, self.openai, temporary_context))
        actions.insert(telegram_tools.remove_and_ban_chat(self.telegram))

        async def open_chat_by_id(ctx: Ctx) -> str:
            if sum(1 for tc in ctx.all_tool_calls if tc.function.name == "open_chat_by_id") > 1:
                return "You can only call this tool once per turn."
            chat_id = json_as_int(ctx.args.get("chat_id"))
            if chat_id is None:
                raise ValueError("chat_id integer is required")
            # Check lockdown mode - only allow PAPIK_CHAT_ID if lockdown is enabled
            if not await is_accessible_from_lockdown(self.telegram, chat_id):
                log.error("Error: Lockdown mode is enabled. You can only open chat with ID %s (PAPIK_CHAT_ID).",
                          config().papikChatId)
                return "No such chat"
            scope = temporary_context.chat_id
            if scope == 0 or (scope == config().papikChatId and await self._is_channel(chat_id)):
                self._queue_chat(chat_id)
                return ("Chat queued separately to protect its private context: you will be woken up to read it in a "
                        "moment. This worked, so do not repeat the call; call wait now.")
            return await self.open_telegram_chat(ctx.tools, chat_id, ctx.temporary_context)

        actions.insert(Tool(
            "open_chat_by_id",
            "Opens a chat by its id. Use this to start conversation. Use get_telegram_chats to retrieve `chat_id`s.",
            open_chat_by_id, {"chat_id": Property("integer", "The ID of the Telegram chat")}, ["chat_id"]))

        if c.canJoinChats and self._may_join(temporary_context.chat_id):
            async def join_by_link(ctx: Ctx) -> str:
                scope = temporary_context.chat_id
                if not self._may_join(scope):
                    return "Only the owner can ask you to join chats."
                link = ctx.args.get("invite_link")
                if not isinstance(link, str) or not link.strip() or len(link) > 200 or any(ch.isspace() for ch in link.strip()):
                    raise ValueError("invite_link must be a link or @username without spaces")
                if err := self._join_budget_error():
                    return err
                try:
                    chat = await self.telegram.join_chat_by_invite_link(link.strip())
                except TelegramError as e:
                    log.error('Failed to join chat by invite link "%s": %s', link, e)
                    return f"Error: failed to join chat by invite link: {e}"
                self._joins.append(time.monotonic())
                if scope is None:  # trusted CLI/tests: the old behaviour
                    return await self.open_telegram_chat(ctx.tools, chat.id, ctx.temporary_context)
                self._queue_chat(chat.id)  # it is read in its own context, never inside this one
                return f'Joined "{chat.title}". It is queued for reading separately. Call wait now.'

            actions.insert(Tool(
                "join_chat_by_link",
                "Joins a Telegram channel or group by an invite link (https://t.me/+xxxx), a public link "
                "(https://t.me/name) or @username, like tapping it in the official Telegram client. The chat is then queued "
                f"and you will read it separately. Limited to {JOINS_PER_DAY} joins per day: join only what is worth following.",
                join_by_link, {"invite_link": Property("string", "Invite link, t.me link or @username of the channel/group")},
                ["invite_link"]))
        if c.capabilityUseStickers:
            actions.insert(telegram_tools.sticker_list(self.telegram, self.openai))
            actions.insert(telegram_tools.sticker_save(self.telegram))

    def _note_denied(self, chat_id: int) -> None:
        entry = self.denied_chats.pop(chat_id, {"count": 0})
        entry.update(count=entry["count"] + 1, last=time.time())
        self.denied_chats[chat_id] = entry
        while len(self.denied_chats) > 30:
            self.denied_chats.pop(next(iter(self.denied_chats)))

    async def _is_channel(self, chat_id: int) -> bool:
        try:
            return (await self.telegram.get_chat(chat_id)).type == ChatType.CHANNEL
        except TelegramError:
            return False

    @staticmethod
    def _may_join(scope: int | None) -> bool:
        """Joining is for the owner's chat and Kuni's own initiative (scope 0), never for other contacts' chats."""
        return scope is None or scope in (0, config().papikChatId)

    def _join_budget_error(self) -> str | None:
        now = time.monotonic()
        self._joins = [t for t in self._joins if now - t < 86400]
        if len(self._joins) >= JOINS_PER_DAY:
            return (f"Error: you have already joined {JOINS_PER_DAY} chats in the last 24 hours; Telegram bans accounts "
                    "that join in bulk. Try again tomorrow.")
        return None

    def _limit_joins(self, tool: Tool) -> Tool:
        inner = tool.handler

        async def handler(ctx: Ctx) -> str:
            if err := self._join_budget_error():
                return err
            out = await inner(ctx)
            if out.startswith("You joined"):
                self._joins.append(time.monotonic())
            return out

        tool.handler = handler
        return tool

    def _queue_chat(self, chat_id: int) -> None:
        """Hands a chat to the queue: it is opened later with its own private context."""
        async def open_selected(ctx: Ctx) -> str:
            return await self.open_telegram_chat(ctx.tools, chat_id, ctx.temporary_context)

        pin = f'<chat id="{chat_id}" />'
        if (any(h.notification.pin == pin for h in self.notification_manager._notifications)
                or any(w.current_notification and w.current_notification.pin == pin for w in self.workers)):
            return  # already waiting or being read: a repeated request must not add another LLM turn
        self.notification_manager.pass_notification_to_ai(Notification(
            message="Revisit this chat using its own private context.",
            actions=ToolSet([Tool("open", "Open the selected chat", open_selected)]),
            pin=f'<chat id="{chat_id}" />', chat_id=chat_id))

    # ------------------------------------------------------------------------------------------------------------
    # notifications
    # ------------------------------------------------------------------------------------------------------------
    async def send_notifications_on_init(self) -> None:
        """tdlib (and MTProto) do not send notifications for unread chats on program startup. We'll fix this."""
        if not config().checkChatsOnStartup:
            return
        chats = list(await self.telegram.get_chats(200))
        chats.reverse()  # older first, newest last
        for chat in chats:
            if chat.unread_count == 0 or chat.last_message is None:
                continue
            # make up a new-message event: handle_new_message will format a notification for us.
            await self.handle_new_message(chat.last_message)

    async def try_handle_cmd(self, msg: Message) -> str | None:
        if msg.kind != "text":
            return None
        text = msg.text.strip()
        parts = text.split()
        if not parts:
            return None
        command, *args = parts
        if command not in ("/version", "/status", "/pause", "/resume", "/lockdown", "/graph"):
            return None
        if command == "/version":
            if args:
                return "Использование: /version"
            info = (f"Project name: kuni\nVersion: {__version__}\nRuntime: Python {platform.python_version()} "
                    f"({sys.platform})\n")
            return f"{info}\nKernel: {KERNEL_NAME}"
        owner = config().papikChatId
        if msg.sender_is_chat or msg.sender_id != owner or msg.chat_id != owner or msg.is_outgoing:
            return "Эта команда доступна только владельцу в личном чате."
        try:
            c = config()
            if command == "/status" and not args:
                counts = self.memory.stats()
                return (f"Куни: {'пауза' if c.paused else 'работает'}\n"
                        f"Время работы: {int((time.monotonic() - self._started_at) / 60)} мин\n"
                        f"Режим доступа: {c.lockdown.value}\nМодель: {c.llm.model}\n"
                        f"Очередь: {len(self.notification_manager._notifications)}; "
                        f"занятых воркеров: {sum(w._active_task is not None for w in self.workers)}\n"
                        f"Ваших напоминаний: {len(self.reminders.list(owner))}\n"
                        f"Память: {counts['nodes']} узлов, {counts['edges']} связей, {counts['episodes']} эпизодов")
            if command in ("/pause", "/resume"):
                if command == "/resume":
                    if args:
                        return "Использование: /resume"
                    paused = False
                elif not args:
                    paused = not c.paused
                elif args in (["on"], ["off"]):
                    paused = args == ["on"]
                else:
                    return "Использование: /pause [on|off]. Без аргумента переключает паузу."
                cfgmod.update_general_settings(paused=paused)
                return ("Пауза включена. Ответы и инициатива остановлены; команды доступны. /pause off — продолжить."
                        if paused else "Пауза выключена. Продолжаю обработку сообщений.")
            if command == "/lockdown":
                if len(args) != 1 or args[0] not in {m.value for m in LockdownMode}:
                    return "Использование: /lockdown none|contacts_only|papik_only"
                cfgmod.update_general_settings(lockdown=args[0])
                return f"Режим доступа: {args[0]}. Сохранён, перезапуск не нужен."
            if command == "/graph" and args == ["stats"]:
                counts = self.memory.stats()
                return (f"Узлов: {counts['nodes']}\nСвязей: {counts['edges']}\nЭпизодов: {counts['episodes']}\n"
                        f"Сущностей: {counts['entities']}")
            return "Команды: /status, /pause [on|off], /resume, /lockdown none|contacts_only|papik_only, /graph stats"
        except Exception as e:  # noqa: BLE001
            log.error("Failed to handle owner command %s: %s", command, type(e).__name__)
            return "Не удалось выполнить команду. Настройки не подтверждены; проверьте журнал сервиса."

    async def handle_new_message(self, msg: Message) -> None:
        user_id = 0 if msg.sender_is_chat else msg.sender_id
        if user_id == self.telegram.my_id:
            return
        if self._stopping:
            flow_log.info("event=deferred chat_id=%d message_id=%d reason=stopping", msg.chat_id, msg.id)
            return
        flow_log.info("event=received chat_id=%d message_id=%d", msg.chat_id, msg.id)
        # Control commands bypass mute, sleep, pause, filters and the LLM's API quota.
        if (cmd := await self.try_handle_cmd(msg)) is not None:
            if user_id:
                await telegram_post_message(self.telegram, msg.chat_id, cmd, reply_to=msg.id)
                await self.telegram.view_messages(msg.chat_id, [msg.id])
            flow_log.info("event=command_handled chat_id=%d message_id=%d", msg.chat_id, msg.id)
            return
        # Check lockdown mode - only allow PAPIK_CHAT_ID if lockdown is enabled
        if not await is_accessible_from_lockdown(self.telegram, msg.chat_id):
            flow_log.info("event=discarded chat_id=%d message_id=%d reason=lockdown", msg.chat_id, msg.id)
            self._note_denied(msg.chat_id)
            return
        if not await is_accessible_from_lockdown(self.telegram, msg.chat_id, config().chatNotificationFilter):
            flow_log.info("event=discarded chat_id=%d message_id=%d reason=notification_filter", msg.chat_id, msg.id)
            return
        chat = await self.telegram.get_chat(msg.chat_id)
        if chat.type == ChatType.CHANNEL and not config().channelNotifications:
            # The post stays unread: she reads channels herself while acting proactively (see get_telegram_chats).
            flow_log.info("event=discarded chat_id=%d message_id=%d reason=channel_read_on_demand", msg.chat_id, msg.id)
            return
        if chat.muted:
            # Alex2772 (Apr 23 2026): added a probability to ignore a muted chat. If we always ignored it, LLM would
            # read it only if it occasionally called get_telegram_chats, recognized a chat with a lot of messages and
            # decided to read it - which basically means NEVER. A PROBABILITY allows the account holder to mute the chat
            # so LLM gives it a lot less attention (useful for spammy chats). To ignore it completely, archive the chat.
            if rng.random() < 0.8:
                flow_log.info("event=discarded chat_id=%d message_id=%d reason=muted", msg.chat_id, msg.id)
                return
        notification = f'<notification chat_id="{chat.id}">\n'
        if user_id == msg.chat_id:
            notification += f"You received a direct message from {chat.title} (chat_id = {chat.id})"
        elif user_id != 0:
            user = await self.telegram.get_user(user_id)
            notification += (f'{user.first_name} {user.last_name} (user_id = {user_id}) sent a message in group chat '
                             f'"{chat.title}" (chat_id = {chat.id})')
        else:
            notification += f'Channel "{chat.title}" (chat_id={chat.id}) created a new post\n'
        notification += "\n</notification>\nYou don't have any chat open. Use #open tool to open the chat"

        if user_id == config().papikChatId:
            priority = 1000
        elif config().wakeUpOnPinnedChat and chat.pinned:
            priority = 100
        else:
            priority = 0

        async def open_action(ctx: Ctx) -> str:
            return await self.open_telegram_chat(ctx.tools, chat.id, ctx.temporary_context)

        self.notification_manager.pass_notification_to_ai(Notification(
            message=notification,
            actions=ToolSet([Tool("open", f'Open "{chat.title}" chat. Use this if you\'d like to reply or see messages.',
                                  open_action)]),
            priority=priority,
            pin=f'<chat id="{chat.id}" />',
            chat_id=chat.id,
            message_id=msg.id,
        ))
        reason = "paused" if config().paused else "worker_sleep" if any(w.sleeping for w in self.workers) else "ready_or_busy"
        flow_log.info("event=queued chat_id=%d message_id=%d reason=%s priority=%d", chat.id, msg.id, reason, priority)
        if priority > 0:
            self.wake_up_if_sleeping()

    # ------------------------------------------------------------------------------------------------------------
    # opening a chat
    # ------------------------------------------------------------------------------------------------------------
    async def notification_allowed(self, notification: Notification) -> bool:
        if not await super().notification_allowed(notification):
            return False
        if notification.chat_id in (None, 0):
            return True
        return (await is_accessible_from_lockdown(self.telegram, notification.chat_id)
                and await is_accessible_from_lockdown(self.telegram, notification.chat_id, config().chatNotificationFilter))

    async def open_telegram_chat(self, tools: ToolSet, chat_id: int, temporary_context) -> str:
        """The LLM "taps" a chat: we load its messages, mark them read, and give the tools of that chat."""
        tg = self.telegram
        c = config()
        if temporary_context.chat_id is not None and temporary_context.chat_id != chat_id:
            return "Private chat isolation: another chat cannot be read inside this conversation."
        if not await is_accessible_from_lockdown(tg, chat_id):
            log.error("Error: Lockdown mode is enabled. You can only open chat with ID %s (PAPIK_CHAT_ID).", c.papikChatId)
            return "No such chat"

        await tg.wait_for_connection()
        self.set_online()
        self.spawn(self._safe(tg.open_chat(chat_id)))
        self.notification_manager.remove_notifications(f'<notification chat_id="{chat_id}">\n')

        chat = await tg.get_chat(chat_id)
        self._release_opened_chat()
        self._opened_chat = chat
        self.metric_breadcrumbs.set("chat", chat.title)

        # loaded messages. first goes the newest, last goes the oldest
        messages: list[Message] = []
        from_message = 0
        done = False
        while not done:
            page = await tg.get_chat_history(chat_id, from_message, 30)
            if not page:
                break
            from_message = page[-1].id
            length = 0
            for msg in page:
                length += len(msg.text) + HISTORY_ENTRY_OVERHEAD
                messages.append(msg)
                marker = f'<message message_id="{msg.id}"'
                if any(marker in m.content for m in temporary_context):
                    # this message is already in context, which means we don't need to load further. We just reassure
                    # this one, so the continuation of a dialogue in context won't feel detached.
                    done = True
                    break
                if len(messages) < 3:
                    continue
                if length >= c.chatMaxHistoryLength:
                    done = True
                    break
        log.info("Loaded %d message(s): %s", len(messages), chat.title)

        # response-time metadata for Prometheus. messages[0] is the most recent.
        self._last_opened_chat_last_message_time = float(messages[0].date) if messages else None
        if not messages:
            scenario = "new conversation"  # this means Kuni sent a message to a new person.
        elif messages[0].sender_id == tg.my_id and not messages[0].sender_is_chat:
            scenario = "kuni proactive"  # last message is from Kuni.
        else:
            scenario = "reply to user"
        self.metric_breadcrumbs.set("scenario", scenario)

        result = ""
        if not messages and c.canWriteToANewPerson:
            result += "This chat is empty! Only proceed if you looked up a @username and it led you here.\n"
            result += "Only write what you have to say to the chat; if someone asked you to text this person, just text them.\n"
            result += ("If you try to get back to the original chat and type something, you will be sending an extra "
                       "message to the wrong chat.")

        for msg in reversed(messages):  # oldest first
            formatted = await format_chat_history_message(tg, msg, chat, self.openai, temporary_context)
            for p in self.chat_history_message_processors:
                formatted = await p.process_chat_history_message(chat, msg, formatted)
            result += formatted

        if messages:
            self.spawn(self._safe(tg.view_messages(chat_id, [messages[0].id])))  # the two checkmarks

        result = f'You switched to the chat "{chat.title}" in Telegram. You see last messages:\n' + result

        # Mirror the official Telegram client behavior: if you open a group chat/channel you haven't joined, you only see
        # a "Join" button (plus the ability to react to messages) instead of a text field.
        is_admin = False
        if chat.type.is_group or chat.type == ChatType.CHANNEL:
            is_member = True
            try:
                status = await tg.get_member_status(chat_id, tg.my_id)
                if status in (MemberStatus.LEFT, MemberStatus.BANNED):
                    is_member = False
                elif status in (MemberStatus.CREATOR, MemberStatus.ADMIN):
                    is_admin = True
            except TelegramError:
                is_member = True  # member list is not accessible (large channels): assume we're in
            if not is_member:
                result += (f'\n<instructions>\nYou are NOT a member of "{chat.title}". You can\'t send messages here; you '
                           "can only #react_with_emoji to messages, or use\n#join_chat to become a member (after which "
                           "you'll be able to fully participate the next time you open this chat).\n</instructions>\n")
                new = ToolSet([telegram_tools.react_with_emoji(tg, chat)])
                if c.canJoinChats:
                    new.insert(self._limit_joins(telegram_tools.join_chat(tg, chat)))
                tools.replace_with(new)
                return result

        if chat.type in (ChatType.PRIVATE, ChatType.SECRET):
            result += (f"\n<instructions>\nYou are in private chat with {chat.title} (also known as direct messages or "
                       f"DM).\n\n{prompts().messages_epilogue}\n</instructions>\n")
        elif chat.type.is_group:
            result += (f'\n<instructions>\nYou are in group chat called "{chat.title}".\n\n{prompts().messages_epilogue}\n'
                       "</instructions>\n")
        else:  # a broadcast channel
            result += (f'\n<instructions>\nYou are in telegram channel (also known as supergroup) called "{chat.title}".\n'
                       "Pay close attention to these messages. Acquire context from them. You can't respond in telegram "
                       "channels\n(#send_telegram_message tool is not available). Instead, do what you usually do when "
                       "reading newsletters: reflect and reason\non them.\n"
                       "Some channels have reactions enabled. In that case, you can sometimes react with "
                       "#react_with_emoji to express your feelings about a message, but you can't send a full reply.\n\n"
                       "Forwarding posts:\n"
                       "If you find a post genuinely interesting, funny, or relevant to someone you know — you can forward "
                       "it to another chat\nusing #forward_message. Be selective: only forward posts that are truly worth "
                       "sharing. You can add a short comment\nexpressing your reaction. Use #get_telegram_chats to find "
                       "the destination chat_id if needed.\n"
                       "Do NOT forward ads, sponsored posts, or low-value content.\n</instructions>\n")
            tools.replace_with(ToolSet([telegram_tools.react_with_emoji(tg, chat), telegram_tools.forward_message(tg, chat)]))
            return result  # no send_telegram_message for channels

        new = ToolSet([
            send_telegram_message(tg, self.openai, chat, messages),
            telegram_tools.get_chat_photo(tg, self.openai, chat, temporary_context),
            telegram_tools.react_with_emoji(tg, chat),
            telegram_tools.remove_message(tg, chat),
            telegram_tools.edit_message_text(tg, chat),
            telegram_tools.forward_message(tg, chat),
        ])
        if c.capabilityUseStickers:
            new.insert(telegram_tools.sticker_send(tg, chat))
        if chat.type.is_group:
            if c.canLeaveChats:
                new.insert(telegram_tools.leave_chat(tg, chat))
            if is_admin:
                new.insert(telegram_tools.group_admin_remove_message(tg, chat))
                new.insert(telegram_tools.group_admin_ban_user(tg, chat))
                new.insert(telegram_tools.group_admin_set_user_tag(tg, chat))
        tools.replace_with(new)
        return result
