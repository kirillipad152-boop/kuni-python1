"""Worker: processes notifications with the LLM in a tool-centric loop. Port of Worker.cpp.

The C++ code is a state machine written with `goto naxyi_populate_ctx` / `goto naxyi_preserve_ctx`. Here it is a loop
with a `phase` variable: "populate" = inject memory, then (always) "preserve" = refresh tools, ask the LLM, handle tool
calls.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import config as cfgmod
from .config import config
from .llm import EMBEDDING_TAG, ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER, Message, Params, Response, Session
from .notifications import Notification
from .prompts import prompts
from .trace import Turn
from .tools import Ctx, Tool, ToolSet

if TYPE_CHECKING:
    from .app import AppBase

rng = random.Random()  # replaced in tests
flow_log = logging.getLogger("MessageFlow")

NICE_THOUGHTS = ("Nice thoughts! However you should be tool-centric. Make sure you made tool calls. The message you "
                 "provided is not visible to anyone but you.")


def context_text(messages) -> str:
    """The text the memory is searched with: the tail of the conversation (reasoning included)."""
    out = ""
    for m in messages:
        if m.reasoning:
            out += m.reasoning + "\n\n"
        if m.reasoning_content:
            out += m.reasoning_content + "\n\n"
        out += m.content + "\n\n---\n\n"
    return out


class Worker:
    def __init__(self, name: int, app: "AppBase") -> None:
        self.name = name
        self.app = app
        self.log = logging.getLogger(f"kuni_worker{name}")
        self._attach_file_log()
        self.pins: set[str] = set()
        self.alive = True
        self._wake_up = asyncio.Event()
        self.ask_called_this_turn = False
        self.relevance_threshold = 0.5
        self.temporary_context = app.get_context(config().papikChatId)
        self._active_task: asyncio.Task | None = None
        self._interrupted = False
        self.current_notification: Notification | None = None
        self.sleeping = False
        self._reply_sent = False
        self._outcome = ""
        self.turn = Turn(0, None, "")  # replaced per notification; this one is never shown
        self._task = asyncio.create_task(app.notification_manager.run(self.pins, self._on_notification))

    def _attach_file_log(self) -> None:
        if cfgmod.TESTING or any(isinstance(h, logging.FileHandler) for h in self.log.handlers):
            return
        try:
            Path("logs").mkdir(exist_ok=True)
            h = logging.FileHandler(Path("logs") / f"kuni_worker{self.name}.log", encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.log.addHandler(h)
        except OSError:
            pass

    def close(self) -> None:
        self.alive = False
        self.interrupt()
        self._task.cancel()

    def interrupt(self) -> None:
        if self._active_task and not self._active_task.done():
            self._interrupted = True
            self._active_task.cancel()

    def wake_up_if_sleeping(self) -> None:
        """If Kuni is sleeping, this wakes her up."""
        self._wake_up.set()

    async def _on_notification(self, notification: Notification) -> bool:
        self.current_notification = notification
        try:
            while self.alive:
                await self.app._run_gate.wait()
                if not await self.app.notification_allowed(notification):
                    flow_log.info("event=discarded chat_id=%s message_id=%s reason=access_changed_or_stopping",
                                  notification.chat_id, notification.message_id)
                    break
                self._interrupted = False
                self._active_task = asyncio.create_task(self.handle_notification(notification))
                try:
                    await self._active_task
                    break
                except asyncio.CancelledError:
                    if not self.alive or not self._interrupted:
                        raise
                finally:
                    self._active_task = None
        finally:
            self.current_notification = None
        return self.alive

    # ------------------------------------------------------------------------------------------------------------
    async def _process_randomly_go_sleep(self) -> None:
        if config().randomlyGoSleep and rng.random() < 0.01:
            # 1. randomly go afk is humane
            # 2. reduce resource usage: less conversations would be made; in case of group chats and telegram
            #    channels, messages would be processed in batches
            minutes = rng.randint(15, 120)
            self.log.info("Going to sleep for %d minutes", minutes)
            self._wake_up.clear()
            self.sleeping = True
            flow_log.info("event=sleep chat_id=%s worker=%s minutes=%d", self.temporary_context.chat_id, self.name, minutes)
            try:
                await asyncio.wait_for(self._wake_up.wait(), timeout=minutes * 60)
                self.log.info("Early wake up")
            except asyncio.TimeoutError:
                pass
            finally:
                self.sleeping = False
                flow_log.info("event=wake chat_id=%s worker=%s", self.temporary_context.chat_id, self.name)

    async def _process_shortcut_open(self, notification: Notification) -> None:
        handlers = notification.actions.handlers()
        if len(handlers) == 1:
            action = next(iter(handlers.values()))
            if action.name == "open" and not action.properties:
                # shortcut/optimization: if the notification gives the only option to open it, there's no need to ask
                # LLM whether it wants to open the notification because it does it in 100% cases. Also, this fits the
                # architecture: we can't change notification text at runtime, BUT we can provide more recent data by
                # giving the notification code control by calling "open()".
                notification.message = await action.handler(
                    Ctx(tools=notification.actions, args={}, temporary_context=self.temporary_context, all_tool_calls=[],
                        logger=self.log))

    def _process_ignore_chance(self, last: Message) -> bool:
        """Attempt to make LLM lazy and ignore the message :)  Returns True if the LLM has been begged to reconsider."""
        if cfgmod.TESTING:
            return False
        if rng.random() < config().suggestIgnoreChance:
            if any(tc.function.name in ("wait", "pause") for tc in last.tool_calls):
                return False
            # the assistant message must precede the tool results, otherwise the API rejects orphan tool messages
            self.temporary_context << last
            for tc in last.tool_calls:
                self.temporary_context << Message(
                    role=ROLE_TOOL, tool_call_id=tc.id,
                    content=f"Error: do you really want to continue? Think again; repeat `{tc.function.name}` to continue or "
                            "call wait() to finish.")
            self.log.info("Begging LLM to be lazy (ignore message)")
            self.turn.note("правило лени: её решение отложили и переспросили модель")
            flow_log.info("event=reconsideration chat_id=%s reason=lazy_rule", self.temporary_context.chat_id)
            return True
        return False

    # ------------------------------------------------------------------------------------------------------------
    def take_memory_entry(self, node, relatedness: float) -> str:
        self.log.debug("takeDiaryEntry: %s", node.id)
        if any(node.summary in m.content for m in self.temporary_context):
            # if the context already contains this page verbatim - we don't need to reinclude it - it makes no sense to
            # consume tokens for the same thing. The copypasted page would not receive score.
            return ""
        self.app.memory.touch(node, relatedness)
        self.log.info("Loaded into context: %s.md relatedness=%s\n%s", node.id, relatedness, node.summary)
        tag = "your_diary_page additional_context just_for_reasoning no_plagiarism no_copy"
        return f"<{tag}>\n{node.summary}\n</{tag}>\n"

    def _inject_memory(self) -> None:
        memory = self.app.memory
        if memory.is_empty() or config().diaryInjectionMaxLength <= 0:
            return
        # performs a graph scan based on the tail of the context: entities recognised there activate their
        # neighbourhood, so we find the common cues related to the current conversation.
        recall = memory.recall(context_text(list(self.temporary_context)[-3:]), hops=config().memoryGraphDepth,
                               confidence_factor=0.0, max_episodes=10, max_entities=0, max_facts=6, text_search=False,
                               chat_id=self.temporary_context.chat_id)
        pages = ""
        for node, relatedness in recall.episodes:
            if relatedness < self.relevance_threshold:
                if not pages:
                    # relax threshold for future queries (glm::mix(0.5, relatedness, 0.9))
                    self.relevance_threshold = 0.5 * 0.1 + relatedness * 0.9
                break
            if len(pages) >= config().diaryInjectionMaxLength:
                # set the minimum constraint for the future queries
                self.relevance_threshold = relatedness
                break
            pages += self.take_memory_entry(node, relatedness)
        facts = [(e, r) for e, r in recall.facts if r >= self.relevance_threshold]
        if pages and facts and len(pages) < config().diaryInjectionMaxLength:
            tag = "your_memory_graph additional_context just_for_reasoning no_plagiarism no_copy"
            body = "\n".join(memory.describe_edge(e) for e, _ in facts if memory.describe_edge(e) not in
                             "".join(m.content for m in self.temporary_context))
            if body:
                pages += f"<{tag}>\n{body}\n</{tag}>\n"
        if pages:
            last = self.temporary_context.last()
            last.content = pages + last.content

    def _update_tools(self, tools: ToolSet) -> None:
        self.app.update_tools(tools, self.temporary_context)

        def after(name: str) -> None:
            if name == "ask":
                self.ask_called_this_turn = True

        tools.on_after_tool_call.append(after)

    # ------------------------------------------------------------------------------------------------------------
    async def handle_notification(self, notification: Notification) -> None:
        self.temporary_context = self.app.get_context(notification.chat_id if notification.chat_id is not None else 0)
        self._reply_sent = False
        self._outcome = ""
        chat = notification.chat_id
        self.turn = self.app.traces.begin(chat or 0, notification.message_id, "чат" if chat else "инициатива")
        self.ask_called_this_turn = False
        self.app.metric_breadcrumbs.clear("chat", "scenario")
        try:
            if not cfgmod.TESTING:
                await self._process_randomly_go_sleep()
            flow_log.info("event=processing chat_id=%s message_id=%s age_seconds=%d",
                          notification.chat_id, notification.message_id, max(0, time.monotonic() - notification.queued_at))
            await self._handle(notification)
        except asyncio.CancelledError:
            self._outcome = self._outcome or "прервано (пауза или остановка)"
            flow_log.info("event=interrupted chat_id=%s message_id=%s reason=pause_or_shutdown",
                          notification.chat_id, notification.message_id)
            raise
        except Exception as e:  # noqa: BLE001
            self.log.error('Failed to process notification: "%s" %r', notification.message[:500], e, exc_info=True)
            self._outcome = f"ошибка {type(e).__name__}"
            flow_log.info("event=processing_failed chat_id=%s message_id=%s error=%s",
                          notification.chat_id, notification.message_id, type(e).__name__)
        finally:
            self.turn.finish(f"{self._outcome or 'завершено'}, "
                             f"{'ответ отправлен' if self._reply_sent else 'без ответа'}")
            flow_log.info("event=turn_finished chat_id=%s message_id=%s reply_sent=%s",
                          notification.chat_id, notification.message_id, self._reply_sent)
            self.app.save_context(self.temporary_context)
            self.app.on_offline()

    async def _handle(self, notification: Notification) -> None:
        ctx = self.temporary_context
        c = config
        await self._process_shortcut_open(notification)
        self.log.info("Processing notification: %s", notification.message)
        ctx << Message(role=ROLE_USER, content=notification.message)

        # the reasons why memory injection happens only on non-send tool calls:
        # 1. Each lookup adds delay. So each time LLM uses send_telegram_message, there would be a lookup.
        # 2. Instead of one big message, LLM is encouraged to send multiple small messages. When we insert occasional
        #    memories between LLMs send_telegram_message calls, it loses its focus and starts to spam messages filled
        #    with random cues from the memory (an ADHD participant that can't finish their thought).
        pause = {"flag": False}

        async def escape(_ctx: Ctx) -> str:
            pause["flag"] = True
            self._outcome = "wait"
            flow_log.info("event=wait chat_id=%s message_id=%s reply_sent=%s reason=llm_decision",
                          notification.chat_id, notification.message_id, self._reply_sent)
            if self.app.is_acting_proactively:
                # at the end of "actProactively", let's try to encourage LLM to write someone, still. If LLM haven't
                # written to anyone at this point, this notification will guide the LLM that dismissive action is not
                # acceptable and LLM will try to revisit some older dialog despite no cue. If it has written already,
                # it will initiate a dialog with one more person.
                self.app.notification_manager.pass_notification_to_ai(
                    Notification(message="You should write someone else and be more proactive.", pin="<act_proactively />",
                                 chat_id=0))
            return "Success"

        phase = "populate"
        while True:
            if phase == "populate":
                self._inject_memory()

            # ---- naxyi_preserve_ctx ----
            self._update_tools(notification.actions)
            if not self.ask_called_this_turn and c().remindUseAsk:
                # remind LLM to call #ask before responding. Injected as a system-level checkpoint so LLM sees it right
                # before generating its next action.
                ctx.last().content += ("\n[system] Have you called #ask yet this turn? If the message involves personal "
                                       "topics, past events, questions, or people you know — call #ask BEFORE "
                                       "send_telegram_message.")
            notification.actions.insert(Tool("pause", "Pauses the conversation", escape))
            notification.actions.insert(Tool("wait", "Wait until further notifications", escape))

            with self.app.metric_breadcrumbs.point("function", "notification processing loop"):
                self.app.save_context(ctx)  # preserve the current turn if the external watchdog must restart a hang
                streaming = self.app.openai.chat_streaming(
                    Params(system_prompt=self.app.get_system_prompt(ctx.chat_id), tools=notification.actions.as_json()), ctx)
                streaming.on_changed(self.app.on_response_assembling)
                await streaming.completed
                answer: Response = streaming.response
                if answer.choices:
                    m = answer.choices[0].message
                    self.turn.step(m.reasoning_content or m.reasoning, m.content,
                                   [(tc.function.name, tc.function.arguments) for tc in m.tool_calls])

            if not answer.choices or not answer.choices[0].message.tool_calls:
                # no tool calls. each LLMs turn should end with "wait" or "pause"
                self.log.warning("LLM didn't perform any action.")
                if answer.choices:
                    # guiderails to make LLM tool-centric.
                    content = answer.choices[0].message.content
                    if "#send_telegram_message" in content:
                        # qwen3.5 bug: misused examples
                        ctx << Message(role=ROLE_USER, content=NICE_THOUGHTS)
                        phase = "preserve"
                        continue
                    if "<message" in content and "</message>" in content:
                        # gemma4 bug: does not perform tool calls, instead, replies with the <message ...> markup
                        ctx << Message(role=ROLE_USER, content=NICE_THOUGHTS + " Call #wait if you are unsure.")
                        phase = "preserve"
                        continue
                # punish llm for not performing tool calls.
                self.turn.note("модель не вызвала ни одного инструмента — её поправили")
                ctx << Message(role=ROLE_USER, content=NICE_THOUGHTS + " Call #wait if you are unsure.")
                phase = "preserve"
                continue

            message = answer.choices[0].message
            if self._process_ignore_chance(message):
                phase = "preserve"
                continue

            tool_results = await notification.actions.handle_tool_calls(
                message.tool_calls, self.app.metric_breadcrumbs, ctx, self.log)
            for tc, result in zip(message.tool_calls, tool_results):
                self.turn.result(tc.function.name, result.content)
            self._reply_sent |= any(tc.function.name == "send_telegram_message"
                                    and result.content.startswith("Message sent successfully")
                                    for tc, result in zip(message.tool_calls, tool_results))
            if any(EMBEDDING_TAG in m.content for m in tool_results):
                # Indicates a low quality tool call. This tag is used as an exception condition within a tool handler:
                # the LLM's tool call is dropped and the last message is sent again.
                if answer.usage.prompt_tokens > c().diaryTokenCountTrigger:
                    # we are stuck; ignore the event
                    self.log.warning("LLM can't find proper response to the notification; context is overflown. "
                                     "Ignoring event and dumping context")
                    await self.diary_dump_messages()
                    return
                phase = "preserve"
                continue
            ctx << message
            ctx << tool_results
            self.log.info("Tool call response: %s", ctx.last().content)

            if pause["flag"]:
                if answer.usage.total_tokens >= c().diaryTokenCountTrigger:
                    await self.diary_dump_messages()
                return
            if notification.actions.handlers():
                ctx.last().content += ("\nWhat's your next action? Use a `tool` to act. Use #ask to consult with your "
                                       "knowledge database. The following tools available: "
                                       + ", ".join(notification.actions.handlers().keys()))
            if any(tc.function.name == "send_telegram_message" for tc in message.tool_calls):
                # if LLM sent a message without ever calling #ask this turn, inject a reminder into the next turn.
                if not self.ask_called_this_turn and c().remindUseAsk:
                    ctx.last().content += ("\n[system] Note: you sent a message without consulting #ask this turn. Next "
                                           "time, call #ask before send_telegram_message to enrich your response with "
                                           "memories and context.")
                phase = "preserve"
            else:
                phase = "populate"

    async def diary_dump_messages(self) -> None:
        self.pins.clear()
        await self.app.diary_dump_messages(self.temporary_context)
