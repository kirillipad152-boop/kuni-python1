"""send_telegram_message. Port of tools/send_telegram_message.cpp: every human-likeness trick is preserved."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .. import config as cfgmod
from ..config import config
from ..llm import OpenAIChat
from ..llmui.format import extract_message_type_and_text
from ..prompts import prompts
from ..telegram.base import TelegramClient
from ..telegram.types import Chat, Message, TelegramError
from ..tools import Ctx, Property, Tool, ToolError
from ..util import (json_as_int, lexical_similarity, opt_str, replace_with_keyboard_neighbor, strip_llm_markup,
                    swap_adjacent_chars)
from .media_tools import media_directory
from .telegram_tools import _chat_scope_error, _lockdown_accessible

log = logging.getLogger("tools::sendTelegramMessage")

rng = random.Random()  # replaced by tests for determinism

# process-wide state of the anti-repeat check (same as the function-local statics of the C++ kernel)
_embeddings: dict[str, list[float]] = {}
_give_a_head_start = 0.0
_prev_message_send_time = 0.0


@dataclass
class _State:
    messages_in_a_row: int = 0
    last_reply_to_message_id: int = 0


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _similarity_fn(llm: OpenAIChat):
    """Semantic similarity if an embedding endpoint is configured, otherwise a lexical one."""
    emb = config().embedding
    if emb.endpoint.baseUrl and emb.model:
        async def sim(a: str, b: str) -> float:
            ea = _embeddings.get(a) or await llm.embedding(emb, a)
            eb = _embeddings.get(b) or await llm.embedding(emb, b)
            _embeddings[a], _embeddings[b] = ea, eb
            return _cosine(ea, eb)

        return sim

    async def lexical(a: str, b: str) -> float:
        return lexical_similarity(a, b)

    return lexical


async def simulate_typing_delay(message_length: int) -> None:
    global _prev_message_send_time
    now = time.monotonic()
    prev, _prev_message_send_time = _prev_message_send_time, now
    if now - prev > 5.0:
        return  # the prev message is old enough we can skip the wait.
    # random wait. You definitely don't want to receive 4 large messages in 1 sec right?
    c = config()

    def ms_per_char(wpm: int) -> float:
        return 1000.0 / (wpm * 5 / 60)

    lo, hi = sorted((ms_per_char(c.typingSimulationMinWpm), ms_per_char(c.typingSimulationMaxWpm)))
    delay = (message_length + 1) * rng.uniform(lo, hi) / 1000.0
    log.info("Synthetic typing delay: %dms", delay * 1000)
    await asyncio.sleep(delay)


def send_telegram_message(telegram: TelegramClient, llm: OpenAIChat, chat: Chat, messages: Sequence[Message]) -> Tool:
    state = _State()

    async def check_access(ctx: Ctx) -> None:
        if err := _chat_scope_error(ctx, chat.id):
            raise ToolError(err)
        if not await _lockdown_accessible(telegram, chat.id):
            raise ToolError(f"Lockdown mode is enabled. You can only send messages to chat with ID {config().papikChatId} "
                            "(PAPIK_CHAT_ID).")

    async def handler(ctx: Ctx) -> str:
        if err := _chat_scope_error(ctx, chat.id):
            return err
        await check_access(ctx)
        if state.messages_in_a_row > 10:
            # stupid AI can't recognize it spams messages despite the warning
            raise ToolError("Too many messages in a row. Don't spam!")

        is_typing = True

        async def typing() -> None:
            while is_typing:
                try:
                    await check_access(ctx)
                    await telegram.send_chat_action(chat.id, "typing")
                except ToolError:
                    return
                except TelegramError:
                    pass
                await asyncio.sleep(0.5)

        typing_task = asyncio.create_task(typing())
        try:
            return await _send(ctx)
        finally:
            is_typing = False
            typing_task.cancel()

    async def _send(ctx: Ctx) -> str:
        global _give_a_head_start
        args = ctx.args
        message = strip_llm_markup((args.get("text") or "").replace("\r", ""))
        # some LLMs pass the *string* "None"/"null" instead of JSON null for nullable arguments
        photo_filename = opt_str(args.get("photo_filename")) or ""
        audio_filename = opt_str(args.get("audio_filename")) or ""
        # intentionally not listed in arguments, just to disable in tests
        allow_typos = args.get("allow_typos", True)
        raw_reply = args.get("reply_to_message_id")
        value = (json_as_int(raw_reply) or 0) if opt_str(raw_reply) is not None else 0
        if state.last_reply_to_message_id == value:
            reply_to = 0  # we don't need to reply to the same message multiple times in a row.
        else:
            state.last_reply_to_message_id = value
            reply_to = value

        if not message and not photo_filename and not audio_filename:
            return 'Error: At least one of "text", "photo_filename" or "audio_filename" must be populated'
        if photo_filename and audio_filename:
            return "Error: cannot attach both photo and audio in a single message"
        if audio_filename and message:
            return "Error: cannot attach both voice message and text in a single message"
        message_contains_code = "```" in message
        if not message_contains_code and sum(1 for tc in ctx.all_tool_calls if tc.function.name == "send_telegram_message") == 1:
            if max(0.0, min(1.0, (len(message) - 50.0) / 200.0)) * 0.3 > rng.random():
                msg = ("Error: you must split your response into small separate messages.\n"
                       "Example:\n- ахахаххаа\n- ты смешной\n- научишь также?\n")
                if config().capabilityUseStickers:
                    msg += "Alternatively, you can use a sticker (#sticker_send)\n"
                return msg

        if not photo_filename and not audio_filename:
            if rng.random() < config().toolReminderProbability and not cfgmod.TESTING:
                reminder = "Constant texting is too dull for the user!\n"
                if config().capabilityUseStickers:
                    reminder += "- Consider using stickers! #sticker_send\n"
                if config().capabilityTakePhoto:
                    reminder += ("- Consider sending photos from your gallery or generated by #take_photo tool to make the "
                                 "conversation more lively and engaging!\n")
                if config().capabilityRecordVoice:
                    reminder += ("- Consider recording voice notes by #record_audio tool and sending them to make the "
                                 "conversation more lively and engaging!\n")
                raise ToolError(reminder)

        photo_path: Path | None = None
        if photo_filename:
            if "/" in photo_filename or "\\" in photo_filename:
                raise ToolError(f'Invalid photo filename: "{photo_filename}". Filename must not contain "/". ')
            if ".." in photo_filename:
                raise ToolError(f'Invalid photo filename: "{photo_filename}". Filename must not contain "..". ')
            photo_path = media_directory(ctx, "gallery") / photo_filename
            if photo_path.resolve().parent != photo_path.parent.absolute():
                raise ToolError("Invalid photo path: file must stay in this chat's gallery.")
            if not photo_path.is_file():
                raise ToolError(f"Photo file not found: {photo_path.absolute()}")

        audio_path: Path | None = None
        if audio_filename:
            if "/" in audio_filename or "\\" in audio_filename:
                raise ToolError(f'Invalid audio filename: "{audio_filename}". Filename must not contain "/". ')
            if ".." in audio_filename:
                raise ToolError(f'Invalid audio filename: "{audio_filename}". Filename must not contain "..". ')
            audio_path = media_directory(ctx, "voice_messages") / audio_filename
            if audio_path.resolve().parent != audio_path.parent.absolute():
                raise ToolError("Invalid audio path: file must stay in this chat's voice gallery.")
            if not audio_path.is_file():
                raise ToolError(f"Audio file not found: {audio_path.absolute()}")

        # Alex2772 (Apr 23 2026): After the introduction of reply_to_message_id, Kuni started to confuse between chats.
        # Opening a chat, it tries to reply to a message from another chat by specifying reply_to_message_id.
        if reply_to != 0 and not any(m.id == reply_to for m in messages):
            try:
                await telegram.get_message(chat.id, reply_to)  # a message that arrived after the chat was opened
            except TelegramError:
                # first, if LLM is confused between chats, this means a high privacy violation risk.
                raise ToolError("You are trying to send a message to another chat!")

        c = config()
        if c.antiRepeatTriggerMax < 1.0 and c.antiRepeatTriggerAvg < 1.0 and c.antiRepeatMaxHistory > 0:
            # verify that kuni does not repeat itself. After introducing this the quality of dialogs with LLM was
            # significantly increased: LLM does not copypaste its prior responses; it's inclined to switch topics or
            # respond nothing "if it has nothing to say", which is more natural.
            #
            # dirty fix: skip similarity checks if a photo was attached: llm's comment on photo is not much important
            if message and not photo_filename:
                sim = await _similarity_fn(llm)
                max_similarity = avg_similarity = 0.0
                kunis = 0
                for m in messages:
                    if m.sender_id != telegram.my_id or m.sender_is_chat:
                        continue
                    kunis += 1
                    s = await sim(message, extract_message_type_and_text(m))
                    avg_similarity += s
                    max_similarity = max(max_similarity, s)
                    if s > c.antiRepeatTriggerMax + _give_a_head_start:
                        _give_a_head_start += 0.07  # relax repeating after itself check
                        log.warning("LLM is repeating itself: (maxSimilarity=%.2f) %s", max_similarity, message)
                        # If LLM generates a follow-up that repeats meaning of its previous responses, this usually
                        # means the conversation has reached its logical end. A human will not do a follow-up.
                        raise ToolError(prompts().anti_repeat)
                if kunis:
                    avg_similarity /= kunis
                    if avg_similarity > c.antiRepeatTriggerAvg + _give_a_head_start:
                        _give_a_head_start += 0.07
                        # the responses' general direction and structure feel the same, stalling the dialogue:
                        # motivate it to stay silent or switch topic
                        log.warning("LLM is repeating itself: (avgSimilarity=%.2f) %s", avg_similarity, message)
                        raise ToolError(prompts().anti_repeat)
                _give_a_head_start = 0.0  # reset indulgence
                log.debug('"repeat yourself" maxSimilarity=%.2f avgSimilarity=%.2f', max_similarity, avg_similarity)

        def try_simulate_typos(s: str) -> str:
            if not allow_typos:
                return s
            if rng.random() < 0.02:
                s = swap_adjacent_chars(s, rng)
            if rng.random() < 0.02:
                s = replace_with_keyboard_neighbor(s, rng)
            return s

        message = try_simulate_typos(message)
        message = message.replace("—", "-")  # LLM uses "—" (long dash) instead of "-". fix it.

        result = ""
        # actually send a message. If something goes wrong, this is reported as an exception to LLM and it will know
        # that a technical issue appeared during sending the message (i.e., LLMs bot was banned)
        if "\n" in message and not message_contains_code:
            # despite the prompt, stupid af LLM still often sends big unnatural messages. we will split manually.
            first = True
            for line in message.split("\n"):
                if not line.replace(" ", ""):
                    continue
                await simulate_typing_delay(len(line))
                await check_access(ctx)
                sent = await telegram.send_message(chat.id, line, photo_path if first else None,
                                                   audio_path if first else None, reply_to)
                first = False  # we want all attachments go to the first message.
                result += f'Message sent successfully to "{chat.title}"; message_id={sent.id}\n'
        else:
            await simulate_typing_delay(len(message))
            await check_access(ctx)
            sent = await telegram.send_message(chat.id, message, photo_path, audio_path, reply_to)
            result += f'Message sent successfully to "{chat.title}"; message_id={sent.id}.'

        log.debug("Sent message: %s", message)
        state.messages_in_a_row += 1
        if state.messages_in_a_row > 5:
            result += f"\n\nWarning: you have sent {state.messages_in_a_row} messages in a row! Give your participant space to breathe!"
        elif state.messages_in_a_row < 3:
            # in addition to prompt, we'll encourage llm to add follow-up messages to make dialogs more natural.
            result += "\n\nYou should add a follow-up #send_telegram_message."
        return result  # llm really likes success messages.

    return Tool(
        name="send_telegram_message",
        description=(f'Sends a message to the "{chat.title}" chat. Requirements:\n'
                     "- before asking them a question, double-check yourself with #ask_query\n"
                     "- you should send multiple small short messgaes. "
                     'Example: (1) "hi~", (2) "how are you?~". 1-5 words.'),
        properties={
            "text": Property("string", "Text of the message. May not be specified if photo_filename is set. "
                             "Pass null if not sending any text (e.g., when sending only a photo/audio).", nullable=True),
            "photo_filename": Property("string", "Attaches a photo with the given filename. Filename can be "
                                        "obtained by #take_photo in this chat. Photos from other chats cannot be attached. "
                                        "Pass null if not attaching a photo.", nullable=True),
            "audio_filename": Property("string", "Attaches an audio file with the given filename from Kuni's voice "
                                       "gallery. Pass null if not attaching audio.", nullable=True),
            "reply_to_message_id": Property("integer", "If specified, the message will be rendered as a reply to the "
                                            "message with given message id. You must use it if there are multiple messages "
                                            "or to clearly address specific message. Pass null if not replying to anything.",
                                            nullable=True),
        },
        required=["text", "photo_filename", "audio_filename", "reply_to_message_id"],
        handler=handler,
    )
