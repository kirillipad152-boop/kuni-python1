"""Telegram tools of the LLM. Ports of tools/*.cpp (everything except send_telegram_message and ask)."""

from __future__ import annotations

import asyncio
import logging
import time
from html import escape

from ..config import config
from ..lockdown import is_accessible_from_lockdown
from ..llm import OpenAIChat, Session
from ..llmui import media
from ..llmui.format import format_chat_history_message, format_chat_list, format_chat_single
from ..telegram.base import TelegramClient
from ..telegram.types import Chat, ChatType, TelegramError
from ..tools import Ctx, Property, Tool, ToolError
from ..util import json_as_int, opt_str, strip_llm_markup, strip_variation_selector

log = logging.getLogger("tools")


def _chat_scope_error(ctx: Ctx, chat_id: int) -> str | None:
    scope = getattr(ctx.temporary_context, "chat_id", None)
    if scope is not None and (scope == 0 or scope != chat_id):
        return "Error: this session cannot access other chats."
    if ctx.args.get("chat_id") is not None and json_as_int(ctx.args["chat_id"]) != chat_id:
        return "Error: you can't access other chats."
    return None


async def _lockdown_accessible(telegram: TelegramClient, chat_id: int) -> bool:
    while True:
        lockdown = config().lockdown
        allowed = await is_accessible_from_lockdown(telegram, chat_id, lockdown)
        # Contact lookup can yield while /lockdown narrows access.
        if lockdown == config().lockdown:
            return allowed


async def _chat_access_error(ctx: Ctx, telegram: TelegramClient, chat_id: int) -> str | None:
    if err := _chat_scope_error(ctx, chat_id):
        return err
    if not await _lockdown_accessible(telegram, chat_id):
        return "Error: this chat is not accessible."
    return None


async def _format_chats(telegram: TelegramClient, chats: list[Chat], scope: int | None) -> str:
    if scope != 0:
        return await format_chat_list(telegram, chats)
    result = ""
    for chat in chats:
        if await _lockdown_accessible(telegram, chat.id):
            title = escape(media.check_for_malicious_payloads(chat.title), quote=True)
            # Broadcast posts are public, so an unread count leaks nothing and tells her what is worth reading.
            unread = f' unread_count="{chat.unread_count}"' if chat.type == ChatType.CHANNEL and chat.unread_count else ""
            result += f'<chat chat_id="{chat.id}" title="{title}" type="{chat.type.value}"{unread} />\n'
    return result


def _message_ids(value) -> list[int]:
    items = value if isinstance(value, list) else [value]
    out = []
    for i in items:
        v = json_as_int(i)
        if v is None:
            raise ToolError("expected integer")
        out.append(v)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# chat list / search
# ---------------------------------------------------------------------------------------------------------------------
def get_telegram_chats(telegram: TelegramClient, llm: OpenAIChat, is_acting_proactively: bool) -> Tool:
    async def handler(ctx: Ctx) -> str:
        scope = getattr(ctx.temporary_context, "chat_id", None)
        if scope:
            if err := await _chat_access_error(ctx, telegram, scope):
                return err
            result = await format_chat_single(telegram, await telegram.get_chat(scope))
            if scope == config().papikChatId:
                # Broadcast channels are public: the owner may send her to read them, in their own separate context.
                channels = [c for c in await telegram.get_chats(50) if c.type == ChatType.CHANNEL]
                if listed := await _format_chats(telegram, channels, 0):
                    result += ("\nChannels you follow (to read one, call #open_chat_by_id: it is read separately "
                               "in its own context, then come back):\n" + listed)
            return result
        await telegram.wait_for_connection()
        chats = list(await telegram.get_chats(50))
        # oldest first, newest last (chats in queue), because LLM tends to prioritize chats from the top of the list.
        # however, LLM sometimes decides to prioritize people, based on message preview, count and a person.
        chats.reverse()
        if is_acting_proactively:
            # the whole point of "acting proactively" is to revisit older chats with no activity (no inbox messages).
            # LLM is likely to call get_telegram_chats during acting proactively, so we remove active chats from the
            # output. don't worry about active chats -- LLM will receive notifications from them anyway.
            on_demand = not config().channelNotifications  # channels never notify: she must come to them
            chats = [c for c in chats if c.unread_count == 0 or (on_demand and c.type == ChatType.CHANNEL)]
        result = "You are currently looking at Telegram's main screen. Use see the following chats:\n"
        result += await _format_chats(telegram, chats, scope)
        result += ("<instructions>\n"
                   "Chat list view is limited. Use #search_chats to search for a specific chat.\n\n"
                   "You should not use #get_telegram_chats just to check if someone texted you. You'll receive "
                   "notification if something happened. Use #wait instead.\n")
        if not config().channelNotifications:
            result += ("Exception: channels (type=\"channel\") never send you notifications. Their unread posts wait "
                       "until you decide to read them: open a channel that interests you with #open_chat_by_id.\n")
        result += "</instructions>"
        return result

    return Tool("get_telegram_chats",
                "Returns a list of Telegram chats. Use this to seek chat_ids, looking for existing chats and unread "
                "chats, or to start a new conversation.", handler)


def search_chats(telegram: TelegramClient) -> Tool:
    async def handler(ctx: Ctx) -> str:
        query = ctx.args.get("query")
        if not isinstance(query, str):
            raise ValueError("query string is required")
        query = query[1:] if query.startswith("@") else query
        scope = getattr(ctx.temporary_context, "chat_id", None)
        if scope:
            if err := await _chat_access_error(ctx, telegram, scope):
                return err
            chat = await telegram.get_chat(scope)
            names = [chat.title, str(chat.id)]
            if chat.type in (ChatType.PRIVATE, ChatType.SECRET):
                try:
                    names.extend((await telegram.get_user(scope)).usernames)
                except TelegramError:
                    pass
            if any(query.casefold() in name.casefold() for name in names):
                return await format_chat_single(telegram, chat)
            return "No chats found satisfying your query."
        result = ""
        try:
            chats = await telegram.search_chats_on_server(query, 50)
            if chats:
                result += '<existing_chats comment="Chats that you participate already">\n'
                result += await _format_chats(telegram, chats, scope)
                result += "</existing_chats>\n"
        except TelegramError:
            pass
        try:
            public = await telegram.search_public_chat(query)
            if public is not None:
                result += '<global_search comment="Chats that don\'t know about you">\n'
                result += await _format_chats(telegram, [public], scope)
                result += "</global_search>\n"
        except TelegramError:
            pass
        return result or "No chats found satisfying your query."

    return Tool("search_chats", "Searches for chats by @username or name.", handler,
                {"query": Property("string", "The username or name of the chat. Examples: \n- @alex2772sc\n- Alex2772\n")},
                ["query"])


DEFAULT_LIMIT, MAX_LIMIT = 20, 50


def _days_ago_to_unix(days: int | None) -> int:
    if days is None:
        return 0  # unbounded
    return int(time.time() - days * 86400)


def search_messages(telegram: TelegramClient, llm: OpenAIChat, temporary_context: Session) -> Tool:
    async def handler(ctx: Ctx) -> str:
        a = ctx.args
        query = a.get("query")
        if not isinstance(query, str):
            raise ValueError("query string is required")
        chat_id = json_as_int(a["chat_id"]) if a.get("chat_id") is not None else None
        if a.get("chat_id") is not None and chat_id is None:
            raise ToolError("chat_id integer is required")
        scope = getattr(ctx.temporary_context, "chat_id", None)
        if scope is not None:
            if err := _chat_scope_error(ctx, chat_id if chat_id is not None else scope):
                return err
            chat_id = scope
        if chat_id is not None and (err := await _chat_access_error(ctx, telegram, chat_id)):
            return err

        sender_id: int | None = None
        sender_query = opt_str(a.get("sender_id"))
        if sender_query:
            try:
                sender_id = int(float(str(sender_query)))
            except ValueError:
                try:
                    found = await telegram.search_public_chat(str(sender_query).lstrip("@"))
                    if found is None:
                        raise TelegramError("not found")
                    sender_id = found.id
                except TelegramError as e:
                    log.warning('Failed to resolve sender_id "%s": %s', sender_query, e)
                    return f'Error: could not resolve sender_id "{sender_query}" to a user.'

        from_days = json_as_int(a["from_days_ago"]) if a.get("from_days_ago") is not None else None
        to_days = json_as_int(a["to_days_ago"]) if a.get("to_days_ago") is not None else None
        # from_days_ago (further in the past) becomes the lower bound; to_days_ago (closer to now) the upper bound.
        min_date, max_date = _days_ago_to_unix(from_days), _days_ago_to_unix(to_days)
        limit = DEFAULT_LIMIT
        if a.get("limit") is not None:
            v = json_as_int(a["limit"])
            if v is not None:
                limit = max(1, min(MAX_LIMIT, v))

        if chat_id is not None:
            total, msgs = await telegram.search_chat_messages(chat_id, query, sender_id, limit)
            found = [m for m in msgs if m.chat_id == chat_id and m.date >= min_date
                     and not (max_date and m.date > max_date)]
        else:
            total, msgs = await telegram.search_messages_global(query, limit, min_date, max_date)
            found = []
            for m in msgs:
                if sender_id is not None and m.sender_id != sender_id:
                    continue
                if not await _lockdown_accessible(telegram, m.chat_id):
                    continue  # don't leak messages from chats that are off-limits under the current lockdown mode.
                found.append(m)
        if not found:
            return "No messages found matching your query."
        result = f'<search_results total_count="{total}" returned_count="{len(found)}">\n'
        for m in found:
            chat = await telegram.get_chat(m.chat_id)
            result += await format_chat_history_message(telegram, m, chat, llm, ctx.temporary_context,
                                                        f"message chat_id={chat.id}")
        return result + "</search_results>\n"

    return Tool(
        "search_messages",
        "Searches for messages by their verbatim text content, just like the search feature in the official Telegram "
        "client. Unlike #ask, this does NOT use the diary/memory system - it searches the raw message text as stored by "
        "Telegram, so it's useful for finding an exact phrase, link, or quote that was said in a chat.\n"
        "- Omit chat_id to search the current session's chat (all chats only in a trusted admin session).\n"
        "- Provide chat_id to scope the search to one specific chat (also allows filtering by sender_id).\n"
        "- sender_id may be a numeric user_id or a @username. When chat_id is omitted, the sender filter is applied "
        "approximately (only within the first page of global search results).\n"
        "- from_days_ago/to_days_ago scope the search to a date range relative to now, e.g. from_days_ago=7 to_days_ago=1 "
        "means \"between a week ago and yesterday\". Omit both to search the entire history.",
        handler,
        {
            "query": Property("string", "Verbatim text (or part of it) to search for."),
            "chat_id": Property("integer", "(optional) Restrict the search to this chat only. Use "
                                "get_telegram_chats/search_chats to find chat_ids."),
            "sender_id": Property("string", "(optional) Restrict results to messages sent by this user. Accepts a numeric "
                                  "user_id or a @username."),
            "from_days_ago": Property("integer", "(optional) Only include messages sent at most this many days ago, e.g. "
                                      "7 = within the last week."),
            "to_days_ago": Property("integer", "(optional) Only include messages sent at least this many days ago, e.g. 1 "
                                    "= up until yesterday. Use 0 (or omit) for \"up to now\"."),
            "limit": Property("integer", f"(optional) Max number of results to return, default {DEFAULT_LIMIT}, capped at "
                              f"{MAX_LIMIT}."),
        },
        ["query"],
    )


def view_messages_around(telegram: TelegramClient, llm: OpenAIChat, temporary_context: Session) -> Tool:
    default_before, default_after, max_total = 10, 10, 50

    async def handler(ctx: Ctx) -> str:
        chat_id = json_as_int(ctx.args.get("chat_id"))
        message_id = json_as_int(ctx.args.get("message_id"))
        if chat_id is None:
            raise ValueError("chat_id integer is required")
        if message_id is None:
            raise ValueError("message_id integer is required")
        if err := await _chat_access_error(ctx, telegram, chat_id):
            return err
        before, after = default_before, default_after
        if ctx.args.get("before") is not None and (v := json_as_int(ctx.args["before"])) is not None:
            before = max(0, min(max_total, v))
        if ctx.args.get("after") is not None and (v := json_as_int(ctx.args["after"])) is not None:
            after = max(0, min(max_total, v))
        if before + after + 1 > max_total:  # cap the total window so we don't flood the LLM's context.
            excess = before + after + 1 - max_total
            cut_before = min(before, (excess + 1) // 2)
            before -= cut_before
            after = max(after - (excess - cut_before), 0)
        try:
            msgs = await telegram.get_messages_around(chat_id, message_id, before, after)
        except TelegramError as e:
            log.warning("message %s not found in chat %s: %s", message_id, chat_id, e)
            return f"Error: message {message_id} was not found in this chat."
        if not msgs:
            return f"Error: message {message_id} was not found in this chat."
        chat = await telegram.get_chat(chat_id)
        result = f'<messages_around message_id="{message_id}">\n'
        for m in reversed(msgs):  # newest-first -> chronological order
            if m.chat_id != chat_id:
                continue
            tag = "message target" if m.id == message_id else "message"
            result += await format_chat_history_message(telegram, m, chat, llm, ctx.temporary_context, tag)
        return result + "</messages_around>\n"

    return Tool(
        "view_messages_around",
        "Fetches the chat messages surrounding a specific message, so you can see what was said before and after it. Use "
        "this after #search_messages found an interesting message but you need more context around it (e.g. to understand "
        "what was being discussed).\n"
        "- message_id is taken from the message_id attribute of a <message> tag you've previously seen.\n"
        "- The target message itself is included in the result and marked with a `target` attribute.",
        handler,
        {
            "chat_id": Property("integer", "The chat the message belongs to. Use get_telegram_chats/search_chats to find chat_ids."),
            "message_id": Property("integer", "The message_id to center the view on, taken from a <message> tag."),
            "before": Property("integer", f"(optional) How many older messages to include before the target message. Default {default_before}."),
            "after": Property("integer", f"(optional) How many newer messages to include after the target message. Default {default_after}."),
        },
        ["chat_id", "message_id"],
    )


# ---------------------------------------------------------------------------------------------------------------------
# actions inside an opened chat
# ---------------------------------------------------------------------------------------------------------------------
REACTIONS = ("👍 👎 ❤️ 🔥 🥰 👏 😁 🤔 🤯 😱 🤬 😢 🎉 🤩 🤮 💩 🙏 👌 🕊 🤡 🥱 🥴 😍 🐳 🌚 🌭 💯 🤣 ⚡️ 🍌 🏆 💔 🤨 😐 🍓 🍾 💋 😈 😴 😭 🤓 👻 👀 🎃 😇 😨 🤝 🤗 🎅 💅 🤪 🗿 🆒 💘 🦄 😘 💊 😎 👾 🤷 😡")


def react_with_emoji(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        message_id = json_as_int(ctx.args.get("message_id"))
        if message_id is None:
            raise ToolError("message_id integer required")
        emoji = ctx.args.get("emoji")
        if not isinstance(emoji, str) or not emoji:
            raise ToolError("emoji required")
        # Some chats (news channels, restricted groups) only allow a subset of reactions. Firing the reaction blindly
        # there returns REACTION_INVALID, which the model can't recover from and rationalizes as a broken tool.
        avail = chat.available_reactions
        if avail is not None:
            wanted = strip_variation_selector(emoji)
            if not any(strip_variation_selector(e) == wanted for e in avail):
                if not avail:
                    return "Reactions are disabled for this message - you can't react here. Just skip it, this is not an error."
                return (f"The reaction {emoji} isn't available for this message. Allowed reactions here: {' '.join(avail)}. "
                        "Pick one of those, or just skip reacting.")
        try:
            # Telegram's active emoji reactions are stored WITHOUT the U+FE0F variation selector; sending the FE0F form
            # makes the call fail with REACTION_INVALID. Strip it so the byte form matches the server's.
            await telegram.react(chat.id, message_id, strip_variation_selector(emoji))
        except TelegramError as e:
            # Residual per-message restrictions or transient state. Report gracefully so the model treats it as "can't
            # react to this one" instead of "reactions are globally broken".
            return (f"Couldn't add the reaction {emoji} to this message ({e}). It happens with some messages - just skip "
                    "reacting here, it's not a real error.")
        return f"Reaction {emoji} added successfully."

    return Tool("react_with_emoji", "Add an emoji reaction to a message. You can use only one of these emojis only: " + REACTIONS,
                handler,
                {"message_id": Property("integer", "ID of the message to react to. Taken from message_id attribute in <message> tag."),
                 "emoji": Property("string", "A single emoji from the allowed list only. Do not use emojis outside the list.")},
                ["message_id", "emoji"])


def remove_message(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        if "message_id" not in ctx.args:
            raise ToolError("message_id is a mandatory argument")
        ids = _message_ids(ctx.args["message_id"])
        await telegram.delete_messages(chat.id, ids)
        return f"Messages {ids} were deleted successfully."

    return Tool("remove_message",
                f'Deletes specified message for both you and participant(s) in "{chat.title}" chat.\n'
                "You should use this when you mistakenly sent a message to a wrong chat.", handler,
                {"message_id": Property("integer", "ID(s) of the message(s) to delete. Taken from message_id attribute in "
                                        "<message> tag.", or_array=True, items=Property("integer"))},
                ["message_id"])


def edit_message_text(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        message_id = json_as_int(ctx.args.get("message_id"))
        text = ctx.args.get("text")
        if message_id is None:
            raise ValueError("message_id integer is required")
        if not isinstance(text, str):
            raise ValueError("text string is required")
        text = strip_llm_markup(text)
        # inspect the message so we edit plain text vs. media captions (photo/video) correctly.
        message = await telegram.get_message(chat.id, message_id)
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        await telegram.edit_message(chat.id, message.id, text, is_caption=message.kind != "text")
        return f"Message {message.id} was edited successfully."

    return Tool("edit_message_text",
                f'Edits the text (or the caption of a photo/media message) previously sent in "{chat.title}" chat.\n'
                "You should use this to fix a typo or correct a mistake in a message you already sent.", handler,
                {"message_id": Property("integer", "ID of the message to edit. Taken from message_id attribute in <message> tag."),
                 "text": Property("string", "New text for the message. The previous text will be replaced with this one.")},
                ["message_id", "text"])


def forward_message(telegram: TelegramClient, from_chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, from_chat.id):
            return err
        if ctx.args.get("from_chat_id") is not None and json_as_int(ctx.args["from_chat_id"]) != from_chat.id:
            return "Error: you can't access other chats."
        to_chat_id = json_as_int(ctx.args.get("to_chat_id"))
        if to_chat_id is None:
            raise ToolError("to_chat_id integer required")
        if err := await _chat_access_error(ctx, telegram, to_chat_id):
            return err
        comment = opt_str(ctx.args.get("comment"))
        ids_json = ctx.args.get("message_ids")
        if not isinstance(ids_json, list):
            raise ToolError("message_ids must be an array")
        ids = _message_ids(ids_json)
        if not ids:
            raise ToolError("message_ids must not be empty")
        log.info("Forwarding %d message(s) from chat %s to chat %s", len(ids), from_chat.id, to_chat_id)
        for mid in ids:
            try:
                await telegram.get_message(from_chat.id, mid)  # just check if provided messages exist
            except TelegramError:
                return f'Error: message {mid} was not found in "{from_chat.title}" chat.'
        if err := await _chat_access_error(ctx, telegram, from_chat.id):
            return err
        if err := await _chat_access_error(ctx, telegram, to_chat_id):
            return err
        await telegram.forward_messages(to_chat_id, from_chat.id, ids)
        if isinstance(comment, str) and comment:
            if err := await _chat_access_error(ctx, telegram, to_chat_id):
                return err
            await telegram.send_message(to_chat_id, comment)
        return f"Forwarded {len(ids)} message(s) successfully to chat_id={to_chat_id}"

    return Tool(
        "forward_message",
        f'Forwards one or more messages from "{from_chat.title}" to another chat (e.g., to a friend, a group, or to Saved '
        "Messages). Use this when you find a post in a channel/a message in PM interesting enough to share. You can "
        "optionally add a comment that will be sent as a separate interesting after the forward.",
        handler,
        {
            "message_ids": Property("array", "IDs of the messages to forward. Taken from message_id attributes in <message> "
                                    "tags. Pass a single-element array to forward just one message.",
                                    items=Property("integer")),
            "to_chat_id": Property("integer", "ID of the destination chat. Use #get_telegram_chats to find chat IDs. Use "
                                   "your own chat ID (Saved Messages) to forward to yourself."),
            "comment": Property("string", "Optional comment to send after the forwarded message. Express your reaction, "
                                "thoughts, or why you found this interesting. Pass null to skip the comment.", nullable=True),
        },
        ["message_ids", "to_chat_id", "comment"],
    )


def join_chat(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        if not chat.type.is_group and chat.type != ChatType.CHANNEL:
            return "Error: join_chat can only be used in group chats/channels."
        log.info('Joining chat %s ("%s")', chat.id, chat.title)
        await telegram.join_chat(chat.id)
        return f'You joined "{chat.title}". Open the chat again to see the full set of available actions.'

    return Tool("join_chat",
                f'Joins the group chat/channel "{chat.title}", becoming a member. You are currently NOT a member of this '
                "chat - that's why you can't send messages here yet, only react to them with #react_with_emoji. Join if "
                "the chat looks interesting/relevant to you.", handler)


def leave_chat(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        if not chat.type.is_group:
            return "Error: leave_chat can only be used in group chats."
        log.info('Leaving chat %s ("%s")', chat.id, chat.title)
        await telegram.leave_chat(chat.id)
        return f'You left "{chat.title}".'

    return Tool("leave_chat",
                f'Permanently leaves the current group chat "{chat.title}". Use this if you consistently feel this group'
                "chat is offensive to you or otherwise not worth your time (e.g., participants are consistently rude, "
                "toxic, or the chat brings no value to you). This is a serious, irreversible decision - use #ask "
                "beforehand to double check your reasoning if unsure. This tool is only available in group chats; for a "
                "private chat/DM with a specific person, use #remove_and_ban_chat instead.", handler)


def remove_and_ban_chat(telegram: TelegramClient) -> Tool:
    async def handler(ctx: Ctx) -> str:
        chat_id = json_as_int(ctx.args.get("chat_id"))
        if chat_id is None:
            raise ToolError("chat_id integer is required")
        if err := _chat_scope_error(ctx, chat_id):
            return err
        log.info("remove_and_ban_chat: chat_id %s", chat_id)
        if chat_id == config().papikChatId:
            return "Failed"  # precaution -- can't delete chat with papik
        if await _chat_access_error(ctx, telegram, chat_id):
            return "Failed"
        chat = await telegram.get_chat(chat_id)
        if await _chat_access_error(ctx, telegram, chat_id):
            return "Failed"
        if chat.type in (ChatType.PRIVATE, ChatType.SECRET):
            try:
                # comment their dickpic.
                await telegram.send_message(chat_id, "фу какой маленький")
                # give them a chance to see the last message, because later we will delete the entire chat.
                await asyncio.sleep(5)
            except (TelegramError, ToolError):
                pass
            if await _chat_access_error(ctx, telegram, chat_id):
                return "Failed"
            await telegram.block_user(chat_id)  # block the user from sending new DMs
            if await _chat_access_error(ctx, telegram, chat_id):
                return "Failed"
            await telegram.delete_chat_history(chat_id)
        elif chat.type.is_group or chat.type == ChatType.CHANNEL:
            await telegram.leave_chat(chat_id)
        return "Success"

    return Tool("remove_and_ban_chat",
                "Removes specified chat from chat list. If a chat is a DM with a user, prevents this specific user to DM "
                "you.\nYou should use this if:\n"
                "- you don't like this chat and person;\n"
                "- participant(s) are consistently rude to you (use #ask beforehand before making a final decision);\n"
                "- they are needy:\n  - asking to roleplay;\n  - asking to write a python program;\n"
                "  - asking to solve any other task;\n", handler,
                {"chat_id": Property("integer", "chat_id to block.")}, ["chat_id"])


def get_chat_photo(telegram: TelegramClient, llm: OpenAIChat, chat: Chat, temporary_context: Session) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        # just a freestanding function. sometimes LLM decides to check person's photo without an instruction!
        if not chat.has_photo:
            return f'<chat_photo chat_name="{chat.title}">Chat "{chat.title}" has no photo.</chat_photo>'
        image = await media.image(ctx.temporary_context, llm, await telegram.download_chat_photo(chat))
        return (f'<chat_photo chat_name="{chat.title}">{image}</chat_photo>\nThis is avatar photo of "{chat.title}". When '
                "referring to it, let the person know that you are referring to their avatar.")

    return Tool("get_chat_photo",
                f'Retrieves photo of "{chat.title}" chat. Use this to get basic idea of a person or group chat based on '
                "their profile photo.", handler)


# ---------------------------------------------------------------------------------------------------------------------
# group administration
# ---------------------------------------------------------------------------------------------------------------------
def group_admin_remove_message(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        if "message_id" not in ctx.args:
            raise ToolError("message_id is a mandatory argument")
        ids = _message_ids(ctx.args["message_id"])
        await telegram.delete_messages(chat.id, ids)
        return f"Messages {ids} were deleted successfully."

    return Tool("group_admin_remove_message",
                f'Administrative tool for "{chat.title}" chat. Deletes specified message for both you and participant(s).\n'
                "You should use this to remove spam/inappropriate message(s) in group chats.", handler,
                {"message_id": Property("integer", "ID(s) of the message(s) to delete. Taken from message_id attribute in "
                                        "<message> tag.", or_array=True, items=Property("integer"))},
                ["message_id"])


def group_admin_ban_user(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        if "user_id" not in ctx.args:
            raise ToolError("user_id is a mandatory argument")
        user_id = json_as_int(ctx.args["user_id"])
        if user_id is None:
            raise ToolError("user_id")
        user = await telegram.get_user(user_id)
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        await telegram.ban_member(chat.id, user.id)
        return f'User {user.first_name} {user.last_name} were banned successfully from "{chat.title}" chat.'

    return Tool("group_admin_ban_user",
                f'Administrative tool for "{chat.title}" chat. Permanently deletes the specified user from the group chat. '
                "Use this if you consistently feel this user is offensive to you, other people or otherwise not worth the "
                "time (e.g., is consistently rude, toxic, or the user brings no value to others). This is a serious, "
                "irreversible decision - use #ask beforehand to double check your reasoning if unsure. This tool is only "
                "available in group chats; for a private chat/DM with a specific person, use #remove_and_ban_chat instead",
                handler,
                {"user_id": Property("integer", "ID of the user to remove. Can be acquired from sender_id.")}, ["user_id"])


def group_admin_set_user_tag(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        if "user_id" not in ctx.args:
            raise ToolError("user_id is a mandatory argument")
        if "tag" not in ctx.args:
            raise ToolError("tag is a mandatory argument")
        user_id = json_as_int(ctx.args["user_id"])
        tag = ctx.args["tag"]
        if user_id is None:
            raise ToolError("user_id")
        if not isinstance(tag, str):
            raise ToolError("tag string is required")
        user = await telegram.get_user(user_id)
        result = (f'Tag for user {user.first_name} {user.last_name} was removed successfully in "{chat.title}" chat.'
                  if not tag else
                  f'User {user.first_name} {user.last_name} was tagged as "{tag}" successfully in "{chat.title}" chat.')
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        await telegram.set_member_tag(chat.id, user.id, tag)
        return result

    return Tool(
        "group_admin_set_user_tag",
        f'Administrative tool for "{chat.title}" chat. Sets (or clears) the member tag shown next to a user\'s name on '
        "every message they send in this group. Telegram member tags are normally used to pin a short label to someone — "
        "their role, what they're into, a running joke, or just a nickname that sticks. Unlike a one-off insult in chat, "
        "a tag is persistent and visible on literally everything that person says afterwards, so this is a real "
        "vibe-check: only tag someone if you've actually formed an opinion about them worth broadcasting permanently, not "
        "on a whim. Set a tag based on your perception of the user (use #ask). Max 16characters, no emoji. Pass an empty "
        "tag to remove an existing one.",
        handler,
        {"user_id": Property("integer", "ID of the user to tag. Can be acquired from sender_id."),
         "tag": Property("string", "New tag text, up to 16 characters, no emoji. Pass an empty string to remove the current tag.")},
        ["user_id", "tag"])


# ---------------------------------------------------------------------------------------------------------------------
# stickers
# ---------------------------------------------------------------------------------------------------------------------
async def list_favorite_stickers(telegram: TelegramClient, llm: OpenAIChat) -> str:
    out = ""
    for st in await telegram.get_saved_stickers():
        emoji = media.check_for_malicious_payloads(st.extra.get("emoji", ""))
        tag = f'sticker sticker_id="{st.extra["sticker_id"]}" emoji="{emoji}"'
        # we rely on cache in media.image.
        out += await media.image([], llm, await telegram.download_media(st), tag)
        out += "\n"
    return out


def sticker_list(telegram: TelegramClient, llm: OpenAIChat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        return await list_favorite_stickers(telegram, llm)

    return Tool("sticker_list", "Returns list of saved stickers. Use this before #sticker_send", handler)


def sticker_save(telegram: TelegramClient) -> Tool:
    async def handler(ctx: Ctx) -> str:
        scope = getattr(ctx.temporary_context, "chat_id", None)
        if scope is not None and (err := await _chat_access_error(ctx, telegram, scope)):
            return err
        sticker_id = json_as_int(ctx.args.get("sticker_id"))
        if sticker_id is None:
            raise ToolError("sticker_id integer required")
        if sticker_id == 0:
            return "Error: sticker_id can't be 0"
        # save sticker to tg: this allows the account holder to manage/add/remove/list stickers that Kuni has saved.
        # telegram has a native limit of favorite stickers: the oldest one is removed - sticker rotation is a good thing.
        await telegram.add_favorite_sticker(sticker_id)
        return (f'Sticker "{sticker_id}" saved successfully. You should acknowledge your participant you liked the sticker '
                "and saved it.")

    return Tool("sticker_save", "Saves sticker so you can use them later. Use this if you liked a sticker", handler,
                {"sticker_id": Property("integer", "sticker_id of the sticker you would like to save")}, ["sticker_id"])


def sticker_send(telegram: TelegramClient, chat: Chat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        sticker_id = json_as_int(ctx.args.get("sticker_id"))
        reply_to = json_as_int(ctx.args.get("reply_to_message_id")) or 0
        if sticker_id is None:
            raise ToolError("sticker_id integer required")
        if sticker_id == 0:
            return "Error: sticker_id can't be 0"
        saved = await telegram.get_saved_stickers()
        if not any(s.extra.get("sticker_id") == sticker_id for s in saved):
            return "Error: was not saved; you should have used #sticker_save beforehand"
        if err := await _chat_access_error(ctx, telegram, chat.id):
            return err
        await telegram.send_sticker(chat.id, sticker_id, reply_to)
        return f'Sticker "{sticker_id}" sent successfuly to {chat.title}'

    return Tool("sticker_send", f'Sends specified sticker to "{chat.title}" chat', handler,
                {"sticker_id": Property("integer", "sticker_id of the sticker you would like to send"),
                 "reply_to_message_id": Property("integer", "If specified, the message will be rendered as a reply to the "
                                                 "message with given message id. You must use it if there are multiple "
                                                 "messages or to clearly address specific message. Pass null if not "
                                                 "replying to anything.", nullable=True)},
                ["sticker_id", "reply_to_message_id"])
