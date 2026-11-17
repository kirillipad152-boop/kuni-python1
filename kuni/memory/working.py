"""Working memory: the "middle" layer of human memory — things that matter for 1-3 days. Port of
util/important_things_to_remember.h + the file handling of AppBase."""

from __future__ import annotations

import logging
import re
from datetime import timedelta

from ..config import config
from ..llm import ROLE_USER, Message, OpenAIChat, Params, Session
from ..util import format_past_hours

log = logging.getLogger("memory.working")

# deepseek bug: attempts to use DSML to make a tool call instead of writing the memory
_TOOL_CALL_ATTEMPT = re.compile(r"tool_calls|DSML|#ask\b|\bask\(")


def build_prompt(previous_working_memory: str) -> str:
    name = config().characterName
    prompt = f"What are important things in timespan {format_past_hours(timedelta(hours=72))} (3 days) you should remember?\n"
    prompt += ("Do not attempt to make tool_calls or #ask. Your job is to summarize your current tasks and "
               "revisit tasks from previous session.\n")
    if previous_working_memory:
        prompt += (
            "\nHere is the PREVIOUS <things_to_remember> from the last session. "
            "You MUST preserve ALL items verbatim from it, except:\n"
            "1. Completed tasks — mark them as done or remove\n"
            "2. Items that have NOT been updated for more than 3 days — you may forget them\n"
            "\nPrevious working memory:\n<previous_things_to_remember>\n"
            f"{previous_working_memory}"
            "\n</previous_things_to_remember>\n"
            "Important: the content of previous_things_to_remember will be overwritten by your next response. Make sure to preserve:"
            "- unfisnihed tasks (not older than 3 days);"
            "- reminders (not older that 3 days or without a deadline)\n"
        )
    prompt += (
        "\nYou must include:\n"
        "- promises\n"
        "- reminders\n"
        "- unfinished tasks\n"
        "- responsibilities\n"
        f"- {name}'s current emotional state: sad, arousal, betrayed, grief, sorrow, pensiveness, anger, annoyed, "
        "  fear, amazement, surprise, distracted, love, trust, accepted, joy, serenity, embarrassed, proud, "
        "  envious, jealous, grateful, disappointed, hopeful, lonely, nostalgic, relieved, resentful\n"
        f"- {name}'s current physical state: energetic, tired, exhausted, sleepy, restless, tense, relaxed, "
        "  hungry, thirsty, sore, dizzy, nauseous, feverish, cold, warm, comfortable, uncomfortable, "
        "  in_pain, injured, healthy, sick, weak, stiff, trembling, numb, wet, dry, stimulated\n"
        "- other important details\n"
        "In your response, you must include previous <things_to_remember></things_to_remember>. DO NOT ALTER "
        "THEIR DESCRIPTION, thought you can expand them. You should not include older, "
        "outdated items and completed tasks.\n"
        "You must write briefly (100-500 words), structurize output, INCLUDE DATES.\n"
        "Each item MUST have a \"last updated\" date. "
        "Example format:\n"
        "- Напомнить Алексею про оплату хостинга до 15 мая — последнее обновление: Apr 28\n"
        "- Жду ответ от Марии по поводу встречи в пятницу — последнее обновление: Apr 29\n"
        "- Обещала скинуть рецепт пасты Коду — последнее обновление: Apr 27\n"
        "- Проверить статус заказа на Ozon — последнее обновление: Apr 25\n"
    )
    return prompt


async def important_things_to_remember(system_prompt: str, llm: OpenAIChat, context: Session,
                                       previous_working_memory: str, max_attempts: int = 8) -> str:
    context = context.copy()
    context << Message(role=ROLE_USER, content=build_prompt(previous_working_memory))
    dsml_check_triggered = False
    for _ in range(max_attempts):
        response = await llm.chat(Params(system_prompt=system_prompt, config=config().llm), context)
        content = response.choices[0].message.content if response.choices else ""
        if _TOOL_CALL_ATTEMPT.search(content) and not dsml_check_triggered:
            dsml_check_triggered = True
            continue
        content = content.replace("<things_to_remember>", "").replace("</things_to_remember>", "")
        content = content.lstrip()
        if not content:
            continue  # sometimes llm returns blank response - try again
        return content
    raise RuntimeError("LLM returned no usable working memory")
