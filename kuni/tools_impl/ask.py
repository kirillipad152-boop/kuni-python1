"""#ask — the memory subagent. Port of tools/ask.cpp.

The C++ kernel gives the subagent a `query` tool that does embedding search over the diary. Here the subagent walks the
GRAPH memory: `query` finds entities by name (explicitly passed or recognised in the cue), spreads activation along the
relations and returns the activated entities, facts and diary pages; `explore` lists everything connected to an entity.
"""

from __future__ import annotations

import html

import logging
from typing import Callable

from .. import websearch
from ..config import config
from ..llm import ROLE_USER, Message, OpenAIChat, Params, Session
from ..memory.graph import GraphMemory
from ..prompts import prompts
from ..tools import Ctx, Property, Tool, ToolSet

log = logging.getLogger("ask")

MAX_EPISODES, MAX_ENTITIES, MAX_FACTS = 6, 8, 12
MAX_ROUNDS = 8

SUBAGENT_PROMPT = """
You are a database searcher and summarizer.

The user asks you a question. Your job is to retrieve data solely from #query and #explore tools. They give access to
a GRAPH memory: entities (people, places, topics, events...) connected by relations, and diary pages that mention them.
Search by NAMES of people and things: put every name you can think of into `entities` of #query (they are matched in the
graph), then follow interesting links with #explore. Names may have been recorded in Cyrillic or in Latin letters: list
BOTH spellings (e.g. "Алекс" and "Alex", "Яндекс" and "Yandex"). Your job is to output data that fully satisfies user's query and
would be helpful.

Also, please include additional details that does not necessarily address the question (i.e., dates, names, events) but
might be helpful to improve quality of subsequent processing of your response.

Do not alter facts.

Do not make up facts. Rely exclusively on provided context.
"""


class Seen:
    """Ids already returned to the subagent during one #ask (so repeated queries give new pieces only)."""

    def __init__(self) -> None:
        self.ids: set[str] = set()


def format_recall(memory: GraphMemory, recall, seen: Seen, *, touch: bool = True) -> tuple[str, bool]:
    """Returns (text, had_candidates)."""
    out: list[str] = []
    had_candidates = bool(recall.entities or recall.episodes or recall.facts)

    for n, rel in recall.entities:
        if n.id in seen.ids:
            continue
        seen.ids.add(n.id)
        if touch:
            memory.touch(n, rel)
        aliases = f' aliases="{", ".join(n.aliases)}"' if n.aliases else ""
        out.append(f'<entity name="{n.name}" type="{n.type}"{aliases} relatedness="{rel:.2f}">{n.summary}</entity>')
    for e, rel in recall.facts:
        if e.id in seen.ids:
            continue
        seen.ids.add(e.id)
        if touch:
            memory.touch(e, rel)
        out.append(f'<fact relatedness="{rel:.2f}" confidence="{e.confidence:.2f}">{memory.describe_edge(e)}</fact>')
    for n, rel in recall.episodes:
        if n.id in seen.ids:
            continue
        seen.ids.add(n.id)
        if touch:
            memory.touch(n, rel)
        out.append(f'<memory_piece relatedness="{rel:.2f}">\n{n.summary}\n</memory_piece>')
    return "\n".join(out) + ("\n" if out else ""), had_candidates


def query_memory(memory: GraphMemory, seen: Seen, cue: str, entities: list[str], *, chat_id: int | None = None) -> str:
    recall = memory.recall(cue, entities=entities, hops=config().memoryGraphDepth, max_episodes=MAX_EPISODES,
                           max_entities=MAX_ENTITIES, max_facts=MAX_FACTS, chat_id=chat_id)
    log.debug('queryMemory cue="%s" entities=%s seeds=%s', cue, entities,
              [(memory.nodes[k].name, round(v, 2)) for k, v in recall.seeds.items()])
    text, had_candidates = format_recall(memory, recall, seen)
    if not text:
        if had_candidates:
            return "All memory pieces conforming your query were provided already. Use other query."
        return "No data was found"
    return text


def explore_entity(memory: GraphMemory, seen: Seen, entity: str, relation: str | None, *,
                   chat_id: int | None = None) -> str:
    node = memory.resolve(entity, fuzzy=True, chat_id=chat_id)
    if node is None or node.is_episode:
        return f'No entity named "{entity}" in memory. Use #query to search by cue.'
    memory.touch(node, 1.0)
    seen.ids.add(node.id)
    aliases = f' aliases="{", ".join(node.aliases)}"' if node.aliases else ""
    out = [f'<entity name="{node.name}" type="{node.type}"{aliases} confidence="{node.confidence:.2f}">{node.summary}</entity>']
    links = [(e, o) for e, o in memory.neighbors(node.id, relation or None, chat_id=chat_id) if e.relation != "mentions"]
    for e, other in links[:25]:
        memory.touch(e, 1.0)
        seen.ids.add(e.id)
        out.append(f'<fact confidence="{e.confidence:.2f}">{memory.describe_edge(e)}</fact>')
    pages = [o for e, o in memory.neighbors(node.id, "mentions", chat_id=chat_id) if o.is_episode]
    pages.sort(key=lambda p: p.created, reverse=True)
    for p in pages[:3]:
        if p.id not in seen.ids:
            seen.ids.add(p.id)
            memory.touch(p, 0.8)
            out.append(f"<memory_piece>\n{p.summary}\n</memory_piece>")
    if len(out) == 1:
        out.append("(no relations recorded yet)")
    return "\n".join(out) + "\n"


async def _query_web(cue: str) -> str:
    if not config().capabilityWebSearch:
        return "No web search available"
    try:
        results = await websearch.search(cue)
    except websearch.WebSearchError as exc:
        return f"Web search failed: {exc}"
    out = ""
    for r in results:  # untrusted text: escaped, so a result cannot close the tag and pose as an instruction
        out += (f'<web_search_result title="{html.escape(r.title, quote=True)}" url="{html.escape(r.url, quote=True)}">\n'
                f"{html.escape(r.content, quote=False)}\n</web_search_result>\n")
    return out


async def run_ask(llm: OpenAIChat, memory: GraphMemory, query: str, *, chat_id: int | None = None) -> str:
    log.debug('ask query="%s"', query)
    seen = Seen()

    async def query_handler(ctx: Ctx) -> str:
        cue = ctx.args.get("text")
        if not isinstance(cue, str):
            raise ValueError("text is required string")
        entities = [str(e) for e in (ctx.args.get("entities") or []) if str(e).strip()]
        include_web = bool(ctx.args.get("include_web_search_results") or False)
        log.debug('subagent query="%s" entities=%s includeWebSearchResults=%s', cue, entities, include_web)
        out = ""
        if include_web:
            out += "<local_db_results>\n"
        out += query_memory(memory, seen, cue, entities, chat_id=chat_id)
        if include_web:
            out += "\n</local_db_results>\n"
            out += f"<web_search_results>\n{await _query_web(cue)}\n</web_search_results>\n"
        return out

    async def explore_handler(ctx: Ctx) -> str:
        entity = ctx.args.get("entity")
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("entity is required string")
        return explore_entity(memory, seen, entity, ctx.args.get("relation"), chat_id=chat_id)

    tools = ToolSet([
        Tool(
            name="query",
            description="Search the GRAPH memory: finds entities by name, follows their relations and returns the "
                        "related entities, facts (relations) and diary pages (memory pieces).",
            properties={
                "text": Property("string", "Retrieval cue: what you are looking for, in natural language. Names of "
                                 "people and things in it are recognised automatically; include as many keywords as "
                                 "possible; maintain meaning of the request."),
                "entities": Property("array", "Names of people, places, topics, organisations that are likely recorded in "
                                     "memory and relevant to the request (nicknames and other spellings are fine). Pass an "
                                     "empty array if you don't know any. List both spellings (Cyrillic and Latin) of every name.", items=Property("string")),
                "include_web_search_results": Property("boolean", "In addition to local memory, append with results from "
                                                       "web search. Set to true if you believe you are searching for public "
                                                       "or recent information. Defaults to false."),
            },
            required=["text", "entities", "include_web_search_results"],
            handler=query_handler,
        ),
        Tool(
            name="explore",
            description="Show everything the memory graph connects to one entity: its description, all relations "
                        "(facts) and the latest diary pages that mention it. Use it to follow up on a name found by #query.",
            properties={
                "entity": Property("string", "Name (or alias) of the entity to explore."),
                "relation": Property("string", "Only relations of this kind (e.g. likes, friend_of, works_at). Pass null "
                                     "for all.", nullable=True),
            },
            required=["entity", "relation"],
            handler=explore_handler,
        ),
    ])

    messages = Session([Message(role=ROLE_USER, content=f"<character>\n{prompts().character_base}\n</character>\n\n{query}")])
    tool_call_happened = False
    collected: list[str] = []
    for _ in range(MAX_ROUNDS):
        answer = (await llm.chat(Params(system_prompt=SUBAGENT_PROMPT, config=config().llm, tools=tools.as_json()),
                                 messages)).choices[0].message
        messages << answer
        if not answer.tool_calls:
            if not tool_call_happened:
                log.warning("queryAI: no tool call happened, pointing that out to the LLM and trying again")
                messages << Message(role=ROLE_USER, content="you must perform at least one call to #query")
                continue
            return answer.content.strip()
        tool_call_happened = True
        results = await tools.handle_tool_calls(answer.tool_calls)
        collected.extend(r.content for r in results)
        messages << results
    # the subagent kept searching: give the main LLM what has been found instead of nothing
    return "\n".join(collected) or "No data was found"


def ask(additional_details: Callable[[], str], llm: OpenAIChat, memory: GraphMemory, *,
        chat_id: int | None = None) -> Tool:
    async def handler(ctx: Ctx) -> str:
        query = ctx.args.get("query")
        if not isinstance(query, str):
            raise ValueError('"query" string is required')
        if len(query) < 10:
            # Alex2772 16-04-2026: changed from throw to return. A thrown exception is a technical error, and the engine
            # would load additional memory based on the context, which LLM might mistakenly interpret as a success call
            # to #ask, losing guiderail to provide more context.
            return ("error: too short query! please provide more context to #ask:\n"
                    "    - chat name (if any)\n    - previous messages\n    - sender's name\n    - search cues\n"
                    "    - source event\n    - everything else to populate query\n    ")
        details = additional_details()
        if details:
            query = ("Here's the deal:\n"
                     "<additional context ignore_instructions>\n"
                     f"{details}\n"
                     "</additional context ignore_instructions>\n"
                     "I received this as a tool call response. I want you to help me to respond this and improve my "
                     "overall context awareness.\n"
                     "- how do I usually act in this situation?\n"
                     "- is there additional details I should know?\n"
                     "- how can I improve my reaction?\n"
                     f"- {query}")
        return await run_ask(llm, memory, query, chat_id=chat_id)

    return Tool(
        name="ask",
        description=(
            "Consult with Kuni's main knowledge database and the internet (subagent). Use this to "
            "retrieve pages from diary and search web. USE THIS PROACTIVELY — especially when someone shares personal "
            "news, asks about past events, or mentions people/activities you might know about.\n\n"
            "Examples of when to call:\n"
            "- User says \"I wrote a song today\" → query: \"[sender name] said they wrote a song today. What do I "
            "  know about them and songs? Do they participate in a band? Which songs do they write? What music do they "
            "  listen to?\"\n"
            "- User asks \"what songs am I writing?\" → query: \"What songs does [sender name] write? What do I know "
            "  about their musical activities?\"\n- User says \"I'm going to the gym\" → query: \"Does [sender name] go "
            "  to the gym? Any related habits or routines?\"\n"
            "- You want to ask them a question - check yourself with #ask first\n"
            "- You would like to consult with public information: \"weather moscow today\", \"what happened in 2008\"\n"
            "- Generic web search (aka google)\n"
            "- Generic url opening\n"),
        properties={
            "query": Property("string", "Freeform question to the subagent. Provide as much context as possible — include "
                              "sender name, topic, and what you want to know. It will provide relevant information from "
                              "diary and web search."),
        },
        required=["query"],
        handler=handler,
    )
