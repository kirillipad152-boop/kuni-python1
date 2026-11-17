"""Context -> graph memory. Port of util/diary_save_entries.h, with graph extraction instead of embedding.

Flow (same as the RAG diary up to the pages, then graph instead of vectors):
  1. LLM writes diary pages (prompts/diary_save.md), pages are separated by `---`;
  2. near-duplicate pages are dropped (plagiarism check);
  3. every page is stored as an *episode* node;
  4. a second LLM call (prompts/graph_extract.md) extracts entities and relations from the page; they are merged into
     the graph and linked to the episode with `mentions` edges.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from ..config import config
from ..llm import ROLE_USER, Message, OpenAIChat, Params, Session
from ..prompts import prompts
from .graph import ENTITY_TYPES, MENTIONS, GraphMemory, Node, Visibility, clamp_confidence, validate_scope

log = logging.getLogger("memory.extract")


def parse_json_object(text: str) -> dict | None:
    """LLMs wrap JSON with fences and chatter; take the outermost {...}."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):  # trailing commas
        try:
            v = json.loads(attempt)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


async def extract_graph(memory: GraphMemory, llm: OpenAIChat, episode: Node, retries: int = 2) -> dict:
    """Extracts entities/relations from an episode and merges them into the graph. Returns counters."""
    if not episode.is_episode or memory.nodes.get(episode.id) is not episode:
        raise ValueError("extraction requires a stored episode")
    validate_scope(episode.source_chat_id, episode.visibility)
    scope = (episode.source_chat_id, episode.visibility)
    provenance = {"source_chat_id": episode.source_chat_id, "visibility": episode.visibility}
    stats = {"entities": 0, "new_entities": 0, "relations": 0}
    parsed: dict | None = None
    known = ""
    for nid in memory.find_seeds(episode.summary, text_search=False, _scope=scope):
        n = memory.nodes[nid]
        also = f"; also known as: {', '.join(n.aliases)}" if n.aliases else ""
        known += f"- {n.name} ({n.type}{also})\n"
    message = f"<page>\n{episode.summary}\n</page>\n"
    if known:
        # entity resolution at ingest: the LLM maps «Алексея»/«Alex» to the canonical node instead of us guessing
        message += ("<known_entities instructions=\"These entities already exist in memory. If the page mentions one of "
                    "them (in any grammatical form, nickname or alias), use EXACTLY the name given here instead of "
                    f"inventing a new spelling.\">\n{known}</known_entities>\n")
    for attempt in range(retries + 1):
        try:
            response = await llm.chat(
                Params(system_prompt=prompts().graph_extract, config=config().llm, max_output_tokens=4096),
                Session([Message(role=ROLE_USER, content=message)]),
            )
            content = response.choices[0].message.content if response.choices else ""
        except Exception as e:  # noqa: BLE001
            log.error("graph extraction request failed for %s: %s", episode.id, e)
            continue
        parsed = parse_json_object(content)
        if parsed is not None:
            break
        log.warning("graph extraction for %s returned unparsable JSON (attempt %d)", episode.id, attempt + 1)

    named: dict[str, Node] = {}
    if parsed:
        for ent in parsed.get("entities") or []:
            if not isinstance(ent, dict) or not str(ent.get("name") or "").strip():
                continue
            node, created = memory.upsert_entity(
                str(ent["name"]), str(ent.get("type") or "topic").lower(), aliases=[str(a) for a in ent.get("aliases") or []],
                summary=str(ent.get("summary") or ""), **provenance)
            named[str(ent["name"]).strip().lower()] = node
            stats["entities"] += 1
            stats["new_entities"] += int(created)

        def ref(name) -> Node | None:
            name = str(name or "").strip()
            if not name:
                return None
            node = named.get(name.lower()) or memory.resolve(name, _scope=scope)
            if node is None or node.is_episode:
                node, created = memory.upsert_entity(name, "topic", **provenance)
                stats["entities"] += 1
                stats["new_entities"] += int(created)
            named.setdefault(name.lower(), node)
            return node

        for rel in (parsed.get("relations") or [])[:30]:
            if not isinstance(rel, dict):
                continue
            a, b = ref(rel.get("source")), ref(rel.get("target"))
            if a is None or b is None or a.id == b.id:
                continue
            e = memory.link(a.id, b.id, str(rel.get("relation") or "related_to"), str(rel.get("fact") or ""),
                            confidence=clamp_confidence(_f(rel.get("confidence"))), evidence=episode.id, **provenance)
            stats["relations"] += int(e is not None)

    # always link the page to what it mentions; cheap lexical pass on top of the LLM's entities
    mentioned = {n.id for n in named.values()}
    for nid in memory.find_seeds(episode.summary, text_search=False, _scope=scope):
        mentioned.add(nid)
    for nid in mentioned:
        memory.link(episode.id, nid, MENTIONS, evidence=episode.id, **provenance)
    return stats


def split_pages(llm_summarization: str) -> list[str]:
    # stupid AI sometimes messes up with separators
    text = llm_summarization.replace("- --", "---").replace("-- -", "---")
    return [p.strip() for p in text.split("---") if len(p.strip()) >= 20]  # shorter ones are random shit


async def save_pages(memory: GraphMemory, llm: OpenAIChat, llm_summarization: str, *,
                     source_chat_id: int | None = None, visibility: Visibility = "private") -> list[Node]:
    """Port of diarySaveEntries(diary, text): stores pages and extracts their graph."""
    validate_scope(source_chat_id, visibility)
    provenance = {"source_chat_id": source_chat_id, "visibility": visibility}
    saved: list[Node] = []
    for page in split_pages(llm_summarization):
        if memory.is_duplicate_episode(page, config().diaryPlagiarismThreshold, **provenance):
            log.info("won't store because it's plagiarism")
            continue
        saved.append(memory.add_episode(page, **provenance))
    if saved:
        sem = asyncio.Semaphore(3)

        async def one(ep: Node) -> None:
            async with sem:
                stats = await extract_graph(memory, llm, ep)
                log.info("episode %s -> %s", ep.id, stats)

        await asyncio.gather(*(one(ep) for ep in saved))
    return saved


async def save_session(memory: GraphMemory, llm: OpenAIChat, context: Session, params: Params, *,
                       source_chat_id: int | None = None, visibility: Visibility = "private") -> list[Node]:
    """Port of diarySaveEntries(diary, context, params): asks the LLM to write the diary from the context."""
    validate_scope(source_chat_id, visibility)
    context = context.copy()
    context << Message(role=ROLE_USER, content=prompts().diary_save)
    for _ in range(5):
        answer = await llm.chat(params, context)
        msg = answer.choices[0].message if answer.choices else Message()
        if not msg.content.strip():
            context << msg
            context << Message(role=ROLE_USER, content="Tools are not available; write the summary without tools.")
            continue
        return await save_pages(memory, llm, msg.content, source_chat_id=source_chat_id, visibility=visibility)
    log.error("LLM kept returning empty diary pages; nothing saved")
    return []
