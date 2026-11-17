"""Sleeping consolidation for the graph memory. Port of Diary::sleepingConsolidation.

Emulates what the brain does at night: replays recent memories (80%) and occasionally random old ones (20%), checks
them against each other, merges duplicates, rewrites what got messy, lowers the confidence of contradicted facts and
forgets lies. We don't need to process everything — neither does a human brain; the sleep is time-boxed.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from ..config import SLEEP_MAX_TIME_SECS, config
from ..llm import ROLE_USER, Message, OpenAIChat, Params, Session
from ..prompts import prompts
from .extract import parse_json_object
from ..util import stem_key, tokens
from .graph import GROUND_TRUTH, MENTIONS, Edge, GraphMemory, Node, Visibility, clamp_confidence, norm_name

log = logging.getLogger("memory.sleep")

MAX_NEIGHBORS = 12
MAX_DUPLICATE_CANDIDATES = 4
MAX_EDGES = 40
MAX_PAGES = 3
PAGE_EXCERPT = 600


@dataclass
class SleepReport:
    processed: int = 0
    merged: int = 0
    nodes_updated: int = 0
    edges_updated: int = 0
    edges_dropped: int = 0
    edges_added: int = 0
    nodes_dropped: int = 0
    errors: int = 0
    deduped: int = 0
    reached_max_time: bool = False


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class Neighborhood:
    """A bounded piece of the graph, with short ids so the LLM doesn't have to copy long uuids."""

    memory: GraphMemory
    node_ids: dict[str, str] = field(default_factory=dict)  # short -> real
    edge_ids: dict[str, str] = field(default_factory=dict)
    page_ids: dict[str, str] = field(default_factory=dict)
    _rev_nodes: dict[str, str] = field(default_factory=dict)
    _scope: tuple[int | None, Visibility] | None = field(default=None, init=False)

    def in_scope(self, item: Node | Edge) -> bool:
        return self._scope is not None and (item.source_chat_id, item.visibility) == self._scope

    def short_node(self, real: str) -> str:
        if real not in self._rev_nodes:
            s = f"n{len(self._rev_nodes) + 1}"
            self._rev_nodes[real] = s
            self.node_ids[s] = real
        return self._rev_nodes[real]

    def build(self, target: Node) -> str:
        m = self.memory
        if m.nodes.get(target.id) is not target or target.is_episode:
            raise ValueError("consolidation requires a stored entity")
        self.node_ids.clear()
        self.edge_ids.clear()
        self.page_ids.clear()
        self._rev_nodes.clear()
        self._scope = (target.source_chat_id, target.visibility)
        members = [target]
        # possible duplicates: entities with a similar name, even if nothing connects them to the target yet
        for name in [target.name, *target.aliases]:
            for t in tokens(name):
                if len(t) < 3:
                    continue
                for nid in sorted(m._stem_index.get(stem_key(t), ())):  # noqa: SLF001
                    other = m.nodes.get(nid)
                    if (other is not None and self.in_scope(other) and other not in members
                            and len(members) <= MAX_DUPLICATE_CANDIDATES):
                        members.append(other)
        for e, other in m.neighbors(target.id):
            if e.relation == MENTIONS or other.is_episode or not self.in_scope(e) or not self.in_scope(other):
                continue
            if other not in members:
                members.append(other)
            if len(members) > MAX_NEIGHBORS:
                break
        member_ids = {n.id for n in members}
        out: list[str] = []
        for n in members:
            sid = self.short_node(n.id)
            aliases = f' aliases="{_esc(", ".join(n.aliases))}"' if n.aliases else ""
            body = f"{n.name}" + (f" — {n.summary}" if n.summary else "")
            out.append(f'<node id="{sid}" type="{n.type}" confidence="{n.confidence:.2f}"{aliases}>{_esc(body)}</node>')

        seen_edges: list[Edge] = []
        for nid in member_ids:
            for eid in m._adj.get(nid, ()):  # noqa: SLF001
                e = m.edges[eid]
                if e.relation == MENTIONS or not self.in_scope(e):
                    continue
                if e.src in member_ids and e.dst in member_ids and e not in seen_edges:
                    seen_edges.append(e)
        seen_edges.sort(key=lambda e: m.edge_strength(e), reverse=True)
        for i, e in enumerate(seen_edges[:MAX_EDGES], 1):
            sid = f"r{i}"
            self.edge_ids[sid] = e.id
            out.append(f'<edge id="{sid}" from="{self.short_node(e.src)}" to="{self.short_node(e.dst)}" '
                       f'relation="{_esc(e.relation)}" confidence="{e.confidence:.2f}" seen="{e.weight:g}">{_esc(e.fact)}</edge>')

        pages: dict[str, Node] = {}
        for n in members:
            for e, other in m.neighbors(n.id, MENTIONS):
                if other.is_episode and self.in_scope(e) and self.in_scope(other):
                    pages[other.id] = other
        for i, ep in enumerate(sorted(pages.values(), key=lambda p: p.created, reverse=True)[:MAX_PAGES], 1):
            self.page_ids[f"p{i}"] = ep.id
            out.append(f'<page id="p{i}">{_esc(ep.summary[:PAGE_EXCERPT])}</page>')
        return "\n".join(out)

    def node(self, ref) -> Node | None:
        ref = str(ref)
        real = self.node_ids.get(ref)
        for nid in ([real] if real else self.node_ids.values()):
            n = self.memory.nodes.get(nid)
            if n is not None and self.in_scope(n) and (real or ref == n.id or
                    any(norm_name(ref) == norm_name(name) for name in [n.name, *n.aliases])):
                return n
        return None

    def edge(self, ref) -> Edge | None:
        e = self.memory.edges.get(self.edge_ids.get(str(ref), ""))
        return e if e is not None and self.in_scope(e) else None


def _mutable(item: Node | Edge) -> bool:
    return item.confidence < GROUND_TRUTH


def apply_ops(memory: GraphMemory, hood: Neighborhood, ops: dict, report: SleepReport) -> None:
    """Applies the consolidator's JSON. Everything with confidence=1 is immutable."""
    if hood.memory is not memory:
        raise ValueError("neighborhood belongs to another memory")
    for m in ops.get("merge") or []:
        keep = hood.node(m.get("keep")) if isinstance(m, dict) else None
        if keep is None:
            continue
        drops = [hood.node(d) for d in m.get("drop") or []]
        if not drops or any(drop is None or drop.id == keep.id or drop.is_episode or not _mutable(drop) for drop in drops):
            continue
        merged = False
        for drop in drops:
            if memory.merge_nodes(keep.id, drop.id):
                report.merged += 1
                merged = True
        if (merged and isinstance(m.get("summary"), str) and m["summary"].strip()
                and _mutable(keep) and keep.id in memory.nodes):
            keep.summary = m["summary"].strip()[:800]
            memory.save(keep)

    for u in ops.get("nodes") or []:
        n = hood.node(u.get("id")) if isinstance(u, dict) else None
        if n is None or n.is_episode or not _mutable(n):
            continue
        if "confidence" in u:
            try:
                c = float(u["confidence"])
            except (TypeError, ValueError):
                c = n.confidence
            if c <= -0.99999:
                memory.remove_node(n.id)
                report.nodes_dropped += 1
                continue
            n.confidence = clamp_confidence(c)
        if isinstance(u.get("summary"), str):
            n.summary = u["summary"].strip()[:800]
        if isinstance(u.get("aliases"), list):
            memory._unindex_node(n)  # noqa: SLF001
            n.aliases = [str(a) for a in u["aliases"] if str(a).strip()]
            memory._index_node(n)  # noqa: SLF001
        memory.save(n)
        report.nodes_updated += 1

    for u in ops.get("edges") or []:
        e = hood.edge(u.get("id")) if isinstance(u, dict) else None
        if e is None or not _mutable(e):
            continue
        if "confidence" in u:
            try:
                c = float(u["confidence"])
            except (TypeError, ValueError):
                c = e.confidence
            if c <= -0.99999:
                memory.remove_edge(e.id)
                report.edges_dropped += 1
                continue
            e.confidence = clamp_confidence(c)
        if isinstance(u.get("fact"), str) and u["fact"].strip():
            e.fact = u["fact"].strip()
        memory.save(e)
        report.edges_updated += 1

    for ref in ops.get("drop_edges") or []:
        e = hood.edge(ref)
        if e is not None and _mutable(e):
            memory.remove_edge(e.id)
            report.edges_dropped += 1

    for ne in ops.get("new_edges") or []:
        if not isinstance(ne, dict):
            continue
        a, b = hood.node(ne.get("from") or ne.get("source")), hood.node(ne.get("to") or ne.get("target"))
        if a is None or b is None or a.id == b.id or a.id not in memory.nodes or b.id not in memory.nodes:
            continue
        try:
            c = clamp_confidence(float(ne.get("confidence") or 0.0))
        except (TypeError, ValueError):
            c = 0.0
        if memory.link(a.id, b.id, str(ne.get("relation") or "related_to"), str(ne.get("fact") or ""), confidence=c,
                       source_chat_id=a.source_chat_id, visibility=a.visibility):
            report.edges_added += 1


async def sleeping_consolidation(memory: GraphMemory, llm: OpenAIChat, max_secs: float = SLEEP_MAX_TIME_SECS,
                                 rng: random.Random | None = None) -> SleepReport:
    rng = rng or random.Random()
    report = SleepReport()
    report.deduped = memory.dedupe_exact()
    started = time.monotonic()

    # recent first, old last
    queue = sorted((n for n in memory.entities()), key=lambda n: n.updated, reverse=True)
    while queue:
        if time.monotonic() - started >= max_secs:
            # we reached sleep max time. we don't need to process all entries, neither does a human brain.
            report.reached_max_time = True
            break
        # during your sleep it's likely you see the past couple of days. however, have you encountered some random
        # shit from 2 years ago where you screwed up on a party? the random branch emulates exactly this.
        idx = 0 if rng.random() < 0.8 else rng.randrange(len(queue))
        target = queue.pop(idx)
        if target.id not in memory.nodes:  # merged away meanwhile
            continue
        hood = Neighborhood(memory)
        body = hood.build(target)
        log.info("[%d/%d] sleepingConsolidation: %s", report.processed + 1, report.processed + 1 + len(queue), target.name)
        try:
            response = await llm.chat(
                Params(system_prompt=prompts().sleep_consolidator, config=config().llm, max_output_tokens=4096),
                Session([Message(role=ROLE_USER, content=body)]),
            )
            ops = parse_json_object(response.choices[0].message.content if response.choices else "")
        except Exception as e:  # noqa: BLE001
            log.error("sleepingConsolidation can't chat: %s", e)
            report.errors += 1
            continue
        report.processed += 1
        if ops is None:
            log.warning("sleepingConsolidation: unparsable response for %s", target.name)
            report.errors += 1
            continue
        apply_ops(memory, hood, ops, report)
        # whatever was considered together doesn't need to be reconsidered in this sleep
        considered = set(hood.node_ids.values())
        queue = [n for n in queue if n.id not in considered]
    return report
