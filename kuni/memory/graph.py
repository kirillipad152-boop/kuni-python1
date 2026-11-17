"""Graph memory (replaces the RAG diary of the C++ kernel).

Model
-----
* **Entity nodes** — people, places, orgs, topics, events, objects, emotions and Kuni herself (type ``self``).
* **Episode nodes** — what the diary pages used to be: a first-person narrative page (``summary`` holds the text).
  Every episode is connected to the entities it talks about with a ``mentions`` edge.
* **Edges** — typed relations between entities with a natural-language ``fact``, a ``weight`` (grows each time the
  fact is re-observed or used) and a ``confidence`` ∈ [-1..1]: -1 lie, 0 theory/default, 1 ground truth
  (immutable by sleep) — the same semantics as the RAG diary had.

Retrieval needs no embeddings and no vector search:

1. **seeds** — entities named explicitly (by the LLM in `#ask`) or recognised in the text by name/alias;
2. **spreading activation** — activation flows from the seeds along edges for a few hops, weighted by edge strength and
   damped by the fan-out of the source node (the ACT-R "fan effect"), so hubs don't flood everything;
3. the activated subgraph is returned as episodes (pages), entities and facts, ranked by activation.

Everything is persisted in SQLite and kept in RAM as a working set (a character's memory is thousands of nodes, not
millions).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from ..util import jaccard, shingles, similar_word, stem_key, stems, tokens

log = logging.getLogger("GraphMemory")

ENTITY_TYPES = ("person", "place", "org", "topic", "event", "object", "emotion", "self")
EPISODE = "episode"
MENTIONS = "mentions"
GROUND_TRUTH = 0.9999
Visibility = Literal["private", "public"]


def validate_scope(source_chat_id: int | None, visibility: Visibility) -> None:
    if source_chat_id is not None and (type(source_chat_id) is not int or not -(2**63) <= source_chat_id < 2**63):
        raise ValueError("source_chat_id must be a SQLite integer or None")
    if visibility not in ("private", "public"):
        raise ValueError("visibility must be private or public")


def clamp_confidence(c: float) -> float:
    return max(-0.99, min(0.99, float(c)))


@dataclass
class Node:
    id: str
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""  # entity description, or the full page text for episodes
    confidence: float = 0.0
    importance: float = 0.5
    score: float = 0.0
    usage_count: int = 0
    created: float = 0.0
    updated: float = 0.0
    last_used: str = "never"
    source_chat_id: int | None = None
    visibility: Visibility = "private"

    @property
    def is_episode(self) -> bool:
        return self.type == EPISODE

    def increment_usage(self) -> None:
        self.usage_count += 1
        self.last_used = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


@dataclass
class Edge:
    id: str
    src: str
    dst: str
    relation: str
    fact: str = ""
    weight: float = 1.0
    confidence: float = 0.0
    score: float = 0.0
    usage_count: int = 0
    created: float = 0.0
    updated: float = 0.0
    last_used: str = "never"
    evidence: str = ""  # id of the episode the fact came from
    source_chat_id: int | None = None
    visibility: Visibility = "private"

    def increment_usage(self) -> None:
        self.usage_count += 1
        self.last_used = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


@dataclass
class Recall:
    seeds: dict[str, float] = field(default_factory=dict)
    entities: list[tuple[Node, float]] = field(default_factory=list)
    episodes: list[tuple[Node, float]] = field(default_factory=list)
    facts: list[tuple[Edge, float]] = field(default_factory=list)

    def empty(self) -> bool:
        return not (self.entities or self.episodes or self.facts)


def norm_name(s: str) -> str:
    t = " ".join(tokens(s))
    return t


def to_relatedness(activation: float) -> float:
    """Squashes raw activation to (0..1) so it reads like a similarity (0 = unrelated, 1 = the thing itself)."""
    return 1.0 - math.exp(-3.0 * max(0.0, activation))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
    id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL, aliases TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0, importance REAL NOT NULL DEFAULT 0.5,
    score REAL NOT NULL DEFAULT 0, usage_count INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL, updated REAL NOT NULL, last_used TEXT NOT NULL DEFAULT 'never',
    source_chat_id INTEGER, visibility TEXT NOT NULL DEFAULT 'private' CHECK(visibility IN ('private', 'public')));
CREATE TABLE IF NOT EXISTS edges(
    id TEXT PRIMARY KEY, src TEXT NOT NULL, dst TEXT NOT NULL, relation TEXT NOT NULL, fact TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 1, confidence REAL NOT NULL DEFAULT 0, score REAL NOT NULL DEFAULT 0,
    usage_count INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, updated REAL NOT NULL,
    last_used TEXT NOT NULL DEFAULT 'never', evidence TEXT NOT NULL DEFAULT '',
    source_chat_id INTEGER, visibility TEXT NOT NULL DEFAULT 'private' CHECK(visibility IN ('private', 'public')));
CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS nodes_type ON nodes(type);
"""


class GraphMemory:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        for table in ("nodes", "edges"):
            columns = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
            if "source_chat_id" not in columns:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN source_chat_id INTEGER")
            if "visibility" not in columns:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private' "
                                 "CHECK(visibility IN ('private', 'public'))")

        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self._adj: dict[str, set[str]] = defaultdict(set)  # node id -> edge ids (both directions)
        self._by_name: dict[str, set[str]] = defaultdict(set)  # normalized name / alias -> scoped entity ids
        self._stem_index: dict[str, set[str]] = defaultdict(set)  # stem -> entity ids
        self._text_docs: dict[str, Counter] | None = None  # lazily built tf-idf over entity descriptions
        self._text_df: Counter = Counter()
        self._shingle_cache: dict[str, set[str]] = {}
        self._last_episode_id = 0
        self._load()

    # ------------------------------------------------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------------------------------------------------
    def _load(self) -> None:
        for r in self._db.execute("SELECT * FROM nodes"):
            n = Node(r["id"], r["type"], r["name"], json.loads(r["aliases"]), r["summary"], r["confidence"],
                     r["importance"], r["score"], r["usage_count"], r["created"], r["updated"], r["last_used"],
                     r["source_chat_id"], r["visibility"])
            validate_scope(n.source_chat_id, n.visibility)
            self.nodes[n.id] = n
            self._index_node(n)
            if n.is_episode and n.id.isdigit():
                self._last_episode_id = max(self._last_episode_id, int(n.id))
        for r in self._db.execute("SELECT * FROM edges"):
            e = Edge(r["id"], r["src"], r["dst"], r["relation"], r["fact"], r["weight"], r["confidence"], r["score"],
                     r["usage_count"], r["created"], r["updated"], r["last_used"], r["evidence"],
                     r["source_chat_id"], r["visibility"])
            validate_scope(e.source_chat_id, e.visibility)
            if e.src in self.nodes and e.dst in self.nodes:
                self.edges[e.id] = e
                self._adj[e.src].add(e.id)
                self._adj[e.dst].add(e.id)
        log.debug("Loaded %d node(s), %d edge(s)", len(self.nodes), len(self.edges))

    def _validate_write(self, item: Node | Edge, table: str) -> None:
        old = self._db.execute(f"SELECT source_chat_id, visibility FROM {table} WHERE id=?", (item.id,)).fetchone()
        try:
            validate_scope(item.source_chat_id, item.visibility)
            if old is not None and (old["source_chat_id"], old["visibility"]) != (item.source_chat_id, item.visibility):
                if not (old["source_chat_id"] is None and old["visibility"] == item.visibility == "private"
                        and item.source_chat_id not in (None, 0)):
                    raise ValueError("memory provenance is immutable except for assigning legacy private memory")
        except ValueError:
            if old is not None:
                item.source_chat_id, item.visibility = old["source_chat_id"], old["visibility"]
            raise

    def _validate_edge_scope(self, e: Edge) -> None:
        for nid in (e.src, e.dst, *([e.evidence] if e.evidence else [])):
            n = self.nodes.get(nid)
            if n is None or not self.same_scope(n, e):
                raise ValueError("edge endpoints and evidence must have the same provenance as the edge")
        if e.evidence and not self.nodes[e.evidence].is_episode:
            raise ValueError("edge evidence must be an episode")

    def _save_node(self, n: Node) -> None:
        self._validate_write(n, "nodes")
        self._db.execute(
            "INSERT OR REPLACE INTO nodes(id,type,name,aliases,summary,confidence,importance,score,usage_count,"
            "created,updated,last_used,source_chat_id,visibility) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n.id, n.type, n.name, json.dumps(n.aliases, ensure_ascii=False), n.summary, n.confidence, n.importance,
             n.score, n.usage_count, n.created, n.updated, n.last_used, n.source_chat_id, n.visibility),
        )

    def _save_edge(self, e: Edge) -> None:
        self._validate_write(e, "edges")
        self._validate_edge_scope(e)
        self._db.execute(
            "INSERT OR REPLACE INTO edges(id,src,dst,relation,fact,weight,confidence,score,usage_count,created,"
            "updated,last_used,evidence,source_chat_id,visibility) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (e.id, e.src, e.dst, e.relation, e.fact, e.weight, e.confidence, e.score, e.usage_count, e.created,
             e.updated, e.last_used, e.evidence, e.source_chat_id, e.visibility),
        )

    def save(self, item: Node | Edge) -> None:
        if isinstance(item, Node):
            item.updated = time.time()
            self._save_node(item)
            self._text_docs = None
        else:
            item.updated = time.time()
            self._save_edge(item)

    def close(self) -> None:
        self._db.close()

    def assign_legacy_chat(self, chat_id: int) -> int:
        """Explicit, idempotent startup migration; null-source private memory is never implicitly public."""
        validate_scope(chat_id, "private")
        if chat_id is None or chat_id == 0:
            raise ValueError("legacy memory requires a nonzero chat_id")
        changed = 0
        self._db.execute("BEGIN IMMEDIATE")
        try:
            for table in ("nodes", "edges"):
                changed += self._db.execute(f"UPDATE {table} SET source_chat_id=? "
                                            "WHERE source_chat_id IS NULL AND visibility='private'", (chat_id,)).rowcount
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise
        for item in [*self.nodes.values(), *self.edges.values()]:
            if item.source_chat_id is None and item.visibility == "private":
                item.source_chat_id = chat_id
        return changed

    @staticmethod
    def same_scope(a: Node | Edge, b: Node | Edge) -> bool:
        return (a.source_chat_id, a.visibility) == (b.source_chat_id, b.visibility)

    @staticmethod
    def can_read(item: Node | Edge, *, chat_id: int | None = None) -> bool:
        """None is trusted admin access; zero is public-only, never a private chat."""
        return (chat_id is None or item.visibility == "public" or
                (chat_id != 0 and item.visibility == "private" and item.source_chat_id == chat_id))

    def _readable(self, item: Node | Edge, chat_id: int | None,
                 scope: tuple[int | None, Visibility] | None = None) -> bool:
        return self.can_read(item, chat_id=chat_id) and (scope is None or (item.source_chat_id, item.visibility) == scope)

    # ------------------------------------------------------------------------------------------------------------
    # indices
    # ------------------------------------------------------------------------------------------------------------
    def _node_names(self, n: Node) -> list[str]:
        return [n.name, *n.aliases]

    def _index_node(self, n: Node) -> None:
        if n.is_episode:
            return
        for name in self._node_names(n):
            key = norm_name(name)
            if key:
                self._by_name[key].add(n.id)
            for t in tokens(name):
                if len(t) >= 3:
                    self._stem_index[stem_key(t)].add(n.id)

    def _unindex_node(self, n: Node) -> None:
        for name in self._node_names(n):
            key = norm_name(name)
            self._by_name.get(key, set()).discard(n.id)
            for t in tokens(name):
                self._stem_index.get(stem_key(t), set()).discard(n.id)

    # ------------------------------------------------------------------------------------------------------------
    # nodes
    # ------------------------------------------------------------------------------------------------------------
    def is_empty(self, *, chat_id: int | None = None) -> bool:
        return not any(self.can_read(n, chat_id=chat_id) for n in self.nodes.values())

    def entities(self, *, chat_id: int | None = None) -> list[Node]:
        return [n for n in self.nodes.values() if not n.is_episode and self.can_read(n, chat_id=chat_id)]

    def episodes(self, *, chat_id: int | None = None) -> list[Node]:
        """Newest first."""
        return sorted((n for n in self.nodes.values() if n.is_episode and self.can_read(n, chat_id=chat_id)),
                      key=lambda n: (n.created, n.id), reverse=True)

    def resolve(self, name: str, fuzzy: bool = False, *, chat_id: int | None = None,
                _scope: tuple[int | None, Visibility] | None = None) -> Node | None:
        """Finds an entity by its id, name or alias (case-insensitive).

        `fuzzy=True` additionally accepts other grammatical forms («Алексея» for «Алексей»). It is meant for LOOKUPS
        (a query, a mention in the text). Creating/merging entities uses the exact mode, so that two different people
        are never glued together by a sloppy match; duplicates that slip through are merged by the sleep.
        """
        if not name:
            return None
        if name in self.nodes:  # a raw id
            n = self.nodes[name]
            return n if self._readable(n, chat_id, _scope) else None
        matches = [self.nodes[nid] for nid in self._by_name.get(norm_name(name), ())
                   if self._readable(self.nodes[nid], chat_id, _scope)]
        if matches:
            return min(matches, key=lambda n: (n.visibility == "public" if chat_id else False, n.created, n.id))
        if not fuzzy:
            return None
        qt = tokens(name)
        if not qt:
            return None
        cands: set[str] | None = None
        for s in {stem_key(t) for t in qt if len(t) >= 3}:
            ids = self._stem_index.get(s, set())
            cands = set(ids) if cands is None else cands & ids
            if not cands:
                return None
        for nid in sorted(cands or ()):
            if not self._readable(self.nodes[nid], chat_id, _scope):
                continue
            for nm in self._node_names(self.nodes[nid]):
                nt = tokens(nm)
                if len(nt) == len(qt) and all(similar_word(a, b) for a, b in zip(qt, nt)):
                    return self.nodes[nid]
        return None

    def upsert_entity(self, name: str, type: str = "topic", aliases: Iterable[str] = (), summary: str = "",
                      confidence: float | None = None, importance: float | None = None, *,
                      source_chat_id: int | None = None, visibility: Visibility = "private") -> tuple[Node, bool]:
        """Returns (node, created)."""
        validate_scope(source_chat_id, visibility)
        name = " ".join(name.split())
        type = type if type in ENTITY_TYPES else "topic"
        existing = self.resolve(name, _scope=(source_chat_id, visibility))
        aliases = [a for a in (" ".join(x.split()) for x in aliases) if a]
        if existing is not None and not existing.is_episode:
            self._unindex_node(existing)
            if norm_name(name) != norm_name(existing.name) and name not in existing.aliases:
                existing.aliases.append(name)
            for a in aliases:
                if norm_name(a) != norm_name(existing.name) and a not in existing.aliases:
                    existing.aliases.append(a)
            if summary and summary.strip() not in existing.summary:
                joined = (existing.summary + " " + summary.strip()).strip()
                existing.summary = joined[-800:]
            if existing.type == "topic" and type != "topic":
                existing.type = type  # a more specific type wins
            if confidence is not None and existing.confidence < GROUND_TRUTH:
                existing.confidence = clamp_confidence(max(existing.confidence, confidence))
            if importance is not None:
                existing.importance = max(existing.importance, importance)
            self.save(existing)
            self._index_node(existing)
            return existing, False
        now = time.time()
        n = Node(id="n" + uuid.uuid4().hex[:10], type=type, name=name,
                 aliases=[a for a in aliases if norm_name(a) != norm_name(name)], summary=summary.strip()[:800],
                 confidence=clamp_confidence(confidence or 0.0), importance=importance if importance is not None else 0.5,
                 created=now, updated=now, source_chat_id=source_chat_id, visibility=visibility)
        self.nodes[n.id] = n
        self._save_node(n)
        self._index_node(n)
        self._text_docs = None
        return n, True

    def add_episode(self, body: str, created: float | None = None, confidence: float = 0.0, *,
                    source_chat_id: int | None = None, visibility: Visibility = "private") -> Node:
        """A diary page. Ids are unix seconds, like diary file names were (and stay unique)."""
        validate_scope(source_chat_id, visibility)
        now = created if created is not None else time.time()
        eid = max(int(now), self._last_episode_id + 1)
        self._last_episode_id = eid
        n = Node(id=str(eid), type=EPISODE, name=f"episode {eid}", summary=body.strip(),
                 confidence=clamp_confidence(confidence), created=now, updated=now,
                 source_chat_id=source_chat_id, visibility=visibility)
        self.nodes[n.id] = n
        self._save_node(n)
        return n

    def is_duplicate_episode(self, body: str, threshold: float, *, source_chat_id: int | None = None,
                             visibility: Visibility = "private") -> Node | None:
        """Plagiarism check: shingle-Jaccard against existing pages. Returns the page it duplicates."""
        validate_scope(source_chat_id, visibility)
        sh = shingles(body)
        if not sh:
            return None
        for ep in self.episodes():
            if (ep.source_chat_id, ep.visibility) != (source_chat_id, visibility):
                continue
            es = self._shingle_cache.get(ep.id)
            if es is None:
                es = self._shingle_cache[ep.id] = shingles(ep.summary)
            sim = jaccard(sh, es)
            if sim > threshold:
                log.info("%s: plagiarism of %s (similarity %.3f)", body[:40].replace("\n", " "), ep.id, sim)
                return ep
        return None

    def remove_node(self, nid: str) -> None:
        n = self.nodes.pop(nid, None)
        if n is None:
            return
        self._unindex_node(n)
        for eid in list(self._adj.get(nid, ())):
            self.remove_edge(eid)
        self._adj.pop(nid, None)
        self._shingle_cache.pop(nid, None)
        self._db.execute("DELETE FROM nodes WHERE id=?", (nid,))
        self._text_docs = None

    # ------------------------------------------------------------------------------------------------------------
    # edges
    # ------------------------------------------------------------------------------------------------------------
    def link(self, src: str, dst: str, relation: str, fact: str = "", confidence: float = 0.0, weight: float = 1.0,
             evidence: str = "", *, source_chat_id: int | None = None, visibility: Visibility = "private") -> Edge | None:
        """Adds a relation, or reinforces an existing identical one (same endpoints + relation)."""
        validate_scope(source_chat_id, visibility)
        if src == dst or src not in self.nodes or dst not in self.nodes:
            return None
        candidate = Edge(id="", src=src, dst=dst, relation=relation, evidence=evidence,
                         source_chat_id=source_chat_id, visibility=visibility)
        self._validate_edge_scope(candidate)
        relation = " ".join(relation.lower().split()) or "related_to"
        for eid in self._adj.get(src, ()):
            e = self.edges[eid]
            if e.src == src and e.dst == dst and e.relation == relation and self.same_scope(e, candidate):
                e.weight += weight  # re-observed: the memory gets stronger
                if fact and fact.strip() != e.fact:
                    e.fact = fact.strip()  # the freshest wording wins
                if e.confidence < GROUND_TRUTH:
                    # corroboration nudges confidence up a little; it never reaches ground truth by itself
                    bumped = max(e.confidence, confidence) + 0.05
                    e.confidence = clamp_confidence(min(bumped, max(e.confidence, 0.9)))
                if evidence:
                    e.evidence = evidence
                self.save(e)
                return e
        now = time.time()
        e = Edge(id="r" + uuid.uuid4().hex[:10], src=src, dst=dst, relation=relation, fact=fact.strip(),
                 weight=weight, confidence=clamp_confidence(confidence), created=now, updated=now, evidence=evidence,
                 source_chat_id=source_chat_id, visibility=visibility)
        self.edges[e.id] = e
        self._adj[src].add(e.id)
        self._adj[dst].add(e.id)
        self._save_edge(e)
        return e

    def remove_edge(self, eid: str) -> None:
        e = self.edges.pop(eid, None)
        if e is None:
            return
        self._adj[e.src].discard(eid)
        self._adj[e.dst].discard(eid)
        self._db.execute("DELETE FROM edges WHERE id=?", (eid,))

    @staticmethod
    def edge_strength(e: Edge) -> float:
        s = 1.0 - 0.5 ** max(e.weight, 0.0)  # weight 1 -> .5, 2 -> .75, 3 -> .875 ...
        if e.confidence < 0:
            s *= max(0.0, 1.0 + e.confidence)  # doubted facts conduct less
        return s

    def neighbors(self, nid: str, relation: str | None = None, *,
                  chat_id: int | None = None) -> list[tuple[Edge, Node]]:
        out = []
        if nid not in self.nodes or not self.can_read(self.nodes[nid], chat_id=chat_id):
            return out
        for eid in self._adj.get(nid, ()):
            e = self.edges[eid]
            if relation and e.relation != relation.lower():
                continue
            other = e.dst if e.src == nid else e.src
            if not self.can_read(e, chat_id=chat_id) or not self.can_read(self.nodes[other], chat_id=chat_id):
                continue
            out.append((e, self.nodes[other]))
        out.sort(key=lambda p: self.edge_strength(p[0]), reverse=True)
        return out

    # ------------------------------------------------------------------------------------------------------------
    # merging (used by sleep)
    # ------------------------------------------------------------------------------------------------------------
    def merge_nodes(self, keep_id: str, drop_id: str) -> bool:
        keep, drop = self.nodes.get(keep_id), self.nodes.get(drop_id)
        if keep is None or drop is None or keep_id == drop_id or keep.is_episode or drop.is_episode:
            return False
        if not self.same_scope(keep, drop):
            return False
        if drop.confidence >= GROUND_TRUTH and keep.confidence < GROUND_TRUTH:
            keep, drop = drop, keep  # never lose a ground-truth node
        for eid in self._adj.get(drop.id, ()):
            self._validate_edge_scope(self.edges[eid])
        self._unindex_node(keep)
        self._unindex_node(drop)
        for a in [drop.name, *drop.aliases]:
            if norm_name(a) != norm_name(keep.name) and a not in keep.aliases:
                keep.aliases.append(a)
        if drop.summary and drop.summary not in keep.summary:
            keep.summary = (keep.summary + " " + drop.summary).strip()[-800:]
        keep.importance = max(keep.importance, drop.importance)
        keep.score += drop.score
        keep.usage_count += drop.usage_count
        if keep.type == "topic":
            keep.type = drop.type
        for eid in list(self._adj.get(drop.id, ())):
            e = self.edges[eid]
            src = keep.id if e.src == drop.id else e.src
            dst = keep.id if e.dst == drop.id else e.dst
            self.remove_edge(eid)
            if src != dst:
                new = self.link(src, dst, e.relation, e.fact, e.confidence, e.weight, e.evidence,
                                source_chat_id=e.source_chat_id, visibility=e.visibility)
                if new:
                    new.usage_count += e.usage_count
                    new.score += e.score
                    self._save_edge(new)
        self.nodes.pop(drop.id, None)
        self._adj.pop(drop.id, None)
        self._db.execute("DELETE FROM nodes WHERE id=?", (drop.id,))
        self.save(keep)
        self._index_node(keep)
        return True

    def dedupe_exact(self) -> int:
        """Merges entities whose normalized names collide (cheap, deterministic part of sleeping)."""
        merged = 0
        seen: dict[tuple[str, int | None, Visibility], str] = {}
        for n in sorted(self.entities(), key=lambda n: n.created):
            name = norm_name(n.name)
            if not name:
                continue
            key = (name, n.source_chat_id, n.visibility)
            if key in seen and seen[key] in self.nodes:
                if self.merge_nodes(seen[key], n.id):
                    merged += 1
                continue
            seen[key] = n.id
        return merged

    # ------------------------------------------------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------------------------------------------------
    def _build_text_index(self) -> None:
        self._text_docs = {}
        self._text_df = Counter()
        for n in self.entities():
            c = Counter(stems(f"{n.name} {' '.join(n.aliases)} {n.summary}"))
            self._text_docs[n.id] = c
            self._text_df.update(c.keys())

    def _text_search(self, query: str, limit: int = 3, *, chat_id: int | None = None,
                     _scope: tuple[int | None, Visibility] | None = None) -> list[tuple[str, float]]:
        """Fallback entry points when no entity is named: keyword overlap with entity descriptions (tf-idf)."""
        if self._text_docs is None:
            self._build_text_index()
        assert self._text_docs is not None
        q = Counter(s for s in stems(query) if len(s) >= 3)
        if not q:
            return []
        docs = {nid: doc for nid, doc in self._text_docs.items() if self._readable(self.nodes[nid], chat_id, _scope)}
        df = self._text_df if chat_id is None and _scope is None else Counter(s for doc in docs.values() for s in doc)
        n_docs = max(len(docs), 1)
        idf = {s: math.log(1 + n_docs / (1 + df.get(s, 0))) for s in q}
        qn = math.sqrt(sum((c * idf[s]) ** 2 for s, c in q.items()))
        scored = []
        for nid, doc in docs.items():
            dot = sum(c * idf[s] * doc.get(s, 0) * idf[s] for s, c in q.items() if s in doc)
            if dot <= 0:
                continue
            dn = math.sqrt(sum((c * math.log(1 + n_docs / (1 + df[s]))) ** 2 for s, c in doc.items()))
            scored.append((nid, dot / (qn * dn)))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:limit]

    def find_seeds(self, text: str = "", entities: Iterable[str] = (), text_search: bool = True, *,
                   chat_id: int | None = None,
                   _scope: tuple[int | None, Visibility] | None = None) -> dict[str, float]:
        seeds: dict[str, float] = {}

        for name in entities:
            n = self.resolve(name, fuzzy=True, chat_id=chat_id, _scope=_scope)
            if n is not None:
                seeds[n.id] = max(seeds.get(n.id, 0.0), 1.0)

        # entities recognised in the text by their name/alias (other grammatical forms are fine)
        text_tokens = tokens(text)
        text_set = set(text_tokens)
        text_norm = " " + " ".join(text_tokens) + " "
        cands: set[str] = set()
        for t in text_tokens:
            if len(t) >= 3:
                cands |= self._stem_index.get(stem_key(t), set())
        for nid in cands:
            n = self.nodes[nid]
            if not self._readable(n, chat_id, _scope):
                continue
            best = 0.0
            for nm in self._node_names(n):
                key = norm_name(nm)
                if key and f" {key} " in text_norm:
                    best = 1.0
                    break
                nt = [t for t in tokens(nm) if len(t) >= 3]
                if not nt:
                    continue
                matched = sum(1 for a in nt if any(similar_word(a, b) for b in text_set))
                coverage = matched / len(nt)
                if coverage >= 0.99 or (len(nt) > 1 and coverage >= 0.6):
                    fan = sum(self._readable(self.nodes[i], chat_id, _scope)
                              for i in self._stem_index.get(stem_key(nt[0]), ()))
                    damping = 1.0 / (1.0 + 0.15 * max(0, fan - 1))
                    best = max(best, (0.85 if len(nt) > 1 else 0.7) * coverage * damping)
            if best > 0.2:
                seeds[nid] = max(seeds.get(nid, 0.0), best)

        if text_search and len(seeds) < 2 and text:
            for nid, sc in self._text_search(text, chat_id=chat_id, _scope=_scope):
                if nid not in seeds and sc > 0.15:
                    seeds[nid] = 0.5 * sc
        return seeds

    def activate(self, seeds: dict[str, float], hops: int = 2, decay: float = 0.8, *,
                 chat_id: int | None = None) -> dict[str, float]:
        """Spreading activation with the fan effect (a hub shares its activation among many neighbours)."""
        act = {nid: a for nid, a in seeds.items() if nid in self.nodes and self.can_read(self.nodes[nid], chat_id=chat_id)}
        frontier = dict(act)
        for _ in range(max(0, hops)):
            nxt: dict[str, float] = defaultdict(float)
            for nid, a in frontier.items():
                es = self.neighbors(nid, chat_id=chat_id)
                if not es:
                    continue
                fan = math.sqrt(len(es))
                for e, other_node in es:
                    strength = self.edge_strength(e)
                    if strength <= 0:
                        continue
                    other = other_node.id
                    nxt[other] += a * decay * strength / fan
            frontier = {}
            for nid, v in nxt.items():
                v = min(v, 0.99)
                if v > act.get(nid, 0.0):
                    act[nid] = v
                    frontier[nid] = v
        return act

    def recall(self, text: str = "", *, entities: Iterable[str] = (), hops: int = 2, confidence_factor: float = 0.0,
               max_episodes: int = 10, max_entities: int = 8, max_facts: int = 12,
               text_search: bool = True, chat_id: int | None = None) -> Recall:
        seeds = self.find_seeds(text, entities, text_search=text_search, chat_id=chat_id)
        out = Recall(seeds=seeds)
        if not seeds:
            return out
        act = self.activate(seeds, hops, chat_id=chat_id)

        ents, eps = [], []
        for nid, a in act.items():
            n = self.nodes[nid]
            rel = to_relatedness(a) + n.confidence * confidence_factor
            (eps if n.is_episode else ents).append((n, rel))
        ents.sort(key=lambda p: p[1], reverse=True)
        eps.sort(key=lambda p: (p[1], p[0].created), reverse=True)

        facts = []
        for e in self.edges.values():
            if e.relation == MENTIONS or not self.can_read(e, chat_id=chat_id):
                continue
            a_src, a_dst = act.get(e.src, 0.0), act.get(e.dst, 0.0)
            if a_src <= 0 or a_dst <= 0:
                continue
            rel = to_relatedness((a_src + a_dst) / 2 * (0.5 + 0.5 * self.edge_strength(e))) + e.confidence * confidence_factor
            facts.append((e, rel))
        facts.sort(key=lambda p: p[1], reverse=True)

        out.entities = ents[:max_entities]
        out.episodes = eps[:max_episodes]
        out.facts = facts[:max_facts]
        return out

    def touch(self, item: Node | Edge, relatedness: float) -> None:
        """Bookkeeping when a memory was actually loaded into context (score drifts with how useful it is)."""
        item.score += (relatedness - 0.5) * 2.0
        item.increment_usage()
        self.save(item)
        if isinstance(item, Edge):
            return

    # ------------------------------------------------------------------------------------------------------------
    # rendering helpers
    # ------------------------------------------------------------------------------------------------------------
    def describe_edge(self, e: Edge) -> str:
        a, b = self.nodes[e.src], self.nodes[e.dst]
        s = f"{a.name} —[{e.relation}]→ {b.name}"
        if e.fact:
            s += f": {e.fact}"
        return s + f" (confidence {e.confidence:.2f}, seen {e.weight:g}x)"

    # ------------------------------------------------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------------------------------------------------
    def stats(self, *, chat_id: int | None = None) -> dict:
        nodes = {n.id: n for n in self.nodes.values() if self.can_read(n, chat_id=chat_id)}
        edges = [e for e in self.edges.values() if self.can_read(e, chat_id=chat_id) and e.src in nodes and e.dst in nodes]
        types = Counter(n.type for n in nodes.values())
        rels = Counter(e.relation for e in edges)
        return {"nodes": len(nodes), "edges": len(edges), "episodes": types.get(EPISODE, 0),
                "entities": len(nodes) - types.get(EPISODE, 0), "types": dict(types),
                "relations": dict(rels.most_common(15))}

    def to_json(self, *, chat_id: int | None = None) -> dict:
        nodes = {n.id: n for n in self.nodes.values() if self.can_read(n, chat_id=chat_id)}
        return {"nodes": [asdict(n) for n in nodes.values()],
                "edges": [asdict(e) for e in self.edges.values()
                          if self.can_read(e, chat_id=chat_id) and e.src in nodes and e.dst in nodes]}

    def ensure_self(self, name: str, *, source_chat_id: int | None = None, visibility: Visibility = "private") -> Node:
        n, _ = self.upsert_entity(name, "self", summary="", confidence=0.99,
                                 source_chat_id=source_chat_id, visibility=visibility)
        return n
