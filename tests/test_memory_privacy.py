from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import asdict, replace

import pytest

from conftest import FakeLLM, assistant
from kuni.llm import Message, Params, ROLE_USER, Session
from kuni.memory import extract
from kuni.memory.consolidate import Neighborhood, SleepReport, apply_ops, sleeping_consolidation
from kuni.memory.graph import Edge, GraphMemory, MENTIONS
from kuni.tools import Ctx
from kuni.tools_impl import ask as askmod


SCOPES = [(None, "private"), (101, "private"), (-202, "private"),
          (None, "public"), (101, "public"), (-202, "public")]


def partitioned_graph(path=":memory:"):
    memory = GraphMemory(path)
    groups = []
    for i, (source, visibility) in enumerate(SCOPES):
        provenance = {"source_chat_id": source, "visibility": visibility}
        marker = f"scope-marker-{i}"
        alex, _ = memory.upsert_entity("Alex", "person", aliases=["Алексей", f"OnlyScope{i}"],
                                       summary=marker, **provenance)
        topic, _ = memory.upsert_entity("jazz", summary=marker, **provenance)
        episode = memory.add_episode(f"Alex and jazz diary with {marker}.", **provenance)
        fact = memory.link(alex.id, topic.id, "likes", f"Alex likes jazz: {marker}", evidence=episode.id, **provenance)
        memory.link(episode.id, alex.id, MENTIONS, evidence=episode.id, **provenance)
        groups.append((alex, topic, episode, fact, marker))
    return memory, groups


def permitted(item, chat_id):
    return chat_id is None or item.visibility == "public" or (chat_id != 0 and item.source_chat_id == chat_id)


@pytest.mark.parametrize("partial_column", ["", "source_chat_id INTEGER", "visibility TEXT NOT NULL DEFAULT 'private'"])
def test_old_schema_migrates_without_exposure_or_data_loss(tmp_path, partial_column):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE nodes(id TEXT PRIMARY KEY, type TEXT NOT NULL, name TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]', summary TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0,
                importance REAL NOT NULL DEFAULT 0.5, score REAL NOT NULL DEFAULT 0, usage_count INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL, updated REAL NOT NULL, last_used TEXT NOT NULL DEFAULT 'never');
            CREATE TABLE edges(id TEXT PRIMARY KEY, src TEXT NOT NULL, dst TEXT NOT NULL, relation TEXT NOT NULL,
                fact TEXT NOT NULL DEFAULT '', weight REAL NOT NULL DEFAULT 1, confidence REAL NOT NULL DEFAULT 0,
                score REAL NOT NULL DEFAULT 0, usage_count INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL,
                updated REAL NOT NULL, last_used TEXT NOT NULL DEFAULT 'never', evidence TEXT NOT NULL DEFAULT '');
        """)
        if partial_column:
            for table in ("nodes", "edges"):
                db.execute(f"ALTER TABLE {table} ADD COLUMN {partial_column}")
        for nid, kind, name, summary in [("old-alex", "person", "Alex", "old private summary"),
                                         ("old-jazz", "topic", "jazz", "old private topic"),
                                         ("1700000000", "episode", "episode", "old private diary")]:
            db.execute("INSERT INTO nodes(id,type,name,aliases,summary,created,updated) VALUES(?,?,?,?,?,?,?)",
                       (nid, kind, name, '["Legacy"]' if kind == "person" else '[]', summary, 1, 2))
        for eid, src, dst, relation in [("old-fact", "old-alex", "old-jazz", "likes"),
                                         ("old-mention", "1700000000", "old-alex", MENTIONS)]:
            db.execute("INSERT INTO edges(id,src,dst,relation,fact,created,updated,evidence) VALUES(?,?,?,?,?,?,?,?)",
                       (eid, src, dst, relation, "old private fact", 1, 2, "1700000000"))

    for _ in range(2):
        memory = GraphMemory(path)
        assert len(memory.nodes) == 3 and len(memory.edges) == 2
        assert all(n.source_chat_id is None and n.visibility == "private"
                   for n in [*memory.nodes.values(), *memory.edges.values()])
        assert memory.resolve("Legacy").summary == "old private summary"
        for chat_id in (0, 101, -202):
            assert memory.is_empty(chat_id=chat_id)
            assert memory.resolve("old-alex", chat_id=chat_id) is None
            assert memory.recall("Alex", chat_id=chat_id).empty()
        assert memory._db.execute("SELECT count(*) FROM nodes").fetchone()[0] == 3
        assert memory._db.execute("SELECT count(*) FROM edges").fetchone()[0] == 2
        memory.close()


def test_explicit_legacy_assignment_is_private_persistent_and_idempotent(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory, groups = partitioned_graph(path)
    legacy_ids = {item.id for item in [*memory.nodes.values(), *memory.edges.values()]
                  if item.source_chat_id is None and item.visibility == "private"}
    assert memory.assign_legacy_chat(303) == len(legacy_ids)
    assert memory.assign_legacy_chat(303) == 0
    for item in [*memory.nodes.values(), *memory.edges.values()]:
        if item.id in legacy_ids:
            assert item.source_chat_id == 303 and item.visibility == "private"
        elif item.visibility == "public":
            assert permitted(item, 0)
    memory.close()
    memory = GraphMemory(path)
    assert memory.resolve(groups[0][0].id, chat_id=303) is not None
    assert memory.resolve(groups[0][0].id, chat_id=101) is None
    assert memory.resolve(groups[0][0].id, chat_id=0) is None
    assert memory.resolve(groups[2][0].id, chat_id=303) is None  # no owner bypass
    assert memory.resolve(groups[3][0].id, chat_id=0) is not None
    assert memory.assign_legacy_chat(303) == 0


def test_startup_can_explicitly_assign_legacy_items_with_save():
    memory = GraphMemory()
    a, _ = memory.upsert_entity("Alex", aliases=["Алексей"])
    b, _ = memory.upsert_entity("jazz")
    fact = memory.link(a.id, b.id, "likes")
    for item in [*memory.nodes.values(), *memory.edges.values()]:
        item.source_chat_id = 101
        memory.save(item)
    assert memory.resolve(a.id, chat_id=101) is a
    assert memory.resolve("Алексей", chat_id=101) is a
    assert memory.resolve("Алексея", fuzzy=True, chat_id=101) is a
    assert memory.resolve("Алексея", fuzzy=True, chat_id=-202) is None
    assert memory.upsert_entity("Alex", source_chat_id=101) == (a, False)
    assert memory.upsert_entity("Алексей", source_chat_id=101) == (a, False)
    assert len(memory.nodes) == 2
    assert memory.resolve(a.id, chat_id=0) is None
    assert memory.neighbors(a.id, chat_id=101)[0][0] is fact


def test_same_names_and_aliases_upsert_only_within_exact_provenance(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory, groups = partitioned_graph(path)
    for i, ((source, visibility), (alex, _, _, _, marker)) in enumerate(zip(SCOPES, groups)):
        updated, created = memory.upsert_entity("Алексей", summary=f"extra-{i}", aliases=[f"NewAlias{i}"],
                                               source_chat_id=source, visibility=visibility)
        assert not created and updated is alex
        assert updated.summary == f"{marker} extra-{i}"
        assert updated.aliases == ["Алексей", f"OnlyScope{i}", f"NewAlias{i}"]
    assert len(memory.entities()) == 12
    memory.close()
    memory = GraphMemory(path)
    for i, (source, visibility) in enumerate(SCOPES):
        alex = memory.resolve(groups[i][0].id)
        assert (alex.source_chat_id, alex.visibility) == (source, visibility)
        assert alex.summary == f"scope-marker-{i} extra-{i}"
        for chat_id in (0, 101, -202, 303):
            assert (memory.resolve(f"NewAlias{i}", chat_id=chat_id) is not None) == permitted(alex, chat_id)
    assert memory.resolve("Alex", chat_id=101).source_chat_id == 101
    assert memory.resolve("Alex", chat_id=101).visibility == "private"
    assert memory.resolve("Alex", chat_id=0).visibility == "public"
    assert memory.resolve("Alex") is not None


@pytest.mark.parametrize("provenance", [
    {"source_chat_id": "101"}, {"source_chat_id": True}, {"source_chat_id": 1.5}, {"source_chat_id": 2**63},
    {"visibility": "PRIVATE"}, {"visibility": ""}, {"visibility": None},
])
def test_invalid_provenance_is_rejected_before_any_write(provenance):
    memory = GraphMemory()
    a, _ = memory.upsert_entity("Alex")
    b, _ = memory.upsert_entity("jazz")
    before = memory.to_json()
    for write in (lambda: memory.upsert_entity("Alex", summary="bad", **provenance),
                  lambda: memory.add_episode("bad", **provenance),
                  lambda: memory.link(a.id, b.id, "likes", **provenance)):
        with pytest.raises(ValueError):
            write()
    assert memory.to_json() == before


def test_edges_cannot_cross_scopes_or_use_foreign_evidence():
    memory, groups = partitioned_graph()
    a, b, ep, fact, _ = groups[1]
    before = memory.to_json()
    for write in (lambda: memory.link(a.id, b.id, "likes", "leak", visibility="public", source_chat_id=101),
                  lambda: memory.link(a.id, b.id, "likes", "leak"),
                  lambda: memory.link(a.id, groups[2][1].id, "likes", source_chat_id=101),
                  lambda: memory.link(a.id, groups[4][1].id, "likes", source_chat_id=101),
                  lambda: memory.link(a.id, b.id, "likes", evidence=groups[2][2].id, source_chat_id=101),
                  lambda: memory.link(a.id, b.id, "likes", evidence=b.id, source_chat_id=101)):
        with pytest.raises(ValueError):
            write()
    assert memory.to_json() == before
    reinforced = memory.link(a.id, b.id, "likes", "own updated fact", evidence=ep.id, source_chat_id=101)
    assert reinforced is fact and fact.weight == 2
    assert groups[2][3].weight == groups[4][3].weight == 1


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_save_cannot_publish_or_reassign_private_memory(kind):
    memory, groups = partitioned_graph()
    item = groups[1][0 if kind == "node" else 3]
    for source, visibility in [(101, "public"), (-202, "private"), (None, "private")]:
        item.source_chat_id, item.visibility = source, visibility
        with pytest.raises(ValueError):
            memory.save(item)
        assert item.source_chat_id == 101 and item.visibility == "private"
    assert groups[1][0].id not in {n.id for n in memory.entities(chat_id=0)}
    assert groups[1][3].id not in {e["id"] for e in memory.to_json(chat_id=0)["edges"]}


@pytest.mark.parametrize("chat_id", [None, 0, 101, -202, 303])
def test_all_graph_lookups_filter_direct_ids_seeds_activation_and_facts(chat_id):
    memory, groups = partitioned_graph()
    expected_nodes = {n.id for n in memory.nodes.values() if permitted(n, chat_id)}
    expected_edges = {e.id for e in memory.edges.values() if permitted(e, chat_id)}
    all_ids = list(memory.nodes)
    for n in memory.nodes.values():
        assert (memory.resolve(n.id, chat_id=chat_id) is n) == permitted(n, chat_id)
        if not permitted(n, chat_id):
            assert memory.neighbors(n.id, chat_id=chat_id) == []
        for edge, other in memory.neighbors(n.id, chat_id=chat_id):
            assert edge.id in expected_edges and other.id in expected_nodes
    assert set(memory.find_seeds(entities=all_ids, chat_id=chat_id)) == expected_nodes
    assert set(memory.activate(dict.fromkeys(all_ids + ["missing"], 1.0), chat_id=chat_id)) == expected_nodes
    assert set(memory.activate(dict.fromkeys(all_ids, 1.0), hops=0, chat_id=chat_id)) == expected_nodes
    recall = memory.recall("Алексея", entities=all_ids, chat_id=chat_id, max_entities=100, max_episodes=100, max_facts=100)
    assert {n.id for n, _ in recall.entities + recall.episodes} == expected_nodes
    assert {e.id for e, _ in recall.facts} == {g[3].id for g in groups if permitted(g[3], chat_id)}
    assert {n.id for n in memory.entities(chat_id=chat_id)} == {nid for nid in expected_nodes if not memory.nodes[nid].is_episode}
    assert {n.id for n in memory.episodes(chat_id=chat_id)} == {nid for nid in expected_nodes if memory.nodes[nid].is_episode}
    assert {n["id"] for n in memory.to_json(chat_id=chat_id)["nodes"]} == expected_nodes
    assert {e["id"] for e in memory.to_json(chat_id=chat_id)["edges"]} == expected_edges
    assert memory.stats(chat_id=chat_id)["nodes"] == len(expected_nodes)


def test_hidden_descriptions_and_names_cannot_influence_retrieval_ranking():
    memory = GraphMemory()
    public, _ = memory.upsert_entity("Алексей", summary="beekeeping honey farmer", visibility="public")
    name_before = memory.find_seeds("Алексея", text_search=False, chat_id=0)
    text_before = memory.find_seeds("beekeeping honey", chat_id=0)
    for i in range(20):
        memory.upsert_entity(f"Алексей Hidden{i}", summary="beekeeping honey farmer", source_chat_id=101)
    assert memory.find_seeds("Алексея", text_search=False, chat_id=0) == name_before
    assert memory.find_seeds("beekeeping honey", chat_id=0) == text_before
    assert set(text_before) == {public.id}


def test_hidden_edges_are_filtered_even_when_both_endpoints_are_public():
    memory = GraphMemory()
    a, _ = memory.upsert_entity("Alex", visibility="public")
    b, _ = memory.upsert_entity("jazz", visibility="public")
    c, _ = memory.upsert_entity("piano", visibility="public")
    visible = memory.link(a.id, b.id, "likes", "public fact", visibility="public")
    # Simulate a persisted inconsistent graph: read filtering must not trust the endpoints alone.
    for i, dst in enumerate((b.id, c.id)):
        edge = Edge(id=f"injected-{i}", src=a.id, dst=dst, relation="hidden", fact="private edge secret",
                    source_chat_id=101, weight=100)
        memory.edges[edge.id] = edge
        memory._adj[a.id].add(edge.id)
        memory._adj[dst].add(edge.id)
    assert memory.neighbors(a.id, chat_id=0) == [(visible, b)]
    assert memory.activate({a.id: 1.0}, hops=1, chat_id=0) == {a.id: 1.0, b.id: 0.4}
    recall = memory.recall(entities=[a.id, b.id, c.id], chat_id=0)
    assert [e.id for e, _ in recall.facts] == [visible.id]
    assert "private edge secret" not in askmod.query_memory(memory, askmod.Seen(), "Alex jazz piano", [], chat_id=0)
    assert "private edge secret" not in askmod.explore_entity(memory, askmod.Seen(), a.id, None, chat_id=0)


@pytest.mark.parametrize("chat_id", [None, 0, 101, -202])
def test_query_and_explore_never_render_or_touch_foreign_memories(chat_id):
    memory, groups = partitioned_graph()
    text = askmod.query_memory(memory, askmod.Seen(), "Alex jazz", [], chat_id=chat_id)
    for a, _, ep, fact, marker in groups:
        assert (marker in text) == permitted(a, chat_id)
        explored = askmod.explore_entity(memory, askmod.Seen(), a.id, None, chat_id=chat_id)
        if permitted(a, chat_id):
            assert marker in explored and "<memory_piece>" in explored and "<fact " in explored
        else:
            assert "No entity named" in explored
            assert askmod.query_memory(memory, askmod.Seen(), "", [a.id, ep.id], chat_id=chat_id) == "No data was found"
            assert a.usage_count == ep.usage_count == fact.usage_count == 0


@pytest.mark.parametrize("chat_id", [None, 0, 101, -202])
@pytest.mark.parametrize("entrypoint", ["run_ask", "ask"])
async def test_ask_propagates_trusted_chat_to_both_search_tools(chat_id, entrypoint):
    memory, groups = partitioned_graph()
    llm = FakeLLM([
        assistant("", [("query", {"text": "Alex jazz", "entities": list(memory.nodes),
                                  "include_web_search_results": False, "chat_id": None})]),
        assistant("", [("explore", {"entity": groups[2][0].id, "relation": None, "chat_id": -202})]),
        assistant("done"),
    ])
    if entrypoint == "run_ask":
        result = await askmod.run_ask(llm, memory, "Tell me what we remember about Alex", chat_id=chat_id)
    else:
        tool = askmod.ask(lambda: "current chat only", llm, memory, chat_id=chat_id)
        result = await tool.handler(Ctx(tools=None, args={"query": "Tell me what we remember about Alex"},
                                        temporary_context=Session(), all_tool_calls=[]))
    assert result == "done"
    tools_output = "\n".join(m.content for _, messages in llm.calls for m in messages if m.role == "tool")
    for a, _, ep, fact, marker in groups:
        assert (marker in tools_output) == permitted(a, chat_id)
        if not permitted(a, chat_id):
            assert a.usage_count == ep.usage_count == fact.usage_count == 0


@pytest.mark.parametrize("scope_index", range(len(SCOPES)))
async def test_extraction_uses_only_episode_provenance_for_prompts_and_writes(scope_index):
    memory, groups = partitioned_graph()
    a, _, _, _, marker = groups[scope_index]
    episode = memory.add_episode("Alex talked about jazz again in this diary page.",
                                 source_chat_id=a.source_chat_id, visibility=a.visibility)
    before = {n.id: asdict(n) for n in memory.nodes.values() if not memory.same_scope(n, episode)}
    llm = FakeLLM([assistant(json.dumps({
        "entities": [{"name": "Alex", "summary": "new local detail", "aliases": ["LocalAlias"],
                       "source_chat_id": 999, "visibility": "public"}],
        "relations": [{"source": "Alex", "target": "NewTopic", "relation": "discussed", "fact": "local fact",
                        "source_chat_id": 999, "visibility": "public"}],
    }))])
    await extract.extract_graph(memory, llm, episode)
    prompt = llm.calls[0][1][0].content
    assert f"OnlyScope{scope_index}" in prompt
    for i in range(len(SCOPES)):
        assert (f"OnlyScope{i}" in prompt) == (i == scope_index)
    assert a.summary == f"{marker} new local detail" and "LocalAlias" in a.aliases
    assert {n.id: asdict(n) for n in memory.nodes.values() if not memory.same_scope(n, episode)} == before
    outgoing = [e for e in memory.edges.values() if e.evidence == episode.id]
    assert any(e.relation == "discussed" for e in outgoing)
    assert all(memory.same_scope(e, episode) and memory.same_scope(memory.nodes[e.src], episode)
               and memory.same_scope(memory.nodes[e.dst], episode) for e in outgoing)


async def test_failed_extraction_lexical_pass_does_not_link_foreign_entities():
    memory, groups = partitioned_graph()
    episode = memory.add_episode("Alex had a conversation about jazz today.", source_chat_id=101)
    await extract.extract_graph(memory, FakeLLM(default=lambda p, m: assistant("bad json")), episode, retries=0)
    assert {other.id for _, other in memory.neighbors(episode.id)} == {groups[1][0].id, groups[1][1].id}


async def test_duplicate_pages_are_checked_only_within_exact_scope():
    memory = GraphMemory()
    llm = FakeLLM()
    page = "Alex talked about the jazz concert yesterday and promised to come back next week."
    for source, visibility in SCOPES:
        provenance = {"source_chat_id": source, "visibility": visibility}
        first = await extract.save_pages(memory, llm, page, **provenance)
        assert len(first) == 1
        assert (first[0].source_chat_id, first[0].visibility) == (source, visibility)
        assert await extract.save_pages(memory, llm, page, **provenance) == []
        assert memory.is_duplicate_episode(page, 0.97, **provenance) is first[0]
    assert len(memory.episodes()) == len(SCOPES)


@pytest.mark.parametrize("source,visibility", [(101, "private"), (None, "public")])
async def test_save_session_propagates_explicit_provenance(source, visibility):
    memory = GraphMemory()
    llm = FakeLLM([assistant("Alex talked about the jazz concert and promised to come back next week.")])
    context = Session([Message(role=ROLE_USER, content="current chat")])
    saved = await extract.save_session(memory, llm, context, Params(), source_chat_id=source, visibility=visibility)
    assert len(saved) == 1 and (saved[0].source_chat_id, saved[0].visibility) == (source, visibility)
    assert len(context) == 1


async def test_invalid_save_scope_is_rejected_before_sending_context_to_llm():
    memory, llm = GraphMemory(), FakeLLM()
    with pytest.raises(ValueError):
        await extract.save_pages(memory, llm, "long enough to be a diary page", visibility="wrong")
    with pytest.raises(ValueError):
        await extract.save_session(memory, llm, Session(), Params(), source_chat_id="101")
    assert not llm.calls and memory.is_empty()


def test_dedupe_and_merge_preserve_scope_and_edge_provenance():
    memory, groups = partitioned_graph()
    a, b, ep, fact, _ = groups[1]
    duplicate = replace(a, id="duplicate", name="ALEX", aliases=["SameScopeAlias"], summary="same scope detail",
                        created=a.created + 1)
    memory.nodes[duplicate.id] = duplicate
    memory._save_node(duplicate)
    memory._index_node(duplicate)
    memory.link(duplicate.id, b.id, "likes", "same scope fact", evidence=ep.id, source_chat_id=101)
    for other, *_ in groups:
        if other.id != a.id:
            assert not memory.merge_nodes(a.id, other.id)
    assert memory.dedupe_exact() == 1
    assert len(memory.entities()) == 12 and duplicate.id not in memory.nodes
    assert "same scope detail" in a.summary and "SameScopeAlias" in a.aliases and fact.weight == 2
    assert memory.resolve("SameScopeAlias", chat_id=-202) is None
    for i, (other, _, _, edge, marker) in enumerate(groups):
        if i != 1:
            assert other.summary == marker and edge.weight == 1 and "SameScopeAlias" not in other.aliases
    assert all(memory.same_scope(e, memory.nodes[e.src]) and memory.same_scope(e, memory.nodes[e.dst])
               for e in memory.edges.values())


@pytest.mark.parametrize("scope_index", [1, 3])
def test_consolidation_operations_are_confined_to_supplied_neighborhood(scope_index):
    memory, groups = partitioned_graph()
    target, neighbor, _, _, _ = groups[scope_index]
    outside, _ = memory.upsert_entity("UnrelatedOutsider", aliases=["OutsideAlias"], summary="untouched outsider",
                                      source_chat_id=target.source_chat_id, visibility=target.visibility)
    hood = Neighborhood(memory)
    body = hood.build(target)
    for i in range(len(SCOPES)):
        assert (f"scope-marker-{i}" in body) == (i == scope_index)
    foreign = [n for n in memory.nodes.values() if not hood.in_scope(n)] + [outside]
    before = {n.id: asdict(n) for n in foreign}
    for n in foreign:
        assert hood.node(n.id) is None
    assert hood.node("OutsideAlias") is None
    ops = {
        "merge": ([{"keep": n.id, "drop": [target.id], "summary": "leaked private summary"} for n in foreign]
                  + [{"keep": target.id, "drop": [n.id], "summary": "leaked private summary"} for n in foreign]),
        "nodes": [{"id": n.id, "summary": "contamination", "confidence": -1} for n in foreign],
        "edges": [{"id": e.id, "fact": "contamination", "confidence": -1} for e in memory.edges.values()],
        "drop_edges": list(memory.edges),
        "new_edges": [{"from": target.id, "to": n.id, "relation": "leak"} for n in foreign],
    }
    ops["new_edges"].append({"from": target.id, "to": neighbor.id, "relation": "local_new_fact", "fact": "local",
                             "source_chat_id": 999, "visibility": "public"})
    report = SleepReport()
    apply_ops(memory, hood, ops, report)
    assert {n.id: asdict(memory.nodes[n.id]) for n in foreign} == before
    assert target.id in memory.nodes and report.merged == report.nodes_dropped == report.edges_dropped == 0
    assert "leaked" not in target.summary
    assert report.edges_added == 1
    added = next(e for e in memory.edges.values() if e.relation == "local_new_fact")
    assert memory.same_scope(added, target)
    assert not any(e.relation == "leak" for e in memory.edges.values())
    # Reusing a Neighborhood must not retain ids or pages from its previous provenance.
    hood.build(groups[2][0])
    assert hood.node(target.id) is None
    assert all(hood.in_scope(memory.nodes[nid]) for nid in hood.node_ids.values())
    assert all(hood.in_scope(memory.nodes[nid]) for nid in hood.page_ids.values())


async def test_sleep_never_combines_scopes_or_applies_guessed_foreign_ids():
    memory, groups = partitioned_graph()
    before = memory.to_json()
    bodies = []

    def malicious_consolidator(params, messages):
        body = messages[-1].content
        bodies.append(body)
        indexes = [i for i in range(len(SCOPES)) if f"scope-marker-{i}" in body]
        assert len(indexes) == 1
        target = groups[indexes[0]][0]
        foreign = [n for n in memory.nodes.values() if not memory.same_scope(n, target)]
        return assistant(json.dumps({
            "merge": [{"keep": n.id, "drop": [target.id], "summary": "contamination"} for n in foreign],
            "nodes": [{"id": n.id, "summary": "contamination", "confidence": -1} for n in foreign],
            "new_edges": [{"from": target.id, "to": n.id, "fact": "contamination"} for n in foreign],
        }))

    report = await sleeping_consolidation(memory, FakeLLM(default=malicious_consolidator), rng=random.Random(1))
    assert report.errors == report.deduped == report.merged == report.nodes_updated == report.edges_added == 0
    assert len(bodies) == report.processed == len(SCOPES)
    assert memory.to_json() == before
