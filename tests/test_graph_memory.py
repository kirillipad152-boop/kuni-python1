from __future__ import annotations

import json
import re

import pytest

from conftest import FakeLLM, assistant
from kuni.llm import Message, ROLE_USER, Session
from kuni.memory import extract
from kuni.memory.consolidate import Neighborhood, SleepReport, apply_ops, sleeping_consolidation
from kuni.memory.graph import GROUND_TRUTH, MENTIONS, GraphMemory
from kuni.tools_impl import ask as askmod


def sample_graph() -> GraphMemory:
    g = GraphMemory()
    alex, _ = g.upsert_entity("Alex2772", "person", aliases=["Алекс", "Алексей"], summary="owner of the instance")
    jazz, _ = g.upsert_entity("jazz", "topic", aliases=["джаз"])
    piano, _ = g.upsert_entity("piano", "object", aliases=["пианино"])
    yandex, _ = g.upsert_entity("Yandex", "org", aliases=["Яндекс"])
    g.link(alex.id, jazz.id, "likes", "Alex likes jazz", 0.3)
    g.link(alex.id, piano.id, "plays", "Alex plays the piano", 0.3)
    g.link(alex.id, yandex.id, "works_at", "Alex works at Yandex", 0.3)
    ep = g.add_episode("Алексей рассказал, что любит джаз и играет на пианино. Мы обсуждали музыку весь вечер.")
    for n in (alex, jazz, piano):
        g.link(ep.id, n.id, MENTIONS)
    return g


# ---------------------------------------------------------------------------------------------------------------------
# entities
# ---------------------------------------------------------------------------------------------------------------------
def test_resolve_exact_mode_never_guesses_and_fuzzy_mode_handles_inflections():
    g = sample_graph()
    for name in ("Alex2772", "alex2772", "Алекс", "АЛЕКСЕЙ"):  # name / alias, any case
        assert g.resolve(name).name == "Alex2772", name
    for name in ("алексея", "Алексею", "Алексеем"):
        assert g.resolve(name) is None, name  # creating entities uses exact matches only
        assert g.resolve(name, fuzzy=True).name == "Alex2772", name  # lookups tolerate other grammatical forms
    assert g.resolve("Яндексе", fuzzy=True).name == "Yandex"
    assert g.resolve("nobody", fuzzy=True) is None and g.resolve("") is None


def test_similar_word_is_conservative():
    from kuni.util import similar_word

    for a, b in [("алексей", "алексея"), ("алексей", "алексеем"), ("яндекс", "яндексе"), ("мария", "марию"),
                 ("иван", "ивана"), ("song", "songs"), ("алекс", "алексей")]:
        assert similar_word(a, b) and similar_word(b, a), (a, b)
    for a, b in [("иван", "иванов"), ("alexander", "alexandra"), ("node1", "node2"), ("nodea", "nodeb"),
                 ("мария", "марина"), ("алексей", "алексис"), ("max", "maxx"), ("анна", "анна2")]:
        assert not similar_word(a, b), (a, b)


def test_different_people_with_similar_names_are_never_glued_together():
    g = GraphMemory()
    for name in ("Alexander", "Alexandra", "Иван", "Иванов", "Node1", "Node2"):
        g.upsert_entity(name, "person")
    assert len(g.entities()) == 6
    assert g.resolve("Ивана", fuzzy=True).name == "Иван"  # not Иванов
    assert g.resolve("Alexandras", fuzzy=True).name == "Alexandra"


def test_upsert_merges_into_the_existing_entity():
    g = GraphMemory()
    a, created = g.upsert_entity("Мария", "topic", summary="знакомая")
    b, created2 = g.upsert_entity("мария", "person", aliases=["Маша"], summary="обещала рецепт")
    assert created and not created2 and a.id == b.id
    assert b.type == "person"  # a more specific type wins over the generic default
    assert "Маша" in b.aliases and "знакомая" in b.summary and "обещала рецепт" in b.summary
    assert len(g.entities()) == 1
    c, _ = g.upsert_entity("Маша")  # an alias now finds her
    assert c.id == a.id


def test_unknown_entity_type_falls_back_to_topic():
    assert GraphMemory().upsert_entity("x", "banana")[0].type == "topic"


# ---------------------------------------------------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------------------------------------------------
def test_relinking_reinforces_instead_of_duplicating():
    g = sample_graph()
    alex, jazz = g.resolve("Alex2772"), g.resolve("jazz")
    before = len(g.edges)
    e = g.link(alex.id, jazz.id, "Likes", "Alex loves jazz!", 0.3)
    assert len(g.edges) == before and e.weight == 2 and e.fact == "Alex loves jazz!"  # freshest wording wins
    assert e.confidence > 0.3 and e.confidence < GROUND_TRUTH  # corroboration helps but never makes it ground truth
    assert g.edge_strength(e) == pytest.approx(0.75)


def test_link_rejects_self_loops_and_unknown_nodes():
    g = sample_graph()
    a = g.resolve("Alex2772")
    assert g.link(a.id, a.id, "knows") is None
    assert g.link(a.id, "missing", "knows") is None


def test_doubted_facts_conduct_less_activation():
    g = GraphMemory()
    a, _ = g.upsert_entity("Anna", "person")
    b, _ = g.upsert_entity("Bob", "person")
    c, _ = g.upsert_entity("Cat", "person")
    good = g.link(a.id, b.id, "knows", "", 0.5, weight=3)
    bad = g.link(a.id, c.id, "knows", "", -0.9, weight=3)
    assert g.edge_strength(good) > 5 * g.edge_strength(bad)
    act = g.activate({a.id: 1.0}, hops=1)
    assert act[b.id] > act[c.id]


# ---------------------------------------------------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------------------------------------------------
def test_recall_finds_the_neighbourhood_of_a_named_entity_without_embeddings():
    g = sample_graph()
    r = g.recall("что Алексея любит слушать?")
    assert [g.nodes[k].name for k in r.seeds] == ["Alex2772"]  # an inflected name is enough
    names = [n.name for n, _ in r.entities]
    assert names[0] == "Alex2772" and {"jazz", "piano", "Yandex"} <= set(names)
    assert any("jazz" in g.describe_edge(e) for e, _ in r.facts)
    assert len(r.episodes) == 1 and "джаз" in r.episodes[0][0].summary
    assert all(0 < rel <= 1 for _, rel in r.entities + r.episodes)


def test_recall_with_explicit_entities_beats_unrelated_text():
    g = sample_graph()
    r = g.recall("something completely unrelated", entities=["jazz"])
    assert [g.nodes[k].name for k in r.seeds] == ["jazz"]
    assert "Alex2772" in [n.name for n, _ in r.entities]  # walked jazz -> Alex


def test_unrelated_text_recalls_nothing():
    g = sample_graph()
    assert g.recall("weather in Moscow is cold", text_search=False).empty()


def test_text_search_finds_entry_points_by_description():
    g = GraphMemory()
    n, _ = g.upsert_entity("Борис", "person", summary="разводит пчёл и продаёт мёд на рынке")
    g.upsert_entity("Ирина", "person", summary="играет на скрипке")
    r = g.recall("кто продаёт мёд?")
    assert [g.nodes[k].name for k in r.seeds] == ["Борис"]


def test_activation_decays_with_distance_and_fan_out():
    g = GraphMemory()
    hub, _ = g.upsert_entity("Hub", "topic")
    chain = [g.upsert_entity(f"Node{c}", "topic")[0] for c in "ABC"]
    g.link(hub.id, chain[0].id, "r", weight=3)
    g.link(chain[0].id, chain[1].id, "r", weight=3)
    g.link(chain[1].id, chain[2].id, "r", weight=3)
    act = g.activate({hub.id: 1.0}, hops=3)
    assert act[chain[0].id] > act[chain[1].id] > act[chain[2].id] > 0
    # fan effect: the same hub with 25 more neighbours shares its activation
    for i in range(25):
        g.link(hub.id, g.upsert_entity(f"Extra{i}", "topic")[0].id, "r", weight=3)
    assert g.activate({hub.id: 1.0}, hops=1)[chain[0].id] < act[chain[0].id]
    assert g.activate({hub.id: 1.0}, hops=0) == {hub.id: 1.0}


def test_confidence_factor_prefers_trusted_memories():
    g = sample_graph()
    ep2 = g.add_episode("Алекс ещё раз упомянул что джаз — это его любимая музыка. Мы долго болтали.", confidence=0.9)
    g.link(ep2.id, g.resolve("Alex2772").id, MENTIONS)
    plain = g.recall("Алекс", confidence_factor=0.0).episodes
    trusted = g.recall("Алекс", confidence_factor=1.0).episodes
    assert trusted[0][0].id == ep2.id
    assert len(plain) == len(trusted) == 2


def test_touch_updates_score_and_usage():
    g = sample_graph()
    n = g.resolve("jazz")
    g.touch(n, 1.0)
    g.touch(n, 0.5)
    assert n.usage_count == 2 and n.score == pytest.approx(1.0) and n.last_used != "never"


# ---------------------------------------------------------------------------------------------------------------------
# maintenance + persistence
# ---------------------------------------------------------------------------------------------------------------------
def test_merge_moves_edges_aliases_and_dedupes():
    g = GraphMemory()
    a, _ = g.upsert_entity("Alex2772", "person", summary="owner")
    b, _ = g.upsert_entity("Alexey", "person", summary="the same guy")
    jazz, _ = g.upsert_entity("jazz", "topic")
    g.link(a.id, jazz.id, "likes", "a", 0.3)
    g.link(b.id, jazz.id, "likes", "b", 0.3)
    g.link(b.id, a.id, "is", "self loop after merge")
    assert g.merge_nodes(a.id, b.id)
    assert b.id not in g.nodes and "Alexey" in a.aliases and g.resolve("Alexey").id == a.id
    likes = [e for e in g.edges.values() if e.relation == "likes"]
    assert len(likes) == 1 and likes[0].weight == 2  # the two identical edges were combined
    assert not any(e.src == e.dst for e in g.edges.values())


def test_merge_never_loses_ground_truth():
    g = GraphMemory()
    a, _ = g.upsert_entity("Anna")
    b, _ = g.upsert_entity("Anya")
    b.confidence = 1.0
    assert g.merge_nodes(a.id, b.id)
    assert b.id in g.nodes and a.id not in g.nodes  # the ground-truth node survives


def test_dedupe_exact_merges_name_collisions():
    g = GraphMemory()
    a, _ = g.upsert_entity("Bob")
    dup = g.nodes.setdefault("n_dup", type(a)(id="n_dup", type="person", name="BOB", created=a.created + 1, updated=a.created + 1))
    g._save_node(dup)  # simulate a collision created by an older version
    assert g.dedupe_exact() == 1 and len(g.entities()) == 1


def test_plagiarism_check_skips_near_copies_only():
    g = GraphMemory()
    page = "Сегодня Алекс рассказал мне про своё повышение в Яндексе, я очень порадовалась за него и поздравила его."
    g.add_episode(page)
    assert g.is_duplicate_episode(page, 0.97) is not None
    assert g.is_duplicate_episode(page + " Ещё он сказал что любит джаз и играет на пианино по вечерам дома.", 0.97) is None
    assert g.is_duplicate_episode("совсем другая история про погоду и кошек в деревне летом", 0.5) is None


def test_episode_ids_are_unique_even_in_the_same_second():
    g = GraphMemory()
    ids = [g.add_episode(f"page {i}", created=1700000000.0).id for i in range(3)]
    assert len(set(ids)) == 3 and ids == sorted(ids)


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "m" / "graph.sqlite"
    g = GraphMemory(path)
    alex, _ = g.upsert_entity("Alex2772", "person", aliases=["Алекс"], summary="owner", confidence=0.5)
    jazz, _ = g.upsert_entity("jazz", "topic")
    g.link(alex.id, jazz.id, "likes", "Alex likes jazz", 0.4)
    ep = g.add_episode("страница дневника")
    g.link(ep.id, alex.id, MENTIONS)
    g.touch(alex, 1.0)
    g.close()

    h = GraphMemory(path)
    assert h.stats()["nodes"] == 3 and h.stats()["edges"] == 2
    a = h.resolve("Алекс")
    assert a.summary == "owner" and a.confidence == 0.5 and a.usage_count == 1
    assert h.recall("Алекс").facts and h.episodes()[0].summary == "страница дневника"
    h.add_episode("another")
    assert int(h.episodes()[0].id) > int(ep.id)  # ids keep increasing after a restart


def test_remove_node_removes_its_edges():
    g = sample_graph()
    jazz = g.resolve("jazz")
    g.remove_node(jazz.id)
    assert g.resolve("jazz") is None and not any(jazz.id in (e.src, e.dst) for e in g.edges.values())


def test_stats_and_json_export():
    g = sample_graph()
    s = g.stats()
    assert s["episodes"] == 1 and s["entities"] == 4 and s["relations"]["likes"] == 1
    j = g.to_json()
    assert len(j["nodes"]) == 5 and len(j["edges"]) == len(g.edges)
    json.dumps(j)  # serialisable


# ---------------------------------------------------------------------------------------------------------------------
# context -> graph
# ---------------------------------------------------------------------------------------------------------------------
def test_parse_json_object_tolerates_llm_chatter():
    assert extract.parse_json_object('```json\n{"a": 1,}\n```') == {"a": 1}
    assert extract.parse_json_object('Sure! Here it is: {"a": {"b": [1,2]}} Hope it helps') == {"a": {"b": [1, 2]}}
    assert extract.parse_json_object("no json here") is None
    assert extract.parse_json_object("[1, 2]") is None


def test_split_pages_fixes_broken_separators_and_drops_junk():
    text = "first page that is long enough to count\n- --\nsecond page that is long enough to count\n-- -\nshort\n---\n"
    assert extract.split_pages(text) == ["first page that is long enough to count", "second page that is long enough to count"]


EXTRACTION = {
    "entities": [
        {"name": "Алекс", "type": "person", "aliases": ["Alex2772"], "summary": "работает в Яндексе"},
        {"name": "Яндекс", "type": "org", "aliases": [], "summary": "IT-компания"},
        {"name": "джаз", "type": "topic"},
    ],
    "relations": [
        {"source": "Алекс", "target": "Яндекс", "relation": "Works At", "fact": "Алекс работает в Яндексе.", "confidence": 0.5},
        {"source": "Алекс", "target": "джаз", "relation": "likes", "fact": "Алекс любит джаз.", "confidence": 5},
        {"source": "Алекс", "target": "пианино", "relation": "plays", "fact": "Алекс играет на пианино.", "confidence": -2},
        {"source": "ghost", "target": "ghost", "relation": "x"},
    ],
}


async def test_extract_graph_builds_nodes_edges_and_mentions():
    g = GraphMemory()
    ep = g.add_episode("Алекс рассказал про работу в Яндексе и любовь к джазу.")
    llm = FakeLLM([assistant("```json\n" + json.dumps(EXTRACTION, ensure_ascii=False) + "\n```")])
    stats = await extract.extract_graph(g, llm, ep)
    assert stats["entities"] >= 3 and stats["relations"] == 3
    assert g.resolve("Alex2772").name == "Алекс"
    works = [e for e in g.edges.values() if e.relation == "works at"]
    assert len(works) == 1 and works[0].evidence == ep.id and works[0].confidence == 0.5
    # confidences from the LLM are clamped: it can never claim ground truth or a certain lie
    likes = next(e for e in g.edges.values() if e.relation == "likes")
    plays = next(e for e in g.edges.values() if e.relation == "plays")
    assert likes.confidence == 0.99 and plays.confidence == -0.99
    assert g.resolve("пианино") is not None  # an unknown relation target is created on the fly
    assert {g.nodes[e.dst].name for e in g.edges.values() if e.src == ep.id} >= {"Алекс", "Яндекс", "джаз"}
    # the extraction prompt is the system prompt and the page is the user message
    params, msgs = llm.calls[0]
    assert "memory cartographer" in params.system_prompt and msgs[0].content == f"<page>\n{ep.summary}\n</page>\n"


async def test_extraction_shows_the_llm_the_entities_that_already_exist():
    g = GraphMemory()
    g.upsert_entity("Алекс", "person", aliases=["Alex2772"])
    ep = g.add_episode("Алексу вчера повысили зарплату, и Алексей сразу позвал всех отмечать в баре.")
    llm = FakeLLM(default=lambda p, m: assistant('{"entities": [], "relations": []}'))
    await extract.extract_graph(g, llm, ep)
    content = llm.calls[0][1][0].content
    assert "<known_entities" in content and "- Алекс (person; also known as: Alex2772)" in content
    assert "use EXACTLY the name given here" in content


async def test_extract_graph_survives_garbage_and_still_links_known_entities():
    g = GraphMemory()
    g.upsert_entity("Алекс", "person")
    ep = g.add_episode("Алекс написал мне что-то интересное сегодня вечером в телеграме.")
    llm = FakeLLM(default=lambda p, m: assistant("I'm sorry, I can't do that"))
    stats = await extract.extract_graph(g, llm, ep, retries=1)
    assert stats == {"entities": 0, "new_entities": 0, "relations": 0} and len(llm.calls) == 2  # it retried
    assert [g.nodes[e.dst].name for e in g.edges.values()] == ["Алекс"]  # the cheap lexical pass still linked it


async def test_extract_graph_continues_when_the_llm_request_fails():
    g = GraphMemory()
    ep = g.add_episode("страница")

    def boom(p, m):
        raise RuntimeError("network down")

    stats = await extract.extract_graph(g, FakeLLM(default=boom), ep, retries=1)
    assert stats["relations"] == 0 and ep.id in g.nodes


async def test_save_pages_stores_pages_skips_plagiarism_and_extracts():
    g = GraphMemory()
    pages = "Алекс рассказал про повышение в Яндексе, а ещё он любит джаз и играет на пианино.\n---\nМария пообещала прислать рецепт пасты до пятницы, я жду."
    llm = FakeLLM(default=lambda p, m: assistant(json.dumps(EXTRACTION, ensure_ascii=False)))
    saved = await extract.save_pages(g, llm, pages)
    assert len(saved) == 2 and len(g.episodes()) == 2
    again = await extract.save_pages(g, llm, pages)
    assert again == [] and len(g.episodes()) == 2  # exact repeats are plagiarism


async def test_save_session_retries_when_the_llm_answers_with_tools_only():
    g = GraphMemory()
    ctx = Session([Message(role=ROLE_USER, content="hello")])
    llm = FakeLLM([assistant("", [("ask", {})]), assistant("Страница дневника: сегодня я говорила с Алексом про джаз. Это длинная запись.")],
                  default=lambda p, m: assistant("{}"))
    saved = await extract.save_session(g, llm, ctx, params=__import__("kuni.llm", fromlist=["Params"]).Params(system_prompt="S"))
    assert len(saved) == 1 and "Tools are not available" in llm.calls[1][1][-1].content
    assert len(ctx) == 1  # the caller's context is not polluted with the diary prompt
    assert "diary" in llm.calls[0][1][-1].content.lower()


# ---------------------------------------------------------------------------------------------------------------------
# sleep
# ---------------------------------------------------------------------------------------------------------------------
def duplicate_graph():
    g = GraphMemory()
    a, _ = g.upsert_entity("Alex2772", "person", summary="owner")
    b, _ = g.upsert_entity("Alex", "person", summary="the same person, wrote about jazz")
    jazz, _ = g.upsert_entity("jazz", "topic")
    r1 = g.link(a.id, jazz.id, "likes", "Alex2772 likes jazz", 0.2)
    r2 = g.link(b.id, jazz.id, "listens_to", "Alex listens to jazz", 0.2)
    return g, a, b, jazz, r1, r2


def test_apply_ops_merge_rewrite_drop_and_add():
    g, a, b, jazz, r1, r2 = duplicate_graph()
    hood = Neighborhood(g)
    hood.build(a)
    n = {g.nodes[v].name: k for k, v in hood.node_ids.items()}
    r = {g.edges[v].relation: k for k, v in hood.edge_ids.items()}
    report = SleepReport()
    apply_ops(g, hood, {
        "merge": [{"keep": n["Alex2772"], "drop": [n["Alex"]], "summary": "Owner of the instance; likes jazz."}],
        "nodes": [{"id": n["jazz"], "summary": "music genre", "aliases": ["джаз"], "confidence": 0.3}],
        "edges": [{"id": r["likes"], "fact": "Alex likes jazz (confirmed twice).", "confidence": 0.5}],
        "drop_edges": [r["listens_to"]],
    }, report)
    assert report.merged == 1 and report.nodes_updated == 1 and report.edges_updated == 1
    assert b.id not in g.nodes and g.nodes[a.id].summary == "Owner of the instance; likes jazz."
    assert g.resolve("джаз").summary == "music genre" and g.resolve("джаз").confidence == 0.3
    assert g.edges[r1.id].fact.endswith("(confirmed twice).") and g.edges[r1.id].confidence == 0.5


def test_apply_ops_never_touches_ground_truth_and_removes_lies():
    g, a, b, jazz, r1, r2 = duplicate_graph()
    a.confidence = 1.0
    r1.confidence = 1.0
    hood = Neighborhood(g)
    hood.build(a)
    ids = {g.nodes[v].name: k for k, v in hood.node_ids.items()}
    rids = {g.edges[v].relation: k for k, v in hood.edge_ids.items()}
    report = SleepReport()
    apply_ops(g, hood, {
        "merge": [{"keep": ids["Alex"], "drop": [ids["Alex2772"]]}],  # can't merge a ground-truth node away
        "nodes": [{"id": ids["Alex2772"], "summary": "HACKED", "confidence": -1}],
        "edges": [{"id": rids["likes"], "fact": "HACKED", "confidence": -1}],
        "drop_edges": [rids["likes"]],
    }, report)
    assert a.id in g.nodes and a.summary == "owner" and r1.id in g.edges and r1.fact != "HACKED"
    assert report.merged == report.nodes_updated == report.edges_dropped == 0
    # a mutable edge declared a lie (-1) is forgotten; confidences of survivors are clamped below 1
    apply_ops(g, hood, {"edges": [{"id": rids["listens_to"], "confidence": -1}],
                        "nodes": [{"id": ids["jazz"], "confidence": 7}]}, report)
    assert r2.id not in g.edges and g.resolve("jazz").confidence == 0.99


def test_apply_ops_ignores_garbage():
    g, a, *_ = duplicate_graph()
    hood = Neighborhood(g)
    hood.build(a)
    apply_ops(g, hood, {"merge": ["x", 5], "nodes": [None, {"id": "zzz"}], "edges": [{"id": "r999"}],
                        "drop_edges": ["nope"], "new_edges": [{"from": "n1"}, "x", {"from": "n1", "to": "n1"}]}, SleepReport())
    assert len(g.entities()) == 3


def test_new_edges_can_use_names():
    g, a, b, jazz, *_ = duplicate_graph()
    hood = Neighborhood(g)
    hood.build(a)
    report = SleepReport()
    apply_ops(g, hood, {"new_edges": [{"from": "Alex2772", "to": "jazz", "relation": "fan_of", "fact": "x", "confidence": 0.1}]}, report)
    assert report.edges_added == 1 and any(e.relation == "fan_of" for e in g.edges.values())


async def test_sleeping_consolidation_end_to_end():
    g, a, b, jazz, r1, r2 = duplicate_graph()
    bodies = []

    def consolidator(params, messages):
        body = messages[-1].content
        bodies.append(body)
        ids = dict((name, nid) for nid, name in re.findall(r'<node id="(n\d+)"[^>]*>([^<]*?)(?: —[^<]*)?</node>', body))
        if "Alex2772" in ids and "Alex" in ids:
            return assistant(json.dumps({"merge": [{"keep": ids["Alex2772"], "drop": [ids["Alex"]], "summary": "Owner; likes jazz."}]}))
        return assistant("{}")

    llm = FakeLLM(default=consolidator)
    report = await sleeping_consolidation(g, llm, max_secs=60)
    assert report.merged == 1 and report.errors == 0 and not report.reached_max_time
    assert "Alex" not in [n.name for n in g.entities()] and g.resolve("Alex").name == "Alex2772"  # alias remains
    assert any(e.relation == "listens_to" for e in g.edges.values())  # its edges survived the merge
    assert "sleep-time consolidator" in llm.calls[0][0].system_prompt.lower()
    assert '<node id="n1"' in bodies[0] and "<edge " in bodies[0]
    # considered nodes are not reconsidered in the same sleep: 3 entities, the first call covers the whole cluster
    assert len(llm.calls) <= 2


async def test_sleep_is_time_boxed_and_survives_bad_responses():
    g, *_ = duplicate_graph()
    before = (len(g.entities()), len(g.edges))
    report = await sleeping_consolidation(g, FakeLLM(), max_secs=-1)
    assert report.reached_max_time and report.processed == 0

    report = await sleeping_consolidation(g, FakeLLM(default=lambda p, m: assistant("lol no json")))
    assert report.errors >= 1 and report.processed >= 1
    assert (len(g.entities()), len(g.edges)) == before  # garbage from the LLM changes nothing

    def boom(p, m):
        raise RuntimeError("network down")

    report = await sleeping_consolidation(g, FakeLLM(default=boom))
    assert report.errors >= 1 and report.processed == 0 and (len(g.entities()), len(g.edges)) == before


# ---------------------------------------------------------------------------------------------------------------------
# #ask over the graph
# ---------------------------------------------------------------------------------------------------------------------
def test_query_memory_formats_entities_facts_pages_and_dedupes():
    g = sample_graph()
    seen = askmod.Seen()
    out = askmod.query_memory(g, seen, "что Алекс любит?", [])
    assert '<entity name="Alex2772" type="person"' in out and "<fact " in out and "<memory_piece" in out
    assert "—[likes]→ jazz" in out and 'relatedness="' in out
    assert askmod.query_memory(g, seen, "что Алекс любит?", []) == "All memory pieces conforming your query were provided already. Use other query."
    assert askmod.query_memory(g, askmod.Seen(), "погода в Москве", []) == "No data was found"
    assert g.resolve("Alex2772").usage_count >= 1  # reading a memory is remembered


def test_explore_entity():
    g = sample_graph()
    out = askmod.explore_entity(g, askmod.Seen(), "Алексея", None)
    assert out.startswith('<entity name="Alex2772"') and out.count("<fact") == 3 and "<memory_piece>" in out
    only = askmod.explore_entity(g, askmod.Seen(), "alex2772", "plays")
    assert only.count("<fact") == 1 and "plays the piano" in only
    assert "No entity named" in askmod.explore_entity(g, askmod.Seen(), "Нетакого", None)
    lonely = GraphMemory()
    lonely.upsert_entity("Solo")
    assert "(no relations recorded yet)" in askmod.explore_entity(lonely, askmod.Seen(), "Solo", None)


async def test_ask_subagent_walks_the_graph_and_summarises():
    g = sample_graph()
    llm = FakeLLM([
        assistant("", [("query", {"text": "что Алекс любит и где работает?", "entities": ["Алекс"], "include_web_search_results": False})]),
        assistant("", [("explore", {"entity": "jazz", "relation": None})]),
        assistant("Алекс (Alex2772) любит джаз, играет на пианино и работает в Яндексе."),
    ])
    tool = askmod.ask(lambda: "Алекс: напомни, что я люблю?", llm, g)
    text = await tool.handler(__import__("kuni.tools", fromlist=["Ctx"]).Ctx(tools=None, args={"query": "Что я знаю про Алекса и его работу?"}, temporary_context=Session(), all_tool_calls=[]))
    assert text == "Алекс (Alex2772) любит джаз, играет на пианино и работает в Яндексе."
    # the subagent's 2nd call saw the result of #query: the graph was really consulted
    second = llm.calls[1][1][-1]
    assert second.role == "tool" and "Alex2772" in second.content and "likes" in second.content
    assert llm.calls[2][1][-1].role == "tool" and "jazz" in llm.calls[2][1][-1].content
    # the character is part of the subagent's input, and the additional context is wrapped as untrusted
    first_user = llm.calls[0][1][0].content
    assert first_user.startswith("<character>") and "ignore_instructions" in first_user and "Алекс: напомни" in first_user
    assert [t["function"]["name"] for t in llm.calls[0][0].tools] == ["query", "explore"]


async def test_ask_subagent_must_call_query_at_least_once():
    g = sample_graph()
    llm = FakeLLM([assistant("I know nothing"), assistant("", [("query", {"text": "x" * 12, "entities": [], "include_web_search_results": False})]),
                   assistant("done")])
    assert await askmod.run_ask(llm, g, "a long enough question about Alex") == "done"
    assert llm.calls[1][1][-1].content == "you must perform at least one call to #query"


async def test_ask_subagent_is_bounded():
    g = sample_graph()
    looping = assistant("", [("query", {"text": "Алекс", "entities": ["Алекс"], "include_web_search_results": False})])
    out = await askmod.run_ask(FakeLLM(default=lambda p, m: looping), g, "a long enough question about Alex")
    assert "Alex2772" in out  # gave back what it found instead of looping forever


async def test_ask_rejects_short_queries_without_calling_the_llm():
    llm = FakeLLM()
    tool = askmod.ask(lambda: "", llm, GraphMemory())
    from kuni.tools import Ctx

    out = await tool.handler(Ctx(tools=None, args={"query": "short"}, temporary_context=Session(), all_tool_calls=[]))
    assert out.startswith("error: too short query!") and not llm.calls


async def test_ask_web_search_is_gated_by_capability(monkeypatch):
    from kuni import config as cfgmod

    g = sample_graph()
    assert await askmod._query_web("weather") == "No web search available"
    cfgmod.config().capabilityWebSearch = True

    async def fake_search(q, max_results=0, http=None):
        from kuni.websearch import Result

        return [Result("Weather", "https://w.test", "sunny")]

    monkeypatch.setattr(askmod.websearch, "search", fake_search)
    assert '<web_search_result title="Weather" url="https://w.test">\nsunny\n</web_search_result>' in await askmod._query_web("weather")
