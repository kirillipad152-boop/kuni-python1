from __future__ import annotations

import os
import random
from pathlib import Path
import tomllib
from datetime import datetime, timedelta, timezone

import pytest

from kuni import config as cfgmod
from kuni import prompts as promptsmod
from kuni import util


# ---------------------------------------------------------------------------------------------------------------------
# util
# ---------------------------------------------------------------------------------------------------------------------
def test_time_ago_matches_the_cpp_wording():
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    ago = lambda **kw: util.time_ago(now - timedelta(**kw), now)  # noqa: E731
    assert ago(seconds=10) == "just now"
    assert ago(minutes=1) == "1 minute ago"
    assert ago(minutes=5) == "5 minutes ago"
    assert ago(hours=1) == "1 hour ago"
    assert ago(hours=5) == "5 hours ago"
    assert ago(days=1) == "1 day ago"
    assert ago(days=3) == "3 days ago"
    assert ago(days=7) == "1 week ago"
    assert ago(days=21) == "3 weeks ago"


def test_json_as_int_coerces_what_small_llms_produce():
    assert util.json_as_int(5) == 5
    assert util.json_as_int(7.576e08) == 757600000
    assert util.json_as_int("7.576e+08") == 757600000
    assert util.json_as_int("12") == 12
    assert util.json_as_int(None) is None
    assert util.json_as_int("abc") is None
    assert util.json_as_int(True) is None


def test_typos_keep_length_and_change_text():
    rng = random.Random(3)
    s = "привет как дела"
    swapped = util.swap_adjacent_chars(s, rng)
    assert sorted(swapped) == sorted(s) and len(swapped) == len(s)
    replaced = util.replace_with_keyboard_neighbor("hello", random.Random(1))
    assert len(replaced) == 5 and sum(a != b for a, b in zip(replaced, "hello")) <= 1
    assert util.swap_adjacent_chars("a") == "a"
    assert util.replace_with_keyboard_neighbor("123!") == "123!"  # no neighbours known


def test_variation_selector_and_control_chars():
    assert util.strip_variation_selector("❤️") == "❤"
    assert util.strip_variation_selector("⚡️") == "⚡"
    assert util.remove_control_characters("a\x00b\x1bc\nd\te") == "abc\nd\te"


def test_opt_str_understands_llms_that_stringify_null():
    assert util.opt_str(None) is None
    for s in ("", "  ", "None", "null", "NULL", "nil", "undefined", "N/A"):
        assert util.opt_str(s) is None, s
    assert util.opt_str("photo.png") == "photo.png"
    assert util.opt_str(5) == "5"


def test_stemmer_matches_inflections():
    assert util.stem("Алексея") == util.stem("Алексей")
    assert util.stem("songs") == util.stem("songz")[:5] or util.stem("songs") == "songs"[:5]
    assert set(util.stems("Алексею нравится джаз")) >= {util.stem("алексей"), util.stem("джаз")}
    assert util.normalize_text("ЁЖ") == "еж"


def test_lexical_similarity_orders_sensibly():
    a = "я правда рада за тебя, поздравляю с повышением"
    assert util.lexical_similarity(a, a) == pytest.approx(1.0)
    assert util.lexical_similarity(a, "я очень рада за тебя, поздравляю с повышением") > 0.8
    assert util.lexical_similarity(a, "weather in moscow is cold today") < 0.3
    assert util.jaccard(util.shingles("a b c d e"), util.shingles("a b c d e")) == 1.0


# ---------------------------------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------------------------------
def test_render_parse_roundtrip_covers_every_key():
    cfg = cfgmod.Config()
    cfg.llm.endpoint.bearerKey = "sk-secret"
    cfg.llm.reasoningEffort = "high"
    cfg.llmTopP = 0.9
    cfg.lockdown = cfgmod.LockdownMode.CONTACTS_ONLY
    text = cfgmod.render(cfg)
    back = cfgmod.parse(tomllib.loads(text))  # must be valid TOML
    for attr, *_ in cfgmod.SCHEMA:
        assert getattr(back, attr) == getattr(cfg, attr), attr


def test_none_for_optional_floats_roundtrips():
    cfg = cfgmod.Config()
    assert cfg.llmTopK is None
    text = cfgmod.render(cfg)
    assert 'top_k = "none"' in text
    assert cfgmod.parse(tomllib.loads(text)).llmTopK is None


def test_env_overrides_file(monkeypatch):
    monkeypatch.setenv("KUNI_API_KEY", "from-env")
    monkeypatch.setenv("KUNI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("KUNI_MODEL", "m1")
    monkeypatch.setenv("KUNI_REASONING_EFFORT", "low")
    cfg = cfgmod.parse({"general": {"llm": {"endpoint": {"baseUrl": "x", "bearerKey": "from-file"}, "model": "m0"}}})
    assert cfg.llm.endpoint.bearerKey == "from-env"
    assert cfg.llm.endpoint.baseUrl == "https://example.test/v1"
    assert cfg.llm.model == "m1" and cfg.llm.reasoningEffort == "low"


def test_base_url_gets_trailing_slash():
    cfg = cfgmod.parse({"general": {"llm": {"endpoint": {"baseUrl": "https://apihub.agnes-ai.com/v1", "bearerKey": "k"},
                                            "model": "agnes-3.0-flash", "reasoningEffort": "high"}}})
    r = cfg.resolved_llm()
    assert r.endpoint.baseUrl == "https://apihub.agnes-ai.com/v1/"
    assert r.model == "agnes-3.0-flash" and r.reasoningEffort == "high"


def test_secondary_endpoints_inherit_from_main_llm():
    cfg = cfgmod.parse({"general": {"llm": {"endpoint": {"baseUrl": "https://a/v1", "bearerKey": "K"}, "model": "M"}}})
    v = cfg.resolve(cfg.llmImageToText)
    assert (v.endpoint.baseUrl, v.endpoint.bearerKey, v.model) == ("https://a/v1/", "K", "M")
    # whisper has its own default host: the main key must NOT leak to it
    w = cfg.resolve(cfg.llmAudioToText)
    assert w.endpoint.baseUrl == "http://localhost:9000/v1/" and w.endpoint.bearerKey == ""
    # an explicitly different host with its own key keeps it
    cfg2 = cfgmod.parse({"capabilities": {"vision": {"llm_image_to_text": {"endpoint": {"baseUrl": "https://b/", "bearerKey": "B"}, "model": "V"}}}})
    assert cfg2.resolve(cfg2.llmImageToText).endpoint.bearerKey == "B"


def test_the_original_authors_chat_never_becomes_the_owner_by_default(tmp_path):
    cfg = cfgmod.Config()
    cfg.telegramApiId, cfg.telegramApiHash = 1, "x"
    for owner in (0, 625207005):
        cfg.papikChatId = owner
        errors = cfgmod.validate(cfg)
        assert len(errors) == 1 and "papik_chat_id" in errors[0]
    cfg.papikChatId = 42
    assert cfgmod.validate(cfg) == []
    cfg.telegramEnabled, cfg.papikChatId = False, 0  # the offline REPL needs no real owner
    assert cfgmod.validate(cfg) == []
    # a freshly generated or copied config carries placeholders, not anybody's identity
    generated = cfgmod.init_config_file(tmp_path / "c.toml", api_key="k")
    assert generated.papikChatId == 0 and generated.papikName == "Owner" and "625207005" not in (tmp_path / "c.toml").read_text()
    example = cfgmod.parse(tomllib.loads((Path(__file__).resolve().parents[1] / "config.example.toml").read_text()))
    assert example.papikChatId == 0 and example.telegramApiId == 0 and example.telegramApiHash == ""
    assert example.llm.endpoint.bearerKey == "" and example.webSearchOllamaKey == "" and not example.llm.endpoint.extraBearerKeys


def test_validate_reports_missing_telegram_credentials():
    cfg = cfgmod.Config()
    cfg.papikChatId = 42
    assert len(cfgmod.validate(cfg)) == 2
    cfg.telegramApiId, cfg.telegramApiHash = 1, "x"
    assert cfgmod.validate(cfg) == []
    cfg.suggestIgnoreChance = 0.99
    assert "suggest_ignore_chance" in cfgmod.validate(cfg)[0]
    cfg.telegramEnabled = False
    cfg.telegramApiId = 0
    cfg.suggestIgnoreChance = 0.1
    assert cfgmod.validate(cfg) == []


def test_init_file_is_private_and_hot_reload_notifies(tmp_path):
    path = tmp_path / "config.toml"
    cfgmod.init_config_file(path, base_url="https://x/v1", api_key="sk-1", model="m", effort="high")
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"  # it holds secrets
    cfg = cfgmod.load(path)
    assert cfg.llm.endpoint.bearerKey == "sk-1" and cfg.llm.reasoningEffort == "high"

    fired = []
    cfgmod.on_config_updated(lambda: fired.append(1))
    text = path.read_text().replace("remind_use_ask = true", "remind_use_ask = false")
    path.write_text(text)
    st = path.stat()
    os.utime(path, (st.st_atime + 5, st.st_mtime + 5))
    assert cfgmod.reload_if_changed(path) is True
    assert cfgmod.config().remindUseAsk is False and fired
    assert cfgmod.reload_if_changed(path) is False  # unchanged -> nothing


def test_config_file_with_broken_toml_keeps_the_old_config(tmp_path):
    path = tmp_path / "config.toml"
    cfgmod.init_config_file(path)
    cfgmod.load(path)
    old = cfgmod.config()
    path.write_text("this is [not toml")
    st = path.stat()
    os.utime(path, (st.st_atime + 5, st.st_mtime + 5))
    assert cfgmod.reload_if_changed(path) is False
    assert cfgmod.config() is old


# ---------------------------------------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------------------------------------
def test_front_matter_is_stripped_only_when_it_leads():
    assert promptsmod.strip_front_matter("---\nnote\n---\n\nbody").strip() == "body"
    assert promptsmod.strip_front_matter("no front matter\n---\nbut a hr").startswith("no front matter")
    assert promptsmod.strip_front_matter("---\nunterminated").startswith("---")


def test_all_default_prompts_load_without_leftover_variables(tmp_path):
    p = promptsmod.prompts()
    for attr in promptsmod.FILES:
        text = getattr(p, attr)
        assert text.strip(), attr
        assert "${" not in text, attr
        assert (tmp_path / "prompts" / f"{promptsmod.FILES[attr]}.md").exists()  # written on the first run
    assert "Kuni" in p.system  # ${CHARACTER_NAME} replaced
    assert "{CHARACTER_APPEARANCE_PROMPT}" in p.image_engineer_system  # fmt-style placeholders are left to the caller


def test_variables_follow_config():
    cfg = cfgmod.config()
    cfg.characterName = "Mimi"
    promptsmod.prompts().reload(force=True)
    assert "Mimi" in promptsmod.prompts().system
    assert "Kuni" not in promptsmod.prompts().system.replace("kuni_", "")  # nothing hard-coded left


def test_user_edit_is_hot_reloaded_and_defaults_not_rewritten(tmp_path):
    p = promptsmod.prompts()
    f = tmp_path / "prompts" / "anti_repeat.md"
    f.write_text("---\nmy note\n---\nstop repeating, ${CHARACTER_NAME}!")
    st = f.stat()
    os.utime(f, (st.st_atime + 5, st.st_mtime + 5))
    assert p.reload() is True
    assert p.anti_repeat.strip() == "stop repeating, Kuni!"
    assert p.reload() is False
    assert f.read_text().startswith("---\nmy note")  # the owner's file is never overwritten


def test_the_original_prompts_are_verbatim():
    """Spot checks against the C++ source: these are the lines the behaviour of the character depends on."""
    p = promptsmod.prompts()
    assert "You ACT EXCLUSIVELY BY CALLING TOOLS." in p.system
    assert "I'm an AI. I think I'm a person." in p.character_base
    assert "You are repeating after yourself" in p.anti_repeat
    assert "Time window: last 48h." in p.diary_save
    assert "You are the Sleep-time Consolidator" in p.sleep_consolidator  # graph version keeps the persona
    assert "GRAPH" in p.sleep_consolidator and '"merge"' in p.sleep_consolidator
    assert '"entities"' in p.graph_extract and '"relations"' in p.graph_extract
