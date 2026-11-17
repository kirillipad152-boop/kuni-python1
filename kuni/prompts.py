"""Prompts: markdown files in ./prompts, auto-created from the defaults and hot-reloaded.

Port of prompts.cpp. The defaults were extracted verbatim from the C++ source (kuni/default_prompts/*.md).
"""

from __future__ import annotations

import asyncio
import logging
from importlib import resources
from pathlib import Path

from . import config as cfgmod

log = logging.getLogger("prompts")

PROMPTS_DIR = Path("prompts")

# attribute -> file name (without .md)
FILES: dict[str, str] = {
    "system": "system",
    "character_base": "character_base",
    "character_appearance": "character_appearance",
    "photo_to_text": "photo_to_text",
    "sticker_to_text": "sticker_to_text",
    "anti_repeat": "anti_repeat",
    "diary_save": "diary_save",
    "sleep_consolidator": "sleep_consolidator",
    "graph_extract": "graph_extract",
    "record_audio_speech": "record_audio_speech",
    "messages_epilogue": "messages_epilogue",
    "image_engineer_system": "image_engineer_system",
    "image_engineer_instructions": "image_engineer_instructions",
    "image_assess_system": "image_assess_system",
}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DO NOT modify the prompt text in kuni/default_prompts! It is only used  ║
# ║  the FIRST TIME, when no file exists in ./prompts. After that it is      ║
# ║  IGNORED. To customise the prompts, edit ./prompts/*.md                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def default_prompt(name: str) -> str:
    return resources.files("kuni").joinpath("default_prompts", f"{name}.md").read_text(encoding="utf-8")


def strip_front_matter(text: str) -> str:
    """Removes a leading `---\\n...\\n---` block (it's a note for the instance owner, not part of the prompt)."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return text
    end = stripped.find("\n---", 4)
    if end == -1:
        return text
    return stripped[end + 4 :]


def apply_variables(text: str) -> str:
    c = cfgmod.config()
    for var, value in (
        ("CHARACTER_NAME", c.characterName),
        ("CHARACTER_NICKNAME", c.characterNickname),
        ("PAPIK_NAME", c.papikName),
    ):
        text = text.replace("${" + var + "}", value)
    return text


def load_prompt(path: Path, default: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default, encoding="utf-8")
        text = default
    return apply_variables(strip_front_matter(text))


class Prompts:
    def __init__(self, directory: Path = PROMPTS_DIR) -> None:
        self.directory = directory
        self._mtimes: dict[str, float] = {}
        for attr in FILES:
            setattr(self, attr, "")
        self.reload(force=True)

    def reload(self, force: bool = False) -> bool:
        changed = False
        for attr, name in FILES.items():
            path = self.directory / f"{name}.md"
            try:
                m = path.stat().st_mtime
            except OSError:
                m = -1.0
            if force or self._mtimes.get(attr) != m:
                setattr(self, attr, load_prompt(path, default_prompt(name)))
                try:
                    self._mtimes[attr] = path.stat().st_mtime
                except OSError:
                    self._mtimes[attr] = -1.0
                if not force:
                    log.info("Loaded: %s", path)
                changed = True
        return changed

    # type hints for IDEs
    system: str
    character_base: str
    character_appearance: str
    photo_to_text: str
    sticker_to_text: str
    anti_repeat: str
    diary_save: str
    sleep_consolidator: str
    graph_extract: str
    record_audio_speech: str
    messages_epilogue: str
    image_engineer_system: str
    image_engineer_instructions: str
    image_assess_system: str


_prompts: Prompts | None = None


def prompts() -> Prompts:
    global _prompts
    if _prompts is None:
        _prompts = Prompts()
        cfgmod.on_config_updated(lambda: _prompts.reload(force=True))
    return _prompts


def set_prompts(p: Prompts | None) -> None:
    global _prompts
    _prompts = p


async def watch(interval: float = 1.0) -> None:
    while True:
        await asyncio.sleep(interval)
        prompts().reload()
