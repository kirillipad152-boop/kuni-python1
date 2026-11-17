"""In-memory record of what Kuni did on recent turns, for the owner's panel ("why is she silent?").

Memory only (lost on restart), bounded per chat and overall, text clipped, keys masked. It holds her own thoughts,
tool calls and results, so it is shown to the owner only.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

SECRETS = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+)")
CLIP = 1500


def clip(text: object, limit: int = CLIP) -> str:
    text = SECRETS.sub("[hidden]", str(text or ""))
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit})"


def _clock() -> str:
    return time.strftime("%H:%M:%S", time.gmtime())


@dataclass
class Turn:
    chat_id: int
    message_id: int | None
    kind: str
    started: float = field(default_factory=time.time)
    steps: list[dict] = field(default_factory=list)
    outcome: str = "идёт"
    finished: float | None = None

    def step(self, thought: str, said: str, calls: list[tuple[str, str]]) -> None:
        """One LLM answer: her reasoning, her plain text and the tool calls she asked for (name, raw arguments)."""
        self.steps.append({"t": _clock(), "thought": clip(thought), "said": clip(said),
                           "calls": [{"name": name, "args": clip(args, 400)} for name, args in calls], "results": []})

    def result(self, name: str, text: str) -> None:
        if self.steps:
            self.steps[-1]["results"].append({"name": name, "text": clip(text, 600)})

    def note(self, text: str) -> None:
        self.steps.append({"t": _clock(), "thought": "", "said": "", "calls": [], "results": [], "note": clip(text, 300)})

    def finish(self, outcome: str) -> None:
        self.outcome, self.finished = outcome, time.time()

    def to_json(self) -> dict:
        return {"chat_id": self.chat_id, "message_id": self.message_id, "kind": self.kind, "started": self.started,
                "finished": self.finished, "outcome": self.outcome, "steps": self.steps}


class TurnTraces:
    def __init__(self, per_chat: int = 6, max_chats: int = 100) -> None:
        self._per_chat, self._max_chats = per_chat, max_chats
        self._chats: OrderedDict[int, deque[Turn]] = OrderedDict()

    def begin(self, chat_id: int, message_id: int | None, kind: str) -> Turn:
        turn = Turn(chat_id, message_id, kind)
        turns = self._chats.setdefault(chat_id, deque(maxlen=self._per_chat))
        self._chats.move_to_end(chat_id)
        turns.append(turn)
        while len(self._chats) > self._max_chats:
            self._chats.popitem(last=False)
        return turn

    def summary(self) -> list[dict]:
        out = []
        for chat_id, turns in reversed(self._chats.items()):
            last = turns[-1]
            out.append({"chat_id": chat_id, "turns": len(turns), "outcome": last.outcome, "started": last.started,
                        "kind": last.kind})
        return out

    def turns(self, chat_id: int) -> list[dict]:
        return [t.to_json() for t in reversed(self._chats.get(chat_id, ()))]
