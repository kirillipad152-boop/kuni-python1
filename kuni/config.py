"""config.toml loading with hot-reload. Port of config.h / config.cpp.

TOML paths are identical to the C++ kernel, so an existing config.toml keeps working. Differences:

* `general.llm` accepts `reasoningEffort` (e.g. "high") next to `model`;
* the API key / base URL can also come from env vars KUNI_API_KEY / KUNI_BASE_URL / KUNI_MODEL /
  KUNI_REASONING_EFFORT (they win over the file);
* secondary endpoints (vision) inherit base URL / key / model from `general.llm` when left empty;
* `general.embedding` is optional now: the memory is a graph and needs no embeddings. When configured, it's used only
  by the anti-repeat check; otherwise a lexical similarity is used.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import tomli_w

from .util import atomic_write_text

log = logging.getLogger("config.toml")

CONFIG_TOML = Path("config.toml")
DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1/"
DEFAULT_MODEL = "agnes-3.0-flash"
DEFAULT_REASONING_EFFORT = "high"
ORIGINAL_AUTHOR_CHAT_ID = 625207005  # the upstream default; it must never become somebody else's owner


class LockdownMode(str, Enum):
    NONE = "none"  # public
    CONTACTS_ONLY = "contacts_only"
    PAPIK_ONLY = "papik_only"


class TTSBackend(str, Enum):
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"


def key_fingerprint(key: str) -> str:
    """Stable public identifier of an API key (never reversible); the panel shows it and the last four characters."""
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def normalize_base_url(url: str) -> str:
    """The kernel concatenates `baseUrl + "chat/completions"`, so the trailing slash is mandatory."""
    url = url.strip()
    if url and not url.endswith("/"):
        url += "/"
    return url


@dataclass
class Endpoint:
    baseUrl: str = ""
    bearerKey: str = ""
    extraBearerKeys: tuple[str, ...] = ()

    def bearer_keys(self) -> list[str]:
        return list(dict.fromkeys(key for key in (self.bearerKey, *self.extraBearerKeys) if key))


@dataclass
class EndpointAndModel:
    endpoint: Endpoint = field(default_factory=Endpoint)
    model: str = ""
    reasoningEffort: str = ""  # "" = don't send; otherwise low/medium/high

    def resolved(self, parent: "EndpointAndModel") -> "EndpointAndModel":
        """Fills the empty fields from `parent` (secondary endpoints inherit from general.llm)."""
        return EndpointAndModel(
            endpoint=Endpoint(
                baseUrl=normalize_base_url(self.endpoint.baseUrl or parent.endpoint.baseUrl),
                bearerKey=self.endpoint.bearerKey or (parent.endpoint.bearerKey if not self.endpoint.baseUrl else ""),
                extraBearerKeys=self.endpoint.extraBearerKeys or (
                    parent.endpoint.extraBearerKeys if not self.endpoint.baseUrl and not self.endpoint.bearerKey else ()),
            ),
            model=self.model or parent.model,
            reasoningEffort=self.reasoningEffort,
        )


# kind: str|int|bool|float|opt_float|seconds|endpoint|endpoint_model|lockdown|tts
# (attr, toml path, default, kind)
SCHEMA: list[tuple[str, str, Any, str]] = [
    ("characterName", "general.character_name", "Kuni", "str"),
    ("characterNickname", "general.character_nickname", "@kunii_chan", "str"),
    ("papikName", "general.papik_name", "Alex2772", "str"),
    ("papikChatId", "general.papik_chat_id", 625207005, "int"),
    ("telegramApiId", "general.telegram_api_id", 0, "int"),
    ("telegramApiHash", "general.telegram_api_hash", "", "str"),
    ("telegramEnabled", "general.telegram_enabled", True, "bool"),
    (
        "llm",
        "general.llm",
        EndpointAndModel(Endpoint(DEFAULT_BASE_URL, ""), DEFAULT_MODEL, DEFAULT_REASONING_EFFORT),
        "endpoint_model",
    ),
    ("embedding", "general.embedding", EndpointAndModel(), "endpoint_model"),
    ("lockdown", "general.lockdown", LockdownMode.PAPIK_ONLY, "lockdown"),
    # Chats that bypass / never pass the lockdown (the owner is always allowed).
    ("allowedChatIds", "general.allowed_chat_ids", [], "int_list"),
    ("blockedChatIds", "general.blocked_chat_ids", [], "int_list"),
    ("paused", "general.paused", False, "bool"),
    ("canWriteToANewPerson", "misc.can_write_to_a_new_person", False, "bool"),
    ("wakeUpOnPinnedChat", "misc.wake_up_on_pinned_chat", False, "bool"),
    ("randomlyGoSleep", "misc.randomly_go_sleep", True, "bool"),
    ("toolReminderProbability", "misc.tool_reminder_probability", 0.02, "float"),
    ("diaryTokenCountTrigger", "misc.diary_token_count_trigger", 40000, "int"),
    ("diaryInjectionMaxLength", "misc.diary_injection_max_length", 0, "int"),
    ("diaryPlagiarismThreshold", "misc.diary_plagiarism_threshold", 0.97, "float"),
    ("diaryMinRelatedness", "misc.diary_min_relatedness", 0.80, "float"),
    ("chatMaxHistoryLength", "misc.chat_max_history_length", 2000, "int"),
    ("llmTemperature", "misc.llm_temperature", 0.2, "opt_float"),
    ("llmTopP", "misc.llm_top_p", None, "opt_float"),
    ("llmTopK", "misc.llm_top_k", None, "opt_float"),
    ("llmMinP", "misc.llm_min_p", None, "opt_float"),
    ("llmPresencePenalty", "misc.presence_penalty", None, "opt_float"),
    ("llmRepetitionPenalty", "misc.repetition_penalty", None, "opt_float"),
    ("antiRepeatTriggerMax", "misc.anti_repeat_trigger_max", 0.95, "float"),
    ("antiRepeatTriggerAvg", "misc.anti_repeat_trigger_avg", 0.85, "float"),
    ("antiRepeatMaxHistory", "misc.anti_repeat_max_history", 32, "int"),
    ("suggestIgnoreChance", "misc.suggest_ignore_chance", 0.1, "float"),
    ("requestTimeoutSecs", "misc.request_timeout_secs", 120, "seconds"),
    ("videoMaxFrames", "misc.video_max_frames", 16, "int"),
    ("videoMinStepMs", "misc.video_min_step_ms", 1000, "int"),
    ("remindUseAsk", "misc.remind_use_ask", True, "bool"),
    ("typingSimulationMinWpm", "misc.typing_simulation_min_wpm", 120, "int"),
    ("typingSimulationMaxWpm", "misc.typing_simulation_max_wpm", 150, "int"),
    ("checkChatsOnStartup", "misc.check_chats_on_startup", True, "bool"),
    ("chatNotificationFilter", "misc.chat_notification_filter", LockdownMode.NONE, "lockdown"),
    ("channelNotifications", "misc.channel_notifications", True, "bool"),
    ("webPanelEnabled", "misc.web_panel_enabled", False, "bool"),
    ("webPanelPort", "misc.web_panel_port", 8765, "int"),
    # Extra Host names the panel accepts (for example a Tailscale name) and the only Tailscale identity allowed there.
    ("webPanelExtraHosts", "misc.web_panel_extra_hosts", [], "str_list"),
    ("webPanelTailscaleLogin", "misc.web_panel_tailscale_login", "", "str"),
    ("backupDir", "misc.backup_dir", "", "str"),
    ("canJoinChats", "misc.can_join_chats", False, "bool"),
    ("canLeaveChats", "misc.can_leave_chats", True, "bool"),
    ("workerCount", "misc.worker_count", 1, "int"),
    # --- new in the Python port ---
    ("memoryGraphDepth", "misc.memory_graph_depth", 2, "int"),
    ("autoSleepConsolidation", "misc.auto_sleep_consolidation", False, "bool"),
    ("bindHost", "misc.bind_host", "127.0.0.1", "str"),
    ("logLlmRequests", "misc.log_llm_requests", True, "bool"),
    # --- capabilities ---
    ("capabilityWebSearch", "capabilities.web_search.enabled", False, "bool"),
    ("capabilityYouTube", "capabilities.youtube.enabled", False, "bool"),
    ("youtubeProxy", "capabilities.youtube.proxy_url", "", "str"),
    ("capabilityWebReader", "capabilities.web_reader.enabled", False, "bool"),
    ("capabilityReminders", "capabilities.reminders.enabled", False, "bool"),
    ("watchdogEnabled", "misc.watchdog_enabled", True, "bool"),
    ("webSearchProvider", "capabilities.web_search.provider", "ollama", "search_provider"),
    ("webSearchTavilyKey", "capabilities.web_search.tavily_key", "", "str"),
    ("webSearchOllamaKey", "capabilities.web_search.ollama_bearer_key", "", "str"),
    ("capabilityVision", "capabilities.vision.enabled", False, "bool"),
    ("llmImageToText", "capabilities.vision.llm_image_to_text", EndpointAndModel(), "endpoint_model"),
    ("llmImageToTextCheap", "capabilities.vision.llm_image_to_text_cheap", EndpointAndModel(), "endpoint_model"),
    ("capabilityUseStickers", "capabilities.use_stickers.enabled", False, "bool"),
    ("capabilityTakePhoto", "capabilities.take_photo.enabled", False, "bool"),
    ("sdEndpoint", "capabilities.take_photo.sd.endpoint", Endpoint("http://localhost:7860/", ""), "endpoint"),
    ("sdCheckpoint", "capabilities.take_photo.sd.checkpoint", "novaAnimeXL_ilV170.safetensors", "str"),
    ("capabilityHearing", "capabilities.hearing.enabled", False, "bool"),
    (
        "llmAudioToText",
        "capabilities.hearing.llm_audio_to_text",
        EndpointAndModel(Endpoint("http://localhost:9000/v1/", ""), "base"),
        "endpoint_model",
    ),
    ("capabilityRecordVoice", "capabilities.record_voice.enabled", False, "bool"),
    ("recordVoiceBackend", "capabilities.record_voice.backend", TTSBackend.ELEVENLABS, "tts"),
    ("recordVoiceElevenLabsKey", "capabilities.record_voice.elevenlabs.key", "", "str"),
    ("recordVoiceElevenLabsVoice", "capabilities.record_voice.elevenlabs.voice_id", "pPdl9cQBQq4p6mRkZy2Z", "str"),
    ("recordVoiceOpenAIUrl", "capabilities.record_voice.openai.url", "https://api.openai.com/v1/", "str"),
    ("recordVoiceOpenAIKey", "capabilities.record_voice.openai.key", "", "str"),
    ("recordVoiceOpenAIModel", "capabilities.record_voice.openai.model", "tts-1", "str"),
    ("recordVoiceOpenAIVoice", "capabilities.record_voice.openai.voice", "alloy", "str"),
    ("recordVoiceOpenAIFormat", "capabilities.record_voice.openai.response_format", "mp3", "str"),
    ("recordVoiceOpenAIPcmSampleRate", "capabilities.record_voice.openai.pcm_sample_rate", 24000, "int"),
    ("proxyEnabled", "capabilities.proxy.enabled", False, "bool"),
]

# These are technical constants that are not interesting for consumers.
SLEEP_MAX_TIME_SECS = 6 * 3600

COMMENTS: dict[str, str] = {
    "general.character_name": "Used by system and other prompts, defines character name.\n\n"
    "Character's phylosophy and appearance can be adjusted in prompts/character_base.md and\n"
    "prompts/character_appearance.md, respectively.",
    "general.character_nickname": "Character nickname in Telegram.",
    "general.papik_name": "Used by system and other prompts, defines instance owner name.\n"
    "Should match with papik_chat_id account name.",
    "general.papik_chat_id": "User id of the instance owner.\n"
    "Kuni's kernel gives several benefits to the \"instance owner user\":\n"
    "- Kuni can't ban instance owner\n"
    "- Messages from instance owner are prioritized (processed ASAP)\n"
    "- A message from instance owner wakes up Kuni if she sleeps\n\n"
    "Your user id can be obtained by @getmyid_bot, @raw_data_bot or @userinfobot.",
    "general.telegram_api_id": "Telegram API key, see\nhttps://core.telegram.org/api/obtaining_api_id",
    "general.telegram_enabled": "Whether to enable the Telegram bot. If set to false, Kuni will run in standalone mode\n"
    "(useful if you only want to use the proxy server without a Telegram account).",
    "general.telegram_api_hash": "Telegram API hash, acquired on the same page as telegram_api_id.",
    "general.llm": "Main LLM endpoint. Any OpenAI-compatible API. PUT YOUR API KEY AND BASE URL HERE:\n"
    "\n"
    "- baseUrl:  \"https://apihub.agnes-ai.com/v1/\", \"https://openrouter.ai/api/v1/\", \"http://localhost:11434/v1/\"\n"
    "- bearerKey: your API key (or set env KUNI_API_KEY, it wins over this file)\n"
    "- extraBearerKeys: optional array of additional keys for this same endpoint; shared quotas may still apply\n"
    "- model: model id\n"
    "- reasoningEffort: \"low\" | \"medium\" | \"high\" (sent as `reasoning_effort`), \"\" to not send it\n"
    "\n"
    "Env overrides: KUNI_BASE_URL, KUNI_API_KEY, KUNI_MODEL, KUNI_REASONING_EFFORT.",
    "general.embedding": "OPTIONAL embeddings endpoint. The memory is a graph and does not need embeddings.\n"
    "If set, it's used only by the anti-repeat check (semantic similarity); if baseUrl is empty,\n"
    "a lexical similarity is used instead.",
    "general.lockdown": "Whether other users can access Kuni.\n"
    "Restricted users will be ignored.\n"
    "LLM will see allowed chats only.\n\n"
    "- \"none\" Kuni will be publicly available (everyone can access her)\n"
    "- \"contacts_only\" only Kuni's contacts can access her\n"
    "- \"papik_only\" papik_chat_id is the only one who can access her",
    "general.paused": "Pause AI replies and proactive activity. Owner commands still work. Use /pause off to resume.",
    "capabilities.take_photo.enabled": "Whether Kuni can generate new photos. Requires `vision` capability.",
    "capabilities.take_photo.sd.endpoint": "Address to stable-diffusion deployment (AUTOMATIC1111 WebUI with --api)",
    "capabilities.take_photo.sd.checkpoint": "Stable diffusion checkpoint. https://civitai.com/models/376130/nova-anime-xl",
    "capabilities.web_search.enabled": "Whether Kuni can use web search.",
    "capabilities.web_search.provider": "\"tavily\" (official API, free tier of 1000 searches a month, needs tavily_key), \"duckduckgo\"\n"
    "(no key, but it scrapes a public page and rate-limits servers after a query or two) or \"ollama\" (needs paid credits).",
    "capabilities.web_search.tavily_key": "API key from tavily.com (free signup, no card).",
    "capabilities.youtube.enabled": "Search YouTube and read available subtitles (no video download or API key).",
    "capabilities.youtube.proxy_url": "Optional HTTP/HTTPS proxy for YouTube metadata and captions only; empty = direct, ignoring environment proxies.\n"
    "Use a local HTTP bridge such as http://127.0.0.1:11990 for other tunnel protocols.",
    "capabilities.vision.enabled": "Whether Kuni can see images/stickers. You should use a vision-capable LLM here.\n"
    "Images/stickers are transcribed to text via LLM and passed to main LLM. Transcriptions are cached.",
    "misc.can_write_to_a_new_person": "Allows Kuni to text a completely new person Kuni didn't have a prior dialog with.\n"
    "This defaults to false to avoid using Kuni as spam bot.",
    "capabilities.use_stickers.enabled": "Allows Kuni to save and send stickers. Requires `vision` capability.",
    "capabilities.hearing.enabled": "Whether Kuni can listen to voice messages if telegram premium is not available.",
    "misc.wake_up_on_pinned_chat": "If true, the pinned chats will wake up Kuni as soon as a message receives. If set to false, only the\n"
    "PAPIK_CHAT_ID will be able to do so.\n\n"
    "The instance owner (also known as PAPIK) can pin chats from Kuni's Telegram account with his friends so Kuni's\n"
    "tokens preservation mechanisms (sleep) are not applied to them.",
    "misc.randomly_go_sleep": "If true, Kuni will randomly go to sleep after some time of inactivity to save LLM tokens.\n"
    "While sleeping, Kuni won't respond to messages (except from papik_chat_id).",
    "misc.tool_reminder_probability": "Probability (0.0-1.0) that Kuni will be reminded about available tools in a given message.\n"
    "Helps prevent Kuni from forgetting she has tools like record audio or photo generation.",
    "misc.diary_token_count_trigger": "When the conversation context exceeds this token count, Kuni dumps her memory to the graph\n"
    "and restarts with a clean context. Keeps context window usage under control.\n"
    "Requires context window >= this value + some headroom.",
    "misc.diary_injection_max_length": "Maximum number of characters of memory (diary pages) automatically injected into the context per\n"
    "message. 0 DISABLES automatic injection (Kuni still reaches the memory through #ask).",
    "misc.diary_plagiarism_threshold": "Similarity threshold (0.0-1.0) above which a new diary page is considered a duplicate\n"
    "of an existing one and will not be saved. Prevents redundant memories.",
    "misc.diary_min_relatedness": "Minimum relatedness (0.0-1.0) for a memory to be injected into context.",
    "misc.chat_max_history_length": "Maximum number of characters loaded per chat loaded.\n"
    "Older messages beyond this limit are omitted.",
    "misc.llm_temperature": "LLM sampling temperature. Higher = more creative/random, lower = more deterministic.\n"
    "\"none\" to use the model's default.",
    "misc.llm_top_p": "LLM nucleus sampling (top-p). Limits token selection to the top cumulative probability mass.\n"
    "\"none\" to use the model's default.",
    "misc.llm_top_k": "LLM top-k sampling. Limits token selection to the k most likely tokens.\n"
    "\"none\" to use the model's default.",
    "misc.llm_min_p": "LLM min-p sampling. Filters tokens below a minimum probability relative to the top token.\n"
    "\"none\" to use the model's default.",
    "misc.presence_penalty": "LLM presence penalty. Positive values discourage the model from repeating topics already mentioned.\n"
    "\"none\" to use the model's default.",
    "misc.repetition_penalty": "LLM repetition penalty. Values > 1.0 penalize repeated tokens.\n"
    "\"none\" to use the model's default.",
    "misc.anti_repeat_trigger_max": "Maximum similarity threshold for the anti-repeat mechanism.\n"
    "If the new response is too similar to a recent one (above this threshold), it is rejected.",
    "misc.anti_repeat_trigger_avg": "Average similarity threshold for the anti-repeat mechanism.\n"
    "Works together with anti_repeat_trigger_max to detect repetitive responses.",
    "misc.anti_repeat_max_history": "Number of recent responses kept in history for the anti-repeat similarity check.",
    "misc.request_timeout_secs": "HTTP timeout (in seconds) for all outgoing requests made by Kuni's kernel\n"
    "(LLM, TTS, stable diffusion, transcription, etc). For streaming it's the maximum silence between chunks.\n"
    "Increase this if your LLM backend is slow to respond (reasoning models can think for a while).",
    "misc.video_max_frames": "Maximum number of frames sampled from a video/GIF/video-sticker for captioning.",
    "misc.video_min_step_ms": "Minimum interval (in milliseconds) between two sampled video frames.",
    "capabilities.web_search.ollama_bearer_key": "Bearer key for the Ollama web search integration (ollama.com/api/web_search).",
    "capabilities.vision.llm_image_to_text": "LLM endpoint and model used to transcribe images/stickers to text.\n"
    "Should be a vision-capable model. Empty fields inherit from general.llm. Transcriptions are cached.",
    "capabilities.vision.llm_image_to_text_cheap": "LLM endpoint and model used to transcribe videos to text.\n"
    "Empty fields inherit from general.llm.",
    "capabilities.hearing.llm_audio_to_text": "Whisper-compatible endpoint used to transcribe voice messages to text\n"
    "(POST audio/transcriptions). Used when Telegram Premium (native transcription) is not available.",
    "capabilities.record_voice.enabled": "Whether Kuni can send voice messages (text-to-speech).\n"
    "Requires configuring a TTS backend (elevenlabs or openai).",
    "capabilities.record_voice.backend": "TTS backend: \"elevenlabs\" or \"openai\" (OpenAI/local OpenAI-compatible TTS service).",
    "capabilities.record_voice.elevenlabs.key": "ElevenLabs API key for TTS. Required when backend is \"elevenlabs\".",
    "capabilities.record_voice.elevenlabs.voice_id": "ElevenLabs voice ID to use for TTS.",
    "capabilities.record_voice.openai.url": "OpenAI-compatible TTS API base URL. Can point to a self-hosted TTS service.",
    "capabilities.record_voice.openai.key": "API key for the OpenAI TTS endpoint.",
    "capabilities.record_voice.openai.model": "OpenAI TTS model name, e.g. \"tts-1\" or \"tts-1-hd\".",
    "capabilities.record_voice.openai.voice": "OpenAI TTS voice name, e.g. \"alloy\", \"echo\", \"fable\", \"onyx\", \"nova\", \"shimmer\".",
    "capabilities.record_voice.openai.response_format": "TTS audio format requested from the provider. \"mp3\" for standard OpenAI TTS.\n"
    "Providers that only return raw PCM: set \"pcm\" to transcode it to OGG/Opus (requires ffmpeg).",
    "capabilities.record_voice.openai.pcm_sample_rate": "Sample rate (Hz) of the raw PCM stream when response_format = \"pcm\".",
    "capabilities.proxy.enabled": "Whether to enable the built-in OpenAI-compatible proxy server (port 10434).\n"
    "Kuni exposes an API endpoint that forwards requests through her LLM, allowing external tools\n"
    "(e.g. VS Code Copilot) to use Kuni as a model backend.",
    "misc.remind_use_ask": "Remind to use #ask tool (main memory fetch) more proactively.\n"
    "Some LLMs forget about using this. If true, the system inserts reminders if LLM doesn't use #ask.",
    "misc.typing_simulation_min_wpm": "Minimum simulated typing speed (words per minute) used to delay sending consecutive messages.",
    "misc.typing_simulation_max_wpm": "Maximum simulated typing speed (words per minute) used to delay sending consecutive messages.",
    "misc.check_chats_on_startup": "If true, Kuni puts all unread chats to queue on program startup.\n"
    "If false, Kuni checks chats only if a notification was received during program runtime.",
    "misc.suggest_ignore_chance": "A value between 0.0 and 0.98 specifying chance of reading and avoiding action.\n"
    "Greater values will make Kuni tend to read and ignore messages.\n"
    "This was implemented as a countermeasure to spamming Kuni's DM.",
    "misc.chat_notification_filter": "In contrast to lockdown mode, controls notifications specifically. You can set lockdown = none and\n"
    "chat_notification_filter = contacts_only to limit notifications to contacts only, but LLM is still\n"
    "able to freely open any chat.",
    "misc.can_join_chats": "Whether Kuni can join a telegram group chat/channel by an invite link or @username (if LLM decides to do so).\n"
    "Only the owner's chat and Kuni's own initiative may trigger it (not other contacts), at most 5 joins per day.\n"
    "Needs lockdown = \"contacts_only\" or \"none\": channels are not readable under \"papik_only\".",
    "misc.can_leave_chats": "Whether Kuni can leave a telegram group chat/channel.",
    "general.allowed_chat_ids": "Chat ids that may talk to Kuni even under contacts_only / papik_only (the owner is always allowed).",
    "general.blocked_chat_ids": "Chat ids Kuni never reads or answers in any mode (the owner cannot be blocked).",
    "misc.backup_dir": "Directory with the daily archives (the panel lists them and can create one now). Empty = disabled.",
    "misc.web_panel_enabled": "Owner-only control panel on 127.0.0.1 (never exposed; reach it through an SSH tunnel).\n"
    "The access token is created in data/web_panel_token.",
    "misc.web_panel_extra_hosts": "Extra host names the panel accepts besides localhost, e.g. [\"kuni.tail1234.ts.net\"] for Tailscale.\n"
    "The panel itself still listens on 127.0.0.1 only; Tailscale Serve forwards to it.",
    "misc.web_panel_tailscale_login": "If set, requests that arrive under an extra host must carry this Tailscale identity\n"
    "(header Tailscale-User-Login, set by Tailscale Serve). Empty = the token alone decides.",
    "misc.web_panel_port": "Local TCP port of the web panel (bound to 127.0.0.1 only).",
    "misc.channel_notifications": "If false, new posts in broadcast channels never wake Kuni or cost an LLM request;\n"
    "she reads channels on her own initiative while acting proactively (get_telegram_chats shows unread counts).",
    "misc.worker_count": "Amount of Kuni's subpersons that process messages. 1 is totally fine.",
    "misc.memory_graph_depth": "How many hops the graph memory activation spreads from the entities found in a query.",
    "misc.auto_sleep_consolidation": "Run the graph \"sleep\" (merge duplicates, reconcile contradictions, rewrite summaries) automatically\n"
    "every night at ~04:00 UTC. The C++ kernel never ran it by itself; use `python -m kuni sleep` to run by hand.",
    "misc.bind_host": "Interface for the proxy server and the Prometheus exporter. 127.0.0.1 = local only (safe).\n"
    "Use 0.0.0.0 only if you really need to reach them from another machine/container.",
    "misc.log_llm_requests": "Dump every LLM request/response to logs/*.json (contains your private chats!).",
    "misc.watchdog_enabled": "Send health alerts/recovery to the owner, without LLM. External systemd watchdog is configured separately.",
    "capabilities.web_reader.enabled": "Read public HTTP/HTTPS text pages by URL, without scripts, cookies or private networks.",
    "capabilities.reminders.enabled": "Persistent per-chat reminders, sent directly via Telegram without waiting for the LLM. Paused by /pause.",
}


# --------------------------------------------------------------------------------------------------------------------


class Config:
    """Plain attribute container populated from SCHEMA."""

    SLEEP_MAX_TIME_SECS = SLEEP_MAX_TIME_SECS

    def __init__(self) -> None:
        for attr, _path, default, _kind in SCHEMA:
            setattr(self, attr, _copy(default))

    def resolved_llm(self) -> EndpointAndModel:
        return self.llm.resolved(EndpointAndModel(Endpoint(DEFAULT_BASE_URL, ""), DEFAULT_MODEL))

    def resolve(self, endpoint_and_model: EndpointAndModel) -> EndpointAndModel:
        return endpoint_and_model.resolved(self.resolved_llm())


def _copy(v):
    if isinstance(v, list):
        return list(v)
    if isinstance(v, EndpointAndModel):
        return EndpointAndModel(replace(v.endpoint), v.model, v.reasoningEffort)
    if isinstance(v, Endpoint):
        return replace(v)
    return v


def _dig(root: dict, path: str):
    cur: Any = root
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _parse_endpoint(v: Any, default: Endpoint) -> Endpoint:
    if not isinstance(v, dict):
        return _copy(default)
    extra = v.get("extraBearerKeys", default.extraBearerKeys)
    if not isinstance(extra, (list, tuple)) or any(
            not isinstance(key, str) or not key or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in key)
            for key in extra):
        raise ValueError("extraBearerKeys must be an array of nonempty keys without whitespace")
    return Endpoint(str(v.get("baseUrl", default.baseUrl)), str(v.get("bearerKey", default.bearerKey)), tuple(extra))


def _parse_value(v: Any, default: Any, kind: str):
    if v is None:
        return _copy(default)
    if kind == "str":
        return str(v)
    if kind == "int":
        return int(v)
    if kind == "bool":
        return bool(v)
    if kind == "float":
        return float(v)
    if kind == "seconds":
        return int(v)
    if kind == "opt_float":
        return None if (isinstance(v, str) and v.lower() == "none") else float(v)
    if kind == "endpoint":
        return _parse_endpoint(v, default)
    if kind == "endpoint_model":
        if not isinstance(v, dict):
            return _copy(default)
        return EndpointAndModel(
            _parse_endpoint(v.get("endpoint"), default.endpoint),
            str(v.get("model", default.model)),
            str(v.get("reasoningEffort", default.reasoningEffort)),
        )
    if kind == "search_provider":
        if v not in ("ollama", "duckduckgo", "tavily"):
            raise ValueError('capabilities.web_search.provider must be "tavily", "duckduckgo" or "ollama"')
        return v
    if kind == "str_list":
        if (not isinstance(v, list) or len(v) > 20
                or any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", i) for i in v)):
            raise ValueError("expected an array of up to 20 host names")
        return list(dict.fromkeys(i.lower() for i in v))
    if kind == "int_list":
        if not isinstance(v, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in v):
            raise ValueError("expected an array of integer chat ids")
        return list(dict.fromkeys(v))
    if kind == "lockdown":
        return LockdownMode(str(v).lower())
    if kind == "tts":
        return TTSBackend(str(v).lower())
    raise ValueError(kind)


def _apply_env(cfg: Config) -> None:
    llm: EndpointAndModel = cfg.llm
    if v := os.environ.get("KUNI_BASE_URL"):
        llm.endpoint.baseUrl = v
        llm.endpoint.extraBearerKeys = ()  # do not send file-based pool credentials to an overridden endpoint
    if v := os.environ.get("KUNI_API_KEY"):
        llm.endpoint.bearerKey = v
        llm.endpoint.extraBearerKeys = ()
    if v := os.environ.get("KUNI_MODEL"):
        llm.model = v
    if v := os.environ.get("KUNI_REASONING_EFFORT"):
        llm.reasoningEffort = v


def parse(data: dict) -> Config:
    cfg = Config()
    for attr, path, default, kind in SCHEMA:
        setattr(cfg, attr, _parse_value(_dig(data, path), default, kind))
    _apply_env(cfg)
    return cfg


# ----- serialization (with comments) ---------------------------------------------------------------------------------


def _scalar(v: Any) -> str:
    return tomli_w.dumps({"x": v})[len("x = ") :].rstrip("\n")


def _endpoint_inline(e: Endpoint) -> str:
    extra = f", extraBearerKeys = {_scalar(list(e.extraBearerKeys))}" if e.extraBearerKeys else ""
    return f"{{ baseUrl = {_scalar(e.baseUrl)}, bearerKey = {_scalar(e.bearerKey)}{extra} }}"


def _serialize_value(v: Any, kind: str) -> str:
    if kind == "opt_float":
        return _scalar("none" if v is None else float(v))
    if kind == "seconds":
        return _scalar(int(v))
    if kind == "int_list":
        return "[" + ", ".join(str(int(i)) for i in v) + "]"
    if kind == "str_list":
        return "[" + ", ".join(_scalar(i) for i in v) + "]"
    if kind in ("lockdown", "tts"):
        return _scalar(v.value)
    if kind == "endpoint":
        return _endpoint_inline(v)
    if kind == "endpoint_model":
        parts = [f"endpoint = {_endpoint_inline(v.endpoint)}", f"model = {_scalar(v.model)}"]
        if v.reasoningEffort:
            parts.append(f"reasoningEffort = {_scalar(v.reasoningEffort)}")
        return "{ " + ", ".join(parts) + " }"
    return _scalar(v)


def render(cfg: Config) -> str:
    header = (
        "# Kuni kernel project (Python port), https://github.com/alex2772/kuni/\n"
        "# WARNING - SENSITIVE DATA! DO NOT SHARE NOR COMMIT THIS FILE!!!\n"
        "#\n"
        "# This file contains all the settings available in Kuni's kernel.\n"
        "# To adjust prompts, go to prompts/ directory.\n"
        "# If kuni is running, it will autoreload changes to this file and prompts automatically.\n"
        "# Setting up [general] section is mandatory, everything else is optional.\n"
    )
    out: list[str] = [header]
    tables: dict[str, list[tuple[str, str, Any, str]]] = {}
    for attr, path, _default, kind in SCHEMA:
        table, key = path.rsplit(".", 1)
        tables.setdefault(table, []).append((key, path, getattr(cfg, attr), kind))
    for table, entries in tables.items():
        out.append(f"\n[{table}]\n")
        for key, path, value, kind in entries:
            out.append("\n")
            for line in COMMENTS.get(path, "").split("\n"):
                if line or path in COMMENTS:
                    out.append(f"# {line}\n".replace("# \n", "#\n"))
            out.append(f"{key} = {_serialize_value(value, kind)}\n")
    return "".join(out)


def init_config_file(path: Path = CONFIG_TOML, *, base_url: str | None = None, api_key: str | None = None,
                     model: str | None = None, effort: str | None = None) -> Config:
    cfg = Config()
    cfg.papikChatId, cfg.papikName = 0, "Owner"  # placeholders: a new file must not name anybody's chat as the owner
    if base_url:
        cfg.llm.endpoint.baseUrl = base_url
    if api_key:
        cfg.llm.endpoint.bearerKey = api_key
    if model:
        cfg.llm.model = model
    if effort:
        cfg.llm.reasoningEffort = effort
    path.write_text(render(cfg), encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # holds secrets
    except OSError:
        pass
    return cfg


# ----- global state / hot reload -------------------------------------------------------------------------------------

_config: Config | None = None
_listeners: list[Callable[[], None]] = []
_mtime = 0.0
TESTING = False


def on_config_updated(cb: Callable[[], None]) -> None:
    _listeners.append(cb)


def set_config(cfg: Config) -> None:
    """For tests / programmatic use."""
    global _config
    _config = cfg


def validate(cfg: Config) -> list[str]:
    errors = []
    if cfg.telegramEnabled:
        if cfg.papikChatId in (0, ORIGINAL_AUTHOR_CHAT_ID):
            errors.append("general.papik_chat_id must be YOUR Telegram user id: it decides who may control Kuni "
                          "(0 means nobody, and the old default is the original author's chat)")
        if cfg.telegramApiId == 0:
            errors.append("general.telegram_api_id should be populated (the actual value is 0)")
        if not cfg.telegramApiHash:
            errors.append("general.telegram_api_hash should be populated (the actual string is empty)")
    if not (0 <= cfg.suggestIgnoreChance <= 0.98):
        errors.append("misc.suggest_ignore_chance should be between 0.0 and 0.98")
    return errors


def load(path: Path = CONFIG_TOML, *, save_back: bool = False) -> Config:
    global _mtime
    if not path.exists():
        cfg = init_config_file(path)
        log.info("Created fresh config file %s", path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = parse(data)
    if save_back:
        # if an instance owner updates kuni kernel, new fields will be added here
        raw = Config()
        for attr, p, _d, kind in SCHEMA:  # keep file values, not env overrides
            setattr(raw, attr, _parse_value(_dig(data, p), _d, kind))
        path.write_text(render(raw), encoding="utf-8")
    _mtime = path.stat().st_mtime
    log.info("%s loaded", path)
    return cfg


def config() -> Config:
    global _config
    if _config is None:
        _config = load(save_back=True)
    return _config


def reload_if_changed(path: Path = CONFIG_TOML) -> bool:
    global _config, _mtime
    try:
        m = path.stat().st_mtime
    except OSError:
        return False
    if m == _mtime:
        return False
    log.info("%s updated - reloading", path)
    try:
        _config = load(path)
    except Exception as e:  # noqa: BLE001
        log.error("config reload failed: %s", e)
        _mtime = m
        return False
    for cb in list(_listeners):
        try:
            cb()
        except Exception:  # noqa: BLE001
            log.exception("config listener failed")
    return True


def update_general_settings(**updates) -> Config:
    """Owner commands may change these two settings, never arbitrary config paths or secrets."""
    if not updates or updates.keys() - {"paused", "lockdown"}:
        raise ValueError("unsupported setting")
    if "paused" in updates and not isinstance(updates["paused"], bool):
        raise ValueError("paused must be a boolean")
    if "lockdown" in updates:
        updates["lockdown"] = LockdownMode(updates["lockdown"]).value
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    data.setdefault("general", {}).update(updates)
    parse(data)  # validate before replacing a file containing credentials
    atomic_write_text(CONFIG_TOML, tomli_w.dumps(data))
    reload_if_changed()
    return config()


# Settings the web panel may change (never keys, endpoints, Telegram credentials or paths):
# attr -> ("bool",) | ("enum", values) | ("int" | "float", lo, hi)
EDITABLE_SETTINGS: dict[str, tuple] = {
    "paused": ("bool",),
    "lockdown": ("enum", [m.value for m in LockdownMode]),
    "chatNotificationFilter": ("enum", [m.value for m in LockdownMode]),
    "channelNotifications": ("bool",),
    "capabilityWebReader": ("bool",),
    "capabilityWebSearch": ("bool",),
    "capabilityReminders": ("bool",),
    "capabilityYouTube": ("bool",),
    "canJoinChats": ("bool",),
    "canLeaveChats": ("bool",),
    "canWriteToANewPerson": ("bool",),
    "wakeUpOnPinnedChat": ("bool",),
    "randomlyGoSleep": ("bool",),
    "checkChatsOnStartup": ("bool",),
    "autoSleepConsolidation": ("bool",),
    "remindUseAsk": ("bool",),
    "watchdogEnabled": ("bool",),
    "logLlmRequests": ("bool",),
    "workerCount": ("int", 1, 4),
    "allowedChatIds": ("ids",),
    "blockedChatIds": ("ids",),
    "suggestIgnoreChance": ("float", 0.0, 0.98),
}


def _checked_setting(attr: str, value):
    spec = EDITABLE_SETTINGS.get(attr)
    if spec is None:
        raise ValueError("unsupported setting")
    if spec[0] == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{attr} must be true or false")
        return value
    if spec[0] == "enum":
        if value not in spec[1]:
            raise ValueError(f"{attr} must be one of {', '.join(spec[1])}")
        return value
    if spec[0] == "ids":
        if (not isinstance(value, list) or len(value) > 200
                or any(isinstance(i, bool) or not isinstance(i, int) or i == 0 or abs(i) >= 2 ** 63 for i in value)):
            raise ValueError(f"{attr} must be an array of up to 200 nonzero integer chat ids")
        return list(dict.fromkeys(value))
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not spec[1] <= value <= spec[2] or (spec[0] == "int" and value != int(value))):
        raise ValueError(f"{attr} must be a number from {spec[1]} to {spec[2]}")
    return int(value) if spec[0] == "int" else float(value)


def update_settings(updates: dict) -> Config:
    """Owner-only changes of whitelisted, non-secret settings: validate everything, then write the file atomically."""
    if not updates:
        raise ValueError("no settings given")
    paths = {attr: path for attr, path, _d, _k in SCHEMA}
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    for attr, value in updates.items():
        *parents, key = paths[attr].split(".") if attr in EDITABLE_SETTINGS else (None,)
        if key is None:
            raise ValueError("unsupported setting")
        table = data
        for name in parents:
            table = table.setdefault(name, {})
        table[key] = _checked_setting(attr, value)
    parse(data)  # validate before replacing a file containing credentials
    atomic_write_text(CONFIG_TOML, tomli_w.dumps(data))
    reload_if_changed()
    return config()


def _endpoint_table(data: dict) -> dict:
    return data.setdefault("general", {}).setdefault("llm", {}).setdefault("endpoint", {})


def _write_validated(data: dict) -> Config:
    parse(data)  # validate before replacing a file containing credentials
    atomic_write_text(CONFIG_TOML, tomli_w.dumps(data))
    reload_if_changed()
    return config()


def add_extra_key(key: str) -> str:
    """Adds an API key to general.llm.endpoint.extraBearerKeys; returns its fingerprint. The key is never logged."""
    if (not isinstance(key, str) or not 8 <= len(key) <= 400
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in key)):
        raise ValueError("a key must be 8-400 characters without spaces or control characters")
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    endpoint = _endpoint_table(data)
    if key == endpoint.get("bearerKey") or key in endpoint.get("extraBearerKeys", []):
        raise ValueError("this key is already in the pool")
    endpoint["extraBearerKeys"] = [*endpoint.get("extraBearerKeys", []), key]
    if len(endpoint["extraBearerKeys"]) > 10:
        raise ValueError("at most 10 additional keys")
    _write_validated(data)
    return key_fingerprint(key)


def remove_extra_key(fingerprint: str) -> None:
    """Removes an additional key by fingerprint. The primary key (bearerKey) cannot be removed from here."""
    data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    endpoint = _endpoint_table(data)
    extras = endpoint.get("extraBearerKeys", [])
    kept = [k for k in extras if key_fingerprint(k) != fingerprint]
    if len(kept) == len(extras):
        raise ValueError("key not found among the additional keys")
    if kept:
        endpoint["extraBearerKeys"] = kept
    else:
        endpoint.pop("extraBearerKeys", None)
    _write_validated(data)


async def watch(interval: float = 1.0) -> None:
    """Hot-reload (the C++ kernel uses inotify/FSEvents; polling mtime is portable)."""
    while True:
        await asyncio.sleep(interval)
        reload_if_changed()


def die_if_invalid(cfg: Config) -> None:
    errors = validate(cfg)
    if errors:
        for e in errors:
            log.error(e)
        sys.exit(-1)
