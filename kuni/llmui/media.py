"""Images / videos / voice for the LLM: everything is transcribed to cached text. Port of llmui/image|video|audio|
telegram_voice_messages|telegram_video_messages. FFmpeg is used through its CLI (the C++ kernel linked libav)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import config
from ..llm import ROLE_USER, Message, OpenAIChat, Params, Session, embed_image, embed_image_bytes
from ..prompts import prompts
from ..telegram.base import TelegramClient
from ..telegram.types import Message as TgMessage
from ..telegram.types import TelegramError

log = logging.getLogger("llmui")

VIDEO_EXTS = {"webm", "mp4", "mkv", "avi", "mov", "m4v", "flv", "3gp"}
NOT_SUPPORTED = "<{tag} description>\nThis media type is not supported\n</{tag}>"

# the cache lives next to the working dir: cache/images, cache/videos
CACHE_DIR = Path("cache")


def check_for_malicious_payloads(s: str) -> str:
    """The internal image tag must never come from a user: it would let them inject arbitrary images/instructions."""
    from ..llm import EMBEDDING_TAG

    return "malicious" if EMBEDDING_TAG in s else s


def _context_text(temporary_context: Sequence[Message]) -> str:
    context = "<context>\n"
    context += f'<character name="{config().characterName}">\n'
    context += prompts().character_appearance
    context += f'\n</character name="{config().characterName}">\n'
    for i in temporary_context:
        context += "<context_item>\n" + i.content + "\n</context_item>\n"
    return context


# ---------------------------------------------------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------------------------------------------------
async def image(temporary_context: Sequence[Message], llm: OpenAIChat, path: str | Path, xml_tag: str = "photo") -> str:
    path = Path(path)
    cache_dir = CACHE_DIR / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{path.name}.md"
    try:
        if cache.is_file():
            return f"<{xml_tag} description>\n{cache.read_text(encoding='utf-8')}\n</{xml_tag}>"

        context = _context_text(temporary_context)
        # route video formats to the video pipeline
        if path.suffix.lower().lstrip(".") in VIDEO_EXTS:
            return await video(temporary_context, llm, path, xml_tag)

        context += "\n\n</context>\n\nPhoto:\n\n"
        try:
            from PIL import Image

            with Image.open(path) as im:
                im.load()
                context += embed_image(im)
        except Exception:  # noqa: BLE001  (tgs / corrupted / not an image)
            return NOT_SUPPORTED.format(tag=xml_tag)
        context += "\n\nDescribe the last photo."

        # hack: ask for shorter descriptions for stickers.
        prompt = prompts().sticker_to_text if "sticker" in xml_tag else prompts().photo_to_text
        for _ in range(3):  # a local model can overflow its reasoning and return nothing: try again
            response = await llm.chat(Params(system_prompt=prompt, config=config().llmImageToText),
                                      Session([Message(role=ROLE_USER, content=context)]))
            content = response.choices[0].message.content.strip() if response.choices else ""
            if content:
                cache.write_text(content, encoding="utf-8")
                return f"<{xml_tag} description>\n{content}\n</{xml_tag}>"
        return NOT_SUPPORTED.format(tag=xml_tag)
    except Exception as e:  # noqa: BLE001
        log.error("Can't describe photo: %s", e)
        return NOT_SUPPORTED.format(tag=xml_tag)


# ---------------------------------------------------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Frame:
    track: str  # "video" or "audio"
    from_ms: int
    to_ms: int
    contents: str


def format_timestamp(ms: int) -> str:
    minutes, secs, millis = ms // 60000, (ms % 60000) // 1000, ms % 1000
    if millis == 0:
        return f"{minutes}:{secs:02d}"
    return f"{minutes}:{secs:02d}.{millis:03d}".rstrip("0")


async def _run(*args: str, stdin: bytes | None = None) -> tuple[int, bytes]:
    proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate(stdin)
    return proc.returncode or 0, out


async def _grab_frame(path: Path, sec: float, duration: float) -> tuple[int, bytes]:
    """One JPEG frame at `sec`. The very last frame sits at the end of the stream, where ffmpeg has nothing to give
    (it drops frames earlier than the requested position): step back until something comes out."""
    attempts = [sec]
    if duration > 0:
        attempts = [max(0.0, min(sec, duration - back)) for back in (0.05, 0.35, 1.0)] + [0.0]
    code, jpg = 1, b""
    for t in dict.fromkeys(attempts):
        code, jpg = await _run("ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                               "-vf", "scale=512:512", "-f", "image2pipe", "-vcodec", "mjpeg", "-")
        if code == 0 and jpg:
            break
    return code, jpg


async def _probe(path: Path) -> dict:
    code, out = await _run("ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path))
    if code != 0:
        return {}
    return json.loads(out or b"{}")


async def video_frames(temporary_context: Sequence[Message], llm: OpenAIChat, path: str | Path, is_sticker: bool = False
                       ) -> list[Frame]:
    path = Path(path)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        log.warning("ffmpeg/ffprobe not found: video can't be described")
        return []
    try:
        info = await _probe(path)
        streams = [s.get("codec_type") for s in info.get("streams", [])]
        if "video" not in streams:
            return []
        duration = float((info.get("format") or {}).get("duration") or 0.0)
        c = config()
        frame_count = (max(1, min(math.ceil(duration * 1000 / c.videoMinStepMs), c.videoMaxFrames)) if duration > 0 else 1)
        frame_prompt = "You are video-to-text captioning module. Don't overthink, provide response as fast as possible."

        video_list: list[Frame] = []
        for i in range(frame_count):
            frame_sec = duration * i / (frame_count - 1) if frame_count > 1 and duration > 0 else 0.0
            next_sec = (duration * (i + 1) / (frame_count - 1)
                        if frame_count > 1 and duration > 0 and i + 1 < frame_count else frame_sec + 1.0)
            code, jpg = await _grab_frame(path, frame_sec, duration)
            if code != 0 or not jpg:
                log.warning("Failed to decode frame %d for %s", i, path)
                continue
            frame_context = embed_image_bytes(jpg) + "\n\nDescribe this frame briefly."
            description = ""
            for _ in range(3):
                response = await llm.chat(Params(system_prompt=frame_prompt, config=c.llmImageToText),
                                          Session([Message(role=ROLE_USER, content=frame_context)]))
                description = response.choices[0].message.content.strip() if response.choices else ""
                if description:
                    break
            if not description:
                continue
            video_list.append(Frame("video", int(frame_sec * 1000), int(next_sec * 1000), description))

        audio_frames: list[Frame] = []
        if not is_sticker and c.capabilityHearing and "audio" in streams:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "audio.ogg"
                code, _ = await _run("ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-c:a", "libopus",
                                     "-b:a", "32k", str(out))
                if code == 0 and out.exists():
                    try:
                        tr = await llm.transcribe_audio(out.read_bytes(), "ogg")
                        for seg in tr.segments:
                            text = str(seg.get("text", "")).strip()
                            if text:
                                audio_frames.append(Frame("audio", int(float(seg.get("start", 0)) * 1000),
                                                          int(float(seg.get("end", 0)) * 1000), text))
                    except Exception as e:  # noqa: BLE001
                        log.warning("Audio transcription failed: %s", e)

        # merge video and audio frames in chronological order
        merged: list[Frame] = []
        vi = ai = 0
        while vi < len(video_list) or ai < len(audio_frames):
            take_video = vi < len(video_list) and (ai >= len(audio_frames) or video_list[vi].from_ms <= audio_frames[ai].from_ms)
            if take_video:
                merged.append(video_list[vi])
                vi += 1
            else:
                merged.append(audio_frames[ai])
                ai += 1
        return merged
    except Exception as e:  # noqa: BLE001
        log.error("Can't describe video frames: %s", e)
        return []


async def video(temporary_context: Sequence[Message], llm: OpenAIChat, path: str | Path, xml_tag: str = "video") -> str:
    path = Path(path)
    cache_dir = CACHE_DIR / "videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{path.name}.md"
    wrap = ("<{tag} transcription>\n{body}\n</{tag} transcription instructions=\"You finished watching this video and "
            "you should acknowledge about it's contents shown above.\">")
    try:
        if cache.is_file():
            return wrap.format(tag=xml_tag, body=cache.read_text(encoding="utf-8"))
        frames = await video_frames(temporary_context, llm, path, is_sticker="sticker" in xml_tag)
        if not frames:
            return NOT_SUPPORTED.format(tag=xml_tag)
        merged: list[Frame] = []
        for f in frames:  # merge consecutive frames of the same (audio) track
            if merged and merged[-1].track == "audio" and f.track == "audio":
                merged[-1].to_ms = f.to_ms
                merged[-1].contents += "\n\n" + f.contents
            else:
                merged.append(f)
        body = "\n".join(
            f'<f track="{f.track}" from="{format_timestamp(f.from_ms)}" to="{format_timestamp(f.to_ms)}">\n{f.contents}\n</f>'
            for f in merged)
        cache.write_text(body, encoding="utf-8")
        return wrap.format(tag=xml_tag, body=body)
    except Exception as e:  # noqa: BLE001
        log.error("Can't describe video: %s", e)
        return NOT_SUPPORTED.format(tag=xml_tag)


# ---------------------------------------------------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------------------------------------------------
async def voice_message(path: str | Path, llm: OpenAIChat) -> str:
    if not config().capabilityHearing:
        return ""
    try:
        tr = await llm.transcribe_audio(Path(path).read_bytes(), "ogg")
        return "<voice>\n" + tr.text + "\n</voice>"
    except Exception as e:  # noqa: BLE001
        log.error("Can't transcribe audio: %s", e)
        return ""


async def _premium_transcription(telegram: TelegramClient, msg: TgMessage) -> str | None:
    me = await telegram.get_user(telegram.my_id)
    if not me.is_premium:
        return None
    text = await telegram.recognize_speech(msg)
    return check_for_malicious_payloads(text) if text else None


async def voice_message_transcription(telegram: TelegramClient, msg: TgMessage, llm: OpenAIChat) -> str:
    text = await _premium_transcription(telegram, msg)
    if text is not None:
        return "[voice transcription]: " + text + "\n"
    # fall back to local Whisper transcription
    return await voice_message(await telegram.download_media(msg), llm)


async def video_note_transcription(telegram: TelegramClient, llm: OpenAIChat, msg: TgMessage) -> str:
    try:
        if config().capabilityHearing and config().capabilityVision:
            return await video([], llm, await telegram.download_media(msg), "video_note")
    except (TelegramError, Exception):  # noqa: BLE001
        pass
    text = await _premium_transcription(telegram, msg)
    if text is not None:
        return "[video note]: " + text + "\n"
    return ""
