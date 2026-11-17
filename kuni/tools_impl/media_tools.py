"""take_photo / record_audio. Ports of tools/take_photo.cpp and tools/record_audio.cpp."""

from __future__ import annotations

from pathlib import Path

from ..config import config
from ..imagegen import ImageGenerator, StableDiffusionClient
from ..llm import OpenAIChat, Params
from ..llmui import media
from ..prompts import prompts
from ..speech import VoiceGenerator
from ..tools import Ctx, Property, Tool, ToolError

TAKE_PHOTO_DESC = (
    "Describes the image Kuni would like to achieve. Refer to yourself as Kuni. take_photo only knows about Kuni.\n"
    "To draw other character, specify their name, and describe their appearance as specifically as possible.\n\n"
    "IMPORTANT: this tool is backed by Stable Diffusion, NOT a general-purpose image model. Write like a Stable "
    "Diffusion prompt, not a literal scene script: a comma-separated list of short keywords/phrases, not full sentences "
    "with grammar and causality. Stable Diffusion sucks at: rendering readable text/writing/signs/posters, precise "
    "spatial composition (e.g. \"one hand here, another hand there\"), more than 1 character in one frame, and any "
    "coherent scene logic/storytelling. Keep it short - only include what actually matters, more keywords means less "
    "control, not more detail.\n\n"
    "Structure it as a checklist, using only the categories that matter for this shot (skip the rest):\n"
    "- subject & pose: who, what they're doing, expression (e.g. \"kuni, playful smirk, leaning forward\")\n"
    "- medium: selfie / photo / illustration / etc (almost always selfie unless stated otherwise)\n"
    "- setting: where, one or two words (e.g. \"bedroom at night\", \"rooftop, city lights\")\n"
    "- lighting: (e.g. \"moonlight\", \"soft window light\", \"neon glow\")\n"
    "- mood/color: (e.g. \"warm tones\", \"cold blue palette\")\n\n"
    "Example (other character): \"Selfie of Kuni - Kuni's sister: anime young female, gold eyes, white hair, white "
    "dress, black socks.\"\n\n"
    "GOOD (short, keyword-based, vibe over plot): \"kuni, selfie, playful smirk, leaning forward, moonlight, bedroom at "
    "night, soft shadows\"\n"
    "BAD (too complex/literal, will come out as a mangled mess): \"Kuni selfie holding a Soviet-style propaganda poster. "
    "Poster shows Evangelion angel with hammer and sickle. Text: \\\"РОДИНА-МАТЬ ЗОВЁТ ЕВУ!\\\" Kuni smirks.\"\n")


def media_directory(ctx: Ctx, kind: str) -> Path:
    directory = Path("data") / kind
    scope = getattr(ctx.temporary_context, "chat_id", None)
    return directory if scope is None else directory / str(scope)


def _store_for_chat(ctx: Ctx, path: Path, kind: str) -> Path:
    if getattr(ctx.temporary_context, "chat_id", None) is None:
        return path
    directory = media_directory(ctx, kind)
    directory.mkdir(parents=True, exist_ok=True)
    return path.replace(directory / path.name)


def take_photo(sd: StableDiffusionClient, llm: OpenAIChat) -> Tool:
    async def handler(ctx: Ctx) -> str:
        desc = ctx.args.get("photo_desc")
        if not isinstance(desc, str):
            raise ValueError("photo_desc is required")
        gen = ImageGenerator(sd, llm, Params(config=config().llmImageToText))
        gallery = await gen.generate(desc)
        gallery.path = _store_for_chat(ctx, gallery.path, "gallery")
        description = await media.image([], llm, gallery.path)
        return (f"{description}\n\nFilename: {gallery.path.name}\n"
                "When writing diary, do not forget to mention this photo and its filename verbatim - you might need this "
                "in the future!\n\nYou have created photo successfully. Review it carefully. Send it only if you are fully "
                "satisfied; use take_photo again to make another photo")

    return Tool("take_photo",
                "Takes a photo by Kuni. This tool is useful for creating selfies, photos of surroundings, or any other "
                "images. The result of this tool is a photo description and a filename. The filename can then be sent to "
                 "this chat using #send_telegram_message.", handler,
                {"photo_desc": Property("string", TAKE_PHOTO_DESC)}, ["photo_desc"])


def record_audio() -> Tool:
    async def handler(ctx: Ctx) -> str:
        speech = ctx.args.get("speech")
        if not isinstance(speech, str):
            raise ToolError("speech is required")
        if not speech.strip():
            raise ToolError("speech must not be empty")
        # really dirty fix: hit Kuni with an exception if it tries to say an introduction in a voice note
        if any(w in speech for w in ("voice", "tone", "Kuni", "голосом", "тоном")):
            raise ToolError('Skip introductions in voice message. Instead, send the message content directly. For example, '
                            'if you want to say "Kuni says hello in a playful tone" in a voice message, just send "hello".')
        voice = await VoiceGenerator().generate(speech, "ru", 1.2)
        voice.path = _store_for_chat(ctx, voice.path, "voice_messages")
        return f"Filename: {voice.path.name}"

    return Tool("record_audio",
                "Records a new voice message and stores it in Kuni's voice gallery. This is useful for expressing emotions "
                 "in a more direct way. The result of this tool is a filename. The filename can then be sent to this chat "
                "using #send_telegram_message.", handler,
                {"speech": Property("string", prompts().record_audio_speech)}, ["speech"])
