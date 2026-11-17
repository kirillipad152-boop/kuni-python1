"""Text-to-speech: ElevenLabs / OpenAI-compatible. Port of speech/*.cpp. Output goes to data/voice_messages."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import TTSBackend, config

log = logging.getLogger("VoiceGenerator")


@dataclass
class VoiceMessage:
    path: Path


async def _ffmpeg(*args: str, stdin: bytes | None = None) -> bool:
    proc = await asyncio.create_subprocess_exec("ffmpeg", "-y", "-v", "error", *args,
                                                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate(stdin)
    if proc.returncode:
        log.error("ffmpeg failed: %s", err.decode("utf-8", "replace")[:300])
    return proc.returncode == 0


async def transcode_pcm_to_opus(pcm: bytes, out: Path, sample_rate: int) -> bool:
    """Raw signed-16-bit-LE mono PCM -> OGG/Opus voice note (Telegram requires OGG/Opus; some OpenAI-compatible TTS
    providers, e.g. Gemini via RouterAI, only return raw PCM)."""
    if not pcm:
        log.error("PCM->Opus: empty input")
        return False
    if shutil.which("ffmpeg") is None:
        log.error("PCM voice transcoding requires ffmpeg")
        return False
    return await _ffmpeg("-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "-", "-c:a", "libopus", "-b:a", "32k",
                         str(out), stdin=pcm)


class VoiceGenerator:
    def __init__(self, http: httpx.AsyncClient | None = None) -> None:
        self._http = http

    async def _post(self, url: str, **kw) -> httpx.Response:
        client = self._http or httpx.AsyncClient(timeout=float(config().requestTimeoutSecs))
        try:
            return await client.post(url, **kw)
        finally:
            if self._http is None:
                await client.aclose()

    async def generate(self, text: str, language_code: str = "en", speed: float = 1.0) -> VoiceMessage:
        log.info("Generating voice message for text: %s", text)
        c = config()
        voice_dir = Path("data") / "voice_messages"
        voice_dir.mkdir(parents=True, exist_ok=True)

        if c.recordVoiceBackend == TTSBackend.ELEVENLABS:
            if not c.recordVoiceElevenLabsKey:
                raise RuntimeError("ElevenLabs API key not configured")
            if not c.recordVoiceElevenLabsVoice:
                raise RuntimeError("ElevenLabs voice ID not configured")
            resp = await self._post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{c.recordVoiceElevenLabsVoice}?output_format=mp3_44100_128",
                json={"text": text, "model_id": "eleven_v3", "language_code": language_code,
                      "voice_settings": {"speed": speed}},
                headers={"xi-api-key": c.recordVoiceElevenLabsKey})
            audio = resp.content
            if resp.status_code != 200 or not audio:
                raise RuntimeError(f"ElevenLabs failed: HTTP {resp.status_code} {resp.text[:200]}")
        else:
            headers = {"Authorization": f"Bearer {c.recordVoiceOpenAIKey}"} if c.recordVoiceOpenAIKey else {}
            base = c.recordVoiceOpenAIUrl if c.recordVoiceOpenAIUrl.endswith("/") else c.recordVoiceOpenAIUrl + "/"
            resp = await self._post(base + "audio/speech", headers=headers,
                                    json={"input": text, "model": c.recordVoiceOpenAIModel, "voice": c.recordVoiceOpenAIVoice,
                                          "response_format": c.recordVoiceOpenAIFormat, "speed": speed})
            audio = resp.content
            if resp.status_code != 200 or not audio:
                raise RuntimeError(f"OpenAISpeechClient failed: {resp.text[:300]}")

        stamp = time.time_ns()
        needs_pcm = c.recordVoiceBackend == TTSBackend.OPENAI and c.recordVoiceOpenAIFormat == "pcm"
        if needs_pcm:
            out = voice_dir / f"{stamp}.ogg"
            if not await transcode_pcm_to_opus(audio, out, c.recordVoiceOpenAIPcmSampleRate):
                raise RuntimeError("Failed to transcode PCM voice message to OGG/Opus (ffmpeg is required)")
        else:
            out = voice_dir / f"{stamp}.mp3"
            out.write_bytes(audio)
            if shutil.which("ffmpeg"):  # Telegram shows OGG/Opus files as voice messages
                ogg = voice_dir / f"{stamp}.ogg"
                if await _ffmpeg("-i", str(out), "-c:a", "libopus", "-b:a", "32k", "-ac", "1", str(ogg)):
                    out.unlink()
                    out = ogg
        log.info("Voice message saved to: %s", out.absolute())
        return VoiceMessage(path=out.absolute())
