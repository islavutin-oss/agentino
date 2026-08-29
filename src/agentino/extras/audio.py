"""Audio transcription — OpenAI-compatible STT for voice messages.

Calls /v1/audio/transcriptions (Whisper API format).
Works with: Groq (free), OpenAI, Router, local whisper.cpp.

Usage:
    transcriber = AudioTranscriber(api_key="...", base_url="https://api.groq.com/openai/v1")
    text = await transcriber.transcribe(audio_bytes, mime="audio/ogg")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default: Groq free tier (whisper-large-v3-turbo)
DEFAULT_STT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"

# Whisper returns full language names — normalize to ISO 639-1
_LANG_TO_ISO = {
    "english": "en",
    "russian": "ru",
    "greek": "el",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "turkish": "tr",
    "dutch": "nl",
    "polish": "pl",
    "ukrainian": "uk",
    "hebrew": "he",
    "hindi": "hi",
    "thai": "th",
    "vietnamese": "vi",
    "tagalog": "tl",
    "indonesian": "id",
    "malay": "ms",
    "swedish": "sv",
    "danish": "da",
    "norwegian": "no",
    "finnish": "fi",
    "czech": "cs",
    "romanian": "ro",
    "hungarian": "hu",
    "bulgarian": "bg",
    "croatian": "hr",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "catalan": "ca",
}


def _normalize_language(lang: str) -> str:
    """Normalize Whisper language to ISO 639-1 code."""
    if not lang:
        return ""
    lower = lang.lower().strip()
    # Already an ISO code
    if len(lower) <= 3:
        return lower
    return _LANG_TO_ISO.get(lower, lower)


# MIME → file extension mapping for STT providers
_MIME_TO_EXT = {
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/ogg; codecs=opus": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/flac": "flac",
}


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""

    text: str
    model: str
    language: str = ""  # ISO 639-1 detected language (e.g. "en", "ru", "el")
    duration_ms: int = 0


class AudioTranscriber:
    """OpenAI-compatible audio transcription client.

    Config priority: constructor args > env vars > defaults (Groq free tier).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("AGENTINO_STT_BASE_URL") or DEFAULT_STT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key or os.environ.get("AGENTINO_STT_API_KEY") or os.environ.get("GROQ_API_KEY", "")
        )
        self.model = model or os.environ.get("AGENTINO_STT_MODEL") or DEFAULT_STT_MODEL
        self.language = language

    async def transcribe(
        self,
        audio: bytes,
        mime: str = "audio/ogg",
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text.

        Args:
            audio: Raw audio bytes (ogg/opus, mp3, wav, etc.)
            mime: MIME type of the audio
            language: Optional language hint (ISO 639-1, e.g. "en", "ru")

        Returns:
            TranscriptionResult with transcript text.

        Raises:
            RuntimeError if transcription fails.
        """
        import time

        import httpx

        ext = _MIME_TO_EXT.get(mime, "ogg")
        filename = f"voice.{ext}"

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Build multipart form data (OpenAI Whisper format)
        files = {"file": (filename, audio, mime)}
        data: dict[str, str] = {
            "model": self.model,
            "response_format": "verbose_json",  # includes detected language
        }
        lang = language or self.language
        if lang:
            data["language"] = lang

        url = f"{self.base_url}/audio/transcriptions"

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, files=files, data=data)

        duration_ms = int((time.monotonic() - t0) * 1000)

        if resp.status_code != 200:
            logger.error("STT failed: %d %s", resp.status_code, resp.text[:200])
            raise RuntimeError(f"Transcription failed: {resp.status_code}")

        result = resp.json()
        text = result.get("text", "").strip()
        if not text:
            raise RuntimeError("Transcription returned empty text")

        detected_lang = _normalize_language(result.get("language", ""))
        logger.info(
            "STT: %d bytes → %d chars (%s) in %dms (%s)",
            len(audio),
            len(text),
            detected_lang or "?",
            duration_ms,
            self.model,
        )
        return TranscriptionResult(
            text=text, model=self.model, language=detected_lang, duration_ms=duration_ms
        )


def build_transcriber(config: dict | None = None) -> AudioTranscriber | None:
    """Build AudioTranscriber from agents.yml audio config section.

    Config:
        audio:
          base_url: https://api.groq.com/openai/v1
          api_key: ${GROQ_API_KEY}
          model: whisper-large-v3-turbo
          language: en

    Returns None if audio is explicitly disabled.
    """
    if config is None:
        # Default: try to build from env vars
        api_key = os.environ.get("AGENTINO_STT_API_KEY") or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return None
        return AudioTranscriber()

    if config.get("enabled") is False:
        return None

    return AudioTranscriber(
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        model=config.get("model"),
        language=config.get("language"),
    )
