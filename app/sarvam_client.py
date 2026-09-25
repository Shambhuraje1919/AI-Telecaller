import base64
import logging
import re
from typing import Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


def clean_spoken_text(text: str) -> str:
    """Sanitize text so Sarvam TTS pronounces Marathi cleanly without glitching on markdown or symbols."""
    if not text:
        return ""
    # Remove markdown symbols and brackets
    t = re.sub(r'[*#_~`>\[\]\(\)]', ' ', text)
    # Remove quotes
    t = t.replace('"', '').replace("'", "")
    # Replace hyphens between numbers with 'ते' (e.g. 1600-3500 -> 1600 ते 3500)
    t = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 ते \2', t)
    # Collapse multiple whitespaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t


class SarvamClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.headers = {
            "api-subscription-key": self.api_key
        }

    async def speech_to_text(
        self,
        wav_audio_bytes: bytes,
        language_code: Optional[str] = None
    ) -> str:
        """
        Transcribe audio using Sarvam AI Speech-to-Text API (Saarika).
        Accepts WAV audio bytes and returns transcribed text.
        """
        if not self.api_key:
            logger.error("SARVAM_API_KEY is not configured.")
            return ""

        url = settings.SARVAM_STT_URL
        lang = language_code or settings.SARVAM_STT_LANGUAGE_CODE

        files = {
            "file": ("audio.wav", wav_audio_bytes, "audio/wav")
        }
        data = {
            "model": settings.SARVAM_STT_MODEL,
        }
        if lang:
            data["language_code"] = lang

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(
                    url,
                    headers=self.headers,
                    files=files,
                    data=data
                )
                response.raise_for_status()
                res_data = response.json()
                transcript = res_data.get("transcript", "").strip()
                logger.info(f"Sarvam STT Output: '{transcript}'")
                return transcript
            except httpx.HTTPStatusError as e:
                logger.error(f"Sarvam STT HTTP error: {e.response.status_code} - {e.response.text}")
                return ""
            except Exception as e:
                logger.error(f"Sarvam STT failed: {e}")
                return ""

    async def text_to_speech(
        self,
        text: str,
        speaker: Optional[str] = None,
        language_code: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech using Sarvam AI Text-to-Speech API (Bulbul).
        Returns raw audio bytes (WAV container).
        """
        if not self.api_key:
            logger.error("SARVAM_API_KEY is not configured.")
            return b""

        cleaned_text = clean_spoken_text(text)
        if not cleaned_text:
            return b""

        url = settings.SARVAM_TTS_URL
        speaker_name = speaker or settings.SARVAM_TTS_SPEAKER
        if speaker_name and speaker_name.lower() in ("rutu", "ruto"):
            speaker_name = "ritu"
        lang = language_code or settings.SARVAM_TTS_LANGUAGE_CODE

        payload = {
            "inputs": [cleaned_text],
            "target_language_code": lang,
            "speaker": speaker_name,
            "pace": 0.98,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": settings.SARVAM_TTS_MODEL,
        }
        # bulbul:v3 does not accept pitch and loudness
        if not settings.SARVAM_TTS_MODEL.startswith("bulbul:v3"):
            payload["pitch"] = 0.0
            payload["loudness"] = 1.2

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                headers = {
                    **self.headers,
                    "Content-Type": "application/json"
                }
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                res_data = response.json()
                audios = res_data.get("audios", [])
                if not audios:
                    logger.error("Sarvam TTS returned empty audio list.")
                    return b""

                audio_bytes = base64.b64decode(audios[0])
                logger.info(f"Sarvam TTS generated {len(audio_bytes)} audio bytes for: '{text[:40]}...'")
                return audio_bytes
            except httpx.HTTPStatusError as e:
                logger.error(f"Sarvam TTS HTTP error: {e.response.status_code} - {e.response.text}")
                return b""
            except Exception as e:
                logger.error(f"Sarvam TTS failed: {e}")
                return b""


sarvam_client = SarvamClient()
