import asyncio
import base64
import logging
import re
import time
from typing import Optional, Dict, Any
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


def calculate_natural_pace(text: str) -> float:
    """
    Vary speech pace dynamically to avoid the robotic monotonous fixed-speed effect:
    - Questions / inquiries ending with '?' or 'का': 0.95 (thoughtful, respectful, slightly slower)
    - Short confirmations ('हो', 'नक्कीच', 'चालेल'): 0.96
    - Informative statements: 0.98 (confident, crisp)
    """
    stripped = text.strip()
    if stripped.endswith("?") or stripped.endswith("का") or stripped.endswith("का?"):
        return 0.95
    if len(stripped.split()) <= 4:
        return 0.96
    return 0.98


def clean_spoken_text(text: str) -> str:
    """Sanitize and format text so Sarvam TTS pronounces Marathi cleanly with human micro-pauses."""
    if not text:
        return ""
    # Remove markdown symbols and brackets
    t = re.sub(r'[*#_~`>\[\]\(\)]', ' ', text)
    # Remove quotes
    t = t.replace('"', '').replace("'", "")
    # Replace hyphens between numbers with 'ते' (e.g. 1600-3500 -> 1600 ते 3500)
    t = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 ते \2', t)

    # Inject breathing commas after common bridge words if comma is missing
    bridge_words = [
        (r'^(नमस्ते सर|नमस्ते मॅडम|नमस्ते)\s*(?!,)', r'\1, '),
        (r'^(नक्कीच सर|नक्कीच)\s*(?!,)', r'\1, '),
        (r'^(चालेल सर|चालेल)\s*(?!,)', r'\1, '),
        (r'^(हो सर|हो नक्कीच)\s*(?!,)', r'\1, '),
        (r'^(काही हरकत नाही सर|काही हरकत नाही)\s*(?!,)', r'\1, '),
    ]
    # Convert domain dots to ' डॉट ' (e.g. lok8.me -> lok8 डॉट me)
    t = re.sub(r'([a-zA-Z0-9]+)\.([a-zA-Z]+)', r'\1 डॉट \2', t)
    # Replace literal 'बिंदू' if spoken for dots in web addresses
    t = re.sub(r'\bबिंदू\b', 'डॉट', t)
    # Expand 'किमी' / 'km' to 'किलोमीटर' so TTS never says 'kimi'
    t = re.sub(r'कि\.?मी\.?', 'किलोमीटर', t)
    t = re.sub(r'(\d+)\s*(km|kms)\b', r'\1 किलोमीटर', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(km|kms)\b', 'किलोमीटर', t, flags=re.IGNORECASE)
    # Expand 'sq ft' / 'sqft' / 'sq.ft' to 'स्क्वेअर फूट'
    t = re.sub(r'(\d+)\s*(sq\.?\s*ft|sqft)\b', r'\1 स्क्वेअर फूट', t, flags=re.IGNORECASE)
    # Expand 'ft' to 'फूट'
    t = re.sub(r'(\d+)\s*(ft|feet)\b', r'\1 फूट', t, flags=re.IGNORECASE)

    # Collapse multiple whitespaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t


class CircuitBreaker:
    """Simple circuit breaker to prevent hammering a failing API."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.state = "closed"  # closed=healthy, open=failing, half_open=testing

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        # half_open: allow one request through
        return True


class SarvamClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.headers = {"api-subscription-key": self.api_key}
        self._client: Optional[httpx.AsyncClient] = None
        
        # Circuit breakers
        self.stt_circuit = CircuitBreaker()
        self.tts_circuit = CircuitBreaker()
        
        # Metrics
        self._stt_latencies: list[float] = []
        self._tts_latencies: list[float] = []
        self._stt_call_count: int = 0
        self._tts_call_count: int = 0
        self._stt_error_count: int = 0
        self._tts_error_count: int = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _retry_request(self, func, max_attempts: int = 3, base_delay: float = 0.5) -> Any:
        """Execute an async function with exponential backoff retry."""
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(f"API call failed (attempt {attempt}/{max_attempts}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
        raise last_error

    async def speech_to_text(self, wav_audio_bytes: bytes, language_code: Optional[str] = None) -> str:
        """
        Transcribe audio using Sarvam AI Speech-to-Text API (Saarika).
        Accepts WAV audio bytes and returns transcribed text.
        """
        if not self.api_key:
            logger.error("SARVAM_API_KEY is not configured.")
            return ""

        if not self.stt_circuit.can_proceed():
            logger.warning("STT Circuit breaker is OPEN. Skipping request.")
            return ""

        url = settings.SARVAM_STT_URL
        lang = language_code or settings.SARVAM_STT_LANGUAGE_CODE
        data = {"model": settings.SARVAM_STT_MODEL}
        if lang:
            data["language_code"] = lang

        self._stt_call_count += 1
        start_time = time.time()

        async def _make_request():
            client = await self._get_client()
            files = {"file": ("audio.wav", wav_audio_bytes, "audio/wav")}
            response = await client.post(url, headers=self.headers, files=files, data=data)
            response.raise_for_status()
            return response.json()

        try:
            res_data = await self._retry_request(_make_request)
            self.stt_circuit.record_success()
            transcript = res_data.get("transcript", "").strip()
            
            latency = time.time() - start_time
            self._stt_latencies.append(latency)
            if len(self._stt_latencies) > 100:
                self._stt_latencies.pop(0)
                
            logger.info(f"Sarvam STT Output: '{transcript}'")
            return transcript
        except Exception as e:
            self.stt_circuit.record_failure()
            self._stt_error_count += 1
            if isinstance(e, httpx.HTTPStatusError):
                logger.error(f"Sarvam STT HTTP error: {e.response.status_code} - {e.response.text}")
            else:
                logger.error(f"Sarvam STT failed: {e}")
            return ""

    async def text_to_speech(self, text: str, speaker: Optional[str] = None, language_code: Optional[str] = None) -> bytes:
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

        if not self.tts_circuit.can_proceed():
            logger.warning("TTS Circuit breaker is OPEN. Skipping request.")
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
            "pace": calculate_natural_pace(cleaned_text),
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": settings.SARVAM_TTS_MODEL,
        }
        
        # bulbul:v3 does not accept pitch and loudness
        if not settings.SARVAM_TTS_MODEL.startswith("bulbul:v3"):
            payload["pitch"] = 0.0
            payload["loudness"] = 1.2

        self._tts_call_count += 1
        start_time = time.time()

        async def _make_request():
            client = await self._get_client()
            headers = {**self.headers, "Content-Type": "application/json"}
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

        try:
            res_data = await self._retry_request(_make_request)
            self.tts_circuit.record_success()
            
            audios = res_data.get("audios", [])
            if not audios:
                logger.error("Sarvam TTS returned empty audio list.")
                return b""

            audio_bytes = base64.b64decode(audios[0])
            
            latency = time.time() - start_time
            self._tts_latencies.append(latency)
            if len(self._tts_latencies) > 100:
                self._tts_latencies.pop(0)
                
            logger.info(f"Sarvam TTS generated {len(audio_bytes)} audio bytes for: '{text[:40]}...'")
            return audio_bytes
        except Exception as e:
            self.tts_circuit.record_failure()
            self._tts_error_count += 1
            if isinstance(e, httpx.HTTPStatusError):
                logger.error(f"Sarvam TTS HTTP error: {e.response.status_code} - {e.response.text}")
            else:
                logger.error(f"Sarvam TTS failed: {e}")
            return b""

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "stt": {
                "calls": self._stt_call_count,
                "errors": self._stt_error_count,
                "avg_latency_ms": round(sum(self._stt_latencies[-100:]) / max(len(self._stt_latencies[-100:]), 1) * 1000, 1),
                "circuit": self.stt_circuit.state,
            },
            "tts": {
                "calls": self._tts_call_count,
                "errors": self._tts_error_count,
                "avg_latency_ms": round(sum(self._tts_latencies[-100:]) / max(len(self._tts_latencies[-100:]), 1) * 1000, 1),
                "circuit": self.tts_circuit.state,
            }
        }
        
    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.stt_circuit.state == "closed" and self.tts_circuit.state == "closed" else "degraded",
            "metrics": self.get_metrics()
        }


sarvam_client = SarvamClient()
