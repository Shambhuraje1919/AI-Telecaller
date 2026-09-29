import asyncio
import logging
import re
import time
from typing import Optional, Dict, Any
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


def clean_cartesia_text(text: str) -> str:
    """Preprocess text before Cartesia TTS: pronounce units, URLs, and abbreviations cleanly."""
    if not text:
        return ""
    # Convert domain dots to ' डॉट ' (e.g. lok8.me -> lok8 डॉट me)
    t = re.sub(r'([a-zA-Z0-9]+)\.([a-zA-Z]+)', r'\1 डॉट \2', text)
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
    return t.strip()


class CircuitBreaker:
    """Simple circuit breaker to prevent hammering Cartesia if it experiences issues."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.state = "closed"

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Cartesia Circuit breaker OPENED after {self.failure_count} failures")

    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                logger.info("Cartesia Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        return True


class CartesiaClient:
    """
    Client for Cartesia Sonic Text-to-Speech API.
    Streams ultra-low latency, raw 8kHz mu-law audio directly tailored for Twilio telephony.
    """
    def __init__(self):
        self.api_key = settings.CARTESIA_API_KEY
        self.voice_id = settings.CARTESIA_VOICE_ID
        self.model_id = settings.CARTESIA_MODEL
        self.url = "https://api.cartesia.ai/tts/bytes"
        self._client: Optional[httpx.AsyncClient] = None

        self.circuit = CircuitBreaker()
        self._latencies: list[float] = []
        self._call_count: int = 0
        self._error_count: int = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _retry_request(self, func, max_attempts: int = 3, base_delay: float = 0.3) -> Any:
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(f"Cartesia call failed (attempt {attempt}/{max_attempts}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
        raise last_error

    async def text_to_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "mr"
    ) -> bytes:
        """
        Synthesize text into raw 8kHz mu-law audio bytes for Twilio.
        Directly returns pure mu-law bytes — no file containers, no resampling needed!
        """
        if not self.api_key:
            logger.error("CARTESIA_API_KEY is not configured.")
            return b""

        if not text or not text.strip():
            return b""

        if not self.circuit.can_proceed():
            logger.warning("Cartesia TTS Circuit breaker is OPEN. Skipping request.")
            return b""

        active_voice = voice_id or self.voice_id
        headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": "2026-08-14",
            "Content-Type": "application/json"
        }

        cleaned_text = clean_cartesia_text(text)
        if not cleaned_text:
            return b""

        payload = {
            "model_id": self.model_id,
            "transcript": cleaned_text,
            "voice": {
                "mode": "id",
                "id": active_voice
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_mulaw",
                "sample_rate": 8000
            },
            "language": language
        }

        self._call_count += 1
        start_time = time.time()

        async def _make_request():
            client = await self._get_client()
            response = await client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

        try:
            mulaw_audio = await self._retry_request(_make_request)
            self.circuit.record_success()

            latency = time.time() - start_time
            self._latencies.append(latency)
            if len(self._latencies) > 100:
                self._latencies.pop(0)

            logger.info(
                f"Cartesia TTS generated {len(mulaw_audio)} raw mulaw bytes in {latency*1000:.0f}ms for: '{text[:40]}...'"
            )
            return mulaw_audio

        except Exception as e:
            self.circuit.record_failure()
            self._error_count += 1
            if isinstance(e, httpx.HTTPStatusError):
                logger.error(f"Cartesia TTS HTTP error: {e.response.status_code} - {e.response.text}")
            else:
                logger.error(f"Cartesia TTS failed: {e}")
            return b""

    def get_metrics(self) -> Dict[str, Any]:
        recent = self._latencies[-100:]
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "avg_latency_ms": round(sum(recent) / max(len(recent), 1) * 1000, 1),
            "circuit": self.circuit.state,
        }


cartesia_client = CartesiaClient()
