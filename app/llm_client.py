import asyncio
import logging
import time
from typing import List, Dict, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6  # Keep last 3 user + 3 bot turns for token efficiency and TPM headroom


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self._client: Optional[httpx.AsyncClient] = None
        
        # Metrics
        self._call_count: int = 0
        self._error_count: int = 0
        self._latencies: list[float] = []
        self._last_provider_used: str = ""

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _truncate_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep only the last MAX_HISTORY_TURNS to prevent token overflow."""
        if len(history) > MAX_HISTORY_TURNS:
            return history[-MAX_HISTORY_TURNS:]
        return history

    async def _retry_call(self, func, max_attempts: int = 3, base_delay: float = 0.7) -> str:
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 and attempt < max_attempts:
                    # Token bucket rate limit: wait 700ms for Groq's 600ms reset window to clear
                    logger.warning(f"LLM rate limited (attempt {attempt}/{max_attempts}), retrying in 700ms...")
                    await asyncio.sleep(0.7)
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(f"LLM call failed (attempt {attempt}/{max_attempts}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
        raise last_error

    async def get_response(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate conversational reply given conversation history.
        conversation_history: List of {"role": "user"|"assistant"|"system", "content": "..."}
        Get LLM response with automatic provider failover.
        """
        history = self._truncate_history(conversation_history)
        self._call_count += 1
        start = time.time()

        # Build ordered provider list: primary first, then fallbacks
        providers = self._get_provider_order()
        
        last_error = None
        for provider_name, call_func in providers:
            try:
                # Use an async wrapper to call the provider function
                async def wrapper():
                    return await call_func(history)
                    
                result = await self._retry_call(wrapper)
                self._last_provider_used = provider_name
                self._latencies.append(time.time() - start)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"LLM provider {provider_name} failed: {e}. Trying fallback...")
                continue

        self._error_count += 1
        logger.error(f"All LLM providers failed. Last error: {last_error}")

        # Check last user utterance for closing/thank you intent
        last_user_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break

        closing_keywords = ["थँक्यू", "thank", "ओके", "ok", "बाय", "bye", "चालेल", "chalel"]
        if any(w in last_user_msg for w in closing_keywords):
            return "वेलकम सर, धन्यवाद आणि आपला दिवस चांगला जावो!"

        return "माफ करा सर, आवाज अडकला होता. आपण काय म्हणालात?"

    def _get_provider_order(self):
        """Return ordered list of (name, callable) based on configured primary + available fallbacks."""
        PLACEHOLDER_PREFIXES = ("your_", "sk-placeholder")

        def is_valid_key(k: str) -> bool:
            if not k or len(k) < 8:
                return False
            kl = k.lower()
            return not any(kl.startswith(p) for p in PLACEHOLDER_PREFIXES)

        all_providers = [
            ("groq", self._call_groq, settings.GROQ_API_KEY),
            ("gemini", self._call_gemini, settings.GEMINI_API_KEY),
            ("together", self._call_together, settings.TOGETHER_API_KEY),
            ("openai", self._call_openai, settings.OPENAI_API_KEY),
        ]
        # Primary provider first
        primary = [(n, f) for n, f, k in all_providers if n == self.provider and is_valid_key(k)]
        # Then others with valid keys configured (excluding primary)
        fallbacks = [(n, f) for n, f, k in all_providers if n != self.provider and is_valid_key(k)]
        return primary + fallbacks

    async def _call_gemini(self, history: List[Dict[str, str]]) -> str:
        """Call Google Gemini API."""
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        model = settings.GEMINI_MODEL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            if not contents and role == "model":
                # Gemini requires conversations to start with a user turn
                contents.append({
                    "role": "user",
                    "parts": [{"text": "Hello"}]
                })
            elif contents and contents[-1]["role"] == role:
                # Merge consecutive turns with the same role
                contents[-1]["parts"][0]["text"] += f"\n{msg['content']}"
                continue

            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "system_instruction": {
                "parts": [{"text": settings.SYSTEM_PROMPT}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 400,
                "thinkingConfig": {
                    "thinkingBudget": 0  # Disable thinking tokens to eliminate latency and preserve full token budget
                }
            }
        }

        models_to_try = [settings.GEMINI_MODEL]
        if "gemini-2.5-flash" not in models_to_try:
            models_to_try.append("gemini-2.5-flash")

        client = await self._get_client()
        last_exc = None
        for m in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                logger.info(f"Gemini ({m}) LLM reply: '{reply}'")
                return reply
            except httpx.HTTPStatusError as e:
                last_exc = e
                if e.response.status_code in (503, 404, 429) and m != models_to_try[-1]:
                    logger.warning(f"Gemini model '{m}' returned {e.response.status_code} (high demand/unavailable). Trying fallback model...")
                    continue
                raise
            except Exception as e:
                last_exc = e
                if m != models_to_try[-1]:
                    logger.warning(f"Gemini model '{m}' failed: {e}. Trying fallback model...")
                    continue
                raise
        if last_exc:
            raise last_exc

    async def _call_together(self, history: List[Dict[str, str]]) -> str:
        """Call Together AI."""
        api_key = settings.TOGETHER_API_KEY
        if not api_key:
            raise ValueError("TOGETHER_API_KEY is not set.")

        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}] + history

        payload = {
            "model": settings.TOGETHER_MODEL,
            "messages": messages,
            "max_tokens": 80,
            "temperature": 0.7
        }

        client = await self._get_client()
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        logger.info(f"Together AI reply: '{reply}'")
        return reply

    async def _call_groq(self, history: List[Dict[str, str]]) -> str:
        """Call Groq fast LLM API."""
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }

        messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}] + history

        payload = {
            "model": settings.GROQ_MODEL,
            "messages": messages,
            "max_tokens": 160,
            "temperature": 0.7,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5
        }

        client = await self._get_client()
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        logger.info(f"Groq reply: '{reply}'")
        return reply

    async def _call_openai(self, history: List[Dict[str, str]]) -> str:
        """Call OpenAI API."""
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}] + history

        payload = {
            "model": settings.OPENAI_MODEL,
            "messages": messages,
            "max_tokens": 80,
            "temperature": 0.7
        }

        client = await self._get_client()
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        logger.info(f"OpenAI reply: '{reply}'")
        return reply

    def get_metrics(self) -> dict:
        recent = self._latencies[-100:]
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "avg_latency_ms": round(sum(recent) / max(len(recent), 1) * 1000, 1) if recent else 0.0,
            "last_provider": self._last_provider_used,
        }


llm_client = LLMClient()
