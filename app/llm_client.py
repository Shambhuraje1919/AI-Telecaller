import asyncio
import logging
import time
from typing import List, Dict, Optional
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20  # Prevent token overflow


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

    async def _retry_call(self, func, max_attempts: int = 3, base_delay: float = 0.5) -> str:
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await func()
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
                logger.warning(f"Provider '{provider_name}' failed: {e}")
                last_error = e
                continue

        self._error_count += 1
        logger.error(f"All LLM providers failed. Last error: {last_error}")
        return "माफ करा सर, एक मिनिट थांबा."  # Fallback Marathi: "Sorry sir, wait a minute"

    def _get_provider_order(self):
        """Return ordered list of (name, callable) based on configured primary + available fallbacks."""
        all_providers = [
            ("gemini", self._call_gemini, settings.GEMINI_API_KEY),
            ("together", self._call_together, settings.TOGETHER_API_KEY),
            ("groq", self._call_groq, settings.GROQ_API_KEY),
            ("openai", self._call_openai, settings.OPENAI_API_KEY),
        ]
        # Primary provider first
        primary = [(n, f) for n, f, k in all_providers if n == self.provider and k]
        # Then others with keys configured
        fallbacks = [(n, f) for n, f, k in all_providers if n != self.provider and k]
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
                "maxOutputTokens": 50,  # Fast conversational turns
            }
        }

        client = await self._get_client()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.info(f"Gemini LLM reply: '{reply}'")
        return reply

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
            "Content-Type": "application/json"
        }

        messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}] + history

        payload = {
            "model": settings.GROQ_MODEL,
            "messages": messages,
            "max_tokens": 80,
            "temperature": 0.7
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
