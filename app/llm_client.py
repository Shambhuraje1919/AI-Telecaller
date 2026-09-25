import logging
from typing import List, Dict
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()

    async def get_response(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate conversational reply given conversation history.
        conversation_history: List of {"role": "user"|"assistant"|"system", "content": "..."}
        """
        if self.provider == "gemini":
            return await self._call_gemini(conversation_history)
        elif self.provider in ("together", "doghara"):
            # Together AI fast open-source LLaMA inference
            return await self._call_together(conversation_history)
        elif self.provider == "groq":
            return await self._call_groq(conversation_history)
        elif self.provider == "openai":
            return await self._call_openai(conversation_history)
        else:
            # Automatic fallback to configured keys
            if settings.GEMINI_API_KEY:
                return await self._call_gemini(conversation_history)
            elif settings.TOGETHER_API_KEY:
                return await self._call_together(conversation_history)
            elif settings.GROQ_API_KEY:
                return await self._call_groq(conversation_history)
            return "I apologize, my intelligence service is not connected."

    async def _call_gemini(self, history: List[Dict[str, str]]) -> str:
        """Call Google Gemini API."""
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.error("GEMINI_API_KEY is not set.")
            return "Thank you for sharing that with me."

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
                "temperature": 0.7,
                "maxOutputTokens": 80,  # Short telephonic replies
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                logger.info(f"Gemini LLM reply: '{reply}'")
                return reply
            except Exception as e:
                logger.error(f"Gemini call failed: {e}")
                return "I understand. Could you tell me more about that?"

    async def _call_together(self, history: List[Dict[str, str]]) -> str:
        """Call Together AI."""
        api_key = settings.TOGETHER_API_KEY
        if not api_key:
            logger.error("TOGETHER_API_KEY is not set.")
            return "Thank you, I understand."

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

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                reply = data["choices"][0]["message"]["content"].strip()
                logger.info(f"Together AI reply: '{reply}'")
                return reply
            except Exception as e:
                logger.error(f"Together AI call failed: {e}")
                return "I hear you. Let me check that for you."

    async def _call_groq(self, history: List[Dict[str, str]]) -> str:
        """Call Groq fast LLM API."""
        api_key = settings.GROQ_API_KEY
        if not api_key:
            logger.error("GROQ_API_KEY is not set.")
            return "Thank you."

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

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                reply = data["choices"][0]["message"]["content"].strip()
                logger.info(f"Groq reply: '{reply}'")
                return reply
            except Exception as e:
                logger.error(f"Groq call failed: {e}")
                return "Understood. How else can I assist?"

    async def _call_openai(self, history: List[Dict[str, str]]) -> str:
        """Call OpenAI API."""
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            return "Thank you."

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

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")
                return "Understood."


llm_client = LLMClient()
