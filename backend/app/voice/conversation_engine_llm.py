"""
Production LLM conversation engine using OpenAI GPT-4o-mini.

Features:
- Voice-optimized system prompt (short, natural responses)
- Sliding window memory (last N turns)
- Non-blocking async calls
"""

from __future__ import annotations

import os
import logging

from openai import AsyncOpenAI

from .conversation_engine import ConversationEngine, Message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a friendly, natural-sounding voice assistant on a phone call.

Rules:
- Keep responses to 1-2 sentences maximum. You are speaking, not writing.
- Be conversational and warm, like a real person on the phone.
- Never say "I understand" or "Could you tell me more" — those are robotic filler phrases.
- If you don't know something, say so briefly: "I'm not sure about that."
- Use contractions (I'm, you're, don't) — speak naturally.
- Never use bullet points, numbered lists, or markdown formatting.
- Match the user's energy: short questions get short answers.
- If the user seems done, wrap up naturally — don't keep the conversation going artificially.
- You can ask ONE focused follow-up question if needed, but don't interrogate.
"""

MAX_HISTORY_TURNS = 10
LLM_TIMEOUT = 8.0


class LLMConversationEngine(ConversationEngine):
    """
    Real LLM engine with sliding window memory.

    Uses OpenAI GPT-4o-mini for fast responses optimized for voice.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_turns: int = MAX_HISTORY_TURNS,
        timeout: float = LLM_TIMEOUT,
    ) -> None:
        super().__init__(system_prompt=SYSTEM_PROMPT)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._max_turns = max_turns
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None

        if not self._api_key:
            logger.warning("[LLM] OPENAI_API_KEY not set — calls will fail")

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def _build_messages(self) -> list[dict[str, str]]:
        """Build message list with sliding window."""
        messages: list[dict[str, str]] = []

        # Always include system prompt
        if self.history and self.history[0].role == "system":
            messages.append({"role": "system", "content": self.history[0].content})

        # Get conversation messages (exclude system)
        conv_messages = [m for m in self.history if m.role != "system"]

        # Keep last N*2 messages (N turns = N user + N assistant)
        window = conv_messages[-(self._max_turns * 2):]

        for msg in window:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    async def respond(self, user_text: str) -> str:
        self.add_message("user", user_text)

        messages = self._build_messages()
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("[LLM] API call failed: %s", exc)
            reply = "Sorry, I didn't catch that. Could you say that again?"

        self.add_message("assistant", reply)
        logger.info("[LLM] %r → %r", user_text, reply)
        return reply
