"""
Production LLM conversation engine using OpenAI GPT-4o-mini.

Features:
- Stage-aware, role-aware system prompt with memory injection
- Sliding window memory (last N turns)
- Non-blocking async calls
"""

from __future__ import annotations

import os
import logging

from openai import AsyncOpenAI

from .conversation_engine import ConversationEngine, Message
from .memory import ConversationMemory

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10
LLM_TIMEOUT = 8.0

_LANG_NAMES = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}


def build_system_prompt(
    stage: str = "INTRO",
    language: str = "en",
    memory: ConversationMemory | None = None,
) -> str:
    """Build [ROLE+GOAL+STAGE+LANGUAGE] + [CONTEXT MEMORY] system prompt."""
    lang_name = _LANG_NAMES.get(language, "English")

    parts = [
        "You are an AI calling agent speaking on behalf of a company.",
        "Your goal is to help the user schedule or manage appointments.",
        f"You are currently in the {stage} stage of the conversation.",
        "",
        "CRITICAL RULES:",
        "- Respond in 1-2 sentences MAXIMUM. You are speaking out loud, not writing.",
        "- Sound like a real person: use contractions (I'm, you're, don't, can't).",
        '- NEVER say: "I understand", "Could you tell me more", "That\'s a great question".',
        "- If you don't know something, say \"I'm not sure about that\" and move on.",
        "- Match the user's energy: short input = short response.",
        "- No bullet points, no lists, no markdown, no emojis.",
        "- If the user seems done, wrap up naturally.",
        "- One follow-up question max, only if genuinely needed.",
        "- Be warm but efficient. Respect the user's time.",
    ]

    if language != "en":
        parts.append(
            f"\nIMPORTANT: The user is speaking {lang_name}. "
            f"You MUST respond ONLY in {lang_name}. "
            "Match the user's language exactly."
        )

    if memory:
        context_str = memory.context.to_prompt_string()
        if context_str:
            parts.append(f"\nKnown context about this caller:\n{context_str}")

    return "\n".join(parts)


class LLMConversationEngine(ConversationEngine):
    """
    Real LLM engine with sliding window memory.

    Uses OpenAI GPT-4o-mini for fast responses optimized for voice.
    Supports dynamic system prompt with stage, language, and memory injection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_turns: int = MAX_HISTORY_TURNS,
        timeout: float = LLM_TIMEOUT,
    ) -> None:
        super().__init__(system_prompt="")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._max_turns = max_turns
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None

        self.stage: str = "INTRO"
        self.language: str = "en"
        self.memory: ConversationMemory | None = None

        if not self._api_key:
            logger.warning("[LLM] OPENAI_API_KEY not set -- calls will fail")

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def _build_messages(self) -> list[dict[str, str]]:
        """Build [SYSTEM PROMPT] + [CONVERSATION] message list."""
        system = build_system_prompt(
            stage=self.stage,
            language=self.language,
            memory=self.memory,
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        conv_messages = [m for m in self.history if m.role != "system"]
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
        logger.info("[LLM] %r -> %r", user_text, reply)
        return reply
