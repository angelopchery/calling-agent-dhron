"""
Conversation memory system — short-term + context memory.

Short-term: sliding window of recent turns.
Context: extracted structured info (user name, intent, topic).
Both are injected into LLM prompts for context-aware responses.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_SHORT_TERM_TURNS = 5


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class ContextMemory:
    """Structured info extracted from conversation."""
    user_name: str | None = None
    intent: str | None = None
    topic: str | None = None
    sentiment: str = "neutral"  # positive, negative, neutral
    extra: dict = field(default_factory=dict)

    def to_prompt_string(self) -> str:
        """Format context for injection into LLM system prompt."""
        parts = []
        if self.user_name:
            parts.append(f"User's name: {self.user_name}")
        if self.intent:
            parts.append(f"Current intent: {self.intent}")
        if self.topic:
            parts.append(f"Topic: {self.topic}")
        if self.sentiment != "neutral":
            parts.append(f"User sentiment: {self.sentiment}")
        return "\n".join(parts)


class ConversationMemory:
    """
    Manages both short-term turn history and structured context.

    Short-term memory: last N turns (sliding window).
    Context memory: extracted structured data persisted across the call.
    """

    def __init__(self, max_turns: int = MAX_SHORT_TERM_TURNS) -> None:
        self._max_turns = max_turns
        self._turns: list[Turn] = []
        self.context = ContextMemory()

    @property
    def turns(self) -> list[Turn]:
        return self._turns

    @property
    def turn_count(self) -> int:
        return len(self._turns) // 2

    def add_user(self, text: str) -> None:
        self._turns.append(Turn(role="user", text=text))
        self._extract_context(text)
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._turns.append(Turn(role="assistant", text=text))
        self._trim()

    def get_history_for_llm(self) -> list[dict[str, str]]:
        """Return turn history formatted for OpenAI-style messages."""
        return [{"role": t.role, "content": t.text} for t in self._turns]

    def _trim(self) -> None:
        max_messages = self._max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def _extract_context(self, text: str) -> None:
        """Extract structured info from user utterance."""
        lower = text.lower()

        # Name extraction
        name_patterns = [
            r"(?:my name is|i'm|i am|this is|call me)\s+([A-Z][a-z]+)",
            r"(?:my name is|i'm|i am|this is|call me)\s+([a-z]+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 1 and name.lower() not in ("a", "the", "just", "not"):
                    self.context.user_name = name.capitalize()
                    logger.info("[MEMORY] Extracted name: %s", self.context.user_name)
                    break

        # Topic extraction (simple keyword-based)
        topic_keywords = {
            "appointment": "scheduling",
            "schedule": "scheduling",
            "book": "scheduling",
            "cancel": "cancellation",
            "payment": "billing",
            "bill": "billing",
            "charge": "billing",
            "refund": "billing",
            "problem": "support",
            "issue": "support",
            "help": "support",
            "broken": "support",
            "price": "pricing",
            "cost": "pricing",
            "how much": "pricing",
        }
        for keyword, topic in topic_keywords.items():
            if keyword in lower:
                self.context.topic = topic
                break

        # Sentiment detection (basic)
        negative_words = {"angry", "frustrated", "annoyed", "terrible", "awful", "worst", "hate"}
        positive_words = {"great", "thanks", "awesome", "love", "perfect", "excellent", "wonderful"}

        words = set(lower.split())
        if words & negative_words:
            self.context.sentiment = "negative"
        elif words & positive_words:
            self.context.sentiment = "positive"

    def reset(self) -> None:
        self._turns.clear()
        self.context = ContextMemory()
