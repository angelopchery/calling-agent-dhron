"""
Conversation memory system — short-term + context + booking data.

Short-term: sliding window of recent turns.
Context: extracted structured info (user name, intent, topic).
Booking: Pramukh Group appointment booking state.
All are injected into LLM prompts for context-aware responses.
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
class BookingData:
    """Structured booking state for Pramukh Group appointments."""
    location: str | None = None       # "Surat", "Vapi", "Silvassa"
    property_type: str | None = None  # "2 BHK", "3 BHK", etc.
    appointment_date: str | None = None
    appointment_time: str | None = None
    user_name: str | None = None
    confirmed: bool = False

    def summary(self) -> str:
        parts = []
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.property_type:
            parts.append(f"Type: {self.property_type}")
        if self.appointment_date:
            parts.append(f"Date: {self.appointment_date}")
        if self.appointment_time:
            parts.append(f"Time: {self.appointment_time}")
        if self.user_name:
            parts.append(f"Name: {self.user_name}")
        return ", ".join(parts) if parts else "No details yet"

    def next_missing_field(self) -> str | None:
        if not self.location:
            return "location"
        if not self.property_type:
            return "property_type"
        if not self.appointment_date and not self.appointment_time:
            return "appointment"
        return None


@dataclass
class ContextMemory:
    """Structured info extracted from conversation."""
    user_name: str | None = None
    intent: str | None = None
    topic: str | None = None
    sentiment: str = "neutral"
    extra: dict = field(default_factory=dict)

    def to_prompt_string(self) -> str:
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
    Manages short-term turn history, structured context, and booking state.
    """

    def __init__(self, max_turns: int = MAX_SHORT_TERM_TURNS) -> None:
        self._max_turns = max_turns
        self._turns: list[Turn] = []
        self.context = ContextMemory()
        self.booking = BookingData()

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
        return [{"role": t.role, "content": t.text} for t in self._turns]

    def _trim(self) -> None:
        max_messages = self._max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def _extract_context(self, text: str) -> None:
        lower = text.lower()

        name_patterns = [
            r"(?:my name is|i'm|i am|this is|call me|mera naam|naam)\s+([A-Za-z][a-z]+)",
        ]
        _name_blocklist = {
            "a", "the", "just", "not", "is", "interested", "busy",
            "calling", "looking", "available", "free", "here", "there",
            "good", "fine", "okay", "sure", "ready", "happy", "sorry",
            "glad", "trying", "going", "coming", "waiting", "talking",
            "asking", "wondering", "thinking", "planning", "searching",
        }
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 1 and name.lower() not in _name_blocklist:
                    self.context.user_name = name.capitalize()
                    self.booking.user_name = self.context.user_name
                    logger.info("[MEMORY] Extracted name: %s", self.context.user_name)
                    break

        topic_keywords = {
            "appointment": "scheduling", "schedule": "scheduling",
            "book": "scheduling", "visit": "scheduling",
            "cancel": "cancellation",
            "price": "pricing", "cost": "pricing",
            "how much": "pricing", "rate": "pricing", "kitna": "pricing",
            "problem": "support", "issue": "support",
            "flat": "property", "apartment": "property",
            "bhk": "property", "property": "property",
            "makan": "property", "ghar": "property",
        }
        for keyword, topic in topic_keywords.items():
            if keyword in lower:
                self.context.topic = topic
                break

        negative_words = {"angry", "frustrated", "annoyed", "terrible", "awful", "worst", "hate"}
        positive_words = {"great", "thanks", "awesome", "love", "perfect", "excellent", "wonderful",
                          "accha", "badhiya", "saru", "maja"}
        words = set(lower.split())
        if words & negative_words:
            self.context.sentiment = "negative"
        elif words & positive_words:
            self.context.sentiment = "positive"

    def reset(self) -> None:
        self._turns.clear()
        self.context = ContextMemory()
        self.booking = BookingData()
