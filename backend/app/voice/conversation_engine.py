"""
Conversation engine — manages dialogue history and LLM interaction.

The base class defines an abstract interface; `MockConversationEngine`
provides deterministic responses for testing. Replace with an OpenAI /
Anthropic / local-model implementation by subclassing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str


class ConversationEngine(ABC):
    """Abstract LLM conversation interface."""

    def __init__(self, system_prompt: str = "") -> None:
        self.history: list[Message] = []
        if system_prompt:
            self.history.append(Message(role="system", content=system_prompt))

    @abstractmethod
    async def respond(self, user_text: str) -> str:
        """Generate a response to `user_text`, updating history."""
        ...

    def add_message(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))


class MockConversationEngine(ConversationEngine):
    """
    Returns canned responses keyed on simple keyword matching.
    Useful for end-to-end pipeline testing without an LLM dependency.
    """

    RESPONSES = {
        "hello": "Hi there! How can I help you today?",
        "name": "Nice to meet you! What can I do for you?",
        "what can you do": "I can help with scheduling, answering questions, and much more.",
        "help": "Of course! I'm here to assist. What do you need help with?",
    }
    DEFAULT = "I understand. Could you tell me more about that?"

    async def respond(self, user_text: str) -> str:
        self.add_message("user", user_text)
        lower = user_text.lower()

        response = self.DEFAULT
        for keyword, reply in self.RESPONSES.items():
            if keyword in lower:
                response = reply
                break

        self.add_message("assistant", response)
        logger.info("[LLM] User: %r → Response: %r", user_text, response)
        return response
