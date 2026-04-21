"""
Deterministic shortcut layer — instant responses without LLM.

Handles common phrases (greetings, farewells, thanks) with zero latency.
Checked before every LLM call; returns None if no shortcut matches.
"""

from __future__ import annotations

SHORTCUTS: dict[str, str] = {
    "hello": "Hello! How can I help you?",
    "hi": "Hi there! What can I do for you?",
    "hey": "Hey! How can I assist you?",
    "good morning": "Good morning! How can I help?",
    "good afternoon": "Good afternoon! What can I do for you?",
    "good evening": "Good evening! How can I help?",
    "thank you": "You're welcome!",
    "thanks": "You're welcome!",
    "thanks a lot": "Happy to help!",
    "thank you so much": "My pleasure!",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Take care!",
    "see you": "See you later!",
    "good night": "Good night! Take care!",
    "okay bye": "Bye! Have a good one!",
    "that's all": "Alright! Let me know if you need anything else.",
    "nothing else": "Okay, take care! Bye!",
}


def check_shortcut(transcript: str) -> str | None:
    """
    Check if transcript matches a shortcut phrase.
    Returns the response string, or None if no match.
    """
    normalized = transcript.strip().lower().rstrip(".!?,")

    # Exact match
    if normalized in SHORTCUTS:
        return SHORTCUTS[normalized]

    # Prefix match for slight variations ("hello there", "hi how are you")
    for trigger, response in SHORTCUTS.items():
        if normalized.startswith(trigger) and len(normalized) - len(trigger) < 8:
            return response

    return None
