"""
Conversation router — intent detection, shortcuts, and LLM.

Routes user utterances through a priority chain:
  1. Deterministic shortcuts (greetings, farewells, thanks) → instant response
  2. Small talk detection → canned natural responses
  3. LLM call with context memory injection → full response

This minimizes LLM usage for trivial interactions and keeps latency low.
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from .memory import ConversationMemory

logger = logging.getLogger(__name__)

LLM_TIMEOUT = 8.0
LLM_MAX_TOKENS = 120


@dataclass
class EngineResult:
    text: str
    is_shortcut: bool
    intent: str


class ConversationState:
    """Tracks conversation state across turns."""

    def __init__(self) -> None:
        self.last_intent: str | None = None
        self.turn_count: int = 0
        self.is_closing: bool = False
        self.language: str = "en"
        self.language_history: list[str] = []


# ---------------------------------------------------------------------------
# Intent patterns
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = {
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "howdy",
}

_FAREWELL_PATTERNS = {
    "bye", "goodbye", "see you", "good night", "okay bye",
    "talk to you later", "take care", "that's all", "nothing else",
    "i'm done",
}

_GRATITUDE_PATTERNS = {
    "thank you", "thanks", "thanks a lot", "thank you so much",
    "appreciate it", "much appreciated",
}

_AFFIRMATION_PATTERNS = {
    "yes", "yeah", "yep", "sure", "okay", "ok", "right",
    "correct", "exactly", "of course",
}

_NEGATION_PATTERNS = {
    "no", "nope", "nah", "not really", "no thanks",
    "i don't think so",
}

_SMALLTALK_RESPONSES = {
    "how are you": "I'm doing great, thanks for asking! How can I help?",
    "what's your name": "I'm your AI assistant. What can I help you with?",
    "who are you": "I'm an AI voice assistant here to help you out.",
    "what can you do": "I can help answer questions, schedule things, or just chat. What do you need?",
    "are you a robot": "I'm an AI assistant! But I try to keep things natural. How can I help?",
    "how's it going": "Going well! What can I do for you?",
    "what's up": "Not much! What can I help you with?",
}

_GREETING_RESPONSES = [
    "Hey there! How can I help you today?",
    "Hello! What can I do for you?",
    "Hi! How can I help?",
]

_FAREWELL_RESPONSES = [
    "Goodbye! Have a great day!",
    "Bye! Take care!",
    "See you later! Take care!",
]

_GRATITUDE_RESPONSES = [
    "You're welcome!",
    "Happy to help!",
    "My pleasure!",
    "Anytime!",
]

# ---------------------------------------------------------------------------
# Multilingual shortcut responses
# ---------------------------------------------------------------------------

_LOCALIZED_RESPONSES: dict[str, dict[str, list[str] | str]] = {
    "hi": {
        "greeting": ["नमस्ते! कैसे मदद कर सकता हूँ?", "नमस्ते! क्या मदद चाहिए?", "हेलो! बताइए क्या कर सकता हूँ?"],
        "farewell": "अच्छा, फिर मिलते हैं!",
        "farewell_named": "अलविदा, {name}! ख्याल रखिए!",
        "gratitude": ["आपका स्वागत है!", "खुशी हुई मदद करके!", "कोई बात नहीं!"],
        "affirmation": "समझ गया! आगे क्या करना है?",
        "negation": "कोई बात नहीं। और कुछ मदद चाहिए?",
        "affirmation_booking": "बढ़िया, चलिए शेड्यूल करते हैं।",
        "negation_booking": "कोई बात नहीं, बाद में कर लेंगे।",
        "closing": "अच्छा, फिर बात करते हैं!",
    },
    "gu": {
        "greeting": ["નમસ્તે! હું કેવી રીતે મદદ કરી શકું?", "નમસ્તે! શું મદદ જોઈએ?", "હેલો! બોલો શું કરી શકું?"],
        "farewell": "સારું, ફરી મળીશું!",
        "farewell_named": "આવજો, {name}! ધ્યાન રાખજો!",
        "gratitude": ["આપનું સ્વાગત છે!", "મદદ કરીને ખુશી થઈ!", "કોઈ વાત નહીં!"],
        "affirmation": "સમજાયું! હવે શું કરવું છે?",
        "negation": "કોઈ વાત નહીં. બીજું કંઈ મદદ જોઈએ?",
        "affirmation_booking": "સરસ, ચાલો શેડ્યૂલ કરીએ.",
        "negation_booking": "કોઈ વાત નહીં, પછી કરીશું.",
        "closing": "સારું, પછી વાત કરીએ!",
    },
}


def detect_intent(text: str) -> str:
    """Classify user input into a simple intent category. Fast and deterministic."""
    if not text or not text.strip():
        return "general"

    text_lower = text.lower().strip()

    def _has(keywords: list[str]) -> bool:
        return any(re.search(rf"\b{re.escape(k)}\b", text_lower) for k in keywords)

    def _contains(keywords: list[str]) -> bool:
        return any(k in text_lower for k in keywords)

    if _has(["hello", "hi", "hey", "good morning", "good evening",
             "namaste", "namasthe", "namaskar"]):
        return "greeting"
    if _contains(["नमस्ते", "नमस्कार", "હેલો", "નમસ્તે"]):
        return "greeting"

    if _has(["bye", "goodbye", "see you", "talk later",
             "alvida", "phir milte"]):
        return "farewell"
    if _contains(["अलविदा", "फिर मिलते", "चलता हूँ", "આવજો", "ફરી મળીશું"]):
        return "farewell"

    if _has(["thank", "thanks", "appreciate",
             "dhanyavaad", "shukriya", "aabhar"]):
        return "gratitude"
    if _contains(["धन्यवाद", "शुक्रिया", "આભાર", "મહેરબાની"]):
        return "gratitude"

    if _has(["yes", "yeah", "yep", "sure", "okay", "ok",
             "haan", "ji", "theek"]):
        return "affirmation"
    if _contains(["हां", "हाँ", "जी", "ठीक", "હા", "બરાબર", "ठीक है"]):
        return "affirmation"

    if _has(["no", "nope", "nah", "not really",
             "nahi", "nako"]):
        return "negation"
    if _contains(["नहीं", "ना", "ના", "નહીં"]):
        return "negation"

    if _has(["book", "schedule", "appointment", "meeting"]):
        return "booking"
    if _has(["price", "cost", "how much", "charge"]):
        return "pricing"

    return "general"


def detect_language(text: str) -> str:
    """Detect language from Unicode script: 'en', 'hi', or 'gu'."""
    if not text or not text.strip():
        return "en"
    if any("઀" <= c <= "૿" for c in text):
        return "gu"
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"
    return "en"


# ---------------------------------------------------------------------------
# Transliteration pattern dictionaries for language detection v2
# ---------------------------------------------------------------------------

_HINDI_PATTERNS = [
    "hai", "tha", "thi", "hoga",
    "kya", "kyun", "kaise",
    "karna", "chahiye", "hona",
    "mujhe", "tum", "aap",
    "karni", "karo", "baat",
    "nahi", "accha", "theek",
    "namaste", "namasthe", "dhanyavaad",
    "shukriya", "alvida", "haan",
]

_GUJARATI_PATTERNS = [
    "che", "chu", "hatu", "hase",
    "shu", "kem",
    "tame", "hu",
    "pan", "che ke",
    "majama", "karo", "chhe",
    "tamne", "mane",
]

_LANG_CLASSIFY_PROMPT = (
    "Classify the dominant language of this text.\n"
    "Options: English, Hindi, Gujarati\n"
    "The input may be transliterated (romanized) or mixed-language.\n"
    "Choose the dominant spoken language.\n"
    'Text: "{text}"\n'
    "Return ONLY one word: English, Hindi, or Gujarati."
)

_LANG_NAME_TO_CODE = {"english": "en", "hindi": "hi", "gujarati": "gu"}


def _pattern_score(text: str, patterns: list[str]) -> int:
    """Count how many transliteration patterns appear as whole words."""
    words = set(text.split())
    return sum(1 for p in patterns if p in words)


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip(".!?,;")


def _match_set(text: str, patterns: set[str]) -> bool:
    normalized = _normalize(text)
    if normalized in patterns:
        return True
    for pattern in patterns:
        if normalized.startswith(pattern):
            extra = normalized[len(pattern):]
            if len(extra) < 10:
                return True
    return False


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly, natural-sounding voice assistant on a phone call.

CRITICAL RULES:
- Respond in 1-2 sentences MAXIMUM. You are speaking out loud, not writing.
- Sound like a real person: use contractions (I'm, you're, don't, can't).
- NEVER say: "I understand", "Could you tell me more", "That's a great question", "I'd be happy to help".
- If you don't know something, say "I'm not sure about that" and move on.
- Match the user's energy: short input = short response.
- No bullet points, no lists, no markdown, no emojis.
- If the user seems done, wrap up. Don't artificially extend the conversation.
- One follow-up question max, and only if genuinely needed.
- Be warm but efficient. Respect the user's time.
"""


class ConversationRouter:
    """
    Routes utterances through: shortcuts → smalltalk → LLM.

    Maintains conversation memory (short-term + context) and injects
    it into LLM calls for context-aware responses.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = LLM_TIMEOUT,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None
        self.memory = ConversationMemory()
        self.state = ConversationState()
        self._response_counter = 0
        self._lang_cache: dict[str, tuple[str, float, str]] = {}

        if not self._api_key:
            logger.warning("[CONV] OPENAI_API_KEY not set — LLM calls will fail")

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    async def process(self, text: str) -> EngineResult:
        """
        Route user text through the response chain.
        Returns an EngineResult with the response and metadata.
        """
        self.memory.add_user(text)

        intent = detect_intent(text)
        lang, conf, src = await self.detect_language_v3(text)
        self.state.turn_count += 1
        logger.info('[LANG] "%s" -> %s (src=%s, conf=%.2f)', text, lang, src, conf)

        # Stabilization: majority vote over last 5 turns
        self.state.language_history.append(lang)
        history = self.state.language_history[-5:]
        self.state.language_history = history
        final_lang = max(set(history), key=history.count)
        self.state.language = final_lang
        logger.info('[LANG] stabilized -> %s (history=%s)', final_lang, history)

        logger.info('[INTENT] "%s" -> %s', text, intent)
        logger.info(
            "[STATE] turn=%d last=%s closing=%s lang=%s",
            self.state.turn_count,
            self.state.last_intent,
            self.state.is_closing,
            self.state.language,
        )

        # --- State-aware early routing ---

        if intent == "farewell":
            self.state.is_closing = True
            self.state.last_intent = intent
            resp = self._localized("closing", fallback="Alright, talk to you soon!")
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="farewell")

        if self.state.is_closing:
            if intent == "greeting":
                self.state.is_closing = False
            else:
                self.state.last_intent = intent
                return EngineResult(text="", is_shortcut=True, intent="closed")

        if intent == "affirmation" and self.state.last_intent == "booking":
            self.state.last_intent = intent
            resp = self._localized("affirmation_booking", fallback="Great, let's get that scheduled.")
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="affirmation")

        if intent == "negation" and self.state.last_intent == "booking":
            self.state.last_intent = intent
            resp = self._localized("negation_booking", fallback="No problem, we can do it later.")
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="negation")

        self.state.last_intent = intent

        normalized = _normalize(text)

        # --- Layer 1: Greetings ---
        if _match_set(text, _GREETING_PATTERNS) or intent == "greeting":
            resp = self._localized_list("greeting", _GREETING_RESPONSES)
            if self.memory.context.user_name:
                resp = resp.replace("!", f", {self.memory.context.user_name}!", 1)
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="greeting")

        # --- Layer 2: Farewells ---
        if _match_set(text, _FAREWELL_PATTERNS) or intent == "farewell":
            name = self.memory.context.user_name
            if name:
                loc = _LOCALIZED_RESPONSES.get(self.state.language, {})
                named_tmpl = loc.get("farewell_named")
                if named_tmpl and isinstance(named_tmpl, str):
                    resp = named_tmpl.format(name=name)
                else:
                    resp = f"Goodbye, {name}! Take care!"
            else:
                resp = self._localized("farewell", fallback=self._pick_response(_FAREWELL_RESPONSES))
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="farewell")

        # --- Layer 3: Gratitude ---
        if _match_set(text, _GRATITUDE_PATTERNS) or intent == "gratitude":
            resp = self._localized_list("gratitude", _GRATITUDE_RESPONSES)
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="gratitude")

        # --- Layer 4: Affirmation/Negation ---
        if _match_set(text, _AFFIRMATION_PATTERNS) or intent == "affirmation":
            resp = self._localized("affirmation", fallback="Got it! What would you like to do next?")
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="affirmation")

        if _match_set(text, _NEGATION_PATTERNS) or intent == "negation":
            resp = self._localized("negation", fallback="No problem. Is there anything else I can help with?")
            self.memory.add_assistant(resp)
            return EngineResult(text=resp, is_shortcut=True, intent="negation")

        # --- Layer 5: Small talk ---
        for trigger, response in _SMALLTALK_RESPONSES.items():
            if trigger in normalized:
                self.memory.add_assistant(response)
                return EngineResult(text=response, is_shortcut=True, intent="smalltalk")

        # --- Layer 6: LLM ---
        response = await self._call_llm(text)
        self.memory.add_assistant(response)
        return EngineResult(text=response, is_shortcut=False, intent="llm")

    async def detect_language_v3(self, text: str) -> tuple[str, float, str]:
        """
        3-layer language detection with confidence scoring and source tracking.

        Returns (lang_code, confidence, source) where:
          - lang_code: 'en', 'hi', or 'gu'
          - confidence: 0.0-1.0
          - source: 'script', 'pattern', 'llm', or 'default'
        """
        text_lower = text.lower().strip()
        if not text_lower:
            return "en", 1.0, "default"

        if text_lower in self._lang_cache:
            result = self._lang_cache[text_lower]
            logger.info("[LANG] cache hit: %r -> %s (conf=%.2f, src=%s)", text_lower, *result)
            return result

        # Layer 1: Unicode script detection — highest confidence
        if any("઀" <= c <= "૿" for c in text_lower):
            result = ("gu", 1.0, "script")
        elif any("ऀ" <= c <= "ॿ" for c in text_lower):
            result = ("hi", 1.0, "script")
        else:
            # Layer 2: Transliteration pattern scoring
            hi_score = _pattern_score(text_lower, _HINDI_PATTERNS)
            gu_score = _pattern_score(text_lower, _GUJARATI_PATTERNS)
            total = hi_score + gu_score

            if total > 0:
                confidence = abs(hi_score - gu_score) / total
                lang = "hi" if hi_score >= gu_score else "gu"

                if confidence >= 0.6:
                    result = (lang, confidence, "pattern")
                elif confidence > 0:
                    result = (lang, confidence, "pattern")
                else:
                    result = (lang, 0.5, "pattern")
            elif text_lower.isascii():
                result = ("en", 1.0, "default")
            else:
                # Layer 3: LLM fallback for ambiguous non-ASCII input
                lang = await self._classify_language_llm(text_lower)
                result = (lang, 0.9, "llm")

        self._lang_cache[text_lower] = result
        return result

    async def _classify_language_llm(self, text: str) -> str:
        """Ask GPT-4o-mini to classify language. Returns 'en', 'hi', or 'gu'."""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": _LANG_CLASSIFY_PROMPT.format(text=text),
                }],
                max_tokens=5,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            lang = _LANG_NAME_TO_CODE.get(raw, "en")
            logger.info("[LANG] LLM classified %r -> %s (raw: %r)", text, lang, raw)
            return lang
        except Exception as exc:
            logger.warning("[LANG] LLM classification failed: %s — defaulting to en", exc)
            return "en"

    def _localized(self, key: str, fallback: str) -> str:
        """Get a localized string response, falling back to English."""
        loc = _LOCALIZED_RESPONSES.get(self.state.language, {})
        val = loc.get(key)
        if val is None:
            return fallback
        if isinstance(val, list):
            return val[self._response_counter % len(val)]
        return val

    def _localized_list(self, key: str, en_list: list[str]) -> str:
        """Get a localized response from a list, with round-robin."""
        loc = _LOCALIZED_RESPONSES.get(self.state.language, {})
        val = loc.get(key)
        if isinstance(val, list):
            resp = val[self._response_counter % len(val)]
            self._response_counter += 1
            return resp
        return self._pick_response(en_list)

    async def _call_llm(self, text: str) -> str:
        """Call LLM with full context memory."""
        client = self._get_client()

        # Build system prompt with context + language instruction
        system = SYSTEM_PROMPT
        if self.state.language != "en":
            lang_name = {"hi": "Hindi", "gu": "Gujarati"}.get(self.state.language, "English")
            system += (
                f"\n\nIMPORTANT: The user is speaking {lang_name}. "
                f"You MUST respond in {lang_name}. "
                "Match the user's language exactly."
            )
        context_str = self.memory.context.to_prompt_string()
        if context_str:
            system += f"\n\nKnown context about this caller:\n{context_str}"

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_history_for_llm())

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
            logger.info("[LLM] Response: %r", reply)
            return reply

        except Exception as exc:
            logger.error("[LLM] Failed: %s", exc)
            return "Sorry, I didn't quite catch that. Could you say that again?"

    def _pick_response(self, options: list[str]) -> str:
        """Round-robin response selection to avoid repetition."""
        resp = options[self._response_counter % len(options)]
        self._response_counter += 1
        return resp
