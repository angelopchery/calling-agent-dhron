"""
Conversation router — intent detection, shortcuts, and LLM.

Routes user utterances through a priority chain:
  1. Deterministic shortcuts (greetings, farewells, thanks) -> instant response
  2. Objection handling -> canned empathetic responses
  3. Small talk detection -> canned natural responses
  4. LLM call with context memory injection -> full response

Features parallel intent+LLM execution, filler responses for slow LLM,
speculative response pre-generation, and LLM-based intent classification.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from .memory import ConversationMemory

# Stage constants
STAGE_INTRO = "INTRO"
STAGE_DISCOVERY = "DISCOVERY"
STAGE_SCHEDULING = "SCHEDULING"
STAGE_CLOSING = "CLOSING"

_VALID_STAGES = {STAGE_INTRO, STAGE_DISCOVERY, STAGE_SCHEDULING, STAGE_CLOSING}

# Stage transition rules: intent -> target stage
_STAGE_TRANSITIONS: dict[str, str] = {
    "greeting": STAGE_INTRO,
    "booking": STAGE_SCHEDULING,
    "pricing": STAGE_DISCOVERY,
    "general": STAGE_DISCOVERY,
    "negation": STAGE_DISCOVERY,
    "objection": STAGE_DISCOVERY,
    "farewell": STAGE_CLOSING,
}

_STAGE_ORDER: dict[str, int] = {
    STAGE_INTRO: 0,
    STAGE_DISCOVERY: 1,
    STAGE_SCHEDULING: 2,
    STAGE_CLOSING: 3,
}

# Markdown/emoji stripping for response validation
_MARKDOWN_RE = re.compile(r"[*#`~_\[\]()>|]")
_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff\U00002600-\U000026ff]+",
    re.UNICODE,
)

MAX_RESPONSE_LENGTH = 250

logger = logging.getLogger(__name__)

LLM_TIMEOUT = 8.0
LLM_MAX_TOKENS = 120
FILLER_TIMEOUT_S = 2.0
LLM_STREAM_TIMEOUT_S = 6.0
INTENT_CLASSIFY_TIMEOUT_S = 1.5


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
        self.stage: str = STAGE_INTRO
        self.lang_confidence: float = 1.0
        self.lang_source: str = "default"
        # Per-turn metrics (reset each turn)
        self.filler_used: bool = False
        self.llm_cancelled: bool = False
        self.parallel_execution: bool = False
        self.objection_handled: bool = False

    def update_stage(self, intent: str) -> str | None:
        """Forward-only stage transition. Returns old stage if changed, else None."""
        new_stage = _STAGE_TRANSITIONS.get(intent)
        if not new_stage or new_stage == self.stage:
            return None

        new_order = _STAGE_ORDER.get(new_stage, 0)
        cur_order = _STAGE_ORDER.get(self.stage, 0)

        if new_order > cur_order:
            old = self.stage
            self.stage = new_stage
            return old

        if intent in ("negation", "objection") and self.stage == STAGE_SCHEDULING:
            old = self.stage
            self.stage = STAGE_DISCOVERY
            return old

        return None


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

_OBJECTION_PATTERNS = [
    "not interested", "busy", "call later", "already using",
    "don't need", "no time", "don't want", "stop calling",
    "not looking", "no need",
]

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
# Objection responses (multilingual)
# ---------------------------------------------------------------------------

_OBJECTION_RESPONSES: dict[str, list[str]] = {
    "en": [
        "I understand. Just quickly -- this might actually save you time.",
        "No worries. When would be a better time to talk?",
    ],
    "hi": [
        "समझ गया। बस एक मिनट लगेगा, शायद आपके काम आ जाए।",
        "कोई बात नहीं, कब बात करना सही रहेगा?",
    ],
    "gu": [
        "સમજી ગયો. બસ એક મિનિટ લાગશે, તમને મદદરૂપ થઈ શકે.",
        "બરાબર, પછી ક્યારે વાત કરીએ?",
    ],
}

# ---------------------------------------------------------------------------
# Filler responses for slow LLM (multilingual)
# ---------------------------------------------------------------------------

_FILLER_RESPONSES: dict[str, list[str]] = {
    "en": ["Let me check that for you...", "Just a moment..."],
    "hi": ["एक सेकंड, मैं देखता हूँ...", "जरा रुकिए..."],
    "gu": ["એક મિનિટ, હું ચેક કરું...", "જોઈ લઉં..."],
}

# ---------------------------------------------------------------------------
# LLM intent classification prompt
# ---------------------------------------------------------------------------

_INTENT_CLASSIFY_PROMPT = (
    "Classify the user utterance into one of these intents:\n"
    "[greeting, farewell, objection, interest, question, scheduling, general]\n\n"
    'Utterance: "{text}"\n\n'
    "Return ONLY one word."
)

_LLM_INTENT_MAP: dict[str, str] = {
    "greeting": "greeting",
    "farewell": "farewell",
    "objection": "objection",
    "interest": "interest",
    "question": "general",
    "scheduling": "booking",
    "general": "general",
}

# ---------------------------------------------------------------------------
# LLM timeout fallbacks (multilingual)
# ---------------------------------------------------------------------------

_TIMEOUT_FALLBACKS: dict[str, str] = {
    "en": "Let me get back to you on that.",
    "hi": "मैं इस पर वापस आता हूँ।",
    "gu": "હું આના પર પાછો આવું છું.",
}

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

    # Objection detection (before other intents — higher priority)
    if any(p in text_lower for p in _OBJECTION_PATTERNS):
        return "objection"
    if _contains(["दिलचस्पी नहीं", "बिज़ी", "बाद में", "ज़रूरत नहीं",
                   "રસ નથી", "વ્યસ્ત", "પછી", "જરૂર નથી"]):
        return "objection"

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
# Response validation
# ---------------------------------------------------------------------------

_RESPONSE_FALLBACK = "Sorry, could you repeat that?"

_LANG_FALLBACKS: dict[str, str] = {
    "hi": "माफ़ कीजिए, क्या आप दोबारा बोल सकते हैं?",
    "gu": "માફ કરશો, શું તમે ફરીથી બોલી શકો?",
    "en": "Sorry, could you repeat that?",
}


def _validate_response(text: str, expected_lang: str = "en") -> str:
    """Validate and clean LLM response before sending to TTS."""
    if not text or not text.strip():
        return _RESPONSE_FALLBACK

    # Strip markdown formatting
    cleaned = _MARKDOWN_RE.sub("", text)

    # Strip emojis
    cleaned = _EMOJI_RE.sub("", cleaned).strip()

    # Truncate if too long
    if len(cleaned) > MAX_RESPONSE_LENGTH:
        truncated = cleaned[:MAX_RESPONSE_LENGTH]
        last_sentence = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_sentence > MAX_RESPONSE_LENGTH // 2:
            cleaned = truncated[: last_sentence + 1]
        else:
            cleaned = truncated.rstrip() + "..."

    if not cleaned:
        return _RESPONSE_FALLBACK

    return cleaned


class ConversationRouter:
    """
    Routes utterances through: shortcuts -> smalltalk -> LLM.

    Features:
    - Parallel intent detection + LLM execution
    - Filler responses when LLM is slow (>2s)
    - Objection handling without LLM
    - LLM-based intent classification fallback
    - Speculative yes/no pre-generation
    - LLM stream timeout protection (6s)
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

        # Speculative response state
        self._speculative_yes: asyncio.Task | None = None
        self._speculative_no: asyncio.Task | None = None
        self._filler_counter = 0

        if not self._api_key:
            logger.warning("[CONV] OPENAI_API_KEY not set -- LLM calls will fail")

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, text: str) -> EngineResult:
        """Non-streaming process. Collects all stream chunks into a single result."""
        parts: list[str] = []
        intent = "general"
        is_shortcut = False
        async for result in self.process_stream(text):
            parts.append(result.text)
            intent = result.intent
            is_shortcut = result.is_shortcut
        return EngineResult(text=" ".join(parts), is_shortcut=is_shortcut, intent=intent)

    async def process_stream(self, text: str):
        """
        Streaming process with parallel intent+LLM, fillers, and speculative
        responses. Yields EngineResult chunks for TTS.
        """
        self.memory.add_user(text)

        # --- Language detection + stabilization ---
        lang, conf, src = await self.detect_language_v3(text)
        self.state.turn_count += 1
        logger.info('[LANG] "%s" -> %s (src=%s, conf=%.2f)', text, lang, src, conf)

        self.state.language_history.append(lang)
        history = self.state.language_history[-5:]
        self.state.language_history = history

        counts: dict[str, int] = {}
        for h in history:
            counts[h] = counts.get(h, 0) + 1
        max_count = max(counts.values())
        candidates = [l for l, c in counts.items() if c == max_count]

        if len(candidates) == 1:
            final_lang = candidates[0]
        else:
            final_lang = self.state.language if self.state.language in candidates else candidates[0]

        self.state.language = final_lang
        self.state.lang_confidence = conf
        self.state.lang_source = src
        logger.info('[LANG] stabilized -> %s (history=%s, tie=%s)',
                     final_lang, history, len(candidates) > 1)

        # --- Reset per-turn metrics ---
        self.state.filler_used = False
        self.state.llm_cancelled = False
        self.state.parallel_execution = True
        self.state.objection_handled = False

        # --- Intent detection (synchronous, instant) ---
        intent = detect_intent(text)

        # --- Fire LLM immediately in background (parallel with routing) ---
        llm_queue: asyncio.Queue[str | None] = asyncio.Queue()
        llm_done = asyncio.Event()
        llm_task = asyncio.create_task(
            self._run_llm_to_queue(text, llm_queue, llm_done),
            name="parallel_llm",
        )

        def _cancel_llm():
            if not llm_task.done():
                llm_task.cancel()
            self.state.llm_cancelled = True

        # --- Stage transition ---
        old_stage = self.state.update_stage(intent)
        if old_stage:
            logger.info("[STAGE] %s -> %s", old_stage, self.state.stage)

        logger.info('[INTENT] "%s" -> %s', text, intent)
        logger.info(
            "[STATE] turn=%d last=%s closing=%s lang=%s stage=%s",
            self.state.turn_count, self.state.last_intent,
            self.state.is_closing, self.state.language, self.state.stage,
        )

        # =============================================================
        # SHORTCUT ROUTING — cancel background LLM on match
        # =============================================================

        # --- Farewell ---
        if intent == "farewell":
            _cancel_llm()
            self.state.is_closing = True
            self.state.last_intent = intent
            resp = self._localized("closing", fallback="Alright, talk to you soon!")
            self.memory.add_assistant(resp)
            logger.info("[PARALLEL] intent_resolved=farewell llm_cancelled=True")
            yield EngineResult(text=resp, is_shortcut=True, intent="farewell")
            return

        # --- Closing state ---
        if self.state.is_closing:
            if intent == "greeting":
                self.state.is_closing = False
                # Fall through — greeting handler below will cancel LLM
            else:
                _cancel_llm()
                self.state.last_intent = intent
                yield EngineResult(text="", is_shortcut=True, intent="closed")
                return

        # --- Booking-specific affirmation/negation ---
        if intent == "affirmation" and self.state.last_intent == "booking":
            _cancel_llm()
            self.state.last_intent = intent
            resp = self._localized("affirmation_booking", fallback="Great, let's get that scheduled.")
            self.memory.add_assistant(resp)
            logger.info("[PARALLEL] intent_resolved=affirmation llm_cancelled=True")
            yield EngineResult(text=resp, is_shortcut=True, intent="affirmation")
            return

        if intent == "negation" and self.state.last_intent == "booking":
            _cancel_llm()
            self.state.last_intent = intent
            resp = self._localized("negation_booking", fallback="No problem, we can do it later.")
            self.memory.add_assistant(resp)
            logger.info("[PARALLEL] intent_resolved=negation llm_cancelled=True")
            yield EngineResult(text=resp, is_shortcut=True, intent="negation")
            return

        self.state.last_intent = intent
        normalized = _normalize(text)

        # --- Objection handling (deterministic, no LLM) ---
        if intent == "objection":
            _cancel_llm()
            self.state.objection_handled = True
            resp = self._pick_objection_response()
            self.memory.add_assistant(resp)
            logger.info("[PARALLEL] intent_resolved=objection llm_cancelled=True")
            yield EngineResult(text=resp, is_shortcut=True, intent="objection")
            self._cancel_speculative()
            return

        # --- Speculative response (before generic shortcuts) ---
        if intent in ("affirmation", "negation"):
            spec = self._try_speculative(intent)
            if spec:
                _cancel_llm()
                logger.info("[SPECULATIVE] used=%s",
                            "yes" if intent == "affirmation" else "no")
                self.memory.add_assistant(spec)
                yield EngineResult(text=spec, is_shortcut=False, intent="llm")
                self._start_speculative()
                return

        # --- Generic greetings ---
        if _match_set(text, _GREETING_PATTERNS) or intent == "greeting":
            _cancel_llm()
            resp = self._localized_list("greeting", _GREETING_RESPONSES)
            if self.memory.context.user_name:
                resp = resp.replace("!", f", {self.memory.context.user_name}!", 1)
            self.memory.add_assistant(resp)
            logger.info("[PARALLEL] intent_resolved=greeting llm_cancelled=True")
            yield EngineResult(text=resp, is_shortcut=True, intent="greeting")
            return

        # --- Generic farewells ---
        if _match_set(text, _FAREWELL_PATTERNS) or intent == "farewell":
            _cancel_llm()
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
            yield EngineResult(text=resp, is_shortcut=True, intent="farewell")
            return

        # --- Generic gratitude ---
        if _match_set(text, _GRATITUDE_PATTERNS) or intent == "gratitude":
            _cancel_llm()
            resp = self._localized_list("gratitude", _GRATITUDE_RESPONSES)
            self.memory.add_assistant(resp)
            logger.info("[PARALLEL] intent_resolved=gratitude llm_cancelled=True")
            yield EngineResult(text=resp, is_shortcut=True, intent="gratitude")
            return

        # --- Generic affirmation ---
        if _match_set(text, _AFFIRMATION_PATTERNS) or intent == "affirmation":
            _cancel_llm()
            resp = self._localized("affirmation", fallback="Got it! What would you like to do next?")
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="affirmation")
            return

        # --- Generic negation ---
        if _match_set(text, _NEGATION_PATTERNS) or intent == "negation":
            _cancel_llm()
            resp = self._localized("negation", fallback="No problem. Is there anything else I can help with?")
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="negation")
            return

        # --- Small talk ---
        for trigger, response in _SMALLTALK_RESPONSES.items():
            if trigger in normalized:
                _cancel_llm()
                self.memory.add_assistant(response)
                yield EngineResult(text=response, is_shortcut=True, intent="smalltalk")
                return

        # =============================================================
        # LLM PATH — consume background stream with filler support
        # =============================================================

        # If keyword intent was "general", try LLM classifier for better accuracy
        if intent == "general":
            try:
                llm_intent = await asyncio.wait_for(
                    self._classify_intent_llm(text),
                    timeout=INTENT_CLASSIFY_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.info("[INTENT-LLM] classifier timeout, keeping intent=general")
                llm_intent = "general"

            if llm_intent and llm_intent != "general":
                intent = llm_intent
                self.state.last_intent = intent
                logger.info('[INTENT-LLM] "%s" -> %s', text, intent)

                # Objection caught by classifier
                if intent == "objection":
                    _cancel_llm()
                    self.state.objection_handled = True
                    resp = self._pick_objection_response()
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="objection")
                    self._cancel_speculative()
                    return

        # --- Consume LLM queue with filler timeout ---
        self.state.llm_cancelled = False
        full_parts: list[str] = []

        first_chunk = None
        try:
            first_chunk = await asyncio.wait_for(
                llm_queue.get(), timeout=FILLER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            filler = self._pick_filler()
            logger.info("[FILLER] triggered after %dms", int(FILLER_TIMEOUT_S * 1000))
            self.state.filler_used = True
            yield EngineResult(text=filler, is_shortcut=True, intent="filler")
            first_chunk = await llm_queue.get()

        if first_chunk is not None:
            full_parts.append(first_chunk)
            yield EngineResult(text=first_chunk, is_shortcut=False, intent=intent)

            while True:
                chunk = await llm_queue.get()
                if chunk is None:
                    break
                full_parts.append(chunk)
                yield EngineResult(text=chunk, is_shortcut=False, intent=intent)

        if full_parts:
            self.memory.add_assistant(" ".join(full_parts))

        # --- Schedule speculative pre-generation for next turn ---
        self._start_speculative()

    # ------------------------------------------------------------------
    # Parallel LLM helpers
    # ------------------------------------------------------------------

    async def _run_llm_to_queue(
        self, text: str, queue: asyncio.Queue, done: asyncio.Event,
    ) -> None:
        """Run LLM stream in background, feeding sentence chunks into a queue."""
        try:
            async for chunk in self._call_llm_stream(text):
                await queue.put(chunk)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[LLM] Background stream failed")
            fallback = _LANG_FALLBACKS.get(self.state.language, _RESPONSE_FALLBACK)
            try:
                queue.put_nowait(fallback)
            except asyncio.QueueFull:
                pass
        finally:
            await queue.put(None)
            done.set()

    async def _call_llm_stream(self, text: str):
        """Async generator yielding sentence-level chunks with timeout protection."""
        client = self._get_client()
        system = self._build_system_prompt()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_history_for_llm())

        _SENTENCE_END = re.compile(r'[.!?।]\s')
        deadline = time.monotonic() + LLM_STREAM_TIMEOUT_S

        try:
            stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
                stream=True,
            )

            buffer = ""
            timed_out = False
            async for chunk in stream:
                if time.monotonic() > deadline:
                    logger.warning("[LLM-TIMEOUT] fallback triggered after %.0fs",
                                   LLM_STREAM_TIMEOUT_S)
                    timed_out = True
                    break
                delta = chunk.choices[0].delta
                if delta.content:
                    buffer += delta.content
                    while True:
                        match = _SENTENCE_END.search(buffer)
                        if match:
                            sentence = buffer[:match.end()].strip()
                            buffer = buffer[match.end():]
                            if sentence:
                                yield _validate_response(sentence, self.state.language)
                        else:
                            break

            if timed_out:
                if buffer.strip():
                    yield _validate_response(buffer.strip(), self.state.language)
                else:
                    yield _TIMEOUT_FALLBACKS.get(
                        self.state.language, _TIMEOUT_FALLBACKS["en"]
                    )
            elif buffer.strip():
                yield _validate_response(buffer.strip(), self.state.language)

        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("[LLM] Stream failed: %s", exc)
            yield _LANG_FALLBACKS.get(self.state.language, _RESPONSE_FALLBACK)

    # ------------------------------------------------------------------
    # LLM intent classifier (Task 3)
    # ------------------------------------------------------------------

    async def _classify_intent_llm(self, text: str) -> str:
        """Ask LLM to classify intent. Returns mapped intent string."""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": _INTENT_CLASSIFY_PROMPT.format(text=text),
                }],
                max_tokens=5,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            mapped = _LLM_INTENT_MAP.get(raw, "general")
            logger.info('[INTENT-LLM] "%s" -> %s (raw: %r)', text, mapped, raw)
            return mapped
        except Exception as exc:
            logger.warning("[INTENT-LLM] classification failed: %s", exc)
            return "general"

    # ------------------------------------------------------------------
    # Objection + filler helpers
    # ------------------------------------------------------------------

    def _pick_objection_response(self) -> str:
        """Pick a localized objection response (round-robin)."""
        lang = self.state.language
        responses = _OBJECTION_RESPONSES.get(lang, _OBJECTION_RESPONSES["en"])
        resp = responses[self._response_counter % len(responses)]
        self._response_counter += 1
        return resp

    def _pick_filler(self) -> str:
        """Pick a localized filler response (round-robin)."""
        lang = self.state.language
        fillers = _FILLER_RESPONSES.get(lang, _FILLER_RESPONSES["en"])
        resp = fillers[self._filler_counter % len(fillers)]
        self._filler_counter += 1
        return resp

    # ------------------------------------------------------------------
    # Speculative response system (Task 6)
    # ------------------------------------------------------------------

    def _start_speculative(self) -> None:
        """Pre-generate yes/no responses if last assistant turn was a question."""
        self._cancel_speculative()

        last_assistant = None
        for turn in reversed(self.memory.turns):
            if turn.role == "assistant":
                last_assistant = turn.text
                break

        if last_assistant and last_assistant.rstrip().endswith("?"):
            self._speculative_yes = asyncio.create_task(
                self._generate_speculative("yes"),
                name="spec_yes",
            )
            self._speculative_no = asyncio.create_task(
                self._generate_speculative("no"),
                name="spec_no",
            )
            logger.info("[SPECULATIVE] pre-generating yes/no responses")

    def _cancel_speculative(self) -> None:
        """Cancel any running speculative tasks."""
        for task in (self._speculative_yes, self._speculative_no):
            if task and not task.done():
                task.cancel()
        self._speculative_yes = None
        self._speculative_no = None

    def _try_speculative(self, intent: str) -> str | None:
        """Try to use a pre-generated speculative response. Returns text or None."""
        task = None
        label = ""
        if intent == "affirmation" and self._speculative_yes:
            task = self._speculative_yes
            label = "yes"
        elif intent == "negation" and self._speculative_no:
            task = self._speculative_no
            label = "no"

        if task and task.done():
            try:
                text = task.result()
                if text:
                    self._cancel_speculative()
                    return text
            except Exception:
                pass

        self._cancel_speculative()
        return None

    async def _generate_speculative(self, hypothetical_input: str) -> str:
        """Generate a speculative LLM response for a hypothetical user input."""
        client = self._get_client()
        system = self._build_system_prompt()
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_history_for_llm())
        messages.append({"role": "user", "content": hypothetical_input})

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            validated = _validate_response(text, self.state.language)
            logger.info("[SPECULATIVE] generated for %r: %r", hypothetical_input, validated)
            return validated
        except asyncio.CancelledError:
            return ""
        except Exception as exc:
            logger.warning("[SPECULATIVE] generation failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

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

        # Layer 1: Unicode script detection
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
                # Layer 3: LLM fallback
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
            logger.warning("[LANG] LLM classification failed: %s -- defaulting to en", exc)
            return "en"

    # ------------------------------------------------------------------
    # LLM non-streaming (for backward compat + language consistency)
    # ------------------------------------------------------------------

    async def _call_llm(self, text: str) -> str:
        """Call LLM with streaming collection and response validation."""
        reply = await self._call_llm_collect(self._build_system_prompt())
        validated = _validate_response(reply, self.state.language)

        if self.state.language != "en" and validated:
            resp_lang = detect_language(validated)
            if resp_lang != self.state.language:
                logger.warning("[LLM] Language mismatch: expected %s, got %s -- retrying",
                               self.state.language, resp_lang)
                lang_name = {"hi": "Hindi", "gu": "Gujarati"}.get(self.state.language, "English")
                enforced_system = self._build_system_prompt() + (
                    f"\n\nCRITICAL: Your previous response was in the WRONG language. "
                    f"You MUST respond ONLY in {lang_name}. Do NOT use English."
                )
                retry = await self._call_llm_collect(enforced_system)
                retry_validated = _validate_response(retry, self.state.language)
                retry_lang = detect_language(retry_validated)
                if retry_lang == self.state.language:
                    logger.info("[LLM] Retry succeeded: response now in %s", self.state.language)
                    return retry_validated
                logger.warning("[LLM] Retry still wrong language -- using fallback")
                return _LANG_FALLBACKS.get(self.state.language, _RESPONSE_FALLBACK)

        return validated

    async def _call_llm_collect(self, system_prompt: str) -> str:
        """Stream LLM response and collect all chunks into a single string."""
        client = self._get_client()
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self.memory.get_history_for_llm())

        try:
            stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
                stream=True,
            )

            chunks: list[str] = []
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)

            reply = "".join(chunks).strip()
            logger.info("[LLM] Response: %r", reply)
            return reply

        except Exception as exc:
            logger.error("[LLM] Failed: %s", exc)
            return _LANG_FALLBACKS.get(self.state.language, _RESPONSE_FALLBACK)

    # ------------------------------------------------------------------
    # Prompt building + localization helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build stage-aware, role-aware system prompt with memory injection."""
        lang_name = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}.get(
            self.state.language, "English"
        )

        parts = [
            "You are an AI calling agent speaking on behalf of a company.",
            "Your goal is to help the user schedule or manage appointments.",
            f"You are currently in the {self.state.stage} stage of the conversation.",
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

        if self.state.language != "en":
            parts.append(
                f"\nIMPORTANT: The user is speaking {lang_name}. "
                f"You MUST respond ONLY in {lang_name}. "
                "Match the user's language exactly."
            )

        context_str = self.memory.context.to_prompt_string()
        if context_str:
            parts.append(f"\nKnown context about this caller:\n{context_str}")

        return "\n".join(parts)

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

    def _pick_response(self, options: list[str]) -> str:
        """Round-robin response selection to avoid repetition."""
        resp = options[self._response_counter % len(options)]
        self._response_counter += 1
        return resp
