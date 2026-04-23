"""
Pramukh Group — proactive outbound calling agent.

Drives conversation through a structured appointment booking flow:
  GREETING -> LOCATION -> PROPERTY_TYPE -> SCHEDULING -> CONFIRMATION -> CLOSING

The agent leads, the user responds. At any point, user questions are
answered and the flow resumes. Uses parallel LLM execution, filler
responses, and deterministic shortcuts for low latency.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from .memory import ConversationMemory, BookingData
from .stt.post_processor import match_location, match_bhk

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

STAGE_GREETING = "GREETING"
STAGE_LOCATION = "LOCATION"
STAGE_PROPERTY_TYPE = "PROPERTY_TYPE"
STAGE_SCHEDULING = "SCHEDULING"
STAGE_CONFIRMATION = "CONFIRMATION"
STAGE_CLOSING = "CLOSING"

_STAGE_ORDER = {
    STAGE_GREETING: 0,
    STAGE_LOCATION: 1,
    STAGE_PROPERTY_TYPE: 2,
    STAGE_SCHEDULING: 3,
    STAGE_CONFIRMATION: 4,
    STAGE_CLOSING: 5,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESPONSE_LENGTH = 200
LLM_TIMEOUT = 5.0
LLM_MAX_TOKENS = 80
FILLER_TIMEOUT_S = 1.5
LLM_STREAM_TIMEOUT_S = 4.0
INTENT_CLASSIFY_TIMEOUT_S = 1.0

logger = logging.getLogger(__name__)

_MARKDOWN_RE = re.compile(r"[*#`~_\[\]()>|]")
_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff\U00002702-\U000027b0\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff\U00002600-\U000026ff]+",
    re.UNICODE,
)


@dataclass
class EngineResult:
    text: str
    is_shortcut: bool
    intent: str


class ConversationState:
    def __init__(self) -> None:
        self.last_intent: str | None = None
        self.turn_count: int = 0
        self.is_closing: bool = False
        self.language: str = "en"
        self.language_history: list[str] = []
        self.stage: str = STAGE_GREETING
        self.lang_confidence: float = 1.0
        self.lang_source: str = "default"
        self.filler_used: bool = False
        self.llm_cancelled: bool = False
        self.parallel_execution: bool = False
        self.objection_handled: bool = False


# ---------------------------------------------------------------------------
# Available appointment slots (mock — replace with real calendar API)
# ---------------------------------------------------------------------------

_AVAILABLE_SLOTS = [
    {"day": "Tomorrow", "time": "10:00 AM"},
    {"day": "Tomorrow", "time": "2:00 PM"},
    {"day": "Day after tomorrow", "time": "11:00 AM"},
    {"day": "This Saturday", "time": "10:00 AM"},
    {"day": "This Saturday", "time": "3:00 PM"},
]

_SLOTS_EN = ", ".join(f"{s['day']} at {s['time']}" for s in _AVAILABLE_SLOTS[:3])
_SLOTS_HI = "कल सुबह 10 बजे, कल दोपहर 2 बजे, या परसों सुबह 11 बजे"
_SLOTS_GU = "કાલે સવારે 10 વાગે, કાલે બપોરે 2 વાગે, અથવા પરમ દિવસે સવારે 11 વાગે"


# ---------------------------------------------------------------------------
# Intent detection — domain-specific for Pramukh Group
# ---------------------------------------------------------------------------

_LOCATION_NAMES = {
    "surat", "vapi", "silvassa",
    "સુરત", "વાપી", "સિલવાસા",
    "सूरत", "वापी", "सिलवासा",
}

_OBJECTION_PATTERNS = [
    "not interested", "busy", "call later", "already using",
    "don't need", "no time", "don't want", "stop calling",
    "not looking", "no need", "wrong number", "remove my number",
]

_TIME_PATTERNS = [
    "morning", "afternoon", "evening", "tomorrow", "today",
    "this weekend", "saturday", "sunday", "next week",
    "10 am", "11 am", "2 pm", "3 pm",
    "subah", "dopahar", "shaam", "kal", "parso", "aaj",
    "savare", "bapore", "saanje", "kaale", "aaje",
    "सुबह", "दोपहर", "शाम", "कल", "परसों", "आज",
    "સવારે", "બપોરે", "સાંજે", "કાલે", "આજે",
]


def detect_intent(text: str, stage: str = STAGE_GREETING) -> str:
    """Classify user input into a domain intent. Fast and deterministic."""
    if not text or not text.strip():
        return "general"

    lower = text.lower().strip()

    # Objection — highest priority
    if any(p in lower for p in _OBJECTION_PATTERNS):
        return "objection"
    if any(w in lower for w in ["दिलचस्पी नहीं", "बिज़ी", "ज़रूरत नहीं",
                                  "રસ નથી", "વ્યસ્ત", "જરૂર નથી"]):
        return "objection"

    # Location selection
    loc = match_location(lower)
    if loc:
        return "location_selection"
    if any(w in lower for w in _LOCATION_NAMES):
        return "location_selection"

    # BHK / property type
    bhk = match_bhk(lower)
    if bhk:
        return "property_type"
    if re.search(r"\d\s*bhk|\d\s*bed", lower, re.IGNORECASE):
        return "property_type"

    # Time preference
    if any(p in lower for p in _TIME_PATTERNS):
        return "time_preference"

    # Affirmation
    if re.search(r"\b(yes|yeah|yep|sure|okay|ok|haan|ji|theek|ha|barobar)\b", lower):
        return "affirmation"
    if any(w in lower for w in ["हां", "हाँ", "जी", "ठीक", "હા", "બરાબર"]):
        return "affirmation"

    # Question — check before negation so कितना isn't caught by ना substring
    if lower.endswith("?") or re.search(r"\b(what|how|when|where|why|which|kitna|kab|kahan|kaun|kya|kyun)\b", lower):
        return "question"
    if any(w in lower for w in ["कौन", "क्या", "कैसे", "कब", "कहां", "कितना", "क्यों",
                                  "કોણ", "શું", "કેવી", "ક્યારે", "ક્યાં", "કેટલા", "કેમ"]):
        return "question"

    # Negation
    if re.search(r"\b(no|nope|nah|nahi|na)\b", lower):
        return "negation"
    if any(w in lower for w in ["नहीं", "ના", "નહીં"]):
        return "negation"
    if re.search(r"(?:^|\s)ना(?:\s|$)", lower):
        return "negation"

    # Farewell
    if re.search(r"\b(bye|goodbye|see you|alvida|aavjo)\b", lower):
        return "farewell"
    if any(w in lower for w in ["अलविदा", "फिर मिलते", "આવજો", "ફરી મળીશું"]):
        return "farewell"

    # Greeting
    if re.search(r"\b(hello|hi|hey|namaste|namaskar)\b", lower):
        return "greeting"
    if any(w in lower for w in ["नमस्ते", "નમસ્તે", "હેલો"]):
        return "greeting"

    # Gratitude
    if re.search(r"\b(thank|thanks|dhanyavaad|shukriya|aabhar)\b", lower):
        return "gratitude"
    if any(w in lower for w in ["धन्यवाद", "शुक्रिया", "આભાર", "थैंक", "થેંક"]):
        return "gratitude"

    return "general"


def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "en"
    if any("઀" <= c <= "૿" for c in text):
        return "gu"
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"
    return "en"


# ---------------------------------------------------------------------------
# Transliteration patterns for language detection v2
# ---------------------------------------------------------------------------

_HINDI_PATTERNS = [
    "hai", "tha", "thi", "hoga", "kya", "kyun", "kaise",
    "karna", "chahiye", "hona", "mujhe", "tum", "aap",
    "nahi", "accha", "theek", "namaste", "dhanyavaad",
    "shukriya", "haan", "dekhna", "dikhao", "samay",
    "subah", "dopahar", "shaam", "kal", "parso", "aaj",
]

_GUJARATI_PATTERNS = [
    "che", "chu", "hatu", "hase", "shu", "kem",
    "tame", "hu", "pan", "chhe", "tamne", "mane",
    "majama", "barobar", "saru", "savare", "bapore",
    "saanje", "aaje", "kaale", "joie", "joiye", "aavjo",
]

_LANG_CLASSIFY_PROMPT = (
    "Classify the dominant language of this text.\n"
    "Options: English, Hindi, Gujarati\n"
    "The input may be transliterated (romanized) or mixed-language.\n"
    'Text: "{text}"\n'
    "Return ONLY one word: English, Hindi, or Gujarati."
)

_LANG_NAME_TO_CODE = {"english": "en", "hindi": "hi", "gujarati": "gu"}


def _pattern_score(text: str, patterns: list[str]) -> int:
    words = set(text.split())
    return sum(1 for p in patterns if p in words)


# ---------------------------------------------------------------------------
# Objection responses (multilingual, Pramukh-specific)
# ---------------------------------------------------------------------------

_OBJECTION_RESPONSES: dict[str, list[str]] = {
    "en": [
        "I completely understand. Just so you know, Pramukh Group has some very attractive options right now. Would you like me to share some quick details?",
        "No problem at all. When would be a better time to discuss? We have some exciting projects coming up.",
    ],
    "hi": [
        "बिल्कुल समझ सकता हूँ। बस इतना बताना चाहता था कि Pramukh Group में अभी बहुत अच्छे विकल्प उपलब्ध हैं।",
        "कोई बात नहीं, कब बात करना सही रहेगा? कुछ बहुत अच्छे प्रोजेक्ट्स आ रहे हैं।",
    ],
    "gu": [
        "બિલકુલ સમજું છું. બસ એટલું કહેવું હતું કે Pramukh Group માં અત્યારે ખૂબ સારા વિકલ્પો છે.",
        "કોઈ વાત નહીં, પછી ક્યારે વાત કરીએ? કેટલાક ખૂબ સારા પ્રોજેક્ટ્સ આવી રહ્યા છે.",
    ],
}

# ---------------------------------------------------------------------------
# Filler responses (multilingual, Pramukh-specific)
# ---------------------------------------------------------------------------

_FILLER_RESPONSES: dict[str, list[str]] = {
    "en": ["Let me check the available options for you...", "One moment please..."],
    "hi": ["एक सेकंड, मैं उपलब्ध विकल्प देखता हूँ...", "बस एक मिनट..."],
    "gu": ["એક મિનિટ, ઉપલબ્ધ વિકલ્પો જોઈ લઉં...", "જોઈ લઉં..."],
}

# ---------------------------------------------------------------------------
# Timeout / error fallbacks
# ---------------------------------------------------------------------------

_TIMEOUT_FALLBACKS: dict[str, str] = {
    "en": "Let me get back to you on that.",
    "hi": "मैं इस पर वापस आता हूँ।",
    "gu": "હું આના પર પાછો આવું છું.",
}

_RESPONSE_FALLBACK = "Sorry, could you repeat that?"

_LANG_FALLBACKS: dict[str, str] = {
    "hi": "माफ़ कीजिए, क्या आप दोबारा बोल सकते हैं?",
    "gu": "માફ કરશો, શું તમે ફરીથી બોલી શકો?",
    "en": "Sorry, could you repeat that?",
}

# ---------------------------------------------------------------------------
# Localized responses for shortcut intents
# ---------------------------------------------------------------------------

_LOCALIZED_RESPONSES: dict[str, dict[str, list[str] | str]] = {
    "hi": {
        "greeting": ["नमस्ते! Pramukh Group से बोल रहा हूँ।", "नमस्ते! कैसे हैं आप?"],
        "farewell": "अच्छा, फिर मिलते हैं! Pramukh Group से कॉल के लिए धन्यवाद।",
        "gratitude": ["आपका स्वागत है!", "खुशी हुई!"],
        "closing": "बहुत अच्छा! हम आपको डिटेल्स भेज देंगे। धन्यवाद!",
    },
    "gu": {
        "greeting": ["નમસ્તે! Pramukh Group તરફથી બોલું છું.", "નમસ્તે! કેમ છો?"],
        "farewell": "સારું, ફરી મળીશું! Pramukh Group તરફથી કૉલ માટે આભાર.",
        "gratitude": ["આપનું સ્વાગત છે!", "ખુશી થઈ!"],
        "closing": "ખૂબ સરસ! અમે તમને ડિટેલ્સ મોકલીશું. આભાર!",
    },
}

# ---------------------------------------------------------------------------
# Proactive stage prompts — what the agent says to drive the flow forward
# ---------------------------------------------------------------------------

_PROACTIVE_PROMPTS: dict[str, dict[str, str]] = {
    STAGE_LOCATION: {
        "en": "Which location are you interested in? We have beautiful projects in Surat, Vapi, and Silvassa.",
        "hi": "आप किस लोकेशन में देख रहे हैं? हमारे Surat, Vapi और Silvassa में बहुत अच्छे प्रोजेक्ट्स हैं।",
        "gu": "તમે કઈ લોકેશનમાં જોઈ રહ્યા છો? અમારા Surat, Vapi અને Silvassa માં ખૂબ સરસ પ્રોજેક્ટ્સ છે.",
    },
    STAGE_PROPERTY_TYPE: {
        "en": "What type of property are you looking for? We have options in 2 BHK, 3 BHK, 4 BHK, and 5 BHK.",
        "hi": "आप किस तरह की प्रॉपर्टी ढूंढ रहे हैं? हमारे पास 2 BHK, 3 BHK, 4 BHK और 5 BHK के विकल्प हैं।",
        "gu": "તમે કેવા પ્રકારની પ્રોપર્ટી શોધી રહ્યા છો? અમારી પાસે 2 BHK, 3 BHK, 4 BHK અને 5 BHK ના વિકલ્પો છે.",
    },
    STAGE_SCHEDULING: {
        "en": f"When would you like to visit? We have slots available: {_SLOTS_EN}. Which works for you?",
        "hi": f"आप कब विजिट करना चाहेंगे? हमारे पास ये स्लॉट्स उपलब्ध हैं: {_SLOTS_HI}। कौन सा सही रहेगा?",
        "gu": f"તમે ક્યારે મુલાકાત લેવા માંગો છો? અમારી પાસે આ સ્લોટ્સ ઉપલબ્ધ છે: {_SLOTS_GU}. કયો યોગ્ય રહેશે?",
    },
}

# ---------------------------------------------------------------------------
# Opening greeting
# ---------------------------------------------------------------------------

_OPENING_GREETING: dict[str, str] = {
    "en": (
        "Hi! ... This is calling from Pramukh Group. "
        "You had shown interest in our properties, and I just wanted to have a quick chat about what you're looking for. "
        "Is this a good time to talk?"
    ),
    "hi": (
        "नमस्ते! ... मैं Pramukh Group से बोल रहा हूँ। "
        "आपने हमारी प्रॉपर्टीज में रुचि दिखाई थी, और मैं बस आपसे इसके बारे में थोड़ी बात करना चाहता था। "
        "क्या अभी बात कर सकते हैं?"
    ),
    "gu": (
        "નમસ્તે! ... હું Pramukh Group તરફથી બોલું છું. "
        "તમે અમારી પ્રોપર્ટીઝમાં રસ દર્શાવ્યો હતો, અને હું બસ થોડી વાત કરવા માંગતો હતો. "
        "શું અત્યારે વાત કરી શકીએ?"
    ),
}


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


def _validate_response(text: str, expected_lang: str = "en") -> str:
    if not text or not text.strip():
        return _RESPONSE_FALLBACK
    cleaned = _MARKDOWN_RE.sub("", text)
    cleaned = _EMOJI_RE.sub("", cleaned).strip()
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


# ===================================================================
# ConversationRouter — Pramukh Group Proactive Agent
# ===================================================================


class ConversationRouter:
    """
    Proactive outbound calling agent for Pramukh Group.

    Drives the conversation through: GREETING -> LOCATION -> PROPERTY_TYPE
    -> SCHEDULING -> CONFIRMATION -> CLOSING.

    Extracts structured booking data at each stage. Falls back to LLM for
    natural responses and question handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = LLM_TIMEOUT,
        default_language: str = "en",
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None
        self.memory = ConversationMemory()
        self.state = ConversationState()
        self.state.language = default_language
        self._response_counter = 0
        self._lang_cache: dict[str, tuple[str, float, str]] = {}
        self._filler_counter = 0
        self._speculative_yes: asyncio.Task | None = None
        self._speculative_no: asyncio.Task | None = None
        self._last_nudge_stage: str | None = None

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
    # Opening greeting (outbound call starts here)
    # ------------------------------------------------------------------

    async def generate_opening(self) -> EngineResult:
        """Generate the proactive opening greeting. Called once at pipeline start."""
        lang = self.state.language
        greeting = _OPENING_GREETING.get(lang, _OPENING_GREETING["en"])
        self.memory.add_assistant(greeting)
        self.state.stage = STAGE_GREETING
        logger.info("[OPENING] Generated greeting, staying at GREETING stage")
        return EngineResult(text=greeting, is_shortcut=True, intent="greeting")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, text: str) -> EngineResult:
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
        Proactive streaming conversation engine.

        Extracts structured data, advances the booking flow, and falls back
        to LLM for questions and natural phrasing.
        """
        self.memory.add_user(text)

        # --- Language detection + stabilization ---
        detected_lang, conf, src = await self.detect_language_v3(text)
        self.state.turn_count += 1
        logger.info('[LANG] "%s" -> %s (src=%s, conf=%.2f)', text, detected_lang, src, conf)

        self.state.language_history.append(detected_lang)
        history = self.state.language_history[-5:]
        self.state.language_history = history

        counts: dict[str, int] = {}
        for h in history:
            counts[h] = counts.get(h, 0) + 1
        max_count = max(counts.values())
        candidates = [l for l, c in counts.items() if c == max_count]
        stabilized_lang = candidates[0] if len(candidates) == 1 else (
            self.state.language if self.state.language in candidates else candidates[0]
        )

        # High-confidence detection (script/pattern) overrides stabilization
        # so the response always matches the language the user just spoke.
        # Low-confidence detection (pure ASCII "default") preserves the
        # CURRENT language to maintain consistency (e.g. "yes" keeps Hindi
        # if the conversation has been in Hindi).
        prev_lang = self.state.language
        if src in ("script", "pattern") and conf >= 0.7:
            self.state.language = detected_lang
        elif src == "default" and prev_lang != "en":
            self.state.language = prev_lang
        else:
            self.state.language = stabilized_lang

        self.state.lang_confidence = conf
        self.state.lang_source = src
        logger.info('[LANG] Response language: %s (detected=%s, stabilized=%s)',
                     self.state.language, detected_lang, stabilized_lang)

        # Reset per-turn metrics
        self.state.filler_used = False
        self.state.llm_cancelled = False
        self.state.parallel_execution = True
        self.state.objection_handled = False

        # --- Intent detection ---
        intent = detect_intent(text, self.state.stage)

        # --- Fire LLM in background (parallel with deterministic routing) ---
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

        self.state.last_intent = intent
        logger.info('[INTENT] "%s" -> %s (stage=%s)', text, intent, self.state.stage)

        # =============================================================
        # STAGE-AWARE ROUTING
        # =============================================================

        # --- Farewell at any stage ---
        if intent == "farewell":
            _cancel_llm()
            self.state.is_closing = True
            resp = self._localized("closing", "Thank you for your time! We'll be in touch. Have a great day!")
            self.memory.add_assistant(resp)
            self.state.stage = STAGE_CLOSING
            yield EngineResult(text=resp, is_shortcut=True, intent="farewell")
            return

        # --- Objection at any stage (except GREETING, which has its own handler) ---
        if intent == "objection" and self.state.stage != STAGE_GREETING:
            _cancel_llm()
            self.state.objection_handled = True
            resp = self._pick_objection_response()
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="objection")
            return

        # --- Gratitude at any stage ---
        if intent == "gratitude":
            _cancel_llm()
            resp = self._localized_list("gratitude", ["You're welcome!", "Happy to help!"])
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="gratitude")
            return

        # =============================================================
        # GREETING STAGE — user speaks first, we respond accordingly
        # =============================================================
        if self.state.stage == STAGE_GREETING:
            _cancel_llm()

            # User opens with a greeting → respond with our opening + ask location
            if intent == "greeting" or intent == "affirmation":
                lang = self.state.language
                greeting = _OPENING_GREETING.get(lang, _OPENING_GREETING["en"])
                self.memory.add_assistant(greeting)
                self.state.stage = STAGE_LOCATION
                yield EngineResult(text=greeting, is_shortcut=True, intent="greeting")
                return

            if intent == "negation" or intent == "objection":
                lang = self.state.language
                if lang == "hi":
                    resp = "कोई बात नहीं! कब बात करना सही रहेगा?"
                elif lang == "gu":
                    resp = "કોઈ વાત નહીં! ક્યારે વાત કરવી યોગ્ય રહેશે?"
                else:
                    resp = "No problem at all! When would be a better time to talk?"
                self.memory.add_assistant(resp)
                self.state.stage = STAGE_CLOSING
                yield EngineResult(text=resp, is_shortcut=True, intent="reschedule")
                return

            # User starts with a direct intent (location, property, etc.)
            # Skip greeting entirely — advance to the right stage and process
            if intent == "location_selection":
                location = self._extract_location(text)
                if location:
                    self.memory.booking.location = location
                    self.state.stage = STAGE_PROPERTY_TYPE
                    next_prompt = self._get_proactive_prompt(STAGE_PROPERTY_TYPE)
                    resp = self._location_ack(location) + " " + next_prompt
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="location_selection")
                    return

            if intent == "property_type":
                bhk = self._extract_bhk(text)
                if bhk:
                    self.memory.booking.property_type = bhk.upper()
                    self.state.stage = STAGE_LOCATION
                    next_prompt = self._get_proactive_prompt(STAGE_LOCATION)
                    resp = self._bhk_ack(bhk, "your area") + " " + next_prompt
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="property_type")
                    return

            # Any other input at greeting — respond with opening and move on
            lang = self.state.language
            greeting = _OPENING_GREETING.get(lang, _OPENING_GREETING["en"])
            self.memory.add_assistant(greeting)
            self.state.stage = STAGE_LOCATION
            yield EngineResult(text=greeting, is_shortcut=True, intent="greeting")
            return

        # =============================================================
        # LOCATION STAGE
        # =============================================================
        if self.state.stage == STAGE_LOCATION:
            if intent == "location_selection":
                _cancel_llm()
                location = self._extract_location(text)
                if location:
                    self.memory.booking.location = location
                    self.state.stage = STAGE_PROPERTY_TYPE
                    next_prompt = self._get_proactive_prompt(STAGE_PROPERTY_TYPE)
                    resp = self._location_ack(location) + " " + next_prompt
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="location_selection")
                    return

            # User said something else — might be a greeting or question
            if intent == "greeting":
                _cancel_llm()
                resp = self._localized_list("greeting", ["Hello!", "Hi there!"])
                resp += " " + self._get_proactive_prompt(STAGE_LOCATION)
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="greeting")
                return

            if intent == "affirmation":
                _cancel_llm()
                resp = self._get_proactive_prompt(STAGE_LOCATION)
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="affirmation")
                return

        # =============================================================
        # PROPERTY TYPE STAGE
        # =============================================================
        if self.state.stage == STAGE_PROPERTY_TYPE:
            # Always try BHK extraction regardless of detected intent —
            # user may say "haan mujhe 4 bhk chahiye" which detects as affirmation
            bhk = self._extract_bhk(text)
            if bhk:
                _cancel_llm()
                self.memory.booking.property_type = bhk.upper()
                self.state.stage = STAGE_SCHEDULING
                loc = self.memory.booking.location or "your chosen location"
                next_prompt = self._get_proactive_prompt(STAGE_SCHEDULING)
                resp = self._bhk_ack(bhk, loc) + " " + next_prompt
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="property_type")
                return

            if intent == "location_selection":
                _cancel_llm()
                location = self._extract_location(text)
                if location:
                    self.memory.booking.location = location
                resp = self._get_proactive_prompt(STAGE_PROPERTY_TYPE)
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="location_update")
                return

        # =============================================================
        # SCHEDULING STAGE
        # =============================================================
        if self.state.stage == STAGE_SCHEDULING:
            if intent in ("time_preference", "affirmation", "general"):
                time_str = self._extract_time_preference(text)
                if time_str:
                    _cancel_llm()
                    self.memory.booking.appointment_time = time_str
                    self.state.stage = STAGE_CONFIRMATION
                    resp = self._build_confirmation_prompt()
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="time_preference")
                    return
                if intent == "general" and not time_str:
                    _cancel_llm()
                    resp = self._get_proactive_prompt(STAGE_SCHEDULING)
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="scheduling_reprompt")
                    return

            if intent == "property_type":
                _cancel_llm()
                bhk = self._extract_bhk(text)
                if bhk:
                    self.memory.booking.property_type = bhk.upper()
                resp = self._get_proactive_prompt(STAGE_SCHEDULING)
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="property_update")
                return

        # =============================================================
        # CONFIRMATION STAGE
        # =============================================================
        if self.state.stage == STAGE_CONFIRMATION:
            if intent == "affirmation":
                _cancel_llm()
                self.memory.booking.confirmed = True
                self.state.stage = STAGE_CLOSING
                self.state.is_closing = True
                resp = self._build_closing_message()
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="confirmed")
                return

            if intent == "negation":
                _cancel_llm()
                self.state.stage = STAGE_SCHEDULING
                lang = self.state.language
                if lang == "hi":
                    resp = "कोई बात नहीं। कौन सा समय आपके लिए बेहतर होगा?"
                elif lang == "gu":
                    resp = "કોઈ વાત નહીં. કયો સમય તમારા માટે વધુ સારો હશે?"
                else:
                    resp = "No problem. Which time would work better for you?"
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="reschedule")
                return

        # =============================================================
        # CLOSING STAGE — allow re-engagement for new intents
        # =============================================================
        if self.state.stage == STAGE_CLOSING:
            # User wants to re-engage with a location or property
            if intent == "location_selection":
                _cancel_llm()
                location = self._extract_location(text)
                if location:
                    self.memory.booking.location = location
                    self.state.stage = STAGE_PROPERTY_TYPE
                    next_prompt = self._get_proactive_prompt(STAGE_PROPERTY_TYPE)
                    resp = self._location_ack(location) + " " + next_prompt
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="location_selection")
                    return

            if intent == "property_type":
                _cancel_llm()
                bhk = self._extract_bhk(text)
                if bhk:
                    self.memory.booking.property_type = bhk.upper()
                    if self.memory.booking.location:
                        self.state.stage = STAGE_SCHEDULING
                        next_prompt = self._get_proactive_prompt(STAGE_SCHEDULING)
                        loc = self.memory.booking.location
                        resp = self._bhk_ack(bhk, loc) + " " + next_prompt
                    else:
                        self.state.stage = STAGE_LOCATION
                        next_prompt = self._get_proactive_prompt(STAGE_LOCATION)
                        resp = self._bhk_ack(bhk, "your area") + " " + next_prompt
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="property_type")
                    return

            # User asks a question — answer via LLM, don't just dismiss
            if intent == "question":
                async for chunk in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=False,
                ):
                    yield chunk
                return

            # Default closing response for farewell, affirmation, general
            _cancel_llm()
            resp = self._localized("closing",
                                   "Thank you for your interest in Pramukh Group! Have a wonderful day!")
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="closing")
            return

        # =============================================================
        # QUESTION HANDLING — user asks something off-flow
        # =============================================================
        if intent == "question":
            # Let LLM handle the question, then nudge back to flow
            self.state.llm_cancelled = False
            async for result in self._consume_llm_with_filler(
                llm_queue, llm_done, intent, nudge_back=True
            ):
                yield result
            return

        # =============================================================
        # GENERAL / FALLBACK — use LLM with domain context
        # =============================================================
        self.state.llm_cancelled = False
        async for result in self._consume_llm_with_filler(
            llm_queue, llm_done, intent, nudge_back=True
        ):
            yield result

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    def _extract_location(self, text: str) -> str | None:
        loc = match_location(text)
        if loc:
            return loc.capitalize()
        lower = text.lower()
        for city in ["surat", "vapi", "silvassa"]:
            if city in lower:
                return city.capitalize()
        for hi, en in [("सूरत", "Surat"), ("वापी", "Vapi"), ("सिलवासा", "Silvassa")]:
            if hi in text:
                return en
        for gu, en in [("સુરત", "Surat"), ("વાપી", "Vapi"), ("સિલવાસા", "Silvassa")]:
            if gu in text:
                return en
        return None

    def _extract_bhk(self, text: str) -> str | None:
        bhk = match_bhk(text)
        if bhk:
            return bhk
        m = re.search(r"(\d)\s*(?:bhk|bed)", text, re.IGNORECASE)
        if m and 1 <= int(m.group(1)) <= 5:
            return f"{m.group(1)} bhk"
        return None

    def _extract_time_preference(self, text: str) -> str | None:
        lower = text.lower()
        for slot in _AVAILABLE_SLOTS:
            day_l = slot["day"].lower()
            time_l = slot["time"].lower()
            if day_l in lower or time_l in lower:
                return f"{slot['day']} at {slot['time']}"
        for keyword in ["morning", "subah", "savare", "सुबह", "સવારે"]:
            if keyword in lower:
                return "Tomorrow at 10:00 AM"
        for keyword in ["afternoon", "dopahar", "bapore", "दोपहर", "બપોરે"]:
            if keyword in lower:
                return "Tomorrow at 2:00 PM"
        for keyword in ["evening", "shaam", "saanje", "शाम", "સાંજે"]:
            if keyword in lower:
                return "Tomorrow at 2:00 PM"
        if re.search(r"\b(tomorrow|kal|kaale|कल|કાલે)\b", lower):
            return "Tomorrow at 10:00 AM"
        if re.search(r"\b(saturday|weekend|this saturday)\b", lower):
            return "This Saturday at 10:00 AM"
        # For affirmation in scheduling context, pick the first available slot
        if re.search(r"\b(yes|ok|sure|haan|ha|ji|theek|first|pehla)\b", lower):
            return f"{_AVAILABLE_SLOTS[0]['day']} at {_AVAILABLE_SLOTS[0]['time']}"
        return None

    # ------------------------------------------------------------------
    # Acknowledgment builders
    # ------------------------------------------------------------------

    def _location_ack(self, location: str) -> str:
        lang = self.state.language
        if lang == "hi":
            return f"बढ़िया, {location} एक बहुत अच्छी लोकेशन है!"
        elif lang == "gu":
            return f"સરસ, {location} ખૂબ સારી લોકેશન છે!"
        return f"Excellent choice! {location} is a wonderful location."

    def _bhk_ack(self, bhk: str, location: str) -> str:
        lang = self.state.language
        bhk_upper = bhk.upper()
        if lang == "hi":
            return f"बढ़िया! {location} में {bhk_upper} के लिए हमारे पास शानदार विकल्प हैं।"
        elif lang == "gu":
            return f"સરસ! {location} માં {bhk_upper} માટે અમારી પાસે ઉત્તમ વિકલ્પો છે."
        return f"Great! We have some fantastic {bhk_upper} options in {location}."

    def _build_confirmation_prompt(self) -> str:
        b = self.memory.booking
        lang = self.state.language
        if lang == "hi":
            return (
                f"तो मैं नोट कर लेता हूँ -- {b.location or 'आपकी लोकेशन'} में "
                f"{b.property_type or 'प्रॉपर्टी'} के लिए "
                f"{b.appointment_time or 'कल'} को साइट विजिट। "
                "क्या ये सही है?"
            )
        elif lang == "gu":
            return (
                f"તો હું નોંધ કરી લઉં -- {b.location or 'તમારી લોકેશન'} માં "
                f"{b.property_type or 'પ્રોપર્ટી'} માટે "
                f"{b.appointment_time or 'કાલે'} ની સાઇટ મુલાકાત. "
                "આ બરાબર છે?"
            )
        return (
            f"Let me confirm -- a site visit for {b.property_type or 'your property'} "
            f"in {b.location or 'your chosen location'}, "
            f"scheduled for {b.appointment_time or 'tomorrow'}. "
            "Does that work for you?"
        )

    def _build_closing_message(self) -> str:
        b = self.memory.booking
        lang = self.state.language
        if lang == "hi":
            return (
                f"बहुत अच्छा! आपकी {b.location} में {b.property_type} की "
                f"साइट विजिट {b.appointment_time} के लिए बुक हो गई है। "
                "हम आपको एक कन्फर्मेशन मैसेज भेजेंगे। Pramukh Group चुनने के लिए धन्यवाद!"
            )
        elif lang == "gu":
            return (
                f"ખૂબ સરસ! તમારી {b.location} માં {b.property_type} ની "
                f"સાઇટ મુલાકાત {b.appointment_time} માટે બુક થઈ ગઈ છે. "
                "અમે તમને કન્ફર્મેશન મેસેજ મોકલીશું. Pramukh Group પસંદ કરવા માટે આભાર!"
            )
        return (
            f"Wonderful! Your site visit for {b.property_type} in {b.location} "
            f"is booked for {b.appointment_time}. "
            "We'll send you a confirmation message. Thank you for choosing Pramukh Group!"
        )

    def _get_proactive_prompt(self, stage: str) -> str:
        lang = self.state.language
        prompts = _PROACTIVE_PROMPTS.get(stage, {})
        return prompts.get(lang, prompts.get("en", ""))

    # ------------------------------------------------------------------
    # LLM consumption with filler support
    # ------------------------------------------------------------------

    async def _consume_llm_with_filler(
        self,
        llm_queue: asyncio.Queue,
        llm_done: asyncio.Event,
        intent: str,
        nudge_back: bool = False,
    ):
        """Consume LLM queue with filler timeout. Optionally nudge back to flow."""
        full_parts: list[str] = []

        first_chunk = None
        try:
            first_chunk = await asyncio.wait_for(
                llm_queue.get(), timeout=FILLER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            filler = self._pick_filler()
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
            full_text = " ".join(full_parts)
            self.memory.add_assistant(full_text)

            # Nudge back to the booking flow
            if nudge_back:
                nudge = self._build_flow_nudge()
                if nudge:
                    self.memory.add_assistant(nudge)
                    yield EngineResult(text=nudge, is_shortcut=True, intent="nudge")

    def _build_flow_nudge(self) -> str | None:
        """Build a gentle nudge to return to the booking flow after a detour. Won't repeat the same nudge."""
        missing = self.memory.booking.next_missing_field()
        if not missing:
            return None

        stage_map = {
            "location": STAGE_LOCATION,
            "property_type": STAGE_PROPERTY_TYPE,
            "appointment": STAGE_SCHEDULING,
        }
        target_stage = stage_map.get(missing)
        if not target_stage:
            return None

        if self._last_nudge_stage == target_stage:
            return None

        self._last_nudge_stage = target_stage
        return self._get_proactive_prompt(target_stage)

    # ------------------------------------------------------------------
    # Parallel LLM helpers
    # ------------------------------------------------------------------

    async def _run_llm_to_queue(
        self, text: str, queue: asyncio.Queue, done: asyncio.Event,
    ) -> None:
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
    # System prompt — Pramukh Group domain
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        b = self.memory.booking
        lang_name = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}.get(
            self.state.language, "English"
        )

        parts = [
            "You are a friendly calling assistant from Pramukh Group, a trusted real estate developer in Gujarat, India.",
            "You are making an outbound call to someone who showed interest in Pramukh properties.",
            "Your goal is to help them find their perfect property and schedule a site visit.",
            "",
            f"CURRENT BOOKING STATE:",
            f"- Stage: {self.state.stage}",
            f"- Location: {b.location or 'not selected yet'}",
            f"- Property type: {b.property_type or 'not selected yet'}",
            f"- Appointment: {b.appointment_time or 'not scheduled yet'}",
            "",
            "LOCATIONS: Surat, Vapi, Silvassa (Gujarat, India)",
            "PROPERTY TYPES: 2 BHK, 3 BHK, 4 BHK, 5 BHK",
            "",
            "CRITICAL RULES:",
            "- YOU lead the conversation. Be proactive. Always guide toward the next step.",
            "- Respond in 1-2 sentences MAXIMUM. You are on a phone call.",
            "- Sound natural: use contractions, be warm and conversational.",
            "- If the user asks a question you can answer (about Pramukh properties, locations, BHK options), answer briefly and return to the booking flow.",
            '- If you CANNOT answer a question, say "I\'ll have our team get back to you on that" and continue the flow.',
            "- No bullet points, lists, markdown, or emojis.",
            "- Be polite, respectful, and proactive. Respect the user's time.",
            "- Use formal address (aap/tamne) in Hindi/Gujarati.",
        ]

        if self.state.language == "hi":
            parts.append(
                "\nLANGUAGE RULE (MANDATORY): The user is speaking Hindi. "
                "You MUST respond ENTIRELY in Hindi (Devanagari script or transliterated Hindi). "
                "Do NOT mix English words into your response. Do NOT respond in English. "
                "Use natural Hindi as spoken in Gujarat/India."
            )
        elif self.state.language == "gu":
            parts.append(
                "\nLANGUAGE RULE (MANDATORY): The user is speaking Gujarati. "
                "You MUST respond ENTIRELY in Gujarati (Gujarati script or transliterated Gujarati). "
                "Do NOT mix English or Hindi into your response. Do NOT respond in English. "
                "Use natural Gujarati as spoken in Gujarat."
            )

        context_str = self.memory.context.to_prompt_string()
        if context_str:
            parts.append(f"\nKnown context:\n{context_str}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    async def detect_language_v3(self, text: str) -> tuple[str, float, str]:
        text_lower = text.lower().strip()
        if not text_lower:
            return "en", 1.0, "default"

        if text_lower in self._lang_cache:
            return self._lang_cache[text_lower]

        if any("઀" <= c <= "૿" for c in text_lower):
            result = ("gu", 1.0, "script")
        elif any("ऀ" <= c <= "ॿ" for c in text_lower):
            result = ("hi", 1.0, "script")
        else:
            hi_score = _pattern_score(text_lower, _HINDI_PATTERNS)
            gu_score = _pattern_score(text_lower, _GUJARATI_PATTERNS)
            total = hi_score + gu_score

            if total > 0:
                confidence = abs(hi_score - gu_score) / total
                lang = "hi" if hi_score >= gu_score else "gu"
                result = (lang, max(confidence, 0.5), "pattern")
            elif text_lower.isascii():
                result = ("en", 1.0, "default")
            else:
                lang = await self._classify_language_llm(text_lower)
                result = (lang, 0.9, "llm")

        self._lang_cache[text_lower] = result
        return result

    async def _classify_language_llm(self, text: str) -> str:
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
            return _LANG_NAME_TO_CODE.get(raw, "en")
        except Exception as exc:
            logger.warning("[LANG] LLM classification failed: %s", exc)
            return "en"

    # ------------------------------------------------------------------
    # Objection + filler helpers
    # ------------------------------------------------------------------

    def _pick_objection_response(self) -> str:
        lang = self.state.language
        responses = _OBJECTION_RESPONSES.get(lang, _OBJECTION_RESPONSES["en"])
        resp = responses[self._response_counter % len(responses)]
        self._response_counter += 1
        return resp

    def _pick_filler(self) -> str:
        lang = self.state.language
        fillers = _FILLER_RESPONSES.get(lang, _FILLER_RESPONSES["en"])
        resp = fillers[self._filler_counter % len(fillers)]
        self._filler_counter += 1
        return resp

    # ------------------------------------------------------------------
    # Localization helpers
    # ------------------------------------------------------------------

    def _localized(self, key: str, fallback: str) -> str:
        loc = _LOCALIZED_RESPONSES.get(self.state.language, {})
        val = loc.get(key)
        if val is None:
            return fallback
        if isinstance(val, list):
            return val[self._response_counter % len(val)]
        return val

    def _localized_list(self, key: str, en_list: list[str]) -> str:
        loc = _LOCALIZED_RESPONSES.get(self.state.language, {})
        val = loc.get(key)
        if isinstance(val, list):
            resp = val[self._response_counter % len(val)]
            self._response_counter += 1
            return resp
        resp = en_list[self._response_counter % len(en_list)]
        self._response_counter += 1
        return resp
