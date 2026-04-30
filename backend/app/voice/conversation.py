"""
Pramukh Group — proactive outbound calling agent.

Drives conversation through a structured appointment booking flow:
  GREETING -> INTEREST_CONFIRM -> (INTEREST_RECONFIRM) -> LOCATION
  -> PROPERTY_TYPE -> SCHEDULING -> CONFIRMATION -> CLOSING

Agent goes first with bilingual greeting ("Hello, Namaste!").
User's language is detected from their response and mirrored.
Interest is confirmed before advancing to the booking flow.
Uses parallel LLM execution, filler responses, and deterministic
shortcuts for low latency.
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
# NAME_CONFIRM is a structural placeholder — the user's first reply to the
# greeting is silently captured and we advance straight to PROJECT_CONFIRM.
# Replace the silent passthrough with an explicit name-ask once the wording
# is finalised by the business.
STAGE_NAME_CONFIRM = "NAME_CONFIRM"
STAGE_PROJECT_CONFIRM = "PROJECT_CONFIRM"
STAGE_PROPERTY_STATUS = "PROPERTY_STATUS"
STAGE_RENT_OR_BUY = "RENT_OR_BUY"
STAGE_PROPERTY_INFO = "PROPERTY_INFO"
STAGE_INTEREST_CONFIRM = "INTEREST_CONFIRM"
STAGE_INTEREST_RECONFIRM = "INTEREST_RECONFIRM"
STAGE_LOCATION = "LOCATION"
STAGE_PROPERTY_TYPE = "PROPERTY_TYPE"
STAGE_SCHEDULING = "SCHEDULING"
STAGE_CONFIRMATION = "CONFIRMATION"
STAGE_CLOSING = "CLOSING"

_STAGE_ORDER = {
    STAGE_GREETING: 0,
    STAGE_NAME_CONFIRM: 1,
    STAGE_PROJECT_CONFIRM: 2,
    STAGE_PROPERTY_STATUS: 3,
    STAGE_RENT_OR_BUY: 4,
    STAGE_PROPERTY_INFO: 5,
    STAGE_INTEREST_CONFIRM: 6,
    STAGE_INTEREST_RECONFIRM: 7,
    STAGE_LOCATION: 8,
    STAGE_PROPERTY_TYPE: 9,
    STAGE_SCHEDULING: 10,
    STAGE_CONFIRMATION: 11,
    STAGE_CLOSING: 12,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESPONSE_LENGTH = 350
LLM_TIMEOUT = 5.0
LLM_MAX_TOKENS = 150
FILLER_TIMEOUT_S = 1.5
LLM_STREAM_TIMEOUT_S = 7.0
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
        self.language: str = "hi"
        self.language_history: list[str] = []
        self.stage: str = STAGE_GREETING
        self.lang_confidence: float = 1.0
        self.lang_source: str = "default"
        self.filler_used: bool = False
        self.llm_cancelled: bool = False
        self.parallel_execution: bool = False
        self.objection_handled: bool = False
        self.language_locked: bool = False
        self.language_lock_turns: int = 0
        self.decline_count: int = 0


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
    # "later" / "not now" — deferral is an objection, treat it kindly
    "talk later", "later please", "maybe later", "some other time",
    "another time", "right now", "not right now",
    "baad mein", "baad me", "baadme", "baad m",
    "phir baat", "phir karte", "phir baat karenge",
    "abhi nahi", "abhi nai", "abhi busy",
    "fursat",
    "pachhi", "pachhi vaat",
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
    # Deferral patterns — "later / not now" in Devanagari and Gujarati script.
    # Treated as objection so we back off gracefully instead of pushing.
    if any(w in text for w in [
        "बाद में", "बाद मे", "बादमें", "बादमे",
        "अभी नहीं", "अभी नही", "अभी बिज़ी", "अभी busy",
        "फिर बात", "फिर बाद",
        "પછી", "પછીથી", "પછી વાત", "અત્યારે નહીં", "અત્યારે નથી",
    ]):
        return "objection"

    # Language switch request — highest priority after objection.
    # Catches romanized, Devanagari, and Gujarati script requests.
    _lang_switch_phrases = [
        "hindi me", "hindi mai", "hindi mein", "hindi boliye", "hindi bolo",
        "hindi me baat", "hindi mai baat", "hindi mein baat",
        "english me", "english mai", "english mein", "english boliye", "english bolo",
        "english me baat", "english mai baat", "english mein baat",
        "gujarati ma", "gujarati mai", "gujarati mein", "gujarati boliye", "gujarati bolo",
        "gujarati ma vaat", "gujarati mai baat",
        "speak hindi", "speak english", "speak gujarati",
        "talk in hindi", "talk in english", "talk in gujarati",
        "speak in hindi", "speak in english", "speak in gujarati",
        "switch to hindi", "switch to english", "switch to gujarati",
        "in english", "in hindi", "in gujarati",
        # Common short asks — "english please", "english only", etc.
        "english please", "hindi please", "gujarati please",
        "only english", "only hindi", "only gujarati",
        "english only", "hindi only", "gujarati only",
        "can you speak english", "can you speak hindi", "can you speak gujarati",
        "can you talk english", "can you talk hindi", "can you talk gujarati",
        "could you speak english", "could you speak hindi", "could you speak gujarati",
        "talk english", "talk hindi", "talk gujarati",
    ]
    if any(p in lower for p in _lang_switch_phrases):
        return "language_switch"
    # Devanagari script — all spellings of language names + में/बोलो/बात
    _lang_switch_devanagari = [
        "हिंदी में", "हिन्दी में", "हिंदी बोल", "हिन्दी बोल",
        "हिंदी में बात", "हिन्दी में बात",
        "अंग्रेज़ी में", "अंग्रेजी में", "इंग्लिश में",
        "अंग्रेज़ी बोल", "अंग्रेजी बोल", "इंग्लिश बोल",
        "अंग्रेज़ी में बात", "अंग्रेजी में बात", "इंग्लिश में बात",
        "गुजराती में", "गुजराती बोल", "गुजराती में बात",
    ]
    if any(p in text for p in _lang_switch_devanagari):
        return "language_switch"
    # Gujarati script
    if any(w in text for w in [
        "ગુજરાતીમાં", "અંગ્રેજીમાં", "હિન્દીમાં",
        "ઇંગ્લિશમાં", "ગુજરાતી બોલ", "અંગ્રેજી બોલ",
        "હિન્દી બોલ", "ઇંગ્લિશ બોલ",
    ]):
        return "language_switch"
    # Catch-all: any mention of a language name with "baat/bol/bolo" nearby
    if re.search(r"(?:english|hindi|gujarati|इंग्लिश|अंग्रेज़ी|अंग्रेजी|हिंदी|हिन्दी|गुजराती|ગુજરાતી|અંગ્રેજી|હિન્દી|ઇંગ્લિશ).{0,15}(?:बात|बोल|bol|baat|bolo|boliye|વાત|બોલ)", lower):
        return "language_switch"
    if re.search(r"(?:बात|बोल|bol|baat|bolo|boliye|વાત|બોલ).{0,15}(?:english|hindi|gujarati|इंग्लिश|अंग्रेज़ी|अंग्रेजी|हिंदी|हिन्दी|गुजराती|ગુજરાતી|અંગ્રેજી|હિન્દી|ઇંગ્લિશ)", lower):
        return "language_switch"

    # Two language names co-mentioned within a 30-char window — strong signal
    # the user is talking about switching ("english instead of hindi", "switch
    # from hindi to english", "hindi or english", etc.).
    if re.search(
        r"(english|hindi|gujarati).{1,30}(english|hindi|gujarati)",
        lower,
    ):
        return "language_switch"

    # Negation / restriction patterns — user implicitly asks to switch by
    # saying "I don't know X" / "X नहीं आती" / "સિર્ફ Y" / "only Y". These
    # must be classified as language_switch so the deterministic handler
    # responds (instead of the LLM, which sometimes hallucinates refusals
    # like "मैं हिंदी में ही बात करूँगा"). The destination language is
    # resolved by _extract_requested_language's last-position fallback.
    _lang_neg_lat = r"(?:english|hindi|gujarati|inglish|angrezi)"
    _lang_neg_dev = r"(?:इंग्लिश|अंग्रेज़ी|अंग्रेजी|हिंदी|हिन्दी|गुजराती)"
    _lang_neg_guj = r"(?:ગુજરાતી|અંગ્રેજી|હિન્દી|ઇંગ્લિશ)"
    _lang_neg_any = f"(?:{_lang_neg_lat[3:-1]}|{_lang_neg_dev[3:-1]}|{_lang_neg_guj[3:-1]})"
    # English: "don't know/speak X", "can't speak X", "only know/speak X"
    if re.search(
        rf"(?:don'?t|do not|can'?t|cannot|only|just)\s+(?:know|speak|talk|understand)\s+{_lang_neg_lat}",
        lower,
    ):
        return "language_switch"
    # Negation word near a language name (either order, ~25 char window).
    # Covers Devanagari नहीं, Gujarati નથી, romanized nahi/nahin.
    _neg_words = r"(?:नहीं|नहि|नही|न्ही|નથી|નહિ|નહી|નહીં|nahi|nahin|nai)"
    if re.search(rf"{_lang_neg_any}[^\n]{{0,25}}{_neg_words}", text, re.IGNORECASE):
        return "language_switch"
    if re.search(rf"{_neg_words}[^\n]{{0,25}}{_lang_neg_any}", text, re.IGNORECASE):
        return "language_switch"
    # Restriction word ("only" in various scripts) near a language name.
    _only_words = r"(?:सिर्फ|केवल|खाली|बस|ફક્ત|માત્ર|સિર્ફ|બસ|ખાલી|sirf|sirff|fakt|fakta|kewal|khali)"
    if re.search(rf"{_only_words}[^\n]{{0,20}}{_lang_neg_any}", text, re.IGNORECASE):
        return "language_switch"
    if re.search(rf"{_lang_neg_any}[^\n]{{0,20}}{_only_words}", text, re.IGNORECASE):
        return "language_switch"

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

    # Identity question — "who is this", "kon bol raha hai", etc.
    # Highest priority among questions — gets a fast deterministic response
    _identity_phrases = [
        "who is this", "who are you", "who is calling", "who is speaking",
        "who am i speaking", "who am i talking",
        "which company", "what company", "where are you calling from",
        "where you calling from", "kahan se", "kis taraf se",
        "kon hai", "kaun hai", "kon bol", "kaun bol", "kon baat", "kaun baat",
        "aap kon", "aap kaun", "ap kon", "ap kaun", "tum kon", "tum kaun",
        "ye kon", "ye kaun", "yeh kon", "yeh kaun",
        "tame kon", "tame kaun", "aa kon", "aa kaun",
        "kon che", "kaun che", "kon bole che", "kon baat kar",
        "kidhar se", "kahan se bol",
    ]
    # Also check Devanagari/Gujarati script identity questions
    if any(w in lower for w in [
        "कौन है", "कौन बोल", "कौन बात", "आप कौन", "ये कौन", "कहां से",
        "કોણ છે", "કોણ બોલ", "કોણ વાત", "તમે કોણ", "ક્યાંથી",
    ]):
        return "identity_question"
    if any(p in lower for p in _identity_phrases):
        return "identity_question"

    # Question — check before negation so कितना isn't caught by ना substring
    if lower.endswith("?") or re.search(r"\b(what|how|when|where|why|which|who|kitna|kab|kahan|kaun|kya|kyun|kon)\b", lower):
        return "question"
    if any(w in lower for w in ["कौन", "क्या", "कैसे", "कब", "कहां", "कितना", "क्यों",
                                  "કોણ", "શું", "કેવી", "ક્યારે", "ક્યાં", "કેટલા", "કેમ"]):
        return "question"
    # Common caller questions without explicit question words
    _caller_questions = [
        "is this", "are you", "tell me about",
        "can you tell", "i want to know", "details", "more information",
    ]
    if any(p in lower for p in _caller_questions):
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

    # Greeting — includes Gujarati/Hindi common greetings
    if re.search(r"\b(hello|helo|hellow|hi|hey|namaste|namaskar|kem cho|kemcho|hallo|halo|bol|bolo|suno|suniye|boliye)\b", lower):
        return "greeting"
    if any(w in lower for w in ["नमस्ते", "નમસ્તે", "હેલો", "કેમ છો", "बोलो", "बोल", "सुनिए", "हेलो"]):
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
    "baat", "raha", "rahi", "batao", "bolo", "boliye",
    "kaun", "kon", "kahan", "kidhar", "suniye", "bhai",
]

_GUJARATI_PATTERNS = [
    "che", "chu", "chhe", "cho", "chho", "chie", "chiye",
    "hatu", "hase", "hati", "hoy",
    "shu", "kem", "kemcho", "kem cho",
    "tame", "hu", "pan", "tamne", "mane", "amne", "amara", "tamara",
    "majama", "barobar", "saru", "sarun",
    "savare", "bapore", "saanje", "aaje", "kaale",
    "joie", "joiye", "jova", "joi", "jovun",
    "aavjo", "aavo", "jao",
    "saheb", "mari", "tari",
    "batavo", "janavo", "bolo",
    "karo", "karso", "kariye",
    "atyare", "pachhi", "pela",
    "nathi", "nahi", "haji",
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


_SLOT_HOUR_RE = re.compile(r"(\d{1,2}):(\d{2})\s*(AM|PM)", re.IGNORECASE)


def _slot_hour_24(time_str: str) -> int:
    """Parse a slot string like '10:00 AM' into 24-hour integer (10, 14, etc)."""
    m = _SLOT_HOUR_RE.match(time_str)
    if not m:
        return -1
    h = int(m.group(1))
    period = m.group(3).upper()
    if period == "PM" and h < 12:
        h += 12
    elif period == "AM" and h == 12:
        h = 0
    return h


def _slot_period(time_str: str) -> str:
    """Classify a slot time as morning / afternoon / evening."""
    h = _slot_hour_24(time_str)
    if 0 <= h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    return "evening"


# ---------------------------------------------------------------------------
# Objection responses (multilingual, Pramukh-specific)
# ---------------------------------------------------------------------------

_OBJECTION_RESPONSES: dict[str, list[str]] = {
    "en": [
        "Of course, I completely understand. Would another time work better for a quick call? Otherwise no problem at all.",
        "No problem at all — when would be a more convenient time? Just a brief chat.",
    ],
    "hi": [
        "जी बिल्कुल, मैं समझता हूँ। क्या किसी और समय एक छोटी सी बात हो सकती है? वरना कोई बात नहीं।",
        "कोई बात नहीं। आप बताइए कब सुविधाजनक रहेगा, बस एक छोटी सी बात करनी है।",
    ],
    "gu": [
        "જી બિલકુલ, હું સમજું છું. શું બીજા કોઈ સમયે થોડી વાત થઈ શકે? નહીંતર કોઈ વાત નહીં.",
        "કોઈ વાત નહીં. તમે જણાવો ક્યારે અનુકૂળ રહેશે, બસ થોડી વાત કરવી છે.",
    ],
}

# Used after a SECOND deferral/objection in the same call — we back off
# completely with a warm goodbye instead of pushing.
_DECLINE_FAREWELL_RESPONSES: dict[str, str] = {
    "en": "Of course, no problem at all. I won't take any more of your time — have a wonderful day, take care!",
    "hi": "जी बिल्कुल, कोई बात नहीं। मैं आपका और समय नहीं लूँगा। आपका दिन शुभ रहे, फिर मिलते हैं!",
    "gu": "જી બિલકુલ, કોઈ વાત નહીં. હું તમારો વધુ સમય નહીં લઉં. તમારો દિવસ શુભ રહે, ફરી મળીશું!",
}

# ---------------------------------------------------------------------------
# Filler responses (multilingual, Pramukh-specific)
# ---------------------------------------------------------------------------

_FILLER_RESPONSES: dict[str, list[str]] = {
    "en": ["One moment...", "Sure, let me check..."],
    "hi": ["एक सेकंड...", "बिल्कुल, बताता हूँ..."],
    "gu": ["એક સેકન્ડ...", "બિલકુલ, જણાવું છું..."],
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

# ---------------------------------------------------------------------------
# Language switch confirmation responses
# ---------------------------------------------------------------------------

_LANG_SWITCH_RESPONSES: dict[str, str] = {
    "hi": "जी बिल्कुल, हम हिंदी में बात करते हैं। बताइए, मैं आपकी कैसे मदद कर सकता हूँ?",
    "en": "Sure, let's continue in English. How can I help you?",
    "gu": "જી બિલકુલ, ચાલો ગુજરાતીમાં વાત કરીએ. બોલો, હું તમારી કેવી રીતે મદદ કરી શકું?",
}

# ---------------------------------------------------------------------------
# Identity responses — fast deterministic answers for "who is this?"
# ---------------------------------------------------------------------------

_IDENTITY_RESPONSES: dict[str, str] = {
    "en": (
        "I'm calling from Pramukh Group, one of Gujarat's most trusted real estate developers. "
        "You had shown interest in our properties and raised an inquiry. "
        "I just wanted to have a quick chat about what you're looking for."
    ),
    "hi": (
        "जी, मैं Pramukh Group से बात कर रहा हूँ, हम Gujarat के जाने-माने रियल एस्टेट डेवलपर हैं। "
        "आपने हमारी प्रॉपर्टीज में इंटरेस्ट दिखाया था और इंक्वायरी भी की थी। "
        "मैं बस आपसे इसके बारे में थोड़ी बात करना चाहता था।"
    ),
    "gu": (
        "જી, હું Pramukh Group તરફથી બોલું છું, અમે Gujarat ના જાણીતા રિયલ એસ્ટેટ ડેવલપર છીએ. "
        "તમે અમારી પ્રોપર્ટીઝમાં ઇન્ટરેસ્ટ દર્શાવ્યો હતો અને ઇન્ક્વાયરી પણ કરી હતી. "
        "હું બસ આ વિશે થોડી વાત કરવા માંગતો હતો."
    ),
}

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
        "greeting": ["हेलो! Pramukh Group से बात कर रहा हूँ।", "हेलो! बोलिए, कैसे मदद कर सकता हूँ?"],
        "farewell": "अच्छा, फिर मिलते हैं! Pramukh Group से कॉल के लिए धन्यवाद।",
        "gratitude": ["आपका स्वागत है!", "खुशी हुई!"],
        "closing": "बहुत अच्छा! हम आपको डिटेल्स भेज देंगे। धन्यवाद!",
    },
    "gu": {
        "greeting": ["હેલો! Pramukh Group તરફથી બોલું છું.", "હેલો! બોલો, કેવી રીતે મદદ કરી શકું?"],
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
# Opening greeting — bilingual so user can respond in preferred language
# ---------------------------------------------------------------------------

_OPENING_GREETING: str = (
    "Good afternoon! Namaste, मैं Pramukh Group से बात कर रहा हूँ।"
)

# ---------------------------------------------------------------------------
# Interest line — delivered after user responds to greeting
# ---------------------------------------------------------------------------

_INTEREST_LINE: dict[str, str] = {
    "en": (
        "You had shown interest in our properties and raised an inquiry as well. "
        "I wanted to have a quick chat about what you're looking for."
    ),
    "hi": (
        "आपने हमारी प्रॉपर्टीज में इंटरेस्ट दिखाया था और इसके बारे में इंक्वायरी भी की थी। "
        "मैं बस आपसे इसके बारे में थोड़ी बात करना चाहता था।"
    ),
    "gu": (
        "તમે અમારી પ્રોપર્ટીઝમાં ઇન્ટરેસ્ટ દર્શાવ્યો હતો અને ઇન્ક્વાયરી પણ કરી હતી. "
        "હું બસ આ વિશે થોડી વાત કરવા માંગતો હતો."
    ),
}

# ---------------------------------------------------------------------------
# Reconfirmation — when user denies interest the first time
# ---------------------------------------------------------------------------

_INTEREST_RECONFIRM: dict[str, str] = {
    "en": (
        "My apologies! We had received your profile because you had shown interest "
        "in our property and raised an inquiry about the same. "
        "Would you like to know more about our projects?"
    ),
    "hi": (
        "माफ़ कीजिए! हमारे पास आपकी प्रोफाइल आई थी क्योंकि आपने हमारी प्रॉपर्टी में "
        "इंटरेस्ट दिखाया था और इसके बारे में इंक्वायरी भी की थी। "
        "क्या आप हमारे प्रोजेक्ट्स के बारे में जानना चाहेंगे?"
    ),
    "gu": (
        "માફ કરજો! અમારી પાસે તમારી પ્રોફાઇલ આવી હતી કારણ કે તમે અમારી પ્રોપર્ટીમાં "
        "ઇન્ટરેસ્ટ દર્શાવ્યો હતો અને ઇન્ક્વાયરી પણ કરી હતી. "
        "શું તમે અમારા પ્રોજેક્ટ્સ વિશે જાણવા માંગો છો?"
    ),
}

# ---------------------------------------------------------------------------
# Graceful close — when user declines twice
# ---------------------------------------------------------------------------

_GRACEFUL_CLOSE: dict[str, str] = {
    "en": "Thank you so much for your time! Have a great day.",
    "hi": "आपके समय के लिए बहुत बहुत धन्यवाद! आपका दिन शुभ हो।",
    "gu": "તમારા સમય માટે ખૂબ ખૂબ આભાર! તમારો દિવસ શુભ રહે.",
}

# ---------------------------------------------------------------------------
# Location ask — short question after interest is confirmed
# ---------------------------------------------------------------------------

_LOCATION_ASK: dict[str, str] = {
    "en": "Great! Which location are you interested in?",
    "hi": "बढ़िया! आप किस लोकेशन में देखना चाहेंगे?",
    "gu": "સરસ! તમે કઈ લોકેશનમાં જોવા માંગો છો?",
}

# ---------------------------------------------------------------------------
# New flow prompts — project / status / rent-or-buy / property info
# ---------------------------------------------------------------------------

_PROJECT_CONFIRM_PROMPT: dict[str, str] = {
    "en": "Could you confirm which Pramukh Group project you're interested in?",
    "hi": "क्या आप बता सकते हैं कि आप हमारे किस Pramukh Group प्रोजेक्ट में दिलचस्पी रखते हैं?",
    "gu": "શું તમે જણાવી શકો કે તમે અમારા કયા Pramukh Group પ્રોજેક્ટમાં રસ ધરાવો છો?",
}

_PROPERTY_STATUS_PROMPT: dict[str, str] = {
    "en": "Got it. Are you looking for a ready-made flat or one that is under construction?",
    "hi": "समझ गया। क्या आप ready-made फ्लैट देख रहे हैं या under-construction?",
    "gu": "સમજ્યો. શું તમે ready-made ફ્લેટ શોધી રહ્યા છો કે under-construction?",
}

_RENT_OR_BUY_PROMPT: dict[str, str] = {
    "en": "And is this for rent, or are you looking to buy?",
    "hi": "और यह किराये के लिए है या खरीदने के लिए?",
    "gu": "અને આ ભાડે માટે છે કે ખરીદવા માટે?",
}

# Property info delivered after we know status (ready / under-construction)
# and intent (rent / buy). Tells about flats and condition, then pivots to
# scheduling a site visit.
_PROPERTY_INFO_READY: dict[str, str] = {
    "en": (
        "Excellent! Our ready-made flats are fully finished — modern interiors, "
        "all amenities active, and ready to move in immediately. We have 2 BHK, "
        "3 BHK, and 4 BHK options available. Would you like to schedule a quick site visit?"
    ),
    "hi": (
        "बहुत बढ़िया! हमारे ready-made फ्लैट्स पूरी तरह तैयार हैं — modern interiors, "
        "सारी amenities active हैं, और तुरंत move-in कर सकते हैं। हमारे पास 2 BHK, "
        "3 BHK और 4 BHK के विकल्प हैं। क्या एक site visit schedule करें?"
    ),
    "gu": (
        "ખૂબ સરસ! અમારા ready-made ફ્લેટ્સ સંપૂર્ણ તૈયાર છે — modern interiors, "
        "બધી amenities active છે, અને તરત જ move-in કરી શકો. અમારી પાસે 2 BHK, "
        "3 BHK અને 4 BHK ના વિકલ્પો છે. શું એક site visit schedule કરીએ?"
    ),
}

_PROPERTY_INFO_UNDER: dict[str, str] = {
    "en": (
        "Great choice! Our under-construction projects offer pre-launch pricing, "
        "modern designs, and flexible payment plans. Possession is in 18 to 24 months. "
        "We have 2 BHK, 3 BHK, and 4 BHK layouts. Would you like to schedule a site visit "
        "to see the layouts and sample flat?"
    ),
    "hi": (
        "बढ़िया choice! हमारे under-construction projects में pre-launch pricing, "
        "modern designs, और flexible payment plans मिलते हैं। 18 से 24 महीनों में "
        "possession मिलता है। हमारे पास 2 BHK, 3 BHK और 4 BHK के layouts हैं। "
        "क्या एक site visit schedule करें ताकि आप layout और sample flat देख सकें?"
    ),
    "gu": (
        "સરસ choice! અમારા under-construction projects માં pre-launch pricing, "
        "modern designs, અને flexible payment plans મળે છે. 18 થી 24 મહિનામાં "
        "possession મળી જાય છે. અમારી પાસે 2 BHK, 3 BHK અને 4 BHK ના layouts છે. "
        "શું એક site visit schedule કરીએ જેથી તમે layout અને sample flat જોઈ શકો?"
    ),
}

# ---------------------------------------------------------------------------
# Status / transaction-type extraction helpers
# ---------------------------------------------------------------------------

_READY_KEYWORDS = (
    "ready", "ready made", "ready-made", "readymade", "readytomove",
    "ready to move", "move in", "move-in", "movein",
    "tayar", "taiyar", "complete", "finished", "occupancy",
    "तैयार", "रेडी", "completed",
    "તૈયાર", "રેડી",
)

_UNDER_CONSTRUCTION_KEYWORDS = (
    "under construction", "under-construction", "underconstruction",
    "under con", "uc", "construction", "new launch", "newlaunch",
    "pre launch", "pre-launch", "prelaunch", "upcoming",
    "naya", "naya project", "nirman",
    "निर्माण", "अंडर कंस्ट्रक्शन", "नया", "नई",
    "નિર્માણ", "અંડર કન્સ્ટ્રક્શન", "નવો", "નવી",
)

_RENT_KEYWORDS = (
    "rent", "rental", "lease", "leasing", "for rent",
    "kiraya", "kiraye", "kirae", "bhada", "bhade",
    "किराया", "किराये", "किराए", "भाड़ा", "भाडा",
    "ભાડે", "ભાડા", "ભાડું",
)

_BUY_KEYWORDS = (
    "buy", "purchase", "buying", "purchasing", "own", "ownership",
    "kharidna", "kharidne", "kharidu", "kharido", "lena", "khareed",
    "खरीद", "ख़रीद", "खरीदना", "खरीदने",
    "ખરીદ", "ખરીદવા", "ખરીદવું", "લેવા",
)


def _detect_property_status(text: str) -> str | None:
    """Return 'ready_made' or 'under_construction' if either is mentioned."""
    lower = text.lower()
    has_ready = any(kw in lower or kw in text for kw in _READY_KEYWORDS)
    has_uc = any(kw in lower or kw in text for kw in _UNDER_CONSTRUCTION_KEYWORDS)
    if has_uc:
        return "under_construction"
    if has_ready:
        return "ready_made"
    return None


def _detect_transaction_type(text: str) -> str | None:
    """Return 'rent' or 'buy' if either is mentioned.

    Rent is checked first because its keywords ('kiraya', 'rent', 'lease') are
    specific and intentional, whereas buy keywords overlap with generic verbs
    ('lena' = 'to take' is generic and only means 'buy' in context). When the
    user says 'kiraye par lena hai', both fire — but the rent intent wins.
    """
    lower = text.lower()
    has_rent = any(kw in lower or kw in text for kw in _RENT_KEYWORDS)
    has_buy = any(kw in lower or kw in text for kw in _BUY_KEYWORDS)
    if has_rent:
        return "rent"
    if has_buy:
        return "buy"
    return None


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

    Drives the conversation through:
      GREETING -> INTEREST_CONFIRM -> (INTEREST_RECONFIRM) -> LOCATION
      -> PROPERTY_TYPE -> SCHEDULING -> CONFIRMATION -> CLOSING.

    Agent speaks first with a bilingual greeting. User's language is
    detected from their first response and mirrored throughout.
    Extracts structured booking data at each stage. Falls back to LLM
    for natural responses and question handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = LLM_TIMEOUT,
        default_language: str = "hi",
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
        """Generate the bilingual opening greeting. Called once at pipeline start.
        Agent speaks first — bilingual so user responds in their preferred language."""
        greeting = _OPENING_GREETING
        self.memory.add_assistant(greeting)
        self.state.stage = STAGE_GREETING
        logger.info("[OPENING] Bilingual greeting delivered, waiting for user response")
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
        logger.info('[LANG] "%s" -> %s (src=%s, conf=%.2f, locked=%s)',
                     text, detected_lang, src, conf, self.state.language_locked)

        self.state.language_history.append(detected_lang)
        history = self.state.language_history[-5:]
        self.state.language_history = history

        prev_lang = self.state.language

        is_gujarati_detected = (
            detected_lang == "gu"
            or src == "script_gu_markers"
        )

        # Decay the lock after 3 turns — allows natural switching back
        if self.state.language_locked:
            self.state.language_lock_turns += 1
            if self.state.language_lock_turns > 3:
                self.state.language_locked = False
                logger.info("[LANG] Lock expired after %d turns", self.state.language_lock_turns)

        # Gujarati script or markers — switch ONLY if not locked to a different
        # language. Explicit user request (intent="language_switch") is the only
        # thing that can override the lock; Devanagari Gujarati-marker words
        # like "माने" / "छे" appearing inside otherwise-Hindi STT output must
        # not flip the agent back to Gujarati.
        if is_gujarati_detected and (
            not self.state.language_locked or prev_lang == "gu"
        ):
            if self.state.language != "gu":
                logger.info("[LANG] Switching to Gujarati (src=%s)", src)
            self.state.language = "gu"
        elif is_gujarati_detected:
            logger.info(
                "[LANG] Ignoring Gujarati markers (locked to %s, src=%s)",
                prev_lang, src,
            )
        elif src == "pattern" and conf >= 0.7:
            # Pattern-based detection: Hindi/Gujarati romanized words
            self.state.language = detected_lang
        elif src == "script" and detected_lang == "hi":
            # Devanagari text — could be Hindi OR Gujarati via STT.
            # Only switch to Hindi if not currently Gujarati, or if
            # language is not locked.
            if prev_lang != "gu" or not self.state.language_locked:
                self.state.language = "hi"
        elif detected_lang == "en" and src in ("default", "pattern"):
            # Pure-Latin / non-Indic text. Switch to English when it's a
            # confident multi-word signal — single ASCII tokens like "ok" or
            # "haan" shouldn't flip a Hindi/Gujarati session. Crucially, this
            # branch runs even when language_locked, so a user can switch from
            # Hindi to English mid-call just by speaking English.
            word_count = len(text.split())
            if word_count >= 2 and prev_lang != "en":
                logger.info(
                    "[LANG] Switching to English (ascii multi-word, src=%s, words=%d)",
                    src, word_count,
                )
                self.state.language = "en"
                # Soft-decay any prior lock so subsequent turns aren't re-locked
                self.state.language_locked = False
                self.state.language_lock_turns = 0
        # Ambiguous/default: preserve current language

        self.state.lang_confidence = conf
        self.state.lang_source = src
        logger.info('[LANG] Response language: %s (detected=%s, prev=%s)',
                     self.state.language, detected_lang, prev_lang)

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

        # Reset decline counter when the user re-engages with a non-objection
        # intent — we don't want a stale "1 prior decline" to trip the 2-strike
        # farewell after the user has clearly come back into the conversation.
        if intent not in ("objection", "farewell"):
            self.state.decline_count = 0

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
            self.state.decline_count += 1

            # Second decline: stop pushing and end gracefully.
            if self.state.decline_count >= 2:
                self.state.is_closing = True
                self.state.stage = STAGE_CLOSING
                resp = _DECLINE_FAREWELL_RESPONSES.get(
                    self.state.language, _DECLINE_FAREWELL_RESPONSES["en"]
                )
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="farewell")
                return

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

        # --- Identity question at any stage — fast deterministic response ---
        if intent == "identity_question":
            _cancel_llm()
            lang = self.state.language
            resp = _IDENTITY_RESPONSES.get(lang, _IDENTITY_RESPONSES["hi"])
            self.memory.add_assistant(resp)
            if self.state.stage == STAGE_GREETING:
                self.state.stage = STAGE_INTEREST_CONFIRM
            elif self.state.stage == STAGE_INTEREST_CONFIRM:
                self.state.stage = STAGE_LOCATION
            yield EngineResult(text=resp, is_shortcut=True, intent="identity_question")
            return

        # --- Language switch at any stage ---
        # Highest-priority override: regardless of any prior lock, an explicit
        # switch request immediately changes the output language. This handles
        # the case "user says 'can you speak in English' while agent is in
        # Hindi" — without this branch the agent stays stuck.
        if intent == "language_switch":
            _cancel_llm()
            new_lang = self._extract_requested_language(text)
            logger.info(
                "[LANG] Explicit switch request -> %s (was %s, locked=%s)",
                new_lang, self.state.language, self.state.language_locked,
            )
            self.state.language = new_lang
            self.state.language_locked = True
            self.state.language_lock_turns = 0
            self.state.language_history = [new_lang] * 5
            resp = _LANG_SWITCH_RESPONSES.get(new_lang, _LANG_SWITCH_RESPONSES["hi"])
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="language_switch")
            return

        # =============================================================
        # GREETING STAGE — agent already said bilingual "Hello, Namaste!
        # Mai Pramukh Group se baat kar raha hu." Now we detect the
        # user's language from their response and deliver the interest line.
        # =============================================================
        if self.state.stage == STAGE_GREETING:

            # User starts with a direct intent — skip to the right stage
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
                    self.state.stage = STAGE_LOCATION
                    next_prompt = self._get_proactive_prompt(STAGE_LOCATION)
                    resp = self._bhk_ack(bhk, "your area") + " " + next_prompt
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="property_type")
                    return

            # For all other responses (hello, haan, who is this, question,
            # general, etc.) — user's language is now detected from their
            # first response to the bilingual greeting. Lock it so STT
            # Devanagari output doesn't flip it back.
            _cancel_llm()
            if not self.state.language_locked:
                self.state.language_locked = True
                logger.info("[LANG] Locked language to %s from greeting response",
                            self.state.language)
            # Silent NAME_CONFIRM passthrough → straight into PROJECT_CONFIRM.
            # The user's first reply has already been recorded by memory.add_user
            # earlier; name extraction (if any) lives in memory._extract_context.
            # Replace this passthrough with an explicit name-ask once business
            # finalises the wording.
            lang = self.state.language
            resp = _PROJECT_CONFIRM_PROMPT.get(lang, _PROJECT_CONFIRM_PROMPT["en"])
            self.memory.add_assistant(resp)
            self.state.stage = STAGE_PROJECT_CONFIRM
            yield EngineResult(text=resp, is_shortcut=True, intent="project_confirm_ask")
            return

        # =============================================================
        # PROJECT CONFIRM — caller names the Pramukh project they're
        # interested in. We accept any non-trivial reply as the project
        # name (best-effort; LLM/system prompt fills in details on follow-up).
        # =============================================================
        if self.state.stage == STAGE_PROJECT_CONFIRM:

            # User asks a question (e.g. "which projects do you have?") —
            # let the LLM answer, then re-prompt for project on next turn.
            if intent == "question":
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=False,
                ):
                    yield result
                return

            # Capture project name when user gave a substantive answer.
            # Skip pure affirmation/negation tokens — those aren't project names.
            if intent not in ("affirmation", "negation", "greeting"):
                cleaned = text.strip()
                if cleaned and len(cleaned.split()) >= 1:
                    self.memory.booking.project = cleaned

            _cancel_llm()
            self.state.stage = STAGE_PROPERTY_STATUS
            lang = self.state.language
            resp = _PROPERTY_STATUS_PROMPT.get(lang, _PROPERTY_STATUS_PROMPT["en"])
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="property_status_ask")
            return

        # =============================================================
        # PROPERTY STATUS — ready-made vs under-construction
        # =============================================================
        if self.state.stage == STAGE_PROPERTY_STATUS:

            if intent == "question":
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=False,
                ):
                    yield result
                return

            status = _detect_property_status(text)
            if status:
                self.memory.booking.property_status = status

            # Default to ready_made when caller skips/dodges — keeps the flow
            # moving; the LLM can clarify later if it really matters.
            if not self.memory.booking.property_status:
                self.memory.booking.property_status = "ready_made"

            _cancel_llm()
            self.state.stage = STAGE_RENT_OR_BUY
            lang = self.state.language
            resp = _RENT_OR_BUY_PROMPT.get(lang, _RENT_OR_BUY_PROMPT["en"])
            self.memory.add_assistant(resp)
            yield EngineResult(text=resp, is_shortcut=True, intent="rent_or_buy_ask")
            return

        # =============================================================
        # RENT OR BUY — caller's transaction intent. After this we deliver
        # PROPERTY_INFO (the flat-and-condition pitch) and advance into the
        # existing SCHEDULING flow.
        # =============================================================
        if self.state.stage == STAGE_RENT_OR_BUY:

            if intent == "question":
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=False,
                ):
                    yield result
                return

            txn = _detect_transaction_type(text)
            if txn:
                self.memory.booking.transaction_type = txn
            if not self.memory.booking.transaction_type:
                self.memory.booking.transaction_type = "buy"

            _cancel_llm()
            status = self.memory.booking.property_status or "ready_made"
            info_dict = _PROPERTY_INFO_READY if status == "ready_made" else _PROPERTY_INFO_UNDER
            lang = self.state.language
            resp = info_dict.get(lang, info_dict["en"])
            self.memory.add_assistant(resp)
            # PROPERTY_INFO is a one-shot speech — the next user reply lands
            # in SCHEDULING (the info ends with "would you like to schedule
            # a site visit?", which makes that the natural next stage).
            self.state.stage = STAGE_SCHEDULING
            yield EngineResult(text=resp, is_shortcut=True, intent="property_info")
            return

        # =============================================================
        # INTEREST CONFIRM — user responds yes/no to the interest line
        # =============================================================
        if self.state.stage == STAGE_INTEREST_CONFIRM:

            # YES / affirmation / greeting — user is engaged, ask about location
            if intent in ("affirmation", "greeting"):
                _cancel_llm()
                lang = self.state.language
                resp = _LOCATION_ASK.get(lang, _LOCATION_ASK["hi"])
                self.memory.add_assistant(resp)
                self.state.stage = STAGE_LOCATION
                yield EngineResult(text=resp, is_shortcut=True, intent="interest_confirmed")
                return

            # Direct data intent — skip ahead
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
                    self.state.stage = STAGE_LOCATION
                    resp = self._bhk_ack(bhk, "your area") + " " + _LOCATION_ASK.get(self.state.language, _LOCATION_ASK["hi"])
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="property_type")
                    return

            # Question or general — user is engaged, answer via LLM then ask location
            if intent in ("question", "general"):
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=False,
                ):
                    yield result
                self.state.stage = STAGE_LOCATION
                return

            # NO / negation / objection — politely reconfirm
            if intent in ("negation", "objection"):
                _cancel_llm()
                lang = self.state.language
                resp = _INTEREST_RECONFIRM.get(lang, _INTEREST_RECONFIRM["hi"])
                self.memory.add_assistant(resp)
                self.state.stage = STAGE_INTEREST_RECONFIRM
                yield EngineResult(text=resp, is_shortcut=True, intent="interest_reconfirm")
                return

            # Anything else — treat as engagement, ask about location
            _cancel_llm()
            lang = self.state.language
            resp = _LOCATION_ASK.get(lang, _LOCATION_ASK["hi"])
            self.memory.add_assistant(resp)
            self.state.stage = STAGE_LOCATION
            yield EngineResult(text=resp, is_shortcut=True, intent="interest_confirmed")
            return

        # =============================================================
        # INTEREST RECONFIRM — second chance after first denial
        # =============================================================
        if self.state.stage == STAGE_INTEREST_RECONFIRM:

            # YES — user changed their mind, ask about location
            if intent in ("affirmation", "greeting"):
                _cancel_llm()
                lang = self.state.language
                resp = _LOCATION_ASK.get(lang, _LOCATION_ASK["hi"])
                self.memory.add_assistant(resp)
                self.state.stage = STAGE_LOCATION
                yield EngineResult(text=resp, is_shortcut=True, intent="interest_confirmed")
                return

            # Direct data intent — skip ahead
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
                    self.state.stage = STAGE_LOCATION
                    resp = self._bhk_ack(bhk, "your area") + " " + _LOCATION_ASK.get(self.state.language, _LOCATION_ASK["hi"])
                    self.memory.add_assistant(resp)
                    yield EngineResult(text=resp, is_shortcut=True, intent="property_type")
                    return

            # Question or general — user is re-engaged
            if intent in ("question", "general"):
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=False,
                ):
                    yield result
                self.state.stage = STAGE_LOCATION
                return

            # NO again — gracefully close
            if intent in ("negation", "objection"):
                _cancel_llm()
                lang = self.state.language
                resp = _GRACEFUL_CLOSE.get(lang, _GRACEFUL_CLOSE["hi"])
                self.memory.add_assistant(resp)
                self.state.stage = STAGE_CLOSING
                self.state.is_closing = True
                yield EngineResult(text=resp, is_shortcut=True, intent="graceful_close")
                return

            # Anything else — benefit of doubt, ask about location
            _cancel_llm()
            lang = self.state.language
            resp = _LOCATION_ASK.get(lang, _LOCATION_ASK["hi"])
            self.memory.add_assistant(resp)
            self.state.stage = STAGE_LOCATION
            yield EngineResult(text=resp, is_shortcut=True, intent="interest_confirmed")
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

            if intent == "negation":
                _cancel_llm()
                lang = self.state.language
                resp = _LOCATION_ASK.get(lang, _LOCATION_ASK["hi"])
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="negation")
                return

            # Question or general conversation — answer via LLM
            if intent in ("question", "general"):
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=True,
                ):
                    yield result
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

            # Question or general conversation — answer via LLM, then nudge back
            if intent in ("question", "general"):
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=True,
                ):
                    yield result
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

            # Question or general conversation — answer via LLM, then nudge back
            if intent in ("question", "general"):
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=True,
                ):
                    yield result
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
                    resp = "કોઈ વાત નહીं. કયો સમય તમારા માટે વધુ સારો હશે?"
                else:
                    resp = "No problem. Which time would work better for you?"
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="reschedule")
                return

            # Question or general conversation — answer via LLM, then nudge back
            if intent in ("question", "general"):
                self.state.llm_cancelled = False
                async for result in self._consume_llm_with_filler(
                    llm_queue, llm_done, intent, nudge_back=True,
                ):
                    yield result
                return

        # =============================================================
        # CLOSING STAGE — allow re-engagement for new intents
        # =============================================================
        if self.state.stage == STAGE_CLOSING:
            # Explicit re-engagement: user wants to schedule / book / change.
            # Match keywords across English / Hindi / Gujarati / transliterated.
            _closing_lower = text.lower()
            if (
                re.search(
                    r"\b(schedule|book|booking|appointment|another|different|"
                    r"reschedule|change|new\s+(?:appointment|slot|time|date))\b",
                    _closing_lower,
                )
                or any(p in text for p in (
                    "अपॉइंटमेंट", "अपोइंटमेंट", "एपॉइंटमेंट",
                    "शेड्यूल", "नया", "नई", "फिर से", "फिर बुक",
                    "बुक करो", "बुक करना", "दूसरा", "दूसरी",
                    "એપોઇન્ટમેન્ટ", "શેડ્યુલ", "નવી", "નવો", "ફરીથી",
                    "બુક કરો", "બીજો", "બીજી",
                ))
            ):
                _cancel_llm()
                self.state.is_closing = False
                self.state.stage = STAGE_SCHEDULING
                resp = self._get_proactive_prompt(STAGE_SCHEDULING)
                self.memory.add_assistant(resp)
                yield EngineResult(text=resp, is_shortcut=True, intent="reengage")
                return

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
        """
        Match user-stated time/day to an available slot.

        Returns None when the user gave a SPECIFIC time that isn't on the
        slot list — never silently fudges to a different slot. The engine's
        scheduling reprompt path then offers the real available slots back.
        """
        lower = text.lower()

        # --- Step 1: extract specific hour if present ---
        hour_24: int | None = None
        # English "11 am", "6 pm", "5:00 pm"
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", lower)
        if m:
            h = int(m.group(1))
            is_pm = "p" in m.group(3)
            hour_24 = (h % 12) + (12 if is_pm else 0)
        else:
            # Hindi/Gujarati/transliterated: "11 बजे", "11 વાગે", "11 baje"
            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(?:बजे|वजे|વાગે|baje|baaje)", text)
            if m:
                h = int(m.group(1))
                is_evening = bool(re.search(
                    r"(शाम|સાંજે|saanje|shaam|evening|night|रात|રાત)", text + lower))
                is_afternoon = bool(re.search(
                    r"(दोपहर|બપોરે|bapore|dopahar|afternoon)", text + lower))
                is_morning = bool(re.search(
                    r"(सुबह|સવારે|savare|subah|morning)", text + lower))
                if is_evening or is_afternoon:
                    hour_24 = (h % 12) + 12
                elif is_morning:
                    hour_24 = h % 12 if h != 12 else 12
                else:
                    # Default heuristic: 1-7 → PM (afternoon/evening),
                    # 8-12 → AM (morning) — fits typical site-visit phrasing
                    hour_24 = (h % 12) + (12 if 1 <= h <= 7 else 0)

        # --- Step 2: extract day ---
        day: str | None = None
        if re.search(r"\b(today|aaj|aaje)\b", lower) or "आज" in text or "આજે" in text:
            day = "Today"
        elif (re.search(r"\b(tomorrow|kal|kaale)\b", lower)
              or "कल" in text or "કાલે" in text):
            day = "Tomorrow"
        elif (re.search(r"\b(parso|parson|day after tomorrow)\b", lower)
              or "परसों" in text or "परसो" in text
              or "પરમ" in text or "પરસું" in text or "પરસો" in text):
            day = "Day after tomorrow"
        elif re.search(r"\b(saturday|this saturday|weekend)\b", lower):
            day = "This Saturday"

        # --- Step 3: if specific time given, match strictly against slots ---
        if hour_24 is not None:
            for slot in _AVAILABLE_SLOTS:
                if day and slot["day"] != day:
                    continue
                if _slot_hour_24(slot["time"]) == hour_24:
                    return f"{slot['day']} at {slot['time']}"
            # User gave a specific hour with no matching slot — caller will
            # reprompt with available options instead of locking a wrong time.
            return None

        # --- Step 4: no specific hour. Match by part-of-day + day. ---
        period: str | None = None
        if re.search(r"(सुबह|સવારે|savare|subah|morning)", text + lower):
            period = "morning"
        elif re.search(r"(दोपहर|બપોરે|bapore|dopahar|afternoon)", text + lower):
            period = "afternoon"
        elif re.search(r"(शाम|સાંજે|saanje|shaam|evening|night)", text + lower):
            period = "evening"

        if day or period:
            for slot in _AVAILABLE_SLOTS:
                if day and slot["day"] != day:
                    continue
                if period and _slot_period(slot["time"]) != period:
                    continue
                return f"{slot['day']} at {slot['time']}"
            # Day matched but period didn't (e.g. user wants "Tomorrow evening"
            # — no evening slots tomorrow). Don't fudge: caller reprompts.
            if period:
                return None
            # Day-only match: take the first slot of that day.
            for slot in _AVAILABLE_SLOTS:
                if slot["day"] == day:
                    return f"{slot['day']} at {slot['time']}"

        # --- Step 5: bare affirmation → first available slot ---
        if re.search(r"\b(yes|yeah|ok|okay|sure|haan|ha|ji|theek|first|pehla)\b", lower):
            return f"{_AVAILABLE_SLOTS[0]['day']} at {_AVAILABLE_SLOTS[0]['time']}"
        if any(w in text for w in ("ठीक", "हां", "हाँ", "જી", "હા", "બરાબર", "ઠીક")):
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
        """Consume LLM queue with filler timeout.

        nudge_back is accepted for API compatibility but no longer appends
        a second utterance — the flow re-prompts naturally on the next turn.
        This prevents the agent from talking non-stop (LLM answer + nudge
        back-to-back without letting the user speak).
        """
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

        # Reinforce language at the end of messages so it's closest to generation
        lang_names = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}
        lang_name = lang_names.get(self.state.language, "Hindi")
        messages.append({
            "role": "system",
            "content": f"REMINDER: Respond ONLY in {lang_name}. Do not use any other language.",
        })

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
            "You are a proactive sales calling assistant from Pramukh Group, a trusted real estate developer in Gujarat, India.",
            "You are making an outbound call to someone who showed interest in Pramukh properties.",
            "Your PRIMARY GOAL is to SELL — generate excitement about properties and schedule a site visit.",
            "",
            f"CURRENT BOOKING STATE:",
            f"- Stage: {self.state.stage}",
            f"- Project: {b.project or 'not confirmed yet'}",
            f"- Property status: {b.property_status or 'not confirmed yet'} (ready-made vs under-construction)",
            f"- Transaction type: {b.transaction_type or 'not confirmed yet'} (rent vs buy)",
            f"- Location: {b.location or 'not selected yet'}",
            f"- Property type: {b.property_type or 'not selected yet'}",
            f"- Appointment: {b.appointment_time or 'not scheduled yet'}",
            "",
            "CALL FLOW (you must drive the caller through these steps in order):",
            "  1. Good-afternoon greeting (already delivered).",
            "  2. Name confirmation (currently silent placeholder — skip until business finalises wording).",
            "  3. Confirm which Pramukh Group project they're interested in.",
            "  4. Ready-made vs under-construction.",
            "  5. For rent or to buy.",
            "  6. Inform them about the available flats and condition (matched to status).",
            "  7. Schedule a site visit.",
            "",
            "LOCATIONS & KEY SELLING POINTS:",
            "- Surat: Gujarat's diamond city, booming IT hub, excellent infrastructure, 2-5 BHK from ₹35L-₹1.5Cr",
            "- Vapi: Industrial growth corridor, close to Mumbai, affordable luxury, 2-4 BHK from ₹25L-₹85L",
            "- Silvassa: Peaceful green living near Vapi, nature surroundings, weekend home destination, 2-3 BHK from ₹20L-₹60L",
            "PROPERTY TYPES: 2 BHK, 3 BHK, 4 BHK, 5 BHK — all with modern amenities, parking, garden, security",
            "READY-MADE vs UNDER-CONSTRUCTION:",
            "- Ready-made: fully finished, modern interiors, all amenities active, immediate move-in.",
            "- Under-construction: pre-launch pricing, modern designs, flexible payment plans, possession in 18-24 months.",
            "",
            "YOUR IDENTITY:",
            "- You are calling on behalf of Pramukh Group, a well-known real estate developer in Gujarat.",
            "- If asked 'who is this?', introduce yourself and immediately pivot to why you're calling.",
            "- Your name is not important — you represent Pramukh Group.",
            "",
            "SALES APPROACH:",
            "- Be PROACTIVE — don't wait for questions, volunteer useful information about properties.",
            "- When the user mentions a location, immediately highlight what makes it special and what options are available.",
            "- When the user mentions BHK, share price range and key amenities to build excitement.",
            "- Always guide toward scheduling a site visit — 'seeing is believing' is your close.",
            "- If the user hesitates, share a compelling detail (new launch, limited units, special pricing).",
            "- Answer every question thoroughly — a well-informed buyer is more likely to visit.",
            "",
            "CRITICAL RULES:",
            "- YOU lead the conversation. Answer questions AND proactively share property details.",
            "- Respond in 2-3 sentences. You are on a phone call — be informative but concise.",
            "- Sound natural: be warm, enthusiastic, and conversational.",
            '- If you CANNOT answer a specific question, say "main apni team se confirm karke batata hu" and continue the flow.',
            "- No bullet points, lists, markdown, or emojis.",
            "- Be polite, respectful, and proactive. Respect the user's time.",
            "- Use formal address (aap/tamne) in Hindi/Gujarati.",
            "- NEVER lose sight of the goal: get them excited about properties and book a site visit.",
            "- Match the caller's language exactly. If they switch to a different language mid-call, switch with them — do NOT refuse.",
        ]

        # NEVER-REFUSE rule — applies to every language. The agent must never
        # claim it can only speak one language or refuse to switch. Language
        # switching is handled deterministically upstream; if the LLM is being
        # invoked at all, the chosen language is already correct, so the model
        # must simply use it without commenting on its own language abilities.
        never_refuse = (
            "\n*** NEVER REFUSE A LANGUAGE — CRITICAL ***"
            "\nYou are FLUENT in Hindi, Gujarati, AND English. You can switch freely between any of them."
            "\nYou MUST NEVER say or imply that you can only speak one language."
            "\nForbidden phrasings (DO NOT produce these or any paraphrase):"
            "\n  - \"I can only speak/talk in [language]\""
            "\n  - \"मैं सिर्फ हिंदी में बात कर सकता हूँ\" / \"मैं हिंदी में ही बात करूँगा\" / \"मुझे केवल हिंदी आती है\""
            "\n  - \"હું ફક્ત ગુજરાતી માં વાત કરી શકું છું\" / \"હું ગુજરાતી માં જ વાત કરીશ\""
            "\n  - \"main sirf hindi mai baat kar sakta hu\" / \"hu fakt gujarati ma vaat karu chu\""
            "\n  - Any sentence that names a language and says you can ONLY use it, or that you will ONLY/ही/જ/only continue in it."
            "\nIf the caller mentions any language difficulty or preference, simply continue helpfully in the language the system has already selected — do NOT comment on your own language capabilities or restrictions."
        )

        lang_instruction = {
            "hi": (
                f"\n*** LANGUAGE: HINDI — THIS IS THE HIGHEST PRIORITY RULE ***"
                f"\nYour response MUST be ENTIRELY in Hindi."
                f"\nUse Devanagari script or transliterated Hindi."
                f"\nDo NOT write in Gujarati script (ગુજરાતી) or English."
                f"\nEven if previous messages were in Gujarati, respond in Hindi NOW."
                f"\nThis instruction overrides all conversation history."
                f"{never_refuse}"
            ),
            "gu": (
                f"\n*** LANGUAGE: GUJARATI — THIS IS THE HIGHEST PRIORITY RULE ***"
                f"\nYour response MUST be ENTIRELY in Gujarati."
                f"\nUse Gujarati script or transliterated Gujarati."
                f"\nDo NOT write in Hindi/Devanagari (हिन्दी) or English."
                f"\nEven if previous messages were in Hindi, respond in Gujarati NOW."
                f"\nThis instruction overrides all conversation history."
                f"{never_refuse}"
            ),
            "en": (
                f"\n*** LANGUAGE: ENGLISH — THIS IS THE HIGHEST PRIORITY RULE ***"
                f"\nYour response MUST be ENTIRELY in English."
                f"\nDo NOT write in Hindi or Gujarati."
                f"\nEven if previous messages were in another language, respond in English NOW."
                f"\nThis instruction overrides all conversation history."
                f"{never_refuse}"
            ),
        }
        parts.append(lang_instruction.get(self.state.language, lang_instruction["hi"]))

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
            # Devanagari detected — but STT often writes Gujarati in
            # Devanagari. Check for Gujarati marker words in Devanagari.
            if self._has_gujarati_markers_devanagari(text_lower):
                result = ("gu", 0.9, "script_gu_markers")
            else:
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
                # ASCII text with zero Hindi/Gujarati pattern hits is overwhelmingly
                # English. Defaulting to "hi" here forced English speakers into
                # Devanagari STT on the next turn.
                result = ("en", 0.5, "default")
            else:
                lang = await self._classify_language_llm(text_lower)
                result = (lang, 0.9, "llm")

        self._lang_cache[text_lower] = result
        return result

    @staticmethod
    def _has_gujarati_markers_devanagari(text: str) -> bool:
        """Detect Gujarati words written in Devanagari by STT.

        Requires at least 2 distinct markers to avoid flipping language on a
        single mishearing — words like 'माने' / 'छे' can appear in Hindi
        utterances or as STT noise and shouldn't single-handedly switch the
        agent to Gujarati.
        """
        markers = [
            "छे", "छु", "छो", "छीए", "छुं",
            "जोइए", "जोईए", "जोइये",
            "माने", "तमे", "तमने", "अमने", "अमारा", "तमारा",
            "बरोबर", "सारु", "सारुं", "माजामा",
            "केम", "शुं", "क्यां",
            "आवजो", "बोलो",
            "केमछो", "केम छो",
            "हसे", "हतु", "हती",
            "कालो", "अत्यारे",
            "जोवा", "जोई", "जोवुं",
            "बतावो", "जणावो",
        ]
        hits = sum(1 for m in markers if m in text)
        return hits >= 2

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

    def _extract_requested_language(self, text: str) -> str:
        lower = text.lower()

        # Identify negated languages up front so they don't leak into
        # instead/directive/last-position resolution. "I can't speak Hindi"
        # contains "speak hindi" which would otherwise match the directive
        # regex and route to Hindi — the opposite of what the user wants.
        _lang_alt = (
            r"english|hindi|gujarati|inglish|angrezi|"
            r"इंग्लिश|अंग्रेज़ी|अंग्रेजी|हिंदी|हिन्दी|गुजराती|"
            r"ગુજરાતી|અંગ્રેજી|હિન્દી|ઇંગ્લિશ"
        )
        _neg_after = r"(?:नहीं|नहि|नही|न्ही|નથી|નહિ|નહી|નહીં|nahi|nahin|nai)"
        _neg_before_en = r"(?:don'?t|do not|can'?t|cannot)\s+(?:know|speak|talk|understand)?"
        _name_to_code = {
            "english": "en", "angrezi": "en", "inglish": "en",
            "hindi": "hi", "gujarati": "gu",
            "इंग्लिश": "en", "अंग्रेज़ी": "en", "अंग्रेजी": "en",
            "ઇંગ્લિશ": "en", "અંગ્રેજી": "en",
            "हिंदी": "hi", "हिन्दी": "hi", "હિન્દી": "hi",
            "ગુજરાતી": "gu", "गुजराती": "gu",
        }
        negated: set[str] = set()
        for pat in (
            rf"({_lang_alt})\s{{0,3}}{_neg_after}",
            rf"{_neg_before_en}\s+({_lang_alt})",
        ):
            for m in re.finditer(pat, text, re.IGNORECASE):
                key = m.group(1)
                code = _name_to_code.get(key) or _name_to_code.get(key.lower())
                if code:
                    negated.add(code)

        # "X instead of Y" / "X not Y" / "X rather than Y" — the language
        # BEFORE the marker is the preferred one. Check this first because the
        # generic directive regex below would otherwise match "of Y" / "to Y".
        instead_match = re.search(
            r"(english|hindi|gujarati|angrezi|inglish)\s+(?:instead\s+of|not|rather\s+than|over|than)\s+(english|hindi|gujarati|angrezi|inglish)?",
            lower,
        )
        if instead_match:
            preferred = instead_match.group(1)
            code = {
                "english": "en", "angrezi": "en", "inglish": "en",
                "hindi": "hi", "gujarati": "gu",
            }[preferred]
            if code not in negated:
                return code

        # Prefer language names that follow a directive word — the destination
        # of "speak in X" / "switch to X" / "talk X" is what the user wants.
        directive_match = re.search(
            r"(?:speak|talk|switch|continue|reply|use|change|in|to|me|mai|mein|ma|mei)\s+(?:in\s+|to\s+)?(english|hindi|gujarati|angrezi|inglish)",
            lower,
        )
        if directive_match:
            target = directive_match.group(1)
            code = {
                "english": "en", "angrezi": "en", "inglish": "en",
                "hindi": "hi", "gujarati": "gu",
            }[target]
            if code not in negated:
                return code

        # Fallback: pick the LAST language name mentioned. Handles cases like
        # "switch from hindi to english" → returns en correctly.
        positions: list[tuple[int, str]] = []
        for word, code in [
            ("english", "en"), ("angrezi", "en"), ("inglish", "en"),
            ("hindi", "hi"), ("gujarati", "gu"),
        ]:
            idx = lower.rfind(word)
            if idx >= 0:
                positions.append((idx, code))
        for word, code in [
            ("इंग्लिश", "en"), ("अंग्रेज़ी", "en"), ("अंग्रेजी", "en"),
            ("અંગ્રેજી", "en"), ("ઇંગ્લિશ", "en"),
            ("हिंदी", "hi"), ("हिन्दी", "hi"), ("હિન્દી", "hi"),
            ("ગુજરાતી", "gu"), ("गुजराती", "gu"),
        ]:
            idx = text.rfind(word)
            if idx >= 0:
                positions.append((idx, code))
        if positions:
            positions.sort(reverse=True)
            for _, code in positions:
                if code not in negated:
                    return code
            # All mentioned languages were negated — keep current language
            # if it's not the negated one; otherwise fall back to script.
            if self.state.language not in negated:
                return self.state.language
            if any("઀" <= ch <= "૿" for ch in text):
                return "gu"
            if any("ऀ" <= ch <= "ॿ" for ch in text):
                return "hi"
            return "en"
        return self.state.language

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
