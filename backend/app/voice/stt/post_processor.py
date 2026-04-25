"""
STT post-processing for domain-specific vocabulary correction.

Fixes common misrecognitions of city names, BHK types, real estate terms,
and Hindi/Gujarati transliterated words. Applied after STT transcription
and before intent detection.
"""

from __future__ import annotations

import re
import logging
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# City name corrections (Surat, Vapi, Silvassa and common misrecognitions)
# ---------------------------------------------------------------------------

_CITY_CORRECTIONS: dict[str, str] = {
    "super": "surat",
    "soorat": "surat",
    "surate": "surat",
    "sorat": "surat",
    "surt": "surat",
    "suraat": "surat",
    "suraht": "surat",
    "surth": "surat",
    "wapi": "vapi",
    "bapi": "vapi",
    "vaapi": "vapi",
    "wappi": "vapi",
    "vapee": "vapi",
    "waapi": "vapi",
    "silver": "silvassa",
    "silvasa": "silvassa",
    "silver sa": "silvassa",
    "silversa": "silvassa",
    "selvassa": "silvassa",
    "silvasa": "silvassa",
    "silwasa": "silvassa",
    "silvassa": "silvassa",
    "dadra": "silvassa",
}

_CANONICAL_CITIES = {"surat", "vapi", "silvassa"}

# ---------------------------------------------------------------------------
# BHK normalization
# ---------------------------------------------------------------------------

_WORD_TO_NUM: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "do": "2", "teen": "3", "char": "4", "paanch": "5", "panch": "5",
    "be": "2", "tran": "3", "chaar": "4",
    "ek": "1",
    # Hindi script numerals
    "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5",
    "फोर": "4", "थ्री": "3", "टू": "2", "फाइव": "5",
    # Gujarati script numerals
    "એક": "1", "બે": "2", "ત્રણ": "3", "ચાર": "4", "પાંચ": "5",
}

# Hindi/Gujarati script BHK variants as commonly emitted by Indian-locale STT
_DEVANAGARI_BHK_PATTERNS: dict[str, str] = {
    "बीएचके": "bhk",
    "बी एच के": "bhk",
    "जीएचके": "bhk",   # common Sarvam misrecognition of BHK
    "जी एच के": "bhk",
    "बिएचके": "bhk",
    "बीएचकी": "bhk",
    "બીએચકે": "bhk",   # Gujarati
    "બી એચ કે": "bhk",
    "बेडरूम": "bhk",
    "બેડરૂમ": "bhk",
}

_BHK_RE = re.compile(
    r"(\d)\s*b\s*h\s*k",
    re.IGNORECASE,
)

_WORD_BHK_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WORD_TO_NUM) + r")\s+b\s*h\s*k\b",
    re.IGNORECASE,
)

_BEDROOM_RE = re.compile(
    r"(\d)\s*(?:bed\s*room|bedroom)s?",
    re.IGNORECASE,
)

_WORD_BEDROOM_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WORD_TO_NUM) + r")\s+(?:bed\s*room|bedroom)s?\b",
    re.IGNORECASE,
)


def _normalize_bhk(text: str) -> str:
    # First: convert Devanagari/Gujarati BHK variants to ASCII "bhk"
    for pattern, replacement in _DEVANAGARI_BHK_PATTERNS.items():
        if pattern in text:
            text = text.replace(pattern, replacement)

    # Convert Devanagari/Gujarati number words to digits when near "bhk"
    for word, num in _WORD_TO_NUM.items():
        if word in text and not word.isascii():
            bhk_nearby = re.search(
                re.escape(word) + r"\s*(?:bhk|b\s*h\s*k|बीएचके|जीएचके|બીએચકે|बेडरूम|બેડરૂમ)",
                text
            )
            if bhk_nearby:
                text = text.replace(word, num, 1)

    text = _BHK_RE.sub(r"\1 bhk", text)

    def _word_bhk_replace(m: re.Match) -> str:
        word = m.group(1).lower()
        num = _WORD_TO_NUM.get(word, word)
        return f"{num} bhk"

    text = _WORD_BHK_RE.sub(_word_bhk_replace, text)
    text = _BEDROOM_RE.sub(r"\1 bhk", text)

    def _word_bedroom_replace(m: re.Match) -> str:
        word = m.group(1).lower()
        num = _WORD_TO_NUM.get(word, word)
        return f"{num} bhk"

    text = _WORD_BEDROOM_RE.sub(_word_bedroom_replace, text)
    return text


# ---------------------------------------------------------------------------
# Hindi / Gujarati common word corrections
# ---------------------------------------------------------------------------

_HINDI_CORRECTIONS: dict[str, str] = {
    "namasthe": "namaste",
    "namashte": "namaste",
    "namasthey": "namaste",
    "ji haan": "ji haan",
    "jee": "ji",
    "bilkool": "bilkul",
    "makan": "makan",
    "makaan": "makan",
    "kiraya": "kiraya",
    "dekhna": "dekhna",
    "dikhao": "dikhao",
    "ghar": "ghar",
    "flat": "flat",
    "bhada": "bhada",
    "khareedna": "kharidna",
    "interested": "interested",
    "samay": "samay",
    "subah": "subah",
    "dopahar": "dopahar",
    "shaam": "shaam",
    "kal": "kal",
    "parso": "parso",
    "aaj": "aaj",
}

_GUJARATI_CORRECTIONS: dict[str, str] = {
    "kem cho": "kem cho",
    "kemcho": "kem cho",
    "majama": "maja ma",
    "maja maa": "maja ma",
    "bhai": "bhai",
    "joie": "joie",
    "joiye": "joiye",
    "gher": "ghar",
    "makan": "makan",
    "aavjo": "aavjo",
    "saru": "saru",
    "barobar": "barobar",
    "savare": "savare",
    "bapore": "bapore",
    "saanje": "saanje",
    "aaje": "aaje",
    "kaale": "kaale",
}

# ---------------------------------------------------------------------------
# Phonetic cross-language corrections
# STT often mistranscribes Hindi/Gujarati phrases as similar-sounding
# English words. These corrections map common misrecognitions back.
# ---------------------------------------------------------------------------

_PHONETIC_CORRECTIONS: dict[str, str] = {
    # Gujarati "theek che" (it's fine) misheard as English
    "shoe care": "theek che",
    "she care": "theek che",
    "tea care": "theek che",
    "tick che": "theek che",
    "thick che": "theek che",
    "take care": "theek che",
    # "kem cho" (how are you) misheard
    "came show": "kem cho",
    "come show": "kem cho",
    "cam show": "kem cho",
    "kim cho": "kem cho",
    # "haan" / "ha" (yes) misheard
    "han": "haan",
    "hahn": "haan",
    "huh": "haan",
    # "nahi" / "nai" (no) misheard
    "nah he": "nahi",
    "na he": "nahi",
    "na hi": "nahi",
    "nye": "nahi",
    # "chahiye" (want/need) misheard
    "chai": "chahiye",
    "chai yeah": "chahiye",
    "cha he a": "chahiye",
    # "dekhna" (to see/look) misheard
    "they can": "dekhna",
    "deck na": "dekhna",
    # "abhi" (now) misheard
    "abbey": "abhi",
    "a be": "abhi",
    # "accha" / "acha" (good/okay) misheard
    "at cha": "accha",
    "attach a": "accha",
    "each a": "accha",
    # "bilkul" (absolutely) misheard
    "bill cool": "bilkul",
    "bill call": "bilkul",
    # "zaroor" (definitely) misheard
    "the rude": "zaroor",
    "jar or": "zaroor",
    # "samajh" (understand) misheard
    "some much": "samajh",
    "some age": "samajh",
    # "joiye" (want, Gujarati) misheard
    "joey": "joiye",
    "joy": "joiye",
    # "batao" (tell me) misheard
    "but how": "batao",
    "but ao": "batao",
    # Common Hinglish
    "money dubhe che joiye che": "mane dubhe che joiye che",
}

_REAL_ESTATE_CORRECTIONS: dict[str, str] = {
    "pramuk": "pramukh",
    "pramuk group": "pramukh group",
    "pramuk groups": "pramukh group",
    "praamukh": "pramukh",
    "premium": "pramukh",
    "site visit": "site visit",
    "sight visit": "site visit",
    "booking": "booking",
    "book": "book",
    "property": "property",
    "apartment": "apartment",
    "flat": "flat",
}

# ---------------------------------------------------------------------------
# Fuzzy location matching
# ---------------------------------------------------------------------------


def match_location(text: str) -> str | None:
    """Return canonical city name if any word closely matches a known location."""
    words = text.lower().split()
    for word in words:
        if word in _CANONICAL_CITIES:
            return word
        if word in _CITY_CORRECTIONS:
            return _CITY_CORRECTIONS[word]
        for city in _CANONICAL_CITIES:
            if fuzz.ratio(word, city) >= 75:
                return city
    return None


def match_bhk(text: str) -> str | None:
    """Extract BHK type from text. Returns e.g. '2 bhk' or None."""
    normalized = _normalize_bhk(text.lower())
    m = re.search(r"(\d)\s*bhk", normalized, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 5:
            return f"{num} bhk"
    return None


# ---------------------------------------------------------------------------
# Main post-processing pipeline
# ---------------------------------------------------------------------------


def _apply_corrections(text: str, corrections: dict[str, str]) -> str:
    lower = text.lower()
    for wrong, right in corrections.items():
        if wrong in lower:
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
            lower = text.lower()
    return text


def post_process_transcript(text: str, language: str = "en") -> str:
    """
    Apply domain-specific corrections to STT output.

    Order: BHK normalization -> city corrections -> language-specific
    corrections -> real estate terms.
    """
    if not text or not text.strip():
        return text

    result = _apply_corrections(text, _PHONETIC_CORRECTIONS)
    result = _normalize_bhk(result)
    result = _apply_corrections(result, _CITY_CORRECTIONS)
    result = _apply_corrections(result, _REAL_ESTATE_CORRECTIONS)

    if language in ("hi", "en"):
        result = _apply_corrections(result, _HINDI_CORRECTIONS)
    if language in ("gu", "en"):
        result = _apply_corrections(result, _GUJARATI_CORRECTIONS)

    if result != text:
        logger.info("[STT-PP] %r -> %r", text, result)

    return result.strip()
