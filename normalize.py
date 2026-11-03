"""
normalize.py
------------
Fixes the same class of problem you already solved for Piper: raw digits,
phone numbers, and symbols being mispronounced (or read digit-by-digit) by
the TTS engine.

OmniVoice's own `normalize_text=True` flag only fully covers Chinese and
English (via WeTextProcessing). Urdu falls back to `num2words`, which does
not support Urdu at all — so Urdu digits/phone numbers/symbols still need to
be expanded to words BEFORE the text reaches OmniVoice. That's what this
file does.

Usage:
    from normalize import normalize_text
    clean = normalize_text("Call me at 0301-1234567 or send 25% off.", lang="ur")
    clean = normalize_text("Call me at 0301-1234567 or send 25% off.", lang="en")
"""

import re

try:
    from num2words import num2words
    _HAS_NUM2WORDS = True
except ImportError:
    _HAS_NUM2WORDS = False

# ---------------------------------------------------------------------------
# Urdu digit -> word map (0-9), used for digit-by-digit reading (phone
# numbers, IDs) and as building blocks for small numbers.
# ---------------------------------------------------------------------------
UR_DIGITS = {
    "0": "صفر", "1": "ایک", "2": "دو", "3": "تین", "4": "چار",
    "5": "پانچ", "6": "چھ", "7": "سات", "8": "آٹھ", "9": "نو",
}

UR_TEENS_TENS = {
    10: "دس", 11: "گیارہ", 12: "بارہ", 13: "تیرہ", 14: "چودہ", 15: "پندرہ",
    16: "سولہ", 17: "سترہ", 18: "اٹھارہ", 19: "انیس", 20: "بیس",
    30: "تیس", 40: "چالیس", 50: "پچاس", 60: "ساٹھ", 70: "ستر",
    80: "اسی", 90: "نوے", 100: "سو",
}

# Common symbols that should never reach the model as raw characters.
SYMBOL_WORDS = {
    "en": {"%": " percent", "&": " and", "@": " at", "#": " number ", "+": " plus ", "=": " equals "},
    "ur": {"%": " فیصد", "&": " اور", "@": " ایٹ", "#": " نمبر ", "+": " جمع ", "=": " برابر "},
}


def _urdu_number_to_words(n: int) -> str:
    """Best-effort small-number Urdu conversion; falls back to digit-by-digit
    for anything larger than 2 digits (safe, always intelligible)."""
    if n in UR_TEENS_TENS:
        return UR_TEENS_TENS[n]
    if 0 <= n <= 9:
        return UR_DIGITS[str(n)]
    if 21 <= n <= 99:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0 and tens in UR_TEENS_TENS:
            return UR_TEENS_TENS[tens]
        # Compound Urdu numbers (21, 22, ...) aren't simple concatenations
        # linguistically, so for safety this falls back to digit-by-digit
        # rather than producing an incorrect compound word.
        return " ".join(UR_DIGITS[d] for d in str(n))
    # 3+ digit numbers: read digit-by-digit (safe, matches how phone
    # numbers/IDs are conventionally read aloud anyway).
    return " ".join(UR_DIGITS[d] for d in str(n))


def _expand_number(match: "re.Match", lang: str) -> str:
    raw = match.group(0)
    digits_only = raw.replace(",", "")

    # Decimal numbers (e.g. 25.50) -> read as "<int> point <digits>"
    if "." in digits_only:
        whole_str, frac_str = digits_only.split(".", 1)
        try:
            whole = int(whole_str)
        except ValueError:
            return raw
        if lang == "ur":
            whole_words = _urdu_number_to_words(whole)
            frac_words = " ".join(UR_DIGITS[d] for d in frac_str)
            return f"{whole_words} اعشاریہ {frac_words}"
        whole_words = num2words(whole) if _HAS_NUM2WORDS else whole_str
        frac_words = " ".join(num2words(int(d)) for d in frac_str) if _HAS_NUM2WORDS else frac_str
        return f"{whole_words} point {frac_words}"

    try:
        n = int(digits_only)
    except ValueError:
        return raw

    if lang == "ur":
        return _urdu_number_to_words(n)

    # English
    if _HAS_NUM2WORDS:
        try:
            return num2words(n)
        except Exception:
            pass
    return raw  # last-resort fallback, leaves digits untouched


def _expand_phone_number(match: "re.Match", lang: str) -> str:
    """Phone numbers are always read digit-by-digit, in either language."""
    digits = re.sub(r"\D", "", match.group(0))
    if lang == "ur":
        return " ".join(UR_DIGITS[d] for d in digits)
    if _HAS_NUM2WORDS:
        return " ".join(num2words(int(d)) for d in digits)
    return " ".join(digits)


PHONE_PATTERN = re.compile(r"\b0\d{2,4}[-\s]?\d{6,8}\b|\+\d[\d\s-]{8,}\d")
NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")


def normalize_text(text: str, lang: str = "en") -> str:
    """Normalize digits, phone numbers, and symbols to spoken-word form.

    Args:
        text: raw text (e.g. dummy LLM output).
        lang: "en" or "ur" — controls which word set is used.
    Returns:
        Normalized text, safe to pass directly to OmniVoice.
    """
    if lang not in ("en", "ur"):
        lang = "en"

    # 1. Phone numbers first (before generic number expansion swallows them)
    text = PHONE_PATTERN.sub(lambda m: _expand_phone_number(m, lang), text)

    # 2. Remaining standalone numbers
    text = NUMBER_PATTERN.sub(lambda m: _expand_number(m, lang), text)

    # 3. Symbols
    for sym, word in SYMBOL_WORDS[lang].items():
        text = text.replace(sym, word)

    # 4. Collapse extra whitespace introduced by expansion
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    samples = [
        ("Please call 0301-1234567 or dial +92 300 1234567.", "en"),
        ("Total is $25.50, that's 15% off the usual price.", "en"),
        ("براہ کرم 0301-1234567 پر کال کریں۔", "ur"),
        ("قیمت 500 روپے ہے، 20% رعایت کے ساتھ۔", "ur"),
    ]
    print("Normalization preview (no model needed):\n")
    for text, lang in samples:
        print(f"[{lang}] before: {text}")
        print(f"[{lang}] after : {normalize_text(text, lang)}\n")
