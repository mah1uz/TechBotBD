"""
Text preprocessing for Bengali/Banglish/English queries.

Single responsibility: clean and normalize input text so downstream
retrieval and intent detection see a consistent form.

Design notes:
  - Bengali Unicode is preserved untouched (range \u0980-\u09FF).
  - English is lowercased.
  - Banglish gadget terms are mapped to their Bengali equivalents so
    that a query like "best gaming laptop under 60k" becomes
    "সেরা গেমিং ল্যাপটপ under 60k" — keeping numbers + price words
    intact for tools.extract_price_ceiling.
  - Numbers (Arabic and Bengali) are NEVER touched here. Price extraction
    in tools.py needs them as-is.
"""

import re
import unicodedata

# Banglish/English -> Bengali mappings.
# Order matters: longer/more-specific first so we don't half-match.
BANGLISH_MAP = {
    # Categories
    r"\bsmartphone\b": "ফোন",
    r"\bsmart phone\b": "ফোন",
    r"\bphone\b": "ফোন",
    r"\bmobile\b": "ফোন",
    r"\bnotebook\b": "ল্যাপটপ",
    r"\blaptop\b": "ল্যাপটপ",

    # Specs
    r"\bcamera\b": "ক্যামেরা",
    r"\bcam\b": "ক্যামেরা",
    r"\bbattery\b": "ব্যাটারি",
    r"\bbackup\b": "ব্যাটারি",
    r"\bmemory\b": "র‍্যাম",
    r"\bram\b": "র‍্যাম",
    r"\bstorage\b": "স্টোরেজ",
    r"\bspace\b": "স্টোরেজ",
    r"\bprocessor\b": "প্রসেসর",
    r"\bchipset\b": "প্রসেসর",
    r"\bcpu\b": "প্রসেসর",
    r"\bdisplay\b": "ডিসপ্লে",
    r"\bscreen\b": "ডিসপ্লে",

    # Use cases
    r"\bgaming\b": "গেমিং",
    r"\bgamer\b": "গেমিং",
    r"\bstudent\b": "শিক্ষার্থী",
    r"\boffice\b": "অফিস",
    r"\blightweight\b": "হালকা",
    r"\bflagship\b": "ফ্ল্যাগশিপ",

    # Price/Budget
    r"\bbudget\b": "বাজেট",
    r"\bprice\b": "দাম",
    r"\bdaam\b": "দাম",
    r"\bcost\b": "দাম",
    r"\btaka\b": "টাকা",
    r"\btk\b": "টাকা",
    r"\bbdt\b": "টাকা",
    r"\bcheap\b": "সস্তা",
    r"\bsasta\b": "সস্তা",

    # Comparison / Quality
    r"\bcompare\b": "তুলনা",
    r"\bcomparison\b": "তুলনা",
    r"\bdifference\b": "পার্থক্য",
    r"\bversus\b": "vs",
    r"\bbest\b": "সেরা",
    r"\bsero\b": "সেরা",
    r"\bbhalo\b": "ভালো",
    r"\bvalo\b": "ভালো",
    r"\bgood\b": "ভালো",

    # Common Banglish question words (preserved as Bengali stems)
    r"\bkonta\b": "কোনটা",
    r"\bkon\b": "কোন",
    r"\bkemon\b": "কেমন",
    r"\bkivabe\b": "কিভাবে",
    r"\bkothay\b": "কোথায়",
    r"\bki\b": "কি",
    r"\bami\b": "আমি",
    r"\bamar\b": "আমার",
    r"\btomar\b": "তোমার",
    r"\bdao\b": "দাও",
    r"\bdekho\b": "দেখাও",
    r"\bdekhao\b": "দেখাও",
    r"\bsuggest\b": "সাজেস্ট",
    r"\brecommend\b": "সুপারিশ",
}


def _normalize_unicode(text: str) -> str:
    """NFC-normalize so visually-identical Bengali codepoints compare equal."""
    return unicodedata.normalize("NFC", text)


def _lowercase_ascii_only(text: str) -> str:
    """Lowercase only ASCII letters; leave Bengali untouched."""
    return "".join(c.lower() if c.isascii() and c.isalpha() else c for c in text)


def _apply_banglish_map(text: str) -> str:
    for pattern, replacement in BANGLISH_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _strip_special_chars(text: str) -> str:
    """
    Keep: Bengali Unicode, ASCII alphanumerics, whitespace, basic punctuation.
    Drop: emojis, decorative symbols, currency symbols (price extractor
    handles ৳ separately if needed — but we keep the digits).
    """
    # Allowed: \u0980-\u09FF (Bengali), 0-9, a-z, A-Z, whitespace,
    # and . , - + ? / ৳ (Bengali Taka sign U+09F3)
    return re.sub(r"[^\u0980-\u09FF0-9a-zA-Z\s\.,\-\+\?/৳]", " ", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text: str) -> str:
    """
    Main entry point. Returns a cleaned string suitable for embedding
    and intent routing.

    Input may be Bengali, Banglish (romanized Bengali), English, or mixed.
    Numbers are preserved verbatim for downstream price extraction.
    """
    if not text or not isinstance(text, str):
        return ""

    text = _normalize_unicode(text)
    text = _lowercase_ascii_only(text)
    text = _apply_banglish_map(text)
    text = _strip_special_chars(text)
    text = _normalize_whitespace(text)
    return text
