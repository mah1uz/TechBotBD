"""
Intent router.

Classifies a query into one of three intents using simple keyword matching.
Order of checks: comparison -> filter -> recommendation (default).

Why keyword-based: zero training data, instant, transparent. For an NLP
course this is the baseline. A learned classifier could be added later
behind the same `route()` interface without touching pipeline.py.
"""

from typing import Optional


COMPARISON_KEYWORDS = [
    "তুলনা", "তুলনামূলক", "পার্থক্য", "নাকি", "কোনটা ভালো",
    "এর চেয়ে", "বনাম",
    " vs ", " vs.", "versus", "compare", "comparison", "difference",
    "better than", "or which",
]

FILTER_KEYWORDS = [
    # Budget signals
    "বাজেট", "দামের মধ্যে", "টাকার মধ্যে", "হাজারের মধ্যে", "এর নিচে",
    "under", "below", "within", "less than", "cheaper than",
    # Use cases
    "gaming", "গেমিং", "student", "শিক্ষার্থী", "office", "অফিস",
    "lightweight", "হালকা", "flagship", "ফ্ল্যাগশিপ",
    # Spec emphasis
    "ক্যামেরা", "camera", "ব্যাটারি", "battery", "র‍্যাম", "ram",
    # Quality keywords
    "সেরা", "best", "সবচেয়ে", "টপ", "top",
    "cheap", "সস্তা", "বাজেট ফোন",
]


def _norm(text: str) -> str:
    return (text or "").lower()


def route(query: str) -> str:
    """
    Return one of: 'comparison', 'filter', 'recommendation'.

    Examples:
        "iPhone 15 vs Samsung S24"        -> comparison
        "৬০ হাজারের মধ্যে ভালো ফোন"        -> filter
        "একটা গেমিং ল্যাপটপ চাই"           -> filter (gaming)
        "ভালো ফোন চাই"                     -> recommendation
    """
    q = _norm(query)

    for kw in COMPARISON_KEYWORDS:
        if kw in q:
            return "comparison"

    for kw in FILTER_KEYWORDS:
        if kw in q:
            return "filter"

    return "recommendation"


def get_sub_category_hint(query: str) -> Optional[str]:
    """
    Return a sub_category hint if the query implies one, else None.

    Used by pipeline.py for hard constraint filtering. Returns None if the
    query is generic (e.g. "good phone") so no sub-category narrowing happens.
    """
    q = _norm(query)

    is_phone = any(k in q for k in ("ফোন", "phone", "mobile", "smartphone"))
    is_laptop = any(k in q for k in ("ল্যাপটপ", "laptop", "notebook"))

    # Gaming
    if "gaming" in q or "গেমিং" in q or "gamer" in q:
        if is_laptop:
            return "gaming_laptop"
        if is_phone:
            return "gaming_phone"
        # Ambiguous — let category-level filtering handle it
        return None

    # Lightweight / ultraportable -> always laptop
    if "lightweight" in q or "হালকা" in q or "ultrabook" in q:
        return "lightweight_laptop"

    # Flagship -> phone (laptop "flagship" isn't a category here)
    if "flagship" in q or "ফ্ল্যাগশিপ" in q:
        return "flagship_phone"

    return None
