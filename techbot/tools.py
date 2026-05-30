"""
Structured operations on the product DataFrame.

Three concerns:
  1. Extracting structured info from natural-language queries (price, category)
  2. Fuzzy-matching named products for comparison queries
  3. Formatting the markdown link block shown after each LLM response

This module never calls the LLM and never imports rag_pipeline.
"""

from __future__ import annotations

import csv
import difflib
import io
import re
import urllib.parse
from typing import Optional

import pandas as pd

from techbot.config import SHOP_TEMPLATE_COLUMNS
from techbot.prompts import build_whatsapp_message


# ---------------------------------------------------------------------------
# Bengali numerals -> ASCII
# ---------------------------------------------------------------------------

_BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _to_ascii_digits(text: str) -> str:
    return text.translate(_BN_TO_EN)


# ---------------------------------------------------------------------------
# Price extraction
# ---------------------------------------------------------------------------
# We try lakh -> hajar/k -> plain number. Largest multiplier wins.

LAKH_RX = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:লাখ|lakh|lac)", re.IGNORECASE)
HAJAR_RX = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:হাজার|hajar|thousand|k\b)", re.IGNORECASE)
PLAIN_RX = re.compile(
    r"(?:৳|tk\.?|bdt|taka|টাকা)?\s*"
    r"(\d{2,7}(?:,\d{3})+|\d{4,7})"
    r"\s*(?:৳|tk\.?|bdt|taka|টাকা)?",
    re.IGNORECASE,
)


def extract_price_ceiling(query: str) -> Optional[int]:
    """
    Extract a budget ceiling from natural-language input.
    Handles Bengali numerals, multipliers (লাখ/হাজার/k), commas, ৳ symbol.

    Examples:
        "৬০ হাজারের মধ্যে"  -> 60000
        "60k laptop"        -> 60000
        "1.5 lakh budget"   -> 150000
        "under ৳60,000"      -> 60000
    """
    if not query:
        return None
    q = _to_ascii_digits(query)

    if m := LAKH_RX.search(q):
        try:
            return int(float(m.group(1).replace(",", "")) * 100_000)
        except ValueError:
            pass

    if m := HAJAR_RX.search(q):
        try:
            return int(float(m.group(1).replace(",", "")) * 1_000)
        except ValueError:
            pass

    candidates = []
    for m in PLAIN_RX.finditer(q):
        try:
            n = int(m.group(1).replace(",", ""))
            if 1_000 <= n <= 10_000_000:
                candidates.append(n)
        except ValueError:
            continue
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Category extraction
# ---------------------------------------------------------------------------

LAPTOP_KEYWORDS = ("ল্যাপটপ", "laptop", "notebook", "macbook")
PHONE_KEYWORDS = ("ফোন", "phone", "mobile", "smartphone", "iphone")


def extract_category(query: str) -> Optional[str]:
    """Return 'laptop', 'smartphone', or None."""
    if not query:
        return None
    q = query.lower()
    if any(k in q for k in LAPTOP_KEYWORDS):
        return "laptop"
    if any(k in q for k in PHONE_KEYWORDS):
        return "smartphone"
    return None


# ---------------------------------------------------------------------------
# Brand extraction
# ---------------------------------------------------------------------------

# Brands that appear in our data (or could later via shop mode).
KNOWN_BRANDS = (
    # Phone brands
    "Samsung", "Apple", "Vivo", "Realme", "Oppo", "Xiaomi", "Motorola",
    "Tecno", "Infinix", "iQOO", "Google", "ZTE", "OnePlus", "Honor",
    "Huawei", "Nothing", "Poco",
    # Laptop brands
    "Lenovo", "Asus", "Acer", "HP", "MSI", "Dell", "Gigabyte",
    "Microsoft", "Razer", "LG", "TECNO",
)

# Sub-brand and product-line aliases. Map lowercase fragment -> canonical brand.
BRAND_ALIASES = {
    "iphone": "Apple", "macbook": "Apple", "ipad": "Apple",
    "galaxy": "Samsung",
    "pixel": "Google",
    "redmi": "Xiaomi", "poco": "Xiaomi",
    "thinkpad": "Lenovo", "ideapad": "Lenovo",
    "vivobook": "Asus", "zenbook": "Asus", "rog": "Asus",
    "aspire": "Acer", "predator": "Acer", "nitro": "Acer",
    "pavilion": "HP", "envy": "HP", "omen": "HP",
    "inspiron": "Dell", "xps": "Dell",
    "surface": "Microsoft",
}


def extract_brand(query: str) -> Optional[str]:
    """
    Return a canonical brand name if the query mentions one, else None.
    Handles direct brand names and product-line aliases ("iphone" -> Apple).
    """
    if not query:
        return None
    q = query.lower()
    for brand in KNOWN_BRANDS:
        if brand.lower() in q:
            return brand
    for alias, brand in BRAND_ALIASES.items():
        if alias in q:
            return brand
    return None


# ---------------------------------------------------------------------------
# "Cheap" detection — used to flip sort order to ascending price
# ---------------------------------------------------------------------------

CHEAP_KEYWORDS = (
    "সস্তা", "সবচেয়ে সস্তা", "কম দামের", "কম দামে", "কমদামী",
    "cheap", "cheapest", "lowest price", "low budget",
    "আরেকটু কম", "আরো সস্তা",
)


def is_cheap_query(query: str) -> bool:
    """True if the query asks for the cheapest option(s)."""
    if not query:
        return False
    q = query.lower()
    return any(k.lower() in q for k in CHEAP_KEYWORDS)


# ---------------------------------------------------------------------------
# Pandas-based filter (hard constraints)
# ---------------------------------------------------------------------------

def filter_products(df: pd.DataFrame,
                    price_ceiling: Optional[int] = None,
                    category: Optional[str] = None,
                    brand: Optional[str] = None,
                    sub_category: Optional[str] = None) -> pd.DataFrame:
    """
    Apply hard constraints via pandas. Returns a filtered DataFrame
    (caller decides ranking/limit). Empty if nothing matches.

    sub_category match is "soft": if no rows match the strict sub-category,
    fall back to the broader category result (rather than returning empty).
    """
    out = df
    if category:
        out = out[out["category"] == category]
    if brand:
        out = out[out["brand"].astype(str).str.lower() == brand.lower()]
    if price_ceiling is not None:
        out = out[(out["price_bdt"] > 0) & (out["price_bdt"] <= price_ceiling)]
    if sub_category:
        strict = out[out["sub_category"] == sub_category]
        if len(strict) > 0:
            out = strict
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

COMPARISON_SPLITTERS = re.compile(
    r"\s+(?:vs\.?|versus|নাকি|তুলনা|তুলনামূলক|পার্থক্য|বনাম|or)\s+",
    re.IGNORECASE,
)

FILLER_RX = re.compile(
    r"\b(কোনটা ভালো|কোনটা সেরা|which is better|which one|"
    r"better than|এর চেয়ে|কে ভালো|তুলনা করো|তুলনা|কোনটি)\b",
    re.IGNORECASE,
)


def _strip_filler(s: str) -> str:
    s = FILLER_RX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" ?!.,")


# Map regex -> prefix to prepend. When user types "S24 Ultra" the brand
# context is missing; we fix that before fuzzy matching against full names
# like "Samsung Galaxy S26 Ultra 5G".
FRAGMENT_EXPANSIONS = [
    (re.compile(r"^s\d{2}\b", re.IGNORECASE), "Samsung Galaxy"),
    (re.compile(r"^a\d{2}\b", re.IGNORECASE), "Samsung Galaxy"),
    (re.compile(r"^pixel\s+\d", re.IGNORECASE), "Google"),
    (re.compile(r"^macbook\b", re.IGNORECASE), "Apple"),
    (re.compile(r"^thinkpad\b", re.IGNORECASE), "Lenovo"),
    (re.compile(r"^aspire\b", re.IGNORECASE), "Acer"),
    (re.compile(r"^vivobook\b|^zenbook\b|^rog\b", re.IGNORECASE), "Asus"),
    (re.compile(r"^ideapad\b", re.IGNORECASE), "Lenovo"),
]


def _expand_fragment(fragment: str) -> str:
    """Prepend brand context to abbreviated model-number-style fragments."""
    for rx, prefix in FRAGMENT_EXPANSIONS:
        if rx.search(fragment):
            return f"{prefix} {fragment}"
    return fragment


def compare_products(query: str, df: pd.DataFrame) -> dict:
    """
    Split query on a comparison keyword, fuzzy-match each fragment against
    df['name']. Returns {"product_a", "product_b", "error"}.

    Cutoff is 0.5 (was 0.3). Combined with fragment expansion this avoids
    the false positives we hit before — "S24 Ultra" was matching "Vivo T4
    Ultra" because of the shared "Ultra" + "T4"/"24" character overlap.
    """
    if df is None or len(df) == 0:
        return {"product_a": None, "product_b": None,
                "error": "ডাটাবেসে কোনো পণ্য নেই।"}

    parts = [_strip_filler(p) for p in COMPARISON_SPLITTERS.split(query) if p.strip()]
    if len(parts) < 2:
        return {"product_a": None, "product_b": None,
                "error": "তুলনার জন্য দুটি পণ্য চিহ্নিত করতে পারিনি।"}

    name_pool = df["name"].astype(str).tolist()

    def _match(fragment: str) -> Optional[dict]:
        if not fragment:
            return None
        expanded = _expand_fragment(fragment)
        # Fuzzy match (Ratcliff–Obershelp via difflib)
        if matches := difflib.get_close_matches(
                expanded.lower(),
                [n.lower() for n in name_pool],
                n=1, cutoff=0.5):
            # Recover original-case name (case-sensitive lookup)
            for n in name_pool:
                if n.lower() == matches[0]:
                    return df[df["name"] == n].iloc[0].to_dict()
        # Substring fallback: only if user fragment is contained in product name
        # (not the other way — avoids "S24" matching "S26")
        f = fragment.lower()
        for n in name_pool:
            if f in n.lower() and len(f) >= 4:
                return df[df["name"] == n].iloc[0].to_dict()
        return None

    a, b = _match(parts[0]), _match(parts[1])

    # If both fragments fuzzy-matched to the same product, that means at
    # most one was a real match (and probably neither). Tell the user.
    if a and b and a.get("name") == b.get("name"):
        return {
            "product_a": a, "product_b": None,
            "error": (f"'{parts[0]}' এবং '{parts[1]}' দুটোই '{a['name']}'-এর সাথে "
                      f"মিলেছে। সম্ভবত আপনার চাওয়া পণ্যগুলো আমাদের কাছে নেই।"),
        }

    error = None
    if a is None and b is None:
        error = (f"তুলনার জন্য চাওয়া পণ্যগুলো ('{parts[0]}', '{parts[1]}') "
                 f"আমাদের ডাটাবেসে পাওয়া যায়নি।")
    elif a is None:
        error = f"প্রথম পণ্যটি ('{parts[0]}') পাওয়া যায়নি।"
    elif b is None:
        error = f"দ্বিতীয় পণ্যটি ('{parts[1]}') পাওয়া যায়নি।"

    return {"product_a": a, "product_b": b, "error": error}


# ---------------------------------------------------------------------------
# Markdown link block (shown after the LLM response)
# ---------------------------------------------------------------------------

def build_link_block(products: list[dict],
                     shop_whatsapp: Optional[str] = None,
                     user_query: str = "",
                     language: str = "bn") -> str:
    if not products:
        return ""

    lines = ["", "---", "📦 **পণ্যের তথ্য ও লিংক:**", ""]
    for i, p in enumerate(products, 1):
        name = p.get("name", "Unknown")
        price = p.get("price_bdt", 0)
        shop = p.get("shop", "")
        url = p.get("url", "")

        price_str = f"৳{int(price):,}" if price and int(price) > 0 else "দাম জানা নেই"
        if url:
            shop_link = f"[{shop}-এ দেখুন]({url})"
        elif shop:
            shop_link = f"({shop})"
        else:
            shop_link = ""

        line = f"{i}. **{name}** — {price_str} — {shop_link}"

        if shop_whatsapp:
            wa_msg = build_whatsapp_message(p, user_query or "", shop, language)
            wa_url = f"https://wa.me/{shop_whatsapp}?text={urllib.parse.quote(wa_msg)}"
            line += f" — [📱 WhatsApp-এ জিজ্ঞেস করুন]({wa_url})"

        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV template for shop uploads
# ---------------------------------------------------------------------------

def get_template_csv() -> str:
    """Header + one example row to make the format obvious."""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(SHOP_TEMPLATE_COLUMNS)
    writer.writerow([
        "Samsung Galaxy A55", "smartphone", "Samsung", "45000",
        "Exynos 1480", "8", "128GB", "", "6.6",
        "5000", "50", "smartphone", "https://yourshop.com/galaxy-a55",
    ])
    return buf.getvalue()
