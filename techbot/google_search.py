"""
Web search fallback (Google Custom Search API).

Activated by pipeline.py only when:
  - RAG returns low confidence
  - is_available() returns True (both API key and CSE ID set)

Failures never raise — they return [] so the pipeline keeps running.
"""

from __future__ import annotations

import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

CSE_URL = "https://www.googleapis.com/customsearch/v1"
DEFAULT_SHOP_DOMAINS = ["ryans.com", "startech.com.bd", "dazzle.com.bd",
                        "pickaboo.com", "binary.com.bd"]


def is_available() -> bool:
    return bool(GOOGLE_API_KEY and GOOGLE_CSE_ID)


def search_products(query: str,
                    shop_urls: Optional[list[str]] = None,
                    num_results: int = 5) -> list[dict]:
    """
    Run a Google CSE search restricted to gadget-shop domains.
    Returns [{"title", "link", "snippet"}, ...] or [] on any error.
    """
    if not is_available():
        return []

    domains = shop_urls or DEFAULT_SHOP_DOMAINS
    site_filter = " OR ".join(f"site:{d}" for d in domains)
    full_q = f"{query} ({site_filter})"

    try:
        r = requests.get(
            CSE_URL,
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": full_q,
                "num": num_results,
            },
            timeout=10,
        )
        if r.status_code != 200:
            print(f"[GoogleSearch] HTTP {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
    except Exception as e:
        print(f"[GoogleSearch] Error: {e}")
        return []

    items = data.get("items", []) or []
    return [
        {
            "title": it.get("title", ""),
            "link": it.get("link", ""),
            "snippet": it.get("snippet", ""),
        }
        for it in items
    ]
