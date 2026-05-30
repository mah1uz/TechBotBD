"""
Pipeline orchestrator. Single chat() entry point.

Architecture: HYBRID RETRIEVAL.

  pandas filter (hard constraints) -> RAG re-rank (semantic ordering) -> LLM

The previous version tried RAG-first-then-filter, which fails when hard
constraints are tight (e.g. "budget laptop under 60k" — RAG's top 10 most
similar laptops were all premium, post-filter zeroed everything out).

Now:
  1. Extract hard constraints from query (price, category, brand, sub-cat)
     plus carry forward brand/category from recent conversation history.
  2. Apply them as a pandas filter on the active dataframe.
  3. RAG retrieves top-K from the FULL corpus, then we INTERSECT with the
     pandas-filtered set. RAG provides semantic ranking within the
     constraint-narrowed candidate pool.
  4. Sort by price asc/desc when the query implies it; otherwise keep
     RAG order.
  5. Confidence floor: if RAG's top score is too low and we have no hard
     constraints to back us up, return zero products. Prevents showing
     irrelevant cards for off-topic queries (e.g. "drone camera").
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from techbot import google_search, llm, prompts, router, tools
from techbot.preprocessor import preprocess
from techbot.rag_pipeline import TechBotRAG
from techbot.config import CONFIDENCE_MEDIUM


# ---------------------------------------------------------------------------
# Conversational context — carry brand/category forward
# ---------------------------------------------------------------------------

def _carry_context(query: str, history: list[dict]) -> str:
    """
    If the current query lacks a brand/category but a recent turn had one,
    prepend it. Lets follow-ups like "give cheaper ones" inherit the
    "Apple phones" context from the previous turn.
    """
    cur_brand = tools.extract_brand(query)
    cur_cat = tools.extract_category(query)
    if cur_brand and cur_cat:
        return query

    add_brand = None
    add_cat = None
    # Look back at last 4 messages (2 user + 2 assistant)
    for msg in reversed((history or [])[-4:]):
        text = msg.get("content", "") or ""
        if not cur_brand and not add_brand:
            add_brand = tools.extract_brand(text)
        if not cur_cat and not add_cat:
            add_cat = tools.extract_category(text)
        if (cur_brand or add_brand) and (cur_cat or add_cat):
            break

    parts = []
    if add_brand and not cur_brand:
        parts.append(add_brand)
    if add_cat and not cur_cat:
        parts.append(add_cat)
    return f"{' '.join(parts)} {query}" if parts else query


# ---------------------------------------------------------------------------
# Hybrid retrieval — pandas filter + RAG re-rank
# ---------------------------------------------------------------------------

def _hybrid_retrieve(query: str,
                     augmented_query: str,
                     df: pd.DataFrame,
                     rag_engine: TechBotRAG,
                     ceiling: Optional[int],
                     category: Optional[str],
                     brand: Optional[str],
                     sub_category: Optional[str]) -> list[dict]:
    """
    Filter df by hard constraints, then rank by RAG similarity.
    Returns up to 5 products (raw — caller does final sort).

    Variant handling: products with the same `name` may appear at different
    prices (different RAM/storage configs). When intersecting RAG candidates
    with the pandas-filtered set, we always use the cheapest variant within
    the filter — RAG might return a more-expensive variant which would
    silently violate the user's budget.
    """
    filtered = tools.filter_products(df, ceiling, category, brand, sub_category)
    if len(filtered) == 0:
        return []

    # Per name, keep the cheapest row that survives the filter
    cheapest = (filtered.sort_values("price_bdt")
                        .drop_duplicates("name", keep="first"))
    name_to_row = {str(row["name"]): row.to_dict()
                   for _, row in cheapest.iterrows()}

    # If the filter is tight, just return all (sorted by price asc as base order)
    if len(name_to_row) <= 5:
        out = []
        for d in cheapest.head(5).to_dict("records"):
            d.setdefault("faiss_score", 0.6)
            out.append(d)
        return out

    # RAG over full corpus, then map each candidate to its filtered cheapest variant
    candidates = rag_engine.retrieve(augmented_query, top_k=20)
    products: list[dict] = []
    seen = set()
    for c in candidates:
        name = str(c.get("name", ""))
        if name in name_to_row and name not in seen:
            row = dict(name_to_row[name])
            row["faiss_score"] = c.get("faiss_score", 0.5)
            products.append(row)
            seen.add(name)

    # If RAG didn't surface enough, append remaining filtered rows
    if len(products) < 3:
        for name, row in name_to_row.items():
            if name not in seen:
                row = dict(row)
                row["faiss_score"] = 0.5
                products.append(row)
                seen.add(name)
                if len(products) >= 5:
                    break

    return products


# ---------------------------------------------------------------------------
# Final sorting
# ---------------------------------------------------------------------------

def _smart_sort(products: list[dict],
                ceiling: Optional[int],
                cheap: bool) -> list[dict]:
    """
    - If user gave a budget ceiling: most-expensive-within-budget first
      (best value within budget heuristic).
    - If user asked for "cheap": ascending price.
    - Otherwise: keep RAG order (already by similarity).
    """
    if ceiling is not None:
        return sorted(products, key=lambda p: p.get("price_bdt", 0), reverse=True)
    if cheap:
        # Treat 0 (unknown) as worst when sorting ascending
        return sorted(products,
                      key=lambda p: p.get("price_bdt", 0) or 10**12)
    return products


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chat(
    query: str,
    history: list[dict],
    df: pd.DataFrame,
    rag_engine: TechBotRAG,
    backend: str = "groq",
    language: str = "bn",
    shop_whatsapp: Optional[str] = None,
    mode: str = "general",
) -> dict:
    """Orchestrate one turn of conversation. See module docstring."""
    if not (query or "").strip():
        return {
            "response": "প্রশ্ন লিখুন।" if language == "bn" else "Please enter a question.",
            "link_block": "", "products": [], "intent": "recommendation",
            "confidence": "low", "used_web_search": False, "backend_used": backend,
        }

    # --- Step 1: classify intent + carry context ---
    augmented = _carry_context(query, history or [])
    clean_query = preprocess(augmented) or augmented
    intent = router.route(clean_query)
    print(f"[STEP 1] query={query!r}, augmented={augmented!r}, intent={intent}")

    products: list[dict] = []
    comparison_error: Optional[str] = None

    # --- Step 2: get candidates ---
    if intent == "comparison":
        cmp = tools.compare_products(query, df)
        products = [p for p in (cmp.get("product_a"), cmp.get("product_b")) if p]
        for p in products:
            p.setdefault("faiss_score", 0.7)
        comparison_error = cmp.get("error")
        print(f"[STEP 2] comparison: found {len(products)}/2, error={comparison_error}")
    else:
        # Extract structured constraints
        ceiling = tools.extract_price_ceiling(augmented)
        category = tools.extract_category(augmented)
        brand = tools.extract_brand(augmented)
        sub_cat = router.get_sub_category_hint(clean_query)
        cheap = tools.is_cheap_query(query)
        print(f"[STEP 2] constraints: ceiling={ceiling}, cat={category}, "
              f"brand={brand}, sub={sub_cat}, cheap={cheap}")

        if any([ceiling, category, brand, sub_cat]):
            # Hybrid: pandas filter + RAG re-rank
            products = _hybrid_retrieve(
                query, augmented, df, rag_engine,
                ceiling, category, brand, sub_cat,
            )
        else:
            # No structured constraints — pure RAG with confidence floor
            candidates = rag_engine.retrieve(augmented, top_k=10)
            active_names = set(df["name"].astype(str).tolist())
            products = [p for p in candidates
                        if str(p.get("name", "")) in active_names]

            # Confidence floor: drop noise. Without hard constraints, a low
            # similarity score means we don't actually have a match.
            if products and products[0].get("faiss_score", 0) < CONFIDENCE_MEDIUM:
                print(f"[STEP 2] confidence floor: dropping {len(products)} "
                      f"weak matches (top score "
                      f"{products[0].get('faiss_score', 0):.2f} < {CONFIDENCE_MEDIUM})")
                products = []

        # Final sort + cap
        products = _smart_sort(products, ceiling, cheap)[:5]
        print(f"[STEP 2] final products: {len(products)}")

    # --- Step 3: confidence label ---
    if products:
        confidence = TechBotRAG.get_confidence(products[0].get("faiss_score", 0.0))
    else:
        confidence = "low"

    # --- Step 4: web fallback (only if we have no products and CSE configured) ---
    web_results = None
    used_web = False
    if not products and intent != "comparison" and google_search.is_available():
        try:
            web_results = google_search.search_products(query)
            used_web = bool(web_results)
        except Exception as e:
            print(f"[STEP 4] web search failed: {e}")

    # --- Step 5: build prompt + call LLM ---
    system_prompt = prompts.SYSTEM_PROMPT_BN if language == "bn" else prompts.SYSTEM_PROMPT_EN
    full_prompt = prompts.build_prompt(
        query=query,
        products=products,
        history=history or [],
        mode=intent,
        web_results=web_results,
        comparison_error=comparison_error,
    )
    response_text = llm.get_response(full_prompt, system_prompt, backend=backend)

    # --- Step 6: format link block ---
    link_block = tools.build_link_block(
        products, shop_whatsapp=shop_whatsapp,
        user_query=query, language=language,
    )

    return {
        "response": response_text,
        "link_block": link_block,
        "products": products,
        "intent": intent,
        "confidence": confidence,
        "used_web_search": used_web,
        "backend_used": backend,
    }
