"""
Prompt assembly. Pure string formatting — no logic, no imports from anything
non-trivial. Two system prompts (BN/EN) and one prompt builder that handles
all intents via a `mode` parameter.
"""

from __future__ import annotations

import math
from typing import Optional

from techbot.config import MAX_HISTORY_TURNS


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BN = """\
আপনি TechBot BD — বাংলাদেশের জন্য একটি বাংলা গ্যাজেট সুপারিশকারী সহকারী। আপনার কাজ \
ব্যবহারকারীর প্রয়োজন বুঝে স্মার্টফোন এবং ল্যাপটপ সুপারিশ করা।

কঠোরভাবে অনুসরণ করুন এই নিয়মগুলো:

১. **ভাষা:** সবসময় শুধুমাত্র বাংলায় উত্তর দিন। ব্র্যান্ড ও পণ্যের নাম (যেমন iPhone, MacBook, Samsung Galaxy) \
ইংরেজিতে রাখতে পারেন, বাকি সব বাংলায়।

২. **শুধু প্রদত্ত তথ্য ব্যবহার করুন:** [পণ্যের তথ্য] অংশে যা দেওয়া আছে শুধু তা থেকেই উত্তর দিন। \
স্পেসিফিকেশন, দাম বা ফিচার নিজে থেকে বানাবেন না। তথ্যে যা নেই, তা বলবেন না।

৩. **কোনো পণ্য না পাওয়া গেলে:** যদি ব্যবহারকারীর চাহিদা অনুযায়ী কোনো পণ্য না থাকে, ঠিক এই বাক্যটি বলুন: \
"দুঃখিত, আপনার চাহিদা অনুযায়ী আমাদের কাছে কোনো পণ্য পাওয়া যায়নি। অনুগ্রহ করে আপনার বাজেট বা চাহিদা পরিবর্তন করে আবার চেষ্টা করুন।"

৪. **দাম:** সবসময় টাকার পরিমাণ ৳ চিহ্ন দিয়ে লিখুন। যেমন: ৳৬০,০০০।

৫. **সংক্ষিপ্ত উত্তর:** সর্বোচ্চ ৩টি পণ্য সুপারিশ করুন। প্রতিটি পণ্যের জন্য মূল ফিচারগুলো বুলেট পয়েন্টে লিখুন।

৬. **শেষে একটি প্রশ্ন:** প্রতিটি উত্তরের শেষে ব্যবহারকারীকে আরও সাহায্য করার জন্য ঠিক একটি ফলো-আপ প্রশ্ন করুন।

৭. **সুর:** বন্ধুসুলভ এবং সহায়ক হোন।
"""

SYSTEM_PROMPT_EN = """\
You are TechBot BD — a gadget recommendation assistant for Bangladesh. Your job \
is to understand the user's needs and recommend smartphones and laptops.

Follow these rules strictly:

1. **Language:** Always respond in English only. Brand and product names stay verbatim.

2. **Use only the provided context:** Recommend only from the [Products] section. \
Never invent specs, prices, or features. If something isn't in the context, don't mention it.

3. **No product found:** If nothing matches the user's need, say exactly: \
"Sorry, no products matching your requirement were found. Please adjust your budget or requirements and try again."

4. **Price:** Always show prices with the ৳ symbol. Example: ৳60,000.

5. **Concise:** Recommend at most 3 products. List key specs as bullet points.

6. **One follow-up question:** End every reply with exactly one follow-up question.

7. **Tone:** Friendly and helpful.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(v) -> str:
    """Return string version, or '' for nan/None/'N/A'/'nan'."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "n/a", "none") else s


def _format_product(p: dict) -> str:
    """Compact one-line spec string for the LLM context."""
    parts = []
    if name := _clean(p.get("name")):
        parts.append(name)
    if (brand := _clean(p.get("brand"))) and brand.lower() != "unknown":
        parts.append(f"Brand: {brand}")
    if cat := _clean(p.get("category")):
        parts.append(f"Category: {cat}")
    if proc := _clean(p.get("processor")):
        parts.append(f"Processor: {proc}")
    if gen := _clean(p.get("generation")):
        parts.append(f"Gen: {gen}")
    if (ram := p.get("ram_gb", 0)) and float(ram) > 0:
        parts.append(f"RAM: {int(float(ram))}GB")
    if storage := _clean(p.get("storage")):
        parts.append(f"Storage: {storage}")
    if gpu := _clean(p.get("gpu")):
        parts.append(f"GPU: {gpu}")
    if (disp := p.get("display_inch", 0)) and float(disp) > 0:
        parts.append(f"Display: {disp}\"")
    if (bat := p.get("battery_mah", 0)) and float(bat) > 0:
        parts.append(f"Battery: {int(float(bat))}mAh")
    if (cam := p.get("camera_mp", 0)) and float(cam) > 0:
        parts.append(f"Camera: {int(float(cam))}MP")
    if (price := p.get("price_bdt", 0)) and int(price) > 0:
        parts.append(f"Price: ৳{int(price):,}")
    if shop := _clean(p.get("shop")):
        parts.append(f"Shop: {shop}")
    return " | ".join(parts)


def _format_history(history: list[dict]) -> str:
    """Render last MAX_HISTORY_TURNS exchanges as plain text."""
    if not history:
        return ""
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    lines = []
    for msg in recent:
        if msg.get("role") == "user":
            lines.append(f"ব্যবহারকারী: {msg.get('content', '')}")
        elif msg.get("role") == "assistant":
            lines.append(f"TechBot: {msg.get('content', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(query: str,
                 products: list[dict],
                 history: list[dict],
                 mode: str = "recommendation",
                 web_results: Optional[list[dict]] = None,
                 comparison_error: Optional[str] = None) -> str:
    """
    Single prompt builder for all intents.

    mode = "comparison":   labels products as A and B with comparison instruction
    mode = "filter" / "recommendation":  standard product list

    comparison_error: if set, included in the prompt so the LLM can give a
    helpful "those specific products aren't in our database" response
    instead of the generic "no products found" message.

    Sections (in order):
      [গত কথোপকথন]   — only if history present
      [পণ্যের তথ্য]    — labeled A/B for comparison, numbered list otherwise
      [নির্দেশনা]      — only for comparison
      [ওয়েব সার্চ]     — only if web_results present
      [ব্যবহারকারীর প্রশ্ন]
    """
    sections = []

    if hist := _format_history(history):
        sections.append(f"[গত কথোপকথন]\n{hist}")

    if mode == "comparison":
        if products and len(products) >= 2:
            labels = ["A", "B"]
            product_lines = [
                f"পণ্য {labels[i]}: {_format_product(p)}"
                for i, p in enumerate(products[:2])
            ]
            sections.append("[তুলনার জন্য পণ্য]\n" + "\n".join(product_lines))
            sections.append(
                "[নির্দেশনা]\n"
                "উপরের পণ্যগুলোর মধ্যে তুলনা করুন। ব্যবহারকারীর প্রশ্নের প্রেক্ষিতে "
                "প্রাসঙ্গিক স্পেসিফিকেশন (প্রসেসর, র‍্যাম, ক্যামেরা, ব্যাটারি, দাম) "
                "বুলেট পয়েন্টে দেখান। শেষে একটি সুপারিশ দিন।"
            )
        elif products and len(products) == 1:
            # Only one of the two named products was found
            sections.append(f"[পণ্যের তথ্য]\n1. {_format_product(products[0])}")
            sections.append(
                "[নির্দেশনা]\n"
                "ব্যবহারকারী দুটি পণ্যের তুলনা চেয়েছেন কিন্তু একটি পণ্য আমাদের "
                "ডাটাবেসে নেই। যেটি পাওয়া গেছে তার তথ্য দিন এবং অন্যটির ব্যাপারে "
                "দুঃখপ্রকাশ করুন।"
            )
        else:
            # Neither product was found — this is a "we don't have those" case,
            # not a generic "no products" case. Tell the LLM specifically.
            err_text = comparison_error or "তুলনার জন্য চাওয়া পণ্যগুলো ডাটাবেসে নেই।"
            sections.append(
                "[পরিস্থিতি]\n"
                f"ব্যবহারকারী দুটি নির্দিষ্ট পণ্যের তুলনা চেয়েছেন, কিন্তু সেগুলো "
                f"আমাদের ডাটাবেসে নেই। ({err_text})"
            )
            sections.append(
                "[নির্দেশনা]\n"
                "ব্যবহারকারীকে দয়া করে জানান যে নির্দিষ্ট পণ্যগুলো আমাদের কাছে "
                "নেই, কিন্তু কোন বিকল্প পণ্য সাজেস্ট করতে পারেন তা জিজ্ঞেস করুন। "
                "সাধারণ \"পণ্য পাওয়া যায়নি\" বার্তা ব্যবহার করবেন না।"
            )
    elif products:
        lines = [f"{i+1}. {_format_product(p)}" for i, p in enumerate(products)]
        sections.append("[পণ্যের তথ্য]\n" + "\n".join(lines))
    else:
        sections.append("[পণ্যের তথ্য]\nকোনো ম্যাচিং পণ্য পাওয়া যায়নি।")

    if web_results:
        web_lines = [
            f"{i+1}. {r.get('title','')} — {r.get('snippet','')}"
            for i, r in enumerate(web_results)
        ]
        sections.append(
            "[ওয়েব সার্চ থেকে অতিরিক্ত তথ্য (ডাটাবেসে নেই, শুধু রেফারেন্স)]\n"
            + "\n".join(web_lines)
        )

    sections.append(f"[ব্যবহারকারীর প্রশ্ন]\n{query}")
    return "\n\n".join(sections)


def build_whatsapp_message(product: dict, user_query: str, shop_name: str,
                           language: str = "bn") -> str:
    """
    Pre-filled WhatsApp inquiry message. Plain text, NOT a prompt — gets
    URL-encoded by the caller for the wa.me link.
    """
    name = _clean(product.get("name"))
    price = product.get("price_bdt", 0)
    price_str = f"৳{int(price):,}" if price and int(price) > 0 else "দাম জানা নেই"

    spec_bits = []
    if proc := _clean(product.get("processor")):
        spec_bits.append(f"প্রসেসর: {proc}")
    if (ram := product.get("ram_gb", 0)) and float(ram) > 0:
        spec_bits.append(f"র‍্যাম: {int(float(ram))}GB")
    if storage := _clean(product.get("storage")):
        spec_bits.append(f"স্টোরেজ: {storage}")

    if language == "en":
        return (
            f"Hello {shop_name},\n\n"
            f"I'm interested in the *{name}* (Price: {price_str}).\n"
            + ("Specs: " + ", ".join(spec_bits) + "\n" if spec_bits else "")
            + f"\nMy question: {user_query}\n\n"
            f"Could you confirm availability? Thanks. (Sent via TechBot BD)"
        )

    return (
        f"আসসালামু আলাইকুম {shop_name},\n\n"
        f"আমি *{name}* (দাম: {price_str}) সম্পর্কে জানতে আগ্রহী।\n"
        + ("স্পেসিফিকেশন: " + ", ".join(spec_bits) + "\n" if spec_bits else "")
        + f"\nআমার প্রশ্ন: {user_query}\n\n"
        f"পণ্যটি কি স্টকে আছে? ধন্যবাদ। (TechBot BD থেকে পাঠানো)"
    )
