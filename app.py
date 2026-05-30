"""
TechBot BD — Streamlit UI

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from techbot import llm, pipeline, tools
from techbot.config import (
    APP_TITLE, APP_SUBTITLE, DEFAULT_CSV,
    PRICE_SLIDER_MIN, PRICE_SLIDER_MAX,
    REQUIRED_COLUMNS, SHOP_TEMPLATE_COLUMNS,
)
from techbot.rag_pipeline import TechBotRAG


# ===========================================================================
# Page config
# ===========================================================================

st.set_page_config(
    page_title="TechBot BD",
    page_icon="🤖",
    layout="wide",
)


# ===========================================================================
# Session state
# ===========================================================================

DEFAULT_STATE = {
    "messages": [],            # [{role, content, products?, link_block?, meta?}]
    "mode": "general",         # "general" | "shop"
    "llm_backend": "groq",     # "groq" | "ollama"
    "language": "bn",          # "bn" | "en"
    "shop_df": None,
    "shop_rag": None,
    "shop_name": "",
    "shop_whatsapp": "",
    "price_filter": (PRICE_SLIDER_MIN, PRICE_SLIDER_MAX),
    "pending_query": None,     # set by suggestion chips
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ===========================================================================
# Cached resources (heavy)
# ===========================================================================

@st.cache_data(show_spinner="ডেটা লোড হচ্ছে...")
def load_general_data() -> pd.DataFrame:
    p = Path(DEFAULT_CSV)
    if not p.exists():
        st.error(
            f"❌ পণ্য ফাইল পাওয়া যায়নি: `{DEFAULT_CSV}`। "
            f"`python scripts/clean_data.py` চালান।"
        )
        st.stop()
    df = pd.read_csv(p)
    return df


@st.cache_resource(show_spinner="মডেল লোড হচ্ছে (প্রথমবার একটু সময় লাগবে)...")
def load_general_rag(_df_signature: str, _df: pd.DataFrame) -> TechBotRAG:
    """
    `_df_signature` makes Streamlit re-cache when data changes.
    `_df` underscore prefix tells Streamlit not to hash the dataframe.
    """
    return TechBotRAG(_df)


def get_active_df() -> pd.DataFrame:
    """Active df depends on mode + price slider."""
    if st.session_state.mode == "shop" and st.session_state.shop_df is not None:
        df = st.session_state.shop_df
    else:
        df = load_general_data()

    lo, hi = st.session_state.price_filter
    if (lo, hi) != (PRICE_SLIDER_MIN, PRICE_SLIDER_MAX):
        df = df[(df["price_bdt"] >= lo) & (df["price_bdt"] <= hi)]
    return df.reset_index(drop=True)


def get_active_rag() -> TechBotRAG:
    """Active RAG engine matches active mode."""
    if st.session_state.mode == "shop" and st.session_state.shop_rag is not None:
        return st.session_state.shop_rag
    df = load_general_data()
    sig = f"{len(df)}_{','.join(sorted(df.columns))}"
    return load_general_rag(sig, df)


# ===========================================================================
# Shop CSV upload handling
# ===========================================================================

def validate_shop_csv(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Required column missing: `{col}`")
    if "category" in df.columns:
        bad = set(df["category"].unique()) - {"laptop", "smartphone"}
        if bad:
            errors.append(f"Invalid `category` values: {bad}. Must be 'laptop' or 'smartphone'.")
    return errors


def fill_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns with safe defaults."""
    df = df.copy()
    text_defaults = {
        "brand": "", "processor": "", "generation": "", "storage": "",
        "gpu": "", "gpu_memory": "", "url": "", "shop": st.session_state.shop_name or "Shop",
        "sub_category": "",
    }
    num_defaults = {
        "ram_gb": 0, "battery_mah": 0, "camera_mp": 0,
        "display_inch": 0.0, "popularity": 0,
    }
    for col, val in text_defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)
    for col, val in num_defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)
    # Default sub_category from category if empty
    if "sub_category" in df.columns:
        empty = df["sub_category"].astype(str).str.strip() == ""
        df.loc[empty, "sub_category"] = df.loc[empty, "category"]
    return df


def handle_shop_upload(uploaded_file) -> None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"❌ CSV পড়া যায়নি: {e}")
        return

    errors = validate_shop_csv(df)
    if errors:
        for e in errors:
            st.sidebar.error(f"❌ {e}")
        return

    df = fill_optional_columns(df)
    st.session_state.shop_df = df

    with st.sidebar:
        with st.spinner("দোকানের ডেটা ইনডেক্স হচ্ছে..."):
            st.session_state.shop_rag = TechBotRAG(df)
    st.sidebar.success(f"✅ {len(df)} পণ্য লোড হয়েছে।")


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.markdown("# 🤖 TechBot BD")
    st.caption("বাংলা গ্যাজেট সুপারিশ চ্যাটবট")
    st.divider()

    # --- Mode selector ---
    mode_label = st.radio(
        "**মোড নির্বাচন**",
        options=["🌐 General Mode", "🏪 Shop Mode"],
        index=0 if st.session_state.mode == "general" else 1,
        horizontal=False,
    )
    st.session_state.mode = "general" if "General" in mode_label else "shop"

    if st.session_state.mode == "shop":
        st.markdown("##### 🏪 দোকানের তথ্য")
        st.session_state.shop_name = st.text_input(
            "দোকানের নাম",
            value=st.session_state.shop_name,
            placeholder="যেমন: TechZone BD",
        )
        st.session_state.shop_whatsapp = st.text_input(
            "WhatsApp নম্বর",
            value=st.session_state.shop_whatsapp,
            placeholder="8801XXXXXXXXX",
            help="দেশ কোডসহ লিখুন। যেমন: 8801711000000",
        )

        uploaded = st.file_uploader(
            "📤 পণ্য তালিকা আপলোড করুন (CSV)",
            type=["csv"],
        )
        if uploaded is not None and st.button("🔄 ইনডেক্স তৈরি করুন", use_container_width=True):
            handle_shop_upload(uploaded)

        st.download_button(
            "📥 CSV টেমপ্লেট ডাউনলোড করুন",
            data=tools.get_template_csv(),
            file_name="techbot_shop_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # --- LLM backend ---
    backend_label = st.radio(
        "**🧠 LLM ইঞ্জিন**",
        options=["💎 Premium (Groq API)", "⚡ Local (Gemma/Llama)"],
        index=0 if st.session_state.llm_backend == "groq" else 1,
    )
    new_backend = "groq" if "Premium" in backend_label else "ollama"
    if new_backend != st.session_state.llm_backend:
        st.session_state.llm_backend = new_backend

    # Backend availability check
    if st.session_state.llm_backend == "groq":
        if not llm.is_backend_available("groq"):
            st.warning("⚠️ GROQ_API_KEY পাওয়া যায়নি। `.env` ফাইলে সেট করুন।")
    else:
        if not llm.is_backend_available("ollama"):
            st.warning("⚠️ Ollama চলছে না। `ollama serve` চালান এবং `gemma3:4b` মডেল ডাউনলোড করুন।")

    st.divider()

    # --- Language ---
    lang_label = st.radio(
        "**🌐 ভাষা**",
        options=["🇧🇩 বাংলা", "🇬🇧 English"],
        index=0 if st.session_state.language == "bn" else 1,
        horizontal=True,
    )
    st.session_state.language = "bn" if "বাংলা" in lang_label else "en"

    st.divider()

    # --- Price filter ---
    st.markdown("**💰 বাজেট ফিল্টার (৳)**")
    st.session_state.price_filter = st.slider(
        "মূল্য পরিসর",
        min_value=PRICE_SLIDER_MIN,
        max_value=PRICE_SLIDER_MAX,
        value=st.session_state.price_filter,
        step=5_000,
        format="৳%d",
        label_visibility="collapsed",
    )

    st.divider()

    # --- Quick stats ---
    active_df = get_active_df()
    n_phones = (active_df["category"] == "smartphone").sum()
    n_laptops = (active_df["category"] == "laptop").sum()
    shop_label = (
        st.session_state.shop_name or "আপলোড করা দোকান"
        if st.session_state.mode == "shop"
        else "সব দোকান"
    )
    st.markdown("**📊 পরিসংখ্যান**")
    c1, c2 = st.columns(2)
    c1.metric("📱 ফোন", int(n_phones))
    c2.metric("💻 ল্যাপটপ", int(n_laptops))
    st.caption(f"🏪 দোকান: {shop_label}")

    st.divider()

    if st.button("🗑️ চ্যাট মুছুন", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ===========================================================================
# Main area — header
# ===========================================================================

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title(f"🤖 {APP_TITLE}")
    st.caption(APP_SUBTITLE)
with col_badge:
    if st.session_state.llm_backend == "groq":
        st.markdown(
            "<div style='text-align:right; padding-top:1rem;'>"
            "<span style='background:linear-gradient(90deg,#FFD700,#FFA500);"
            "color:#000; padding:6px 14px; border-radius:20px;"
            "font-weight:bold; font-size:0.9rem;'>💎 PREMIUM</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='text-align:right; padding-top:1rem;'>"
            "<span style='background:#444; color:#fff; padding:6px 14px;"
            "border-radius:20px; font-weight:bold; font-size:0.9rem;'>⚡ LOCAL</span></div>",
            unsafe_allow_html=True,
        )

if st.session_state.mode == "shop":
    if st.session_state.shop_df is None:
        st.info("🏪 **Shop Mode সক্রিয়।** সাইডবারে দোকানের তথ্য দিয়ে CSV আপলোড করুন।")
    else:
        st.success(
            f"🏪 **{st.session_state.shop_name or 'Your Shop'}** — "
            f"{len(st.session_state.shop_df)} পণ্য সক্রিয়।"
        )


# ===========================================================================
# Suggestion chips (only shown when chat is empty)
# ===========================================================================

SUGGESTIONS = [
    "৫০ হাজারের মধ্যে ল্যাপটপ দেখাও",
    "সেরা ক্যামেরার ফোন কোনটা?",
    "iPhone vs Samsung তুলনা করো",
    "গেমিং ল্যাপটপ ১ লাখ বাজেটে",
    "হালকা ওজনের ল্যাপটপ চাই",
]

if not st.session_state.messages:
    st.markdown("##### 💡 কিছু উদাহরণ প্রশ্ন:")
    cols = st.columns(len(SUGGESTIONS))
    for col, sug in zip(cols, SUGGESTIONS):
        with col:
            if st.button(sug, use_container_width=True, key=f"sug_{sug}"):
                st.session_state.pending_query = sug
                st.rerun()


# ===========================================================================
# Render chat history
# ===========================================================================

def _render_product_cards(products: list[dict], max_cards: int = 3) -> None:
    """Render up to max_cards product cards side-by-side."""
    if not products:
        return
    products = products[:max_cards]
    cols = st.columns(len(products))
    for col, p in zip(cols, products):
        with col:
            with st.container(border=True):
                st.markdown(f"**{p.get('name', 'Unknown')}**")
                price = p.get("price_bdt", 0)
                price_str = f"৳{int(price):,}" if price and int(price) > 0 else "দাম জানা নেই"
                st.markdown(f"<h3 style='color:#10B981; margin:0;'>{price_str}</h3>",
                            unsafe_allow_html=True)

                spec_lines = []
                if p.get("processor"):
                    spec_lines.append(f"🧠 {p['processor']}")
                if p.get("ram_gb") and float(p.get("ram_gb", 0)) > 0:
                    spec_lines.append(f"💾 {int(float(p['ram_gb']))}GB RAM")
                if p.get("storage"):
                    spec_lines.append(f"💿 {p['storage']}")
                if p.get("camera_mp") and float(p.get("camera_mp", 0)) > 0:
                    spec_lines.append(f"📷 {int(float(p['camera_mp']))}MP")
                if p.get("battery_mah") and float(p.get("battery_mah", 0)) > 0:
                    spec_lines.append(f"🔋 {int(float(p['battery_mah']))}mAh")
                if p.get("display_inch") and float(p.get("display_inch", 0)) > 0:
                    spec_lines.append(f"📐 {p['display_inch']}\"")

                for line in spec_lines[:3]:
                    st.caption(line)

                shop = p.get("shop", "")
                url = p.get("url", "")
                if url:
                    st.markdown(f"🏪 [{shop}]({url})")
                elif shop:
                    st.caption(f"🏪 {shop}")


def _render_confidence(confidence: str) -> None:
    """Show a small colored-dot confidence indicator."""
    mapping = {
        "high": ("🟢🟢🟢🟢🟢", "উচ্চ"),
        "medium": ("🟡🟡🟡⚪⚪", "মাঝারি"),
        "low": ("🔴⚪⚪⚪⚪", "কম"),
    }
    dots, label_bn = mapping.get(confidence, ("⚪⚪⚪⚪⚪", "অজানা"))
    st.caption(f"{dots} ম্যাচ মান: {label_bn}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = msg.get("meta", {})
            if msg.get("products"):
                _render_product_cards(msg["products"])
            if msg.get("link_block"):
                st.markdown(msg["link_block"])
            if meta.get("confidence"):
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    _render_confidence(meta["confidence"])
                with cols[1]:
                    st.caption(f"🎯 ইনটেন্ট: {meta.get('intent', '?')}")
                with cols[2]:
                    if meta.get("used_web_search"):
                        st.caption("🌐 ওয়েব সার্চ ব্যবহৃত")


# ===========================================================================
# Input handling
# ===========================================================================

# Handle suggestion-chip click (set pending_query, then handled below)
chat_input_value = st.chat_input("আপনার প্রশ্ন লিখুন...")
query = chat_input_value or st.session_state.pending_query

if st.session_state.pending_query:
    st.session_state.pending_query = None  # consume

if query:
    # Append user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process
    with st.chat_message("assistant"):
        with st.spinner("ভাবছি..."):
            try:
                result = pipeline.chat(
                    query=query,
                    history=st.session_state.messages[:-1],  # exclude current user msg
                    df=get_active_df(),
                    rag_engine=get_active_rag(),
                    backend=st.session_state.llm_backend,
                    language=st.session_state.language,
                    shop_whatsapp=(
                        st.session_state.shop_whatsapp
                        if st.session_state.mode == "shop"
                        else None
                    ),
                    mode=st.session_state.mode,
                )
            except Exception as e:
                st.error(f"❌ পাইপলাইন ত্রুটি: {e}")
                result = {
                    "response": "দুঃখিত, একটি সমস্যা হয়েছে। আবার চেষ্টা করুন।",
                    "products": [], "link_block": "",
                    "intent": "?", "confidence": "low",
                    "used_web_search": False, "backend_used": st.session_state.llm_backend,
                }

        st.markdown(result["response"])
        if result.get("products"):
            _render_product_cards(result["products"])
        if result.get("link_block"):
            st.markdown(result["link_block"])

        cols = st.columns([1, 1, 1])
        with cols[0]:
            _render_confidence(result.get("confidence", "low"))
        with cols[1]:
            st.caption(f"🎯 ইনটেন্ট: {result.get('intent', '?')}")
        with cols[2]:
            if result.get("used_web_search"):
                st.caption("🌐 ওয়েব সার্চ ব্যবহৃত")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "products": result.get("products", []),
        "link_block": result.get("link_block", ""),
        "meta": {
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "used_web_search": result.get("used_web_search", False),
            "backend_used": result.get("backend_used"),
        },
    })

    # If we processed a suggestion-chip query (no chat_input), rerun to clear chips
    if chat_input_value is None:
        st.rerun()
