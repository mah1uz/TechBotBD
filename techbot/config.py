"""
Central configuration for TechBot BD.

All other modules import from here. No hard-coded paths, model names,
or magic numbers should appear elsewhere in the codebase.
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "techbot/data/"
DEFAULT_CSV = "techbot/data/products.csv"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# Per the agreed plan: multilingual-e5-small as default, swappable.
# To swap, just change EMBEDDING_MODEL to one of:
#   "intfloat/multilingual-e5-small"            (default — strong Bengali, ~120MB)
#   "paraphrase-multilingual-MiniLM-L12-v2"     (spec literal, ~120MB)
#   "BAAI/bge-m3"                               (best quality, ~570MB)
# E5 models require "query: " / "passage: " prefixes; rag_pipeline.py
# detects this automatically based on model name.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
DEFAULT_LLM_BACKEND = "groq"          # "groq" or "ollama"
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_TEMPERATURE = 0.5
LLM_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 5
MAX_HISTORY_TURNS = 4

# Confidence thresholds.
# NOTE: spec specified L2 distances (50/150). With normalized embeddings
# and inner-product search those scales don't apply. We use cosine
# similarity (range 0..1) instead, which matches what FAISS returns when
# we normalize embeddings + use IndexFlatIP.
CONFIDENCE_HIGH = 0.65
CONFIDENCE_MEDIUM = 0.45

# ---------------------------------------------------------------------------
# Shop-mode CSV upload — template columns
# ---------------------------------------------------------------------------
SHOP_TEMPLATE_COLUMNS = [
    "name", "category", "brand", "price_bdt", "processor",
    "ram_gb", "storage", "gpu", "display_inch",
    "battery_mah", "camera_mp", "sub_category", "url",
]

REQUIRED_COLUMNS = ["name", "category", "price_bdt"]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
PRICE_SLIDER_MIN = 5_000
PRICE_SLIDER_MAX = 500_000
APP_TITLE = "TechBot BD"
APP_SUBTITLE = "বাংলায় আপনার পছন্দের গ্যাজেট খুঁজুন"
