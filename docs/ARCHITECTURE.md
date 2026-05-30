# TechBot BD — Architecture & NLP Components

A reading guide for explaining the system. Each section describes one
component, what NLP technique it uses, and where to find it in the code.

## 1. End-to-end query flow

```
       ┌─────────────────────────────────────────────────────────────┐
       │                          app.py                             │
       │              (Streamlit UI, session state)                  │
       └────────────────────────────┬────────────────────────────────┘
                                    │ chat(query, history, df, rag, ...)
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                    pipeline.chat()                          │
       │         orchestrates the steps below in order               │
       └──┬───────┬──────────┬──────────┬──────────┬─────────────────┘
          │       │          │          │          │
          ▼       ▼          ▼          ▼          ▼
     preprocess  router   rag_pipeline tools     prompts → llm
       (1)        (2)        (3)        (4)        (5)     (6)
```

Six NLP-relevant steps, each in its own module:

1. **Preprocessing** — text normalization (Bengali / Banglish / English mix)
2. **Intent classification** — comparison / filter / recommendation
3. **Dense retrieval** — sentence-transformer embeddings + FAISS
4. **Information extraction** — price/category from query, fuzzy matching
5. **Prompt engineering** — context assembly, few-shot-style instructions
6. **LLM generation** — Groq or Ollama, with retry

Below: what each step does, why, and the file/function to point at.

---

## 2. Preprocessing — `techbot/preprocessor.py`

**Goal.** Turn a noisy mixed-language query into a normalized form suitable
for both intent routing and embedding.

**Techniques used.**

- **Unicode NFC normalization.** Bengali has multiple valid encodings for
  the same visual character (e.g. ো can be one codepoint or two). Without
  NFC, two visually-identical strings can fail to match. `unicodedata.normalize("NFC", text)`.
- **Selective casefolding.** ASCII letters are lowercased; Bengali codepoints
  are passed through unchanged (Bengali script has no case).
- **Banglish lexicon mapping.** A regex dictionary rewrites romanized Bengali
  gadget terms to their Bengali equivalents (`phone → ফোন`, `ram → র‍্যাম`,
  `bhalo → ভালো`, …). Roughly 40 entries. This is a hand-built lexicon —
  the simplest form of statistical NLP, also called a **gazetteer**.
- **Character class filter.** Keeps Bengali (U+0980–U+09FF), ASCII alphanumerics,
  whitespace, and basic punctuation. Drops emoji and decorative symbols.
- **Number preservation.** Numerals — both Arabic (60) and Bengali (৬০) —
  are deliberately preserved so `tools.extract_price_ceiling` can parse them.

**Entry point.** `preprocess(text: str) -> str`.

**Why this matters for the demo.** The user can type *"60k er moddhe gaming
laptop dao"* (mixed Banglish + numbers) and the system turns it into
*"৬০k এর মধ্যে গেমিং ল্যাপটপ দাও"* — close enough to Bengali for the
embedding model to match well.

---

## 3. Intent classification — `techbot/router.py`

**Goal.** Decide whether the query is asking for a **comparison** (X vs Y),
a **filter** (anything with a hard constraint or use case), or a generic
**recommendation**.

**Technique.** Keyword-based rule classifier. Three lists of trigger words
(comparison-keywords, filter-keywords) checked in order; default is
`"recommendation"`. Two functions:

- `route(query) -> str` — returns one of the three intents.
- `get_sub_category_hint(query) -> str | None` — returns `"gaming_laptop"`,
  `"flagship_phone"`, etc. when the query implies a sub-category.

**Why keyword-based?** It's the standard NLP baseline. Zero training data,
fully interpretable, instant. A learned classifier (e.g. a fine-tuned
multilingual encoder) could replace `route()` without touching anything else.

**Trade-offs.** Will misclassify rare phrasings ("compare" appears nowhere
but the intent is comparison). Acceptable for a demo with bounded vocabulary.

---

## 4. Dense retrieval — `techbot/rag_pipeline.py`

This is the NLP core of the system. The class `TechBotRAG` does four things:

### 4a. Build a `search_text` per product

`_row_to_search_text(row)` concatenates the relevant spec columns into one
sentence. For example:

```
"Samsung Galaxy S25 FE smartphone flagship_phone Samsung Exynos 2400 8GB RAM
 128/256/512 storage 6.7 inch 4900mAh battery 50MP camera"
```

We deliberately **exclude** `price_bdt`, `url`, `shop`, and `popularity` —
these shouldn't influence semantic similarity. Hard constraints (price,
shop) are applied later as a post-filter.

### 4b. Embed with sentence-transformers

The default model is `intfloat/multilingual-e5-small` (~120 MB, 384-dim
vectors, trained on a multilingual contrastive objective). Strong on Bengali.

E5 models require special prefixes — `"query: "` for queries and
`"passage: "` for documents — built into the contrastive training. The
code detects this from the model name and adds prefixes automatically.

Other supported models (just change `EMBEDDING_MODEL` in `config.py`):

| Model                                       | Size  | Notes                          |
| ------------------------------------------- | ----- | ------------------------------ |
| `intfloat/multilingual-e5-small`            | 120MB | Default. Strong Bengali.       |
| `paraphrase-multilingual-MiniLM-L12-v2`     | 120MB | Original spec choice.          |
| `BAAI/bge-m3`                               | 570MB | Best quality, larger.          |

### 4c. Index with FAISS

`faiss.IndexFlatIP(dim)` — exact (not approximate) inner-product search.
Because we normalize embeddings to unit length before indexing, inner
product equals cosine similarity. So scores are in `[-1, 1]`, with higher
being better. In practice you'll see scores in the 0.3–0.9 range.

### 4d. Retrieve, dedup, return

`retrieve(query, top_k)`:

1. Preprocess and embed the query.
2. Search FAISS with `fetch_k = top_k * 4` (over-fetch).
3. Iterate results in score order. Maintain a `seen_names` dict; if a
   product name already appears, keep only the cheapest variant. (The
   data has multiple rows per laptop model with different RAM/storage
   configs; we don't want them all in the result list.)
4. Return up to `top_k` deduped products as dicts, with a `faiss_score`
   field.

**Confidence labels.** A static method maps the top score to high / medium /
low (thresholds 0.65 and 0.45). Used by the UI for display, and by the
pipeline to decide whether to fall back to web search.

---

## 5. Information extraction — `techbot/tools.py`

**Goal.** Pull structured fields out of unstructured text. Two extractors,
plus fuzzy product-name matching.

### 5a. Price ceiling extraction

`extract_price_ceiling(query)` handles four patterns in priority order:

```
লাখ / lakh      e.g. "1.5 lakh"     → 150000
হাজার / k       e.g. "৬০ হাজার", "60k" → 60000
plain number    e.g. "60,000", "60000" → 60000
none            → None
```

Bengali numerals (০-৯) are translated to ASCII before regex matching. The
`max(...)` over plain candidates avoids treating model numbers as prices
("iPhone 15" doesn't extract 15).

### 5b. Category extraction

`extract_category(query)` returns `"laptop"`, `"smartphone"`, or `None`
based on a keyword set per category (English + Bengali + Banglish).

### 5c. Fuzzy matching for comparison

`compare_products(query, df)` splits the query on a comparison keyword
(`vs`, `versus`, `নাকি`, `তুলনা`, …) and fuzzy-matches each fragment
against the `name` column.

The fuzzy match uses `difflib.get_close_matches(n=1, cutoff=0.3)`, which
implements the **Ratcliff-Obershelp** algorithm: similarity = (2 × matches)
/ (total length). A 0.3 cutoff is generous, allowing partial matches like
"iPhone Air" → "iPhone Air" and "S24" → "Samsung Galaxy S24".

Falls back to substring containment when fuzzy fails.

---

## 6. Prompt engineering — `techbot/prompts.py`

**Goal.** Convert structured data + user query into a single prompt string
that maximizes LLM faithfulness.

**Design.** Two system prompts (`SYSTEM_PROMPT_BN`, `SYSTEM_PROMPT_EN`) and
one builder function `build_prompt()` that handles all intents via a `mode`
parameter.

The system prompt encodes seven hard rules:

1. Reply only in the target language (Bengali by default; brand names allowed).
2. **No hallucination** — only use the [পণ্যের তথ্য] section.
3. Exact "no products" canned response when context is empty.
4. Always show prices with `৳` symbol.
5. At most 3 product recommendations per response.
6. End every response with one follow-up question (keeps conversation alive).
7. Friendly tone.

Rule 2 — "use only the provided context" — is the foundational RAG
constraint. Without it, LLMs invent specs.

**Sections in the user prompt** (assembled in this order):

```
[গত কথোপকথন]               ← only if history exists
[পণ্যের তথ্য] OR [তুলনার জন্য পণ্য]   ← retrieved products
[নির্দেশনা]                  ← only for comparison mode
[ওয়েব সার্চ থেকে অতিরিক্ত তথ্য]    ← only if web fallback used
[ব্যবহারকারীর প্রশ্ন]           ← the actual user query
```

**WhatsApp message builder.** `build_whatsapp_message()` produces a
pre-filled inquiry message in Bengali or English. The result is URL-encoded
into a `wa.me/<number>?text=<message>` link inside `tools.build_link_block`.

---

## 7. Pipeline orchestration — `techbot/pipeline.py`

**Goal.** Wire everything together into a single `chat()` function.

**Architecture pattern: HYBRID RETRIEVAL.**

The pipeline combines two retrieval techniques because each handles a
different kind of query well:

- **Pandas filter** is exhaustive and exact. Good for hard constraints
  (price ≤ 60000, brand = Apple, category = laptop). It will find every
  matching product, never miss one.
- **RAG (sentence embeddings)** is semantic and approximate. Good for
  fuzzy queries ("good phone for video editing") where the user's intent
  doesn't map to a single column. It can rank similar products by relevance.

A pure-RAG approach fails on hard constraints. Asking for "budget laptop
under 60k" via RAG retrieves the top-10 most semantically similar laptops
— which may all be premium models (since "laptop" is the dominant signal),
and the post-filter zeros them out. The fix is to constrain *first* with
pandas, then rank *within* that set with RAG.

The flow:

```
1. Carry context from history    →  Extract brand/category from prior turns
                                    if the current query lacks them.
                                    "Apple এর ফোন" then "আরেকটু কম দামে"
                                    → the second turn inherits "Apple smartphone".

2. Extract structured constraints →  ceiling, category, brand, sub-category
                                    via tools.extract_*

3a. If any hard constraint exists:
     a. Pandas filter df          →  exact set of products meeting constraints
     b. RAG retrieve top-20       →  semantic ranking from full corpus
     c. Intersect by name         →  use the cheapest variant within budget
                                    when multiple price tiers exist
     d. Smart-sort                →  price desc if budget given;
                                     price asc if "cheap" detected;
                                     RAG order otherwise

3b. Else (no constraints):
     a. RAG retrieve top-10       →  pure semantic search
     b. Confidence floor          →  if top score < 0.45, drop everything.
                                    Prevents irrelevant cards on off-topic
                                    queries ("drone camera" → no products).

4. Comparison intent special-cased: fuzzy-match two product names from
   the query, return both. If neither matches, the LLM is told via a
   `comparison_error` field in the prompt so it gives a helpful response
   ("we don't have those specific products") instead of the generic
   "no products found" message.
```

**Why this matters.** The user's query "Apple এর সবচেয়ে সস্তা ফোন":
- Pandas filter narrows 413 products → 5 Apple smartphones
- "সস্তা" detected → sort by price ascending
- Result: iPhone 17e (৳77,990) first, iPhone 17 Pro Max (৳165,888) last

Without the hybrid approach (the previous RAG-only version), the bot
returned only iPhone 17 (the most semantically similar to the query),
missing the cheaper iPhone 17e entirely.

---

## 8. LLM backend — `techbot/llm.py`

Two backends behind one function:

- **Groq** (cloud, paid) — `llama-3.3-70b-versatile` via the official client.
  Strong instruction-following, multilingual.
- **Ollama** (local, free) — `gemma3:4b` via HTTP POST to `localhost:11434`.
  Smaller model, runs on a laptop with 8GB RAM.

`get_response(prompt, system_prompt, backend)` retries once on any exception
and falls back to a Bengali error message rather than raising. The UI shows
a warning if the chosen backend is unreachable, but the rest of the app
continues to function.

---

## 9. What you can say in the demo

If asked which NLP techniques are in the system, point to:

| Technique                     | Where                                       |
| ----------------------------- | ------------------------------------------- |
| Unicode normalization (NFC)   | `preprocessor.py` `_normalize_unicode`      |
| Lexicon-based normalization   | `preprocessor.py` `BANGLISH_MAP`            |
| Bengali numeral conversion    | `tools.py` `_to_ascii_digits`               |
| Rule-based intent classifier  | `router.py` `route`                         |
| Sentence embeddings           | `rag_pipeline.py` E5-small via SBERT        |
| Approximate semantic search   | `rag_pipeline.py` FAISS IndexFlatIP         |
| Cosine similarity             | normalized vectors + inner product          |
| Information extraction        | `tools.py` `extract_price_ceiling/category/brand` |
| Fuzzy string matching         | `tools.py` `compare_products` (difflib)     |
| Fragment expansion            | `tools.py` `_expand_fragment` (model→brand context) |
| Hybrid retrieval              | `pipeline.py` `_hybrid_retrieve`            |
| Conversational context carry  | `pipeline.py` `_carry_context`              |
| Confidence-based abstention   | `pipeline.py` (drop products below threshold) |
| Prompt engineering / RAG      | `prompts.py` `build_prompt`                 |

If the professor asks **"why this design and not just the LLM?"**:

- The LLM doesn't know our inventory or prices.
- Hallucinations on prices/specs would make the bot unusable.
- RAG grounds the LLM in real product data.
- Hard constraints (price slider, category) need rules, not embeddings —
  embeddings are soft, slider is hard.

If asked **"why hybrid retrieval instead of pure RAG?"**:

- Pure RAG retrieves by semantic similarity. For "budget laptop under 60k",
  the top 10 most similar laptops are often premium ones (because "laptop"
  is the dominant signal); post-filtering them by price zeroes out most.
- Pandas filter is exhaustive — finds *every* product matching the
  constraint, not just the top-K most semantically similar.
- RAG re-ranks within the filtered set, providing semantic ordering for
  free-form parts of the query.
- This is the standard pattern in production search: filter then rank.

If asked **"why a confidence floor?"**:

- The user can ask for things we don't have ("drone camera" — we don't
  sell drones). RAG will still return the closest matches it can find,
  but those scores will be low (~0.3 on a 0-1 cosine scale).
- Below the medium-confidence threshold (0.45) we treat the result as
  "no real match" and return zero products. The LLM then gives the
  honest "sorry, we don't have that" response.
