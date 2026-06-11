"""
LLM backend wrapper. Supports Groq (cloud) and Ollama (local) via a single
get_response() entry point.

Design:
  - Backend chosen per call via the `backend` arg (defaults to config).
  - Retries once on any exception before returning a Bengali error string.
  - Never crashes the caller — always returns a string.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from techbot.config import (
    DEFAULT_LLM_BACKEND, GROQ_MODEL, OLLAMA_MODEL, OLLAMA_URL,
    LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ERROR_MESSAGE_BN = (
    "দুঃখিত, এই মুহূর্তে উত্তর দিতে পারছি না। অনুগ্রহ করে আবার চেষ্টা করুন।"
)


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

_groq_client = None

def _get_groq_client():
    """Lazy-init Groq client to avoid import errors when key is absent."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in environment.")
        from groq import Groq  # imported lazily so absence isn't fatal
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _call_groq(prompt: str, system_prompt: str) -> str:
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system_prompt: str) -> str:
    """
    Ollama /api/generate. We concatenate system_prompt + user prompt because
    the simpler /api/generate endpoint doesn't have a separate system field
    in older versions; this works across versions.
    """
    full_prompt = (
        f"{system_prompt}\n\n---\n\n{prompt}\n\n---\n\nএখন বাংলায় উত্তর দিন:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def get_response(prompt: str, system_prompt: str,
                 backend: Optional[str] = None) -> str:
    """
    Call the chosen backend. Retry once on failure. Return error string
    rather than raising on terminal failure.
    """
    backend = (backend or DEFAULT_LLM_BACKEND).lower()
    print(f"[LLM] Using backend: {backend}")

    last_err = None
    for attempt in (1, 2):
        try:
            if backend == "groq":
                return _call_groq(prompt, system_prompt)
            elif backend == "ollama":
                return _call_ollama(prompt, system_prompt)
            else:
                return f"{ERROR_MESSAGE_BN} (Unknown backend: {backend})"
        except Exception as e:
            last_err = e
            print(f"[LLM] Attempt {attempt} failed ({backend}): {e}")
            if attempt == 1:
                time.sleep(0.5)
                continue

    print(f"[LLM] Both attempts failed. Last error: {last_err}")
    return print(f"There was a problem with the {backend} backend: {last_err}")


def is_backend_available(backend: str) -> bool:
    """Quick check used by the UI to display backend status."""
    backend = backend.lower()
    if backend == "groq":
        return bool(GROQ_API_KEY)
    if backend == "ollama":
        try:
            r = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"),
                             timeout=2)
            return r.status_code == 200
        except Exception:
            return False
    return False
