"""
TechBot BD — CLI mode

Run:
    python main.py                  # default backend (groq)
    python main.py --backend ollama
    python main.py --lang en        # English responses
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from techbot import pipeline
from techbot.config import DEFAULT_CSV
from techbot.rag_pipeline import TechBotRAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Path to products CSV")
    ap.add_argument("--backend", default="groq", choices=["groq", "ollama"])
    ap.add_argument("--lang", default="bn", choices=["bn", "en"])
    ap.add_argument("--debug", action="store_true",
                    help="Print full result dict instead of just the response")
    args = ap.parse_args()

    print(f"[main] Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"[main] {len(df)} products loaded.")

    print("[main] Building RAG engine (this may take a minute on first run)...")
    rag = TechBotRAG(df)
    print("[main] Ready. Type 'exit' to quit.\n")

    history: list[dict] = []
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "/exit", "/quit"):
            break

        result = pipeline.chat(
            query=query,
            history=history,
            df=df,
            rag_engine=rag,
            backend=args.backend,
            language=args.lang,
        )

        if args.debug:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            print(f"\nTechBot: {result['response']}")
            if result.get("link_block"):
                print(result["link_block"])
            print(
                f"\n[intent={result['intent']} | confidence={result['confidence']} | "
                f"web={result['used_web_search']}]\n"
            )

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result["response"]})


if __name__ == "__main__":
    main()
