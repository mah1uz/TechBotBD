from techbot.rag_pipeline import get_rag_engine
from techbot.llm import get_response
from techbot.prompts import SYSTEM_PROMPT, build_prompt
from techbot.router import route
from techbot.tools import filter_products, build_link_block
import pandas as pd
import os

# Load database for tool filtering
def load_db():
    return pd.read_csv("techbot/data/merged_clean_products.csv")

def chat(query):
    try:
        df = load_db()
        intent = route(query)
        
        # Determine strict category based on query keywords
        target_cat = None
        if any(w in query.lower() for w in ['phone', 'mobile', 'ফোন']): target_cat = 'smartphone'
        elif any(w in query.lower() for w in ['laptop', 'ল্যাপটপ']): target_cat = 'laptop'
        elif any(w in query.lower() for w in ['tab', 'tablet', 'ট্যাব']): target_cat = 'tablet'

        # 1. Get filtered data
        results = filter_products(query, df, intent_category=target_cat)
        
        # 2. Build Context for LLM
        if not results:
            context = "No matching products found."
        else:
            context_list = []
            for item in results:
                # Build a clean string of specs for the LLM to read
                specs = ", ".join([f"{k}: {v}" for k, v in item.items() if pd.notna(v) and k != 'url'])
                context_list.append(specs)
            context = "\n".join(context_list)

        # 3. Get LLM Response (Uses Groq/Ollama from your llm.py)
        prompt = build_prompt(query, context)
        response = get_response(prompt, SYSTEM_PROMPT)

        # 4. Smart Link Box Logic
        # If the LLM response contains "দুঃখিত" or results are empty, skip the links
        if "দুঃখিত" in response or not results:
            return response
        else:
            return response + build_link_block(results)

    except Exception as e:
        return f"Pipeline Error: {str(e)}"