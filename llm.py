import os
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
GROQ_MODEL = "llama-3.3-70b-versatile"

_first_call = True

def get_response(prompt: str, system_prompt: str, retries: int = 1) -> str:
    global _first_call
    if _first_call:
        print(f"🤖 [System] Active LLM Backend: {LLM_BACKEND.upper()}")
        _first_call = False

    try:
        if LLM_BACKEND == "groq":
            if not GROQ_API_KEY:
                return "Error: GROQ_API_KEY not found in .env file."
            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )
            return completion.choices[0].message.content

        elif LLM_BACKEND == "ollama":
            url = "http://localhost:11434/api/generate"
            # Combine system and user prompt for Ollama's generate endpoint
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
            
        else:
            return f"Error: Unknown LLM_BACKEND '{LLM_BACKEND}'"

    except Exception as e:
        if retries > 0:
            print(f"⚠️ LLM Error ({e}). Retrying...")
            return get_response(prompt, system_prompt, retries=retries - 1)
        return f"LLM Backend Error: {str(e)}"