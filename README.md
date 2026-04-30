# TechBot Project 🤖
A modular RAG-based Gadget Recommendation Chatbot supporting Groq API and Local LLMs.

### Setup Instructions
1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Environment Variables:** Copy `.env.example` to `.env` and add your Groq API key.
3. **Data:** Place your `.csv` product files in `techbot/data/`
4. **Run Terminal Bot:** `python main.py`
5. **Run Streamlit UI:** `streamlit run app.py`

### Switch to Local LLM (Ollama)
1. Install [Ollama](https://ollama.com/)
2. Pull the model: `ollama run gemma3:4b`
3. In your `.env` file, change `LLM_BACKEND=ollama`