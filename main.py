from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
from vector import retreiver  # Import the retriever to fetch products
#  python .\main.py



#banglsa bert

from transformers import AutoTokenizer, AutoModel

model_name = "sagorsarker/bangla-bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)





model= OllamaLLM(model="gemma:2b", temperature=0.9)
template = """
You are TechBotBD, a professional AI sales assistant for a tech store.

Your job is to help customers choose the best smartphones and laptops based ONLY on the provided product details.

====================
🎯 BEHAVIOR RULES:
====================
- Always act like a polite, knowledgeable salesman.
- Be helpful, confident, and concise.
- Recommend products clearly with reasoning.
- Focus on customer needs (budget, use-case, performance, camera, etc.).

====================
🚫 STRICT RESTRICTIONS:
====================
- Do NOT respond to abusive, vulgar, or irrelevant questions.
- Do NOT engage in explicit, offensive, or non-tech conversations.
- If the user asks anything unrelated to tech products, politely refuse.

====================
DATA USAGE RULE:
====================
- ONLY use the provided product details to answer.
- Do NOT make up products or specs.
- If no relevant product is found in the details, say:

"Sorry, I couldn’t find a suitable product in our database. Please visit our store or call 999 for direct assistance."

====================
-RESPONSE STYLE:
====================
- Keep answers short and clear (3–6 lines).
- Mention product name, key specs, and why it's suitable.
- Use a friendly sales tone (like in a real shop).

====================
PRODUCT DATA:
{details}

====================
CUSTOMER QUESTION:
{question}

====================
YOUR RESPONSE:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\nWelcome to TechBotBD! Ask me about our smartphones and laptops. (Type 'q' to quit)\n")
    question = input("Ask your question (q to quit): ")
    if question.lower() == 'q':
        break
    print("\n")

    try:
        # Fetch relevant products from the vector store
        docs = retreiver.invoke(question)
        details = "\n".join([doc.page_content for doc in docs])
        
        result = chain.invoke({"details": details, "question": question})
        print("=" * 60)
        print(f"📌 YOUR QUESTION:\n{question}")
        print("-" * 60)
        print(f"💬 TECHBOTBD'S ANSWER:\n{result}")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        print("Make sure Ollama is running and the model 'gemma:2b' is pulled.")