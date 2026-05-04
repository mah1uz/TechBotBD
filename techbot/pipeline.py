from techbot.rag_pipeline import get_rag_engine
from techbot.llm import get_response

def chat(query):
    try:
        rag = get_rag_engine()
        results = rag.retrieve(query)
        
        # DYNAMIC CONTEXT: Automatically formats whatever columns are in your CSV
        context_parts = []
        for idx, item in enumerate(results):
            # Exclude the temporary 'search_text' column
            details = ", ".join([f"{k}: {v}" for k, v in item.items() if k != 'search_text' and v])
            context_parts.append(f"{idx+1}. {details}")
            
        context = "\n".join(context_parts)
        
        sys_msg = "আপনি একজন গ্যাজেট বিশেষজ্ঞ। প্রদত্ত তথ্যের উপর ভিত্তি করে ব্যবহারকারীকে বাংলায় সাহায্য করুন। বুলেট পয়েন্ট ব্যবহার করুন।"
        prompt = f"ব্যবহারকারীর প্রশ্ন: {query}\n\nআমাদের কাছে থাকা তথ্য:\n{context}"
        
        return get_response(prompt, sys_msg)
    except Exception as e:
        return f"Pipeline Error: {e}"