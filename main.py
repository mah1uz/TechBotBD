from techbot.pipeline import chat
import warnings

# Suppress the huggingface token warning for a cleaner terminal
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

print("========================================")
print("🤖 TechBot Initializing... Please wait.")
print("========================================")

# This dummy call forces the model to download/load into RAM immediately
_ = chat("init") 
print("\n✅ TechBotBD is ready! (Type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("TechBotBD: ধন্যবাদ! আবার দেখা হবে।")
        break
        
    print("TechBotBD is typing...")
    response = chat(user_input)
    print(f"\n🤖 Bot: {response}\n")
    print("-" * 40)