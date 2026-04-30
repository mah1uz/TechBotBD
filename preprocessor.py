import re

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. Basic Cleaning
    text = text.lower().strip()
    
    # 2. Comprehensive Normalization Map
    # Key: Regex pattern (what the user might type)
    # Value: The standardized Bengali term (what's likely in your CSV)
    norm_map = {
        # Device Categories
        r"phone|phon|mobile|set": "ফোন",
        r"laptop|notebook|computer|pc": "ল্যাপটপ",
        r"camera|cam": "ক্যামেরা",
        r"watch|smartwatch": "ঘড়ি",
        r"headphone|earbud|airpod|buds": "হেডফোন",
        
        # Financial / Budget terms
        r"budget|daam|price|cost|range": "বাজেট",
        r"taka|tk|bdt|টাকা": "টাকা",
        r"thousand|k|হাজার": "হাজার",
        r"cheap|low cost|sasta": "কম দাম",
        r"expensive|premium|high end": "বেশি দাম",
        
        # Tech Specs
        r"ram|memory": "র‍্যাম",
        r"rom|storage|space": "স্টোরেজ",
        r"battery|backup": "ব্যাটারি",
        r"display|screen": "ডিসপ্লে",
        r"processor|chipset|cpu": "প্রসেসর",
        r"gaming|gamer|play": "গেমিং",
        
        # Action Verbs
        r"show|dekhao|dekhaw|find|search": "দেখান",
        r"best|ভালো|valo|valow": "সেরা",
        r"buy|kena|kinbo": "কেনা"
    }
    
    # Apply the mapping
    for pattern, replacement in norm_map.items():
        text = re.sub(pattern, replacement, text)
    
    # 3. Noise Removal
    # Keep Bengali characters, English alphanumeric, and basic punctuation
    text = re.sub(r'[^a-z0-9\u0980-\u09FF\s.,!?]', '', text)
    
    # 4. Whitespace Normalization
    return " ".join(text.split())