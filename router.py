import re

def route(query: str) -> str:
    query = query.lower()
    
    comp_keywords = [r'তুলনা', r'তুলনামূলক', r'পার্থক্য', r'\bvs\b', r'versus', r'compare', r'difference', r'কোনটা ভালো', r'এর চেয়ে', r'better']
    filter_keywords = [r'বাজেট', r'দামের মধ্যে', r'টাকার মধ্যে', r'under', r'below', r'within', r'gaming', r'গেমিং', r'camera', r'ক্যামেরা', r'battery', r'ব্যাটারি', r'student', r'office', r'সেরা', r'best', r'সবচেয়ে']
    
    for kw in comp_keywords:
        if re.search(kw, query):
            return "comparison"
            
    for kw in filter_keywords:
        if re.search(kw, query):
            return "filter"
            
    return "recommendation"