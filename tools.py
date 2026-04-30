import pandas as pd
import re

def filter_products(query: str, df, intent_category=None):
    query_lower = query.lower()
    
    # 1. Category Enforcement
    filtered_df = df.copy()
    if intent_category:
        filtered_df = filtered_df[filtered_df['category'] == intent_category]

    # 2. Extract Price Ceiling (e.g., 50k, 50000)
    price_match = re.search(r'(\d+)\s*(k|হাজার|000)', query_lower)
    if price_match:
        val = int(price_match.group(1))
        unit = price_match.group(2)
        ceiling = val * 1000 if unit in ['k', 'হাজার'] else int(price_match.group(0))
        filtered_df = filtered_df[filtered_df['price_bdt'] <= ceiling]

    # Sort by price to show best options
    results = filtered_df.sort_values(by='price_bdt', ascending=False).head(5)
    return results.to_dict('records')

def build_link_block(products: list[dict]):
    # This now only returns a string if there are actual products
    if not products:
        return ""
    
    block = "\n\n---\n📦 **পণ্যের দাম ও লিংক:**\n"
    for i, prod in enumerate(products):
        name = prod.get('name', 'Unknown')
        price = prod.get('price_bdt', 0)
        url = prod.get('url', '')
        price_str = f"Tk {int(price):,}" if price > 0 else "দাম জানা নেই"
        
        if url and str(url).startswith('http'):
            block += f"{i+1}. **{name}** — {price_str} — [Ryans-এ দেখুন]({url})\n"
        else:
            block += f"{i+1}. **{name}** — {price_str}\n"
    return block