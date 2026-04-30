import pandas as pd
import re
import os

# --- HELPER FUNCTIONS ---
def clean_name(text):
    if pd.isna(text): return "Unknown"
    # Remove the trailing numeric codes and newlines common in Ryans data
    name = str(text).split('\n')[0].strip()
    if name.endswith('...'): name = name[:-3].strip()
    return name

def clean_price(text):
    if pd.isna(text): return 0
    # Remove 'Tk', commas, and extra spaces
    text = str(text).replace(',', '').replace('Tk', '').strip()
    match = re.search(r'\d+', text)
    return int(match.group()) if match else 0

def process_file(filepath, category):
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: {filepath} not found. Skipping...")
        return None
    
    df = pd.read_csv(filepath)
    records = []
    
    for _, row in df.iterrows():
        record = {
            'url': row.get('image-box href', ''),
            'name': clean_name(row.get('product-name', '')),
            'price_bdt': clean_price(row.get('pr-text', '')),
            'category': category,
            'image_url': row.get('card-img-top src', '')
        }
        
        # Dynamically extract specs from columns like 'card-text', 'card-text (2)', etc.
        for col in df.columns:
            if 'card-text' in col and pd.notna(row[col]):
                val_str = str(row[col])
                if ' - ' in val_str:
                    # Split 'RAM - 8GB' into Key: ram, Value: 8GB
                    key, val = val_str.split(' - ', 1)
                    clean_key = key.strip().lower().replace(' ', '_').replace('/', '_').replace('.', '').replace('(', '').replace(')', '')
                    record[clean_key] = val.strip()
        records.append(record)
    
    return pd.DataFrame(records)

# --- MAIN EXECUTION ---
files_to_process = {
    "laptops.csv": "laptop",
    "Tabs.csv": "tablet",
    "phones.csv": "smartphone"
}

all_dfs = []

for file, cat in files_to_process.items():
    print(f"🔄 Processing {file}...")
    df_clean = process_file(file, cat)
    if df_clean is not None:
        all_dfs.append(df_clean)

if all_dfs:
    # Merge all cleaned dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # --- STANDARDIZATION ---
    # Merge similar columns (e.g., 'rom' from phones and 'storage' from laptops)
    col_mapping = {
        'processor_model': 'processor',
        'processor_type': 'processor',
        'cpu_series': 'processor',
        'screen_display_type': 'display_type',
        'display_size_inch': 'display_inch',
        'battery': 'battery_mah',
        'front_camera': 'camera_front',
        'back_rear_camera': 'camera_rear',
        'graphics_chipset': 'gpu',
        'rom': 'storage',
        'cellular_network': 'network'
    }
    merged_df.rename(columns=col_mapping, inplace=True)
    
    # Remove rows where product name is missing
    merged_df = merged_df[merged_df['name'] != "Unknown"]
    
    # Save the final file
    output_name = "merged_clean_products.csv"
    merged_df.to_csv(output_name, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"Total products cleaned: {len(merged_df)}")
    print(f"Unified file saved as: {output_name}")
    print(f"Available columns for RAG: {list(merged_df.columns)}")
else:
    print("❌ No files were processed.")