import pandas as pd
import faiss
import numpy as np
import os
import glob
import pickle
from sentence_transformers import SentenceTransformer
from techbot.preprocessor import preprocess

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#mahfuz
class TechBotRAG:
    def __init__(self, data_dir='techbot/data/', force_rebuild=False):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Missing data directory at: {data_dir}")
        
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}.")
            
        index_path = os.path.join(data_dir, "faiss_index.bin")
        cache_path = os.path.join(data_dir, "df_cache.pkl")
        
        # Check cache validity
        cache_valid = False
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(cache_path):
            cache_mtime = min(os.path.getmtime(index_path), os.path.getmtime(cache_path))
            latest_csv_mtime = max([os.path.getmtime(f) for f in csv_files])
            if cache_mtime > latest_csv_mtime:
                cache_valid = True

        if cache_valid:
            print("⚡ Loading FAISS index from cache...")
            self.df = pd.read_pickle(cache_path)
            self.index = faiss.read_index(index_path)
        else:
            print(f"📂 Processing {len(csv_files)} files. Building fresh FAISS index...")
            df_list = [pd.read_csv(f) for f in csv_files]
            self.df = pd.concat(df_list, ignore_index=True).fillna("")
            
            # Smart search_text construction (Issue 4)
            def build_text(row):
                parts = []
                cols = ['name', 'category', 'brand', 'processor', 'generation', 'ram_gb', 'storage', 'gpu', 'display_inch', 'battery_mah', 'camera_mp']
                for col in cols:
                    if col in row.index and str(row[col]).strip() not in ["", "0"]:
                        val = str(row[col])
                        if col == 'ram_gb': val += "GB RAM"
                        elif col == 'storage': val += " storage"
                        parts.append(val)
                # Excluded by omission: url, price_bdt, shop, popularity
                return " ".join(parts)

            self.df['search_text'] = self.df.apply(build_text, axis=1)
            embeddings = model.encode(self.df['search_text'].tolist())
            
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Save cache
            faiss.write_index(self.index, index_path)
            self.df.to_pickle(cache_path)
        
    def retrieve(self, q, k=5): # k changed to 5 (Issue 8)
        clean_q = preprocess(q)
        _, indices = self.index.search(model.encode([clean_q]).astype('float32'), k)
        return self.df.iloc[indices[0]].to_dict('records')

_engine = None
def get_rag_engine():
    global _engine
    if _engine is None:
        _engine = TechBotRAG()
    return _engine