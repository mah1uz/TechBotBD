import pandas as pd
import faiss
import numpy as np
import os
import glob  # <--- Added to search for multiple files
from sentence_transformers import SentenceTransformer
from techbot.preprocessor import preprocess

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class TechBotRAG:
    def __init__(self, data_dir='techbot/data/'):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Missing data directory at: {data_dir}")
        
        # 1. Find ALL .csv files in the data directory
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}. Please add your scraped data.")
            
        print(f"📂 Found {len(csv_files)} data files. Merging them into the brain...")
        
        # 2. Read and combine all CSVs
        df_list = []
        for file in csv_files:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
            
        # 3. Merge them into one giant table
        self.df = pd.concat(df_list, ignore_index=True)
        self.df = self.df.fillna("") # Prevent errors from empty spreadsheet cells
        
        # 4. DYNAMIC COMBINATION: Automatically joins all columns for searching
        self.df['search_text'] = self.df.apply(lambda row: " ".join([str(val) for val in row.values]), axis=1)
        
        print("🧠 Loading combined database into FAISS Memory...")
        embeddings = model.encode(self.df['search_text'].tolist())
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        
    def retrieve(self, q, k=3):
        clean_q = preprocess(q)
        _, indices = self.index.search(model.encode([clean_q]).astype('float32'), k)
        return self.df.iloc[indices[0]].to_dict('records')

_engine = None
def get_rag_engine():
    global _engine
    if _engine is None:
        _engine = TechBotRAG()
    return _engine