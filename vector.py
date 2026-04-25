from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd







df = pd.read_csv("all_data.csv")
embeddings= OllamaEmbeddings(model="mxbai-embed-large")

db_location= r".\chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content = (
    str(row["p_name"]) + " " +
    str(row["release_date"]) + " " +
    str(row["specs"]) + " " +
    str(row["price_in_dollar"]) + " " +
    str(row["category"])
)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="tech_products",
    embedding_function=embeddings,
    persist_directory=db_location
)        

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retreiver = vector_store.as_retriever(search_kwargs={"k": 10})    