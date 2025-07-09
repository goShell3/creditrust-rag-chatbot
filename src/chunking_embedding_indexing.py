import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

#Load the cleaned data 
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "processed" / "filtered_complaints.csv"

df = pd.read_csv(data_path)

#Text Chunking 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # Chosen to balance context and embedding performance
    chunk_overlap=50,     # Helps preserve context across splits
    separators=["\n\n", "\n", ".", "!", "?", " "]
)

documents = []
metadatas = []

df['Product'] = df['Product'].astype('category')
df['Complaint ID'] = df['Complaint ID'].astype(str)
df['cleaned_narrative'] = df['cleaned_narrative'].fillna('').astype(str)

for idx, row in df.iterrows():
    chunks = text_splitter.split_text(row['cleaned_narrative'])
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "complaint_id": row['Complaint ID'],
            "product": row['Product'],
            "chunk_index": i
        })

print(f"Total text chunks: {len(documents)}")

# Generate Embeddings 
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

embeddings = embedder.encode(documents, show_progress_bar=True, convert_to_numpy=True)

# Index in Chroma Vector Store
persist_path = "vector_store/chroma_db"
os.makedirs(persist_path, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=persist_path)
collection = chroma_client.get_or_create_collection(name="complaint_chunks")

ids = [f"chunk_{i}" for i in range(len(documents))]

# collection.add(
#     ids=ids,
#     embeddings=embeddings.tolist(),
#     documents=documents,
#     metadatas=metadatas
# )

batch_size = 5000  # < 5461 as required
total = len(ids)

for i in tqdm(range(0, total, batch_size)):
    batch_ids = ids[i:i+batch_size]
    batch_documents = documents[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]

    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        metadatas=batch_metadatas
    )

chroma_client.persist()
print(f"Vector store saved to {persist_path}")
