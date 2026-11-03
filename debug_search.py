from sentence_transformers import SentenceTransformer
import numpy as np

print("1. Imports successful",flush=True)

model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("2. Model loaded",flush=True)

documents=[
    "LangGraph builds stateful AI agents.",
    "Python is a programming language.",
    "RAG combines retrieval with generation."
]

embeddings=model.encode(documents)

print("3. Embeddings generated",flush=True)

import faiss

print("4. FAISS imported",flush=True)

embeddings=np.array(embeddings).astype("float32")

index=faiss.IndexFlatL2(384)
index.add(embeddings)

print("5. Documents added",flush=True)

query_embedding=model.encode(["How do I build AI agents?"])

distances,indices=index.search(
    np.array(query_embedding).astype("float32"),3
)

print("6. Search completed",flush=True)
print(indices)
print(distances)