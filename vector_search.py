
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

documents=[
    "LangGraph is used to build stateful AI agents.",
    "Python is a popular programming language.",
    "RAG combines retrieval with language generation.",
    "MCP connects AI applications with external tools.",
    "Vector databases store and search embeddings.",
    "Transformers use self attention mechanisms."
]

embeddings=model.encode(documents)

# Convert embeddings to float32
embeddings=np.array(embeddings).astype("float32")

# Create a vector index
dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)

# Add document vectors
index.add(embeddings)

# Search
query="How do I build AI agents?"
query_embedding=model.encode([query])
query_embedding=np.array(query_embedding).astype("float32")

distances,indices=index.search(query_embedding,3)

for i in indices[0]:
    print(documents[i])
