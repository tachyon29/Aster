
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# Load embedding model
model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Connect to ChromaDB
client=chromadb.PersistentClient(path="./chroma_db")

collection=client.get_collection("webpilot_documents")

# User question
question="How does LangGraph help build AI agents?"

# Generate query embedding
query_embedding=model.encode(question).tolist()

# Retrieve relevant documents
results=collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

documents=results["documents"][0]

# Construct context
context="\n".join(documents)

# Construct augmented prompt
prompt=f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

If the context does not contain enough information,
say that you don't have enough information.
"""

# Send prompt to Ollama
response=requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model":"qwen3:4b",
        "messages":[
            {"role":"user","content":prompt}
        ],
        "stream":False
    }
)

response.raise_for_status()

answer=response.json()["message"]["content"]

print("Retrieved documents:")
for doc in documents:
    print("-",doc)

print("\nAnswer:")
print(answer)
