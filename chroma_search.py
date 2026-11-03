
from sentence_transformers import SentenceTransformer
import chromadb

# Load our embedding model
model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Create a persistent ChromaDB client
client=chromadb.PersistentClient(path="./chroma_db")

# Create or access a collection
collection=client.get_or_create_collection(
    name="webpilot_documents",
    metadata={"hnsw:space":"cosine"}
)

documents=[
    "LangGraph is used to build stateful AI agents.",
    "Python is a popular programming language.",
    "RAG combines information retrieval with language models.",
    "MCP allows AI applications to connect with external tools.",
    "Vector databases store and search embeddings."
]

ids=["doc1","doc2","doc3","doc4","doc5"]

metadatas=[
    {"topic":"agents"},
    {"topic":"programming"},
    {"topic":"rag"},
    {"topic":"tools"},
    {"topic":"databases"}
]

# Generate embeddings
embeddings=model.encode(documents)

# Store documents, vectors and metadata
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Documents stored:",collection.count())

# Search
query="How do I build AI agents?"

query_embedding=model.encode(query).tolist()

results=collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Display results
for i in range(len(results["ids"][0])):
    print("ID:",results["ids"][0][i])
    print("Document:",results["documents"][0][i])
    print("Distance:",results["distances"][0][i])
    print("Metadata:",results["metadatas"][0][i])
    print()
