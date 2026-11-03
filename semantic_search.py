
from sentence_transformers import SentenceTransformer,util

model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")

documents=[
    "LangGraph is a framework for building stateful AI agents.",
    "Python is a popular programming language.",
    "RAG combines information retrieval with language models.",
    "MCP allows AI applications to connect with external tools.",
    "LangChain helps developers build applications powered by LLMs.",
    "Vector databases store and search high-dimensional embeddings."
]

# Convert documents into embeddings
doc_embeddings=model.encode(documents)

query="How can I build AI agents with workflows?"

# Convert query into an embedding
query_embedding=model.encode(query)

# Find the top 3 most similar documents
results=util.semantic_search(query_embedding,doc_embeddings,top_k=3)

for result in results[0]:
    index=result["corpus_id"]
    score=result["score"]
    print(f"Score: {score:.4f}")
    print(f"Document: {documents[index]}")
    print()
