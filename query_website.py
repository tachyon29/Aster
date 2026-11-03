
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("webpilot_websites")

question = "What are the quotes and who wrote them?"

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for i, document in enumerate(results["documents"][0]):

    print("\nDocument:", document)

    print("Source:", results["metadatas"][0][i]["source_url"])

    print("Distance:", results["distances"][0][i])
