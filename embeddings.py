
from sentence_transformers import SentenceTransformer,util

model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Running on:",model.device)

sentences=[
    "I love programming",
    "I enjoy writing code",
    "The weather is very hot"
]

embeddings=model.encode(sentences)

print("Shape:",embeddings.shape)

similarity=util.cos_sim(embeddings,embeddings)

print(similarity)
