
from sentence_transformers import SentenceTransformer
import chromadb
from playwright.sync_api import sync_playwright
from collections import deque
from urllib.parse import urlparse, urldefrag
import hashlib
import re

# Configuration
START_URL = "https://quotes.toscrape.com/"
MAX_PAGES = 10
MAX_DEPTH = 2
CHUNK_SIZE = 500
OVERLAP = 75

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="webpilot_websites",
    metadata={"hnsw:space": "cosine"}
)


# 1. Clean webpage text
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# 2. Split text into chunks
def chunk_text(text, chunk_size=500, overlap=75):

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        if end < len(text):
            space = text.rfind(" ", start, end)

            if space > start:
                end = space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


# 3. Normalize URLs
def normalize_url(url):
    url = urldefrag(url)[0]
    return url.rstrip("/") if urlparse(url).path else url


# 4. Crawl and ingest website
with sync_playwright() as p:

    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    domain = urlparse(START_URL).hostname

    queue = deque([(START_URL, 0)])
    visited = set()

    total_chunks = 0

    while queue and len(visited) < MAX_PAGES:

        current_url, depth = queue.popleft()
        current_url = normalize_url(current_url)

        if current_url in visited:
            continue

        parsed = urlparse(current_url)

        if parsed.scheme not in ("http", "https"):
            continue

        if parsed.hostname != domain:
            continue

        visited.add(current_url)

        print("\nCrawling:", current_url)

        try:

            page.goto(
                current_url,
                wait_until="domcontentloaded",
                timeout=15000
            )

            title = page.title()

            # Extract webpage text
            raw_text = page.locator("body").inner_text()

            # Clean text
            text = clean_text(raw_text)

            # Chunk text
            chunks = chunk_text(
                text,
                CHUNK_SIZE,
                OVERLAP
            )

            if chunks:

                # Generate embeddings
                embeddings = model.encode(chunks)

                ids = []
                metadatas = []

                for i, chunk in enumerate(chunks):

                    unique_id = hashlib.sha256(
                        f"{current_url}:{i}".encode()
                    ).hexdigest()

                    ids.append(unique_id)

                    metadatas.append({
                        "source_url": current_url,
                        "page_title": title,
                        "chunk_index": i
                    })

                # Store in ChromaDB
                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )

                total_chunks += len(chunks)

                print("Title:", title)
                print("Chunks stored:", len(chunks))

            # Discover more links
            if depth < MAX_DEPTH:

                links = page.locator("a").evaluate_all(
                    "elements => elements.map(a => a.href)"
                )

                for link in links:

                    link = normalize_url(link)

                    if urlparse(link).hostname != domain:
                        continue

                    if link not in visited:
                        queue.append((link, depth + 1))

        except Exception as e:
            print("Error:", e)

    browser.close()

print("\nIngestion completed!")
print("Pages visited:", len(visited))
print("Chunks processed:", total_chunks)
print("Total records in collection:", collection.count())
