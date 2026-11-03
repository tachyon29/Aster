
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.server import MCPServer
from playwright.sync_api import sync_playwright
import requests

# Initialize MCP server
mcp = MCPServer("WebPilot Tools")

# Load the knowledge base only if its search tool is requested. Importing the
# browser server should not load a model or make any Hugging Face network calls.
BASE_DIR = Path(__file__).parent
_model = None
_collection = None


def get_knowledge_collection():
    global _model, _collection
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            local_files_only=True,
        )
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
        _collection = client.get_or_create_collection(
            name="webpilot_websites",
            metadata={"hnsw:space": "cosine"},
        )
    return _model, _collection


class PageTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts = []
        self.text_parts = []
        self.in_title = False
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data):
        cleaned = data.strip()
        if not cleaned:
            return
        if self.in_title:
            self.title_parts.append(cleaned)
        elif not self.skip_depth:
            self.text_parts.append(cleaned)


class SearchResultsParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.active = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self.active = {"tag": tag, "url": attrs.get("href", ""), "title": ""}
            self.results.append(self.active)
        elif tag in {"a", "div", "span"} and "result__snippet" in classes:
            self.active = {"tag": tag, "snippet": ""}
            self.results.append(self.active)

    def handle_data(self, data):
        if self.active:
            key = "title" if "title" in self.active else "snippet"
            self.active[key] += data.strip() + " "

    def handle_endtag(self, tag):
        if self.active and tag == self.active.get("tag"):
            self.active = None


def fetch_page_without_browser(url: str) -> dict:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; WebPilot/1.0)"},
        timeout=20,
    )
    response.raise_for_status()
    parser = PageTextParser()
    parser.feed(response.text)
    return {
        "title": " ".join(parser.title_parts),
        "url": response.url,
        "content": re.sub(r"\s+", " ", " ".join(parser.text_parts))[:4000],
        "retrieval_mode": "http_fallback",
    }


@mcp.tool()
def open_website(url: str) -> dict:
    """Open a website and extract its title and visible text."""

    if not url.startswith(("https://", "http://")):
        return {"error": "Invalid URL"}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                return {
                    "title": page.title(),
                    "url": page.url,
                    "content": page.locator("body").inner_text()[:4000],
                    "retrieval_mode": "playwright",
                }
            finally:
                browser.close()
    except Exception as browser_error:
        try:
            return fetch_page_without_browser(url)
        except Exception as fallback_error:
            raise RuntimeError(
                "Browser retrieval failed ({}); HTTP fallback failed ({}).".format(
                    browser_error, fallback_error
                )
            ) from fallback_error


@mcp.tool()
def search_web(query: str) -> dict:
    """Search the web and return relevant pages with URLs and short snippets. Use open_website to verify facts on a selected source."""
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        return {"error": "Search query cannot be empty", "results": []}
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0 (compatible; WebPilot/1.0)"},
        timeout=18,
    )
    response.raise_for_status()
    parser = SearchResultsParser()
    parser.feed(response.text)
    links = [item for item in parser.results if "url" in item and item.get("url")]
    snippets = [item for item in parser.results if "snippet" in item and item.get("snippet")]
    results = []
    for index, item in enumerate(links[:6]):
        url = item["url"]
        if "duckduckgo.com/l/?" in url:
            target = urlparse(url)
            url = parse_qs(target.query).get("uddg", [url])[0]
        results.append({
            "title": re.sub(r"\s+", " ", item.get("title", "")).strip(),
            "url": url,
            "snippet": re.sub(r"\s+", " ", snippets[index].get("snippet", "")).strip() if index < len(snippets) else "",
        })
    return {"query": query, "results": results}


@mcp.tool()
def search_knowledge_base(query: str) -> dict:
    """Search previously crawled website content using semantic similarity."""

    model, collection = get_knowledge_collection()
    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    return {
        "results": [
            {
                "content": document,
                "metadata": metadata,
                "distance": distance
            }
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    }


if __name__ == "__main__":
    mcp.run()
