
import requests
import json
import chromadb

from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("webpilot_websites")


# TOOL 1: Open and extract a website
def open_website(url):

    # Basic URL validation
    if not url.startswith(("https://", "http://")):
        return {"error": "Invalid URL"}

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=15000
            )

            result = {
                "title": page.title(),
                "url": page.url,
                "content": page.locator("body").inner_text()[:4000]
            }

            return result

        finally:
            browser.close()


# TOOL 2: Search our knowledge base
def search_website(query):

    embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return results["documents"][0]


# Register tools
available_functions = {
    "open_website": open_website,
    "search_website": search_website
}


# Tool definitions for Qwen3
tools = [
    {
        "type": "function",
        "function": {
            "name": "open_website",
            "description": "Open a website URL and extract its title and visible text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The website URL to visit"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_website",
            "description": "Search the locally stored website knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


messages = [
    {
        "role": "user",
        "content": "Search my stored knowledge base for information about quotes."
    }
]


# Agent execution loop
for step in range(5):

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:4b",
            "messages": messages,
            "tools": tools,
            "stream": False
        }
    )

    response.raise_for_status()

    assistant_message = response.json()["message"]

    messages.append(assistant_message)

    tool_calls = assistant_message.get("tool_calls", [])

    # No more tools requested
    if not tool_calls:
        print("\nWebPilot:", assistant_message["content"])
        break

    for tool_call in tool_calls:

        name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        print("\nExecuting:", name)
        print("Arguments:", arguments)

        if name not in available_functions:
            result = {"error": "Unknown tool"}
        else:
            try:
                result = available_functions[name](**arguments)
            except Exception as e:
                result = {"error": str(e)}

        messages.append({
            "role": "tool",
            "name": name,
            "content": json.dumps(result)
        })
