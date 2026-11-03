
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import json

# Load embedding model
model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Connect to ChromaDB
client=chromadb.PersistentClient(path="./chroma_db")
collection=client.get_collection("webpilot_documents")

# Tool 1: Search knowledge base
def search_knowledge_base(query):
    embedding=model.encode(query).tolist()

    results=collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return results["documents"][0]

# Tool 2: Count documents
def count_documents():
    return collection.count()

# Register available tools
available_functions={
    "search_knowledge_base":search_knowledge_base,
    "count_documents":count_documents
}

# Tell the LLM which tools it can use
tools=[
    {
        "type":"function",
        "function":{
            "name":"search_knowledge_base",
            "description":"Search the local knowledge base for information relevant to a query.",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The topic or question to search for"
                    }
                },
                "required":["query"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"count_documents",
            "description":"Get the total number of documents stored in the knowledge base.",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[]
            }
        }
    }
]

messages=[
    {
        "role":"user",
        "content":"Search my knowledge base for information about LangGraph."
    }
]

# Agent execution loop
for step in range(3):

    response=requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model":"qwen3:4b",
            "messages":messages,
            "tools":tools,
            "stream":False
        }
    )

    response.raise_for_status()

    assistant_message=response.json()["message"]

    messages.append(assistant_message)

    tool_calls=assistant_message.get("tool_calls",[])

    # No more tool calls: display final answer
    if not tool_calls:
        print("Agent:",assistant_message["content"])
        break

    # Execute requested tools
    for tool_call in tool_calls:

        name=tool_call["function"]["name"]
        arguments=tool_call["function"]["arguments"]

        print("Calling tool:",name)
        print("Arguments:",arguments)

        if name not in available_functions:
            result="Error: Unknown tool"
        else:
            result=available_functions[name](**arguments)

        # Send tool result back to LLM
        messages.append({
            "role":"tool",
            "name":name,
            "content":json.dumps(result)
        })
