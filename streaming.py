
import requests
import json

url="http://localhost:11434/api/chat"

payload={
    "model":"qwen3:4b",
    "messages":[
        {
            "role":"user",
            "content":"Explain how an autonomous AI agent works in 300 words."
        }
    ],
    "stream":True
}

response=requests.post(url,json=payload,stream=True)
response.raise_for_status()

for line in response.iter_lines():
    if line:
        data=json.loads(line)
        content=data.get("message",{}).get("content","")
        print(content,end="",flush=True)

print()
