
import requests

url="http://localhost:11434/api/chat"

payload={
    "model":"qwen3:4b",
    "messages":[
        {
            "role":"user",
            "content":"Give me 10 creative names for an AI startup."
        }
    ],
    "options":{
        "temperature":0.0
    },
    "stream":False
}

response=requests.post(url,json=payload)
data=response.json()

print(data["message"]["content"])
