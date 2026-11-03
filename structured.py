
import requests
import json

url="http://localhost:11434/api/chat"

schema={
    "type":"object",
    "properties":{
        "mission":{
            "type":"string"
        },
        "tasks":{
            "type":"array",
            "items":{
                "type":"string"
            }
        },
        "priority":{
            "type":"string",
            "enum":["low","medium","high"]
        }
    },
    "required":["mission","tasks","priority"],
    "additionalProperties":False
}

payload={
    "model":"qwen3:4b",
    "messages":[
        {
            "role":"system",
            "content":"You are WebPilot. Break user missions into actionable tasks. Return only the requested structured data."
        },
        {
            "role":"user",
            "content":"Research 5 AI startups in Bangalore, investigate their products and prepare a report."
        }
    ],
    "format":schema,
    "stream":False,
    "options":{
        "temperature":0
    }
}

response=requests.post(url,json=payload)
response.raise_for_status()

result=response.json()
content=result["message"]["content"]

data=json.loads(content)

print(json.dumps(data,indent=2))
print("\nFirst task:",data["tasks"][0])
