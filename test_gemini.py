import os
import requests
import json

response = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    },
    json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50,
        "temperature": 0
    }
)

print("Status Code:", response.status_code)
print("\nFull Response:")
print(json.dumps(response.json(), indent=2))