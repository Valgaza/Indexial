import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not JINA_API_KEY:
    raise RuntimeError("JINA_API_KEY not found")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

# -----------------------
# 1. Test Jina Embeddings
# -----------------------
print("Testing Jina embeddings...")

jina_url = "https://api.jina.ai/v1/embeddings"
jina_headers = {
    "Authorization": f"Bearer {JINA_API_KEY}",
    "Content-Type": "application/json",
}

jina_payload = {
    "model": "jina-embeddings-v3",
    "task": "text-matching",
    "dimensions": 1024,
    "input": "This is a test sentence for Jina embeddings."
}

jina_resp = requests.post(
    jina_url,
    headers=jina_headers,
    json=jina_payload,
    timeout=60,
)

jina_resp.raise_for_status()
jina_data = jina_resp.json()

vector = jina_data["data"][0]["embedding"]
print(f"Embedding length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")

# -----------------------
# 2. Test Groq LLM
# -----------------------
print("\nTesting Groq LLM...")

groq_url = "https://api.groq.com/openai/v1/chat/completions"
groq_headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

groq_payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain what Groq is in one short paragraph."}
    ],
    "temperature": 0.7,
    "max_tokens": 150,
}

groq_resp = requests.post(
    groq_url,
    headers=groq_headers,
    json=groq_payload,
    timeout=60,
)

groq_resp.raise_for_status()
groq_data = groq_resp.json()

print(groq_data["choices"][0]["message"]["content"])
