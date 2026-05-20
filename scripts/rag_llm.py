from fastembed import TextEmbedding
from qdrant_client import QdrantClient
import requests
import json

# embedding model
model = TextEmbedding()

# qdrant client
client = QdrantClient("localhost", port=6333)

# question
query = input("QUESTION: ")

# query embedding
query_vector = list(model.embed([query]))[0]

# semantic retrieval
results = client.query_points(
    collection_name="technical_rag",
    query=query_vector.tolist(),
    limit=5
).points

# build context
context = ""

for r in results:
    context += f"\nFILE: {r.payload['path']}\n"
    context += r.payload["text"][:1200]
    context += "\n\n"

# prompt
prompt = f"""
You are a highly technical AI systems engineer.

Use the retrieved code context below to answer the question.

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

# ollama request
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen-local:latest",
        "prompt": prompt,
        "stream": False
    }
)

print("\nRAW API RESPONSE:\n")
print(response.text)

# parse response
try:
    data = response.json()

    print("\n" + "=" * 80)
    print("FINAL ANSWER:\n")

    if "response" in data:
        print(data["response"])
    else:
        print(data)

except Exception as e:
    print("JSON ERROR:", e)
