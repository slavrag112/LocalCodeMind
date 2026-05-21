from fastembed import TextEmbedding
from qdrant_client import QdrantClient

model = TextEmbedding()

client = QdrantClient("localhost", port=6333)

query = "How does KV cache optimization work in LLM inference?"

query_vector = list(model.embed([query]))[0]

results = client.query_points(
    collection_name="technical_rag",
    query=query_vector.tolist(),
    limit=5
).points

print("\nTOP SEMANTIC RESULTS:\n")

for r in results:
    print("=" * 80)
    print("SCORE:", round(r.score, 3))
    print("FILE:", r.payload["path"])
    print()
    print(r.payload["text"][:700])
    print()
