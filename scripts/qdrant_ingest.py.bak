from pathlib import Path
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

repo_path = Path.home() / "RAG"

extensions = [".py", ".cpp", ".cu", ".h", ".hpp", ".md", ".txt"]

files = []

for ext in extensions:
    files.extend(repo_path.rglob(f"*{ext}"))

print("FILES:", len(files))

chunks = []

CHUNK_SIZE = 1200
OVERLAP = 200

for file in files[:400]:
    try:
        text = file.read_text(errors="ignore")

        if len(text.strip()) < 300:
            continue

        for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]

            if len(chunk.strip()) > 200:
                chunks.append({
                    "path": str(file),
                    "text": chunk
                })

    except Exception:
        pass

print("CHUNKS:", len(chunks))

model = TextEmbedding()

texts = [c["text"] for c in chunks]

embeddings = list(model.embed(texts))

client = QdrantClient("localhost", port=6333)

collection_name = "technical_rag"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=len(embeddings[0]),
        distance=Distance.COSINE
    ),
)

points = []

for i, emb in enumerate(embeddings):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload=chunks[i]
        )
    )

BATCH_SIZE = 256

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]

    client.upsert(
        collection_name=collection_name,
        points=batch
    )

    print(f"Uploaded {i + len(batch)} / {len(points)}")

print(f"Inserted {len(points)} vectors")
