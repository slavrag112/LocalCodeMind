from pathlib import Path
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

repo_paths = [
    Path.home() / "RAG" / "knowledge",
    Path.home() / "RAG" / "LocalCodeMind",

    Path.home() / "RAG" / "sources" / "ollama",
    
    Path.home() / "RAG" / "sources" / "llama.cpp",
    
    
]

extensions = [
    ".py",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
]

SKIP_DIRS = {
    ".git",
    "venv",
    "__pycache__",
    "tests",
    "test",
    ".pytest_cache",
    "node_modules",
    ".cache",
    ".idea",
    ".vscode",
}

SKIP_FILES = {
    ".aider.chat.history.md",
}

files = []

for repo_path in repo_paths:

    if not repo_path.exists():
        print("SKIP:", repo_path)
        continue

    for ext in extensions:

        for f in repo_path.rglob(f"*{ext}"):

            if f.name in SKIP_FILES:
                continue

            if any(part in SKIP_DIRS for part in f.parts):
                continue

            files.append(f)

print("FILES:", len(files))

chunks = []

CHUNK_SIZE = 1200
OVERLAP = 200

for file in files:

    try:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if len(text.strip()) < 50:
            continue

        for i in range(
            0,
            len(text),
            CHUNK_SIZE - OVERLAP
        ):

            chunk = text[i:i + CHUNK_SIZE]

            if len(chunk.strip()) > 50:

                chunks.append({
                    "path": str(file),
                    "text": chunk
                })

    except Exception:
        pass

print("CHUNKS:", len(chunks))

if not chunks:
    raise RuntimeError("No chunks found")

print("Creating embeddings...")

model = TextEmbedding()

texts = [c["text"] for c in chunks]

embeddings = list(model.embed(texts))

print("Embeddings:", len(embeddings))

client = QdrantClient(
    "localhost",
    port=6333
)

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

    print(
        f"Uploaded {i + len(batch)} / {len(points)}"
    )

print(
    f"Inserted {len(points)} vectors"
)
