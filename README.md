# 🚀 LocalCodeMind

Offline AI Repository Intelligence System

LocalCodeMind is an experimental local-first AI system for semantic repository analysis.

It allows users to:

- Index GitHub repositories
- Perform semantic code search
- Build repository embeddings
- Visualize project architecture
- Explore large codebases locally
- Analyze CUDA / GGML / inference pipelines

---

# Features

## 🔍 Semantic Repository Search

Ask questions like:

- Where is CUDA implemented?
- KV cache implementation
- Quantization code
- GGUF loader
- Server implementation

---

## 🧠 Local Embeddings

Uses:

- sentence-transformers
- Qdrant vector database
- local embedding generation

---

## 🗺 Architecture Visualization

Generate repository maps and dependency graphs.

---

## 🌐 Web UI

Built with Gradio.

---

# Stack

- Python
- Gradio
- Qdrant
- sentence-transformers
- llama.cpp
- GGUF models

---

# Example

```bash
python rag_webui.py
