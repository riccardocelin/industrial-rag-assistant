# Industrial RAG Assistant

A domain-focused Retrieval-Augmented Generation (RAG) project for industrial maintenance and diagnostics use cases.

This repository currently implements an end-to-end offline pipeline that:
1. Ingests industrial PDF manuals.
2. Splits documents into structured text chunks.
3. Builds embeddings for each chunk.
4. Creates and populates a Qdrant collection.
5. Retrieves relevant chunks and generates answers with an OpenAI model.

The current demo knowledge base is based on ABB technical documentation for variable speed drives (VSDs):
https://library.e.abb.com/public/a44d07ce27e7665e85257ccb00539304/3ADW000195_F.pdf

---

## Repository structure

```text
.
├── app/rag/rag_system.py
├── src/ingestion/
│   ├── ingestion.py
│   └── config.ingestion.example.yaml
├── src/embeddings/
│   ├── build_embeddings.py
│   └── config.embeddings.example.yaml
├── src/vectordb/
│   ├── create_collection.py
│   ├── load_embeddings.py
│   └── config.vectordb.example.yaml
├── requirements.txt
└── README.md
```

---

## File-by-file overview

### `src/ingestion/ingestion.py`
Implements the ingestion pipeline with typed config dataclasses and modular steps:
- Loads ingestion settings from YAML.
- Discovers PDF files from a folder.
- Loads PDFs with `PyPDFLoader`.
- Splits content into overlapping chunks using `RecursiveCharacterTextSplitter`.
- Writes chunks to JSONL as `{text, metadata}` records.

### `src/ingestion/config.ingestion.example.yaml`
Example config for ingestion:
- Input PDF directory and glob pattern.
- Chunk size, overlap, and separators.
- Output path for generated chunk JSONL.

### `src/embeddings/build_embeddings.py`
Reads chunk JSONL and generates embeddings in batches using the OpenAI API:
- Loads API key/model from environment.
- Computes text hashes for traceability.
- Stores vectors and embedding metadata into a new JSONL output.

### `src/embeddings/config.embeddings.example.yaml`
Example config defining:
- Input chunk file location.
- Output embeddings file location.
- Batch size and skip-existing behavior.

### `src/vectordb/create_collection.py`
Creates a Qdrant collection for vector search:
- Reads collection settings from YAML.
- Detects embedding dimension from the first record in the embeddings JSONL.
- Creates collection only if it does not already exist.

### `src/vectordb/load_embeddings.py`
Loads embedding records into Qdrant:
- Converts each chunk into `PointStruct`.
- Inserts text and provenance metadata as payload.
- Upserts all points into the configured collection.

### `app/rag/rag_system.py`
Defines a `RAG` class with two key stages:
- **Retrieve**: embed user query and search Qdrant for top-k relevant chunks.
- **Generate**: call chat completion with retrieved context (or optional no-context mode for debugging).

### `requirements.txt`
Python dependencies for the current implementation, including:
- `openai`, `qdrant-client`
- `langchain`, `langchain_community`, `langchain_text_splitters`
- `pypdf`, plus `fastapi` and `uvicorn` for forthcoming serving needs.

---

## Current workflow

A typical local flow is:

1. Configure ingestion/embedding/vector DB YAML files (copy from `.example.yaml`).
2. Run ingestion to create `chunks.jsonl`.
3. Run embedding generation to create `chunks_with_embeddings.jsonl`.
4. Create the Qdrant collection.
5. Load embeddings into Qdrant.
6. Use the `RAG` class to retrieve context and generate answers.

---

## Qdrant quick start

```bash
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

Qdrant dashboard: http://localhost:6333/dashboard

---

## Project direction

This project is intentionally structured to evolve beyond a local prototype. Near-term evolution will focus on three major areas:

### 1) API serving
The repository already includes `fastapi`/`uvicorn` dependencies. The next step is to expose the RAG pipeline as a production-style API service (e.g., health checks, query endpoints, request validation, and configurable model/vector parameters).

### 2) Docker containerization
The current setup includes containerized Qdrant usage. The project will evolve toward full multi-service containerization so ingestion/embedding/runtime components can be packaged and executed consistently across local, staging, and production environments.

### 3) CI/CD workflows
To improve reliability and delivery speed, the project will add CI/CD workflows for linting, tests, build checks, image publishing, and automated deployment gates.

In short, this repository is a solid foundation for an industrial RAG assistant and is designed to grow into a fully served, containerized, and continuously delivered application.
