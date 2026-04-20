# Industrial RAG Assistant

Industrial RAG Assistant is a Python-based Retrieval-Augmented Generation (RAG) project designed for industrial maintenance, troubleshooting, and diagnostics workflows.

The repository contains:
- A **data pipeline** for PDF ingestion, chunking, and embedding generation.
- A **vector database integration** with Qdrant.
- A **runtime RAG service** and **FastAPI HTTP API** for question answering.
- Simple **test scripts** for direct RAG calls and API calls.

The current demonstration context references ABB technical documentation for variable speed drives (VSDs):
https://library.e.abb.com/public/a44d07ce27e7665e85257ccb00539304/3ADW000195_F.pdf

---

## Repository structure

```text
.
├── app/
│   ├── api.py
│   ├── core/
│   │   └── settings.py
│   └── rag/
│       └── rag_system.py
├── src/
│   ├── ingestion/
│   │   ├── ingestion.py
│   │   └── config.ingestion.example.yaml
│   ├── embeddings/
│   │   ├── build_embeddings.py
│   │   └── config.embeddings.example.yaml
│   └── vectordb/
│       ├── create_collection.py
│       ├── load_embeddings.py
│       └── config.vectordb.example.yaml
├── test/
│   ├── test_rag_simple.py
│   ├── test_api.py
│   └── config.test.example.yaml
├── requirements.txt
└── README.md
```

---

## File-by-file overview

### Application layer (`app/`)

- `app/api.py`
  - Defines the FastAPI app and public endpoints.
  - Uses `AskRequest`, `SourceItem`, and `AskResponse` Pydantic models.
  - Builds a single `RAG` instance and serves retrieval + generation through `/ask`.

- `app/rag/rag_system.py`
  - Implements the `RAG` class.
  - `retrieve(query)` embeds the query and searches Qdrant for top-k relevant chunks.
  - `generate(query, retrieved_docs, force_no_context)` calls the LLM with or without retrieved context.
  - Includes helper methods for embedding, context formatting, and internal state inspection.

- `app/core/settings.py`
  - Centralized typed settings with `pydantic-settings`.
  - Reads environment values such as OpenAI key, model names, vector DB host/port/collection, and retrieval configuration.
  - Supports local `.env` loading and container/runtime environment injection.

- `app/__init__.py`, `app/rag/__init__.py`, `app/core/__init__.py`
  - Package initializer files.

### Data pipeline (`src/`)

- `src/ingestion/ingestion.py`
  - End-to-end ingestion pipeline:
    1. Load YAML config.
    2. Discover PDFs.
    3. Load pages via `PyPDFLoader`.
    4. Split into chunks using `RecursiveCharacterTextSplitter`.
    5. Save JSONL records as `{text, metadata}`.
  - Adds stable `chunk_id` metadata for downstream mapping.

- `src/ingestion/config.ingestion.example.yaml`
  - Example ingestion config for PDF input paths, chunking behavior, and output chunk file location.

- `src/embeddings/build_embeddings.py`
  - Loads chunk JSONL.
  - Calls OpenAI embeddings in configurable batches.
  - Writes output JSONL enriched with:
    - `embedding`
    - `text_hash` (SHA256)
    - `embedding_metadata` (model, dimension)

- `src/embeddings/config.embeddings.example.yaml`
  - Example embeddings config for input/output directories and embedding runtime options (`batch_size`, `skip_existing`).

- `src/vectordb/create_collection.py`
  - Creates a Qdrant collection if missing.
  - Infers vector dimension from first embedding record.
  - Uses cosine distance.

- `src/vectordb/load_embeddings.py`
  - Loads embeddings JSONL.
  - Converts rows to Qdrant `PointStruct` objects.
  - Upserts points with payload fields like source, page, text, and embedding metadata.

- `src/vectordb/config.vectordb.example.yaml`
  - Example vector DB config (collection name and embeddings file path).

### Testing layer (`test/`)

- `test/test_rag_simple.py`
  - Directly tests `RAG.retrieve()` and `RAG.generate()` using YAML-configured query options.

- `test/test_api.py`
  - Sends HTTP POST request to `/ask` and prints retrieved context plus final answer.

- `test/config.test.example.yaml`
  - Example values for test query, context toggle, and API URL.

- `test/__init__.py`
  - Package initializer.

### Root files

- `requirements.txt`
  - Declares runtime dependencies including FastAPI, Uvicorn, OpenAI SDK, Qdrant client, and LangChain ecosystem packages.

- `README.md`
  - Project documentation (this file).

---

## API endpoints

The API is implemented in `app/api.py` and exposes two endpoints.

### `GET /health`

Health check endpoint.

**Response (200)**

```json
{
  "status": "ok"
}
```

### `POST /ask`

Main RAG question-answering endpoint.

**Request body**

```json
{
  "question": "What type of maintenance is necessary for the DCS800 system?",
  "force_no_context": false
}
```

- `question` (string, required): user question.
- `force_no_context` (boolean, optional, default `false`): if `true`, bypasses retrieved context during generation (debug/behavior comparison mode).

**Response body (200)**

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": 12,
      "source": "manual.pdf",
      "text": "...",
      "score": 0.87
    }
  ]
}
```

Error handling:
- `400` for validation/domain errors.
- `500` for unexpected internal errors.

---

## End-to-end workflow

A typical local workflow is:

1. Copy and edit example configs:
   - `src/ingestion/config.ingestion.example.yaml` → `src/ingestion/config.ingestion.yaml`
   - `src/embeddings/config.embeddings.example.yaml` → `src/embeddings/config.embeddings.yaml`
   - `src/vectordb/config.vectordb.example.yaml` → `src/vectordb/config.vectordb.yaml`
2. Place PDFs in your configured raw data folder.
3. Run ingestion to produce chunks JSONL.
4. Run embedding generation to produce embeddings JSONL.
5. Create the Qdrant collection.
6. Load embeddings into Qdrant.
7. Start API server and query `/ask`.

---

## Runbook (quick start)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Start Qdrant (Docker)

```bash
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

Qdrant dashboard: http://localhost:6333/dashboard

### 3) Run data pipeline

```bash
python src/ingestion/ingestion.py
python src/embeddings/build_embeddings.py
python src/vectordb/create_collection.py
python src/vectordb/load_embeddings.py
```

### 4) Start API

```bash
uvicorn app.api:app --reload
```

### 5) Call API

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Which preventive maintenance schedule is suggested?","force_no_context":false}'
```

---

## Configuration and environment

The runtime relies on environment variables (directly or via `.env`) for at least:
- `OPENAI_API_KEY`
- `EMBEDDING_MODEL`
- Qdrant connection values (`QDRANT_HOST`, `QDRANT_PORT`, and collection naming where applicable)

Application defaults are centralized in `app/core/settings.py`:
- `openai_embedding_model: text-embedding-3-small`
- `openai_llm_model: gpt-5.4-mini`
- `retrieval_top_k: 5`
- `vector_db_host: localhost`
- `vector_db_port: 6333`

---

## Project evolution roadmap

This project is intentionally set up as a strong foundation that will evolve in the following directions:

1. **Full Docker containerization**
   - Move from “Qdrant-only containerized” usage to multi-service containerization.
   - Package API service, optional background ingestion/embedding jobs, and vector store dependencies in reproducible container images.
   - Introduce Docker Compose profiles for local development and staging parity.

2. **CI/CD workflows**
   - Add automated lint/test/build pipelines on pull requests.
   - Validate ingestion and API behavior in CI with controlled fixtures.
   - Build and publish container images through CI.
   - Add deployment gates and environment promotion checks for safer releases.

3. **Operational hardening**
   - Extend observability (structured logging, metrics, tracing).
   - Add stricter config validation and secret management practices.
   - Improve API robustness and documentation maturity.

In summary, the repository already provides an end-to-end industrial RAG baseline and is expected to mature into a containerized, continuously integrated, and continuously delivered production workflow.
