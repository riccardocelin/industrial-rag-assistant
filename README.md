# Industrial RAG Assistant

Industrial RAG Assistant is a Python project for industrial-domain question answering using a Retrieval-Augmented Generation (RAG) architecture.

It includes:
- A document ingestion and chunking pipeline for PDFs.
- OpenAI embedding generation for chunks.
- Qdrant as vector database for retrieval.
- A FastAPI `/ask` endpoint for RAG answers.
- A Streamlit GUI that communicates with the API endpoint.

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
│   ├── ginterface/
│   │   ├── gui.py
│   │   └── README.md
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
│   ├── test_api.py
│   ├── config.test_api.yaml
│   └── config.test_api.example.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## How the RAG system works

1. **Ingestion** (`src/ingestion/ingestion.py`) reads PDFs and splits text into chunks with metadata.
2. **Embeddings** (`src/embeddings/build_embeddings.py`) generates OpenAI embeddings for each chunk and saves enriched JSONL.
3. **Vector DB setup** (`src/vectordb/create_collection.py`) creates a Qdrant collection with cosine distance.
4. **Load vectors** (`src/vectordb/load_embeddings.py`) upserts chunk embeddings + payload into Qdrant.
5. **Runtime API** (`app/api.py`) receives user questions, retrieves top-k chunks via `RAG.retrieve`, and generates an answer via `RAG.generate`.

---

## Streamlit GUI ↔ endpoint communication

The Streamlit interface is implemented in `app/ginterface/gui.py` and is designed as a thin client for the FastAPI endpoint.

### GUI behavior

- Renders a page title and caption using app settings.
- Provides:
  - A text area for the user question.
  - A `Force no context` checkbox.
  - An `Ask` button.
- Validates that the question is not empty.

### Endpoint integration

When the user clicks **Ask**, the GUI sends an HTTP POST request to `API_URL`, which is loaded from settings (`api_ask_endpoint_url`, default `http://localhost:8000/ask`).

Payload sent by Streamlit:

```json
{
  "question": "<user text>",
  "force_no_context": false
}
```

The request is sent with `requests.post(..., timeout=120)`.

### Response handling in GUI

- On success:
  - Displays `answer` from the JSON response.
  - Displays `sources` in expandable sections when context mode is enabled and sources are present.
- On error:
  - Catches request exceptions and shows a user-facing error message.

This mirrors the same `/ask` contract used by `test/test_api.py`, so both automated API tests and the Streamlit GUI exercise the same endpoint shape.

---

## API endpoints

### Run API locally

```bash
uvicorn app.api.api:app --host 0.0.0.0 --port 8000
```

### `GET /health`

Health check:

```json
{"status": "ok"}
```

### `POST /ask`

Request body:

```json
{
  "question": "What type of maintenance is necessary for the DCS800 system?",
  "force_no_context": false
}
```
- `question` (string, required): user question.
- `force_no_context` (boolean, optional, default `false`): if `true`, bypasses retrieved context during generation (debug/behavior comparison mode).

Response body:

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

---

## Quick start

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

### 3) Prepare pipeline configs

Create local configs from examples:
- `src/ingestion/config.ingestion.example.yaml` → `src/ingestion/config.ingestion.yaml`
- `src/embeddings/config.embeddings.example.yaml` → `src/embeddings/config.embeddings.yaml`
- `src/vectordb/config.vectordb.example.yaml` → `src/vectordb/config.vectordb.yaml`

### 4) Run pipeline

```bash
python src/ingestion/ingestion.py
python src/embeddings/build_embeddings.py
python src/vectordb/create_collection.py
python src/vectordb/load_embeddings.py
```

### 5) Start API

```bash
uvicorn app.api:app --reload
```

### 6) Start Streamlit GUI

```bash
streamlit run app/ginterface/gui.py
```

Then open the URL shown by Streamlit and ask questions through the UI.

---

## Configuration

Settings are managed in `app/core/settings.py` (Pydantic `BaseSettings`), including:
- `OPENAI_API_KEY`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_LLM_MODEL`
- `VECTOR_DB_HOST`, `VECTOR_DB_PORT`, `VECTOR_DB_COLLECTION_NAME`
- `RETRIEVAL_TOP_K`
- `API_ASK_ENDPOINT_URL`

A local `.env` is supported for development.



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

## Docker setup (Compose + Dockerfiles)

This repository includes containerized runtime services for Qdrant, the FastAPI backend, and the Streamlit UI.

### Dockerfiles

- `docker/api.Dockerfile`
  - Uses `python:3.11-slim`.
  - Installs `requirements-api.txt`.
  - Copies `app/api`, `app/rag`, and `app/core`.
  - Starts the API with `uvicorn app.api.api:app --host 0.0.0.0 --port 8000`.
- `docker/ui.Dockerfile`
  - Uses `python:3.11-slim`.
  - Installs `requirements-ui.txt`.
  - Copies `app/ginterface` and `app/core`.
  - Starts UI with `streamlit run app/ginterface/gui.py --server.address=0.0.0.0 --server.port=8501`.

### Docker Compose services

`compose.yaml` defines three services:

1. `qdrant`
   - Image: `qdrant/qdrant:v1.13.4`
   - Port mapping: `${VECTOR_DB_PORT:-6333}:6333`
   - Persistent volume: `./qdrant_storage:/qdrant/storage`

2. `api`
   - Built from `docker/api.Dockerfile`
   - Loads environment from `.env`
   - Uses internal Docker network to reach Qdrant with:
     - `VECTOR_DB_HOST=qdrant`
     - `VECTOR_DB_PORT=6333`
   - Port mapping: `${API_PORT:-8000}:8000`

3. `ui`
   - Built from `docker/ui.Dockerfile`
   - Loads environment from `.env`
   - Uses internal Docker network to reach API with:
     - `API_HOST=api`
     - `API_PORT=8000`
   - Port mapping: `${UI_PORT:-8501}:8501`

### Run the full stack with Docker Compose

1. Create a `.env` file in the project root with at least:

```env
OPENAI_API_KEY=your_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-5.4-mini
VECTOR_DB_COLLECTION_NAME=my_collection
API_PORT=8000
UI_PORT=8501
VECTOR_DB_PORT=6333
```

2. Build and start all services:

```bash
docker compose up --build -d
```

3. Check container status:

```bash
docker compose ps
```

4. Access services:

- API health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

5. Stop services:

```bash
docker compose down
```

To also remove named/anonymous volumes created by Compose, run:

```bash
docker compose down -v
```

