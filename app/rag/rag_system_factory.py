from openai import OpenAI
from qdrant_client import QdrantClient

from app.core.settings import get_settings
from app.rag.rag_system import RAG


def build_rag() -> RAG:
    settings = get_settings()

    openai_client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    )

    vector_db_client = QdrantClient(
        host=settings.vector_db_host,
        port=settings.vector_db_port,
    )

    return RAG(
        openai_client=openai_client,
        vector_db_client=vector_db_client,
        embedding_model=settings.openai_embedding_model,
        llm_model=settings.openai_llm_model,
        collection_name=settings.vector_db_collection_name,
        top_k=settings.retrieval_top_k,
    )