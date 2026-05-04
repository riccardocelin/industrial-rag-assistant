from app.rag import RAG
import pytest
######################################################################

class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = Embeddings()

class Embeddings:
    def create(self, model: str, input: str):
        return Response()

class Response:
    def __init__(self):
        self.data = [
            type("Obj", (), {"embedding": [0.1, 0.2, 0.3]})()
        ]

######################################################################

class FakeQdrantClient:
    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ):
        return FakeQueryResponse(
            points=[
                FakePoint(
                    point_id="chunk_001",
                    payload={
                        "chunk_id": "chunk_001",
                        "text": "Maintenance should be performed every 6 months.",
                        "source": "manual.pdf",
                        "pages": [10, 11],
                        "document_id": "manual",
                    },
                    score=0.91,
                ),
                FakePoint(
                    point_id="chunk_001",
                    payload={
                        "chunk_id": "chunk_001",
                        "text": "Maintenance should be performed every 6 months.",
                        "source": "manual.pdf",
                        "pages": [10, 11],
                        "document_id": "manual",
                    },
                    score=0.91,
                ),
                FakePoint(
                    point_id="chunk_001",
                    payload={
                        "chunk_id": "chunk_001",
                        "text": "Maintenance should be performed every 6 months.",
                        "source": "manual.pdf",
                        "pages": [10, 11],
                        "document_id": "manual",
                    },
                    score=0.91,
                ),
            ]
        )

class FakePoint:
    def __init__(self, point_id: str, payload: dict, score: float):
        self.id = point_id
        self.payload = payload
        self.score = score

class FakeQueryResponse:
    def __init__(self, points: list[FakePoint]):
        self.points = points

######################################################################
@pytest.mark.unit
def test_retrieve_returns_documents():

    TOP_K = 3

    rag = RAG(
        vector_db_client=FakeQdrantClient(),
        openai_client=FakeOpenAIClient(),
        top_k=TOP_K
    )

    fake_query = "What is the maintenance schedule for the DSC800 system?"

    results = rag.retrieve(fake_query)
    
    # check retrieved query embedding
    assert isinstance(results, list)
    assert all(
        isinstance(item, dict)
        and {"chunk_id", "score", "text", "source", "pages"} <= item.keys()
        for item in results
    )

    # check for retrieved documents number according to top_k and score threshold
    assert len(results) == TOP_K

    # check that the internal state of the rag system is retrieved correctly
    state = rag.get_internal_state()
    assert state["top_k"] == TOP_K
    assert state["llm_model"] is None