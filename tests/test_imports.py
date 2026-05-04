import pytest

@pytest.mark.unit
def test_can_import_rag():

    from app.rag import RAG, build_rag
    
    rag = RAG()
    assert rag is not None

    rag = build_rag()
    assert rag is not None