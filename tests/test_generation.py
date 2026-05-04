from app.rag import RAG
import pytest

######################################################################

class FakeOpenAIClient:
    def __init__(self):
        self.chat = Chat()

class Chat:
    def create(self):
        return Completion()
    
class FakeAnswer:
    def __init__(self):
        self.choices = [
            type("Obj", (), {"message": type("Msg", (), {"content": "This is the generated answer."})()})()
        ]

class Completion:
    def create(
        self,
        model: str,
        messages: list[dict],
        temperature=0.0,
        max_completion_tokens=1000,
        verbosity="low",
        seed=42):
        return FakeAnswer()

######################################################################
@pytest.mark.unit
def test_answer_generation():

    rag = RAG(
        openai_client=FakeOpenAIClient()
    )

    fake_query = "What is the maintenance schedule for the DSC800 system?"

    fake_retrieved_docs = [
         {
            "chunk_id": "chunk_id_1",
            "score": 0.3,
            "text": "this is the text of the retrieved document chunk",
            "source": "document.pdf",
            "pages": [10, 11]
         },
         {
            "chunk_id": "chunk_id_2",
            "score": 0.3,
            "text": "this is the text of the retrieved document chunk",
            "source": "document.pdf",
            "pages": [10, 11]
         }
         
    ]

    answer_with_context = rag.generate(fake_query, fake_retrieved_docs, force_no_context=False)
    answer_without_context = rag.generate(fake_query, fake_retrieved_docs, force_no_context=True)
    
    # check output answer with/without context
    assert isinstance(answer_with_context, str) and answer_with_context is not None and len(answer_with_context) > 0
    assert isinstance(answer_without_context, str) and answer_without_context is not None and len(answer_without_context) > 0
