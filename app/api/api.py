from pydantic import BaseModel, Field
from typing import List
from fastapi import FastAPI
from fastapi import HTTPException

from app.rag.rag_system import RAG
from app.core.settings import get_settings

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What type of maintenance is necessary for the DCS800 system?"])
    force_no_context: bool = False

class SourceItem(BaseModel):
    chunk_id: int
    source: str
    text: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


rag = RAG()
app = FastAPI(title=get_settings().app_name)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        retrieved_docs = rag.retrieve(request.question)
        answer = rag.generate(request.question, retrieved_docs, force_no_context=request.force_no_context)
        return AskResponse(answer=answer, sources=retrieved_docs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")