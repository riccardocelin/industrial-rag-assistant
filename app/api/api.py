from pydantic import BaseModel, Field
from typing import List
from functools import lru_cache
from fastapi import FastAPI, Depends
from fastapi import HTTPException

from app.rag import RAG, build_rag

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

app = FastAPI()

@lru_cache
def get_rag() -> RAG:
    return build_rag()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, rag: RAG = Depends(get_rag)):
    try:
        retrieved_docs = rag.retrieve(request.question)
        answer = rag.generate(request.question, retrieved_docs, force_no_context=request.force_no_context)
        return AskResponse(answer=answer, sources=retrieved_docs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")