from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.deps import get_current_user
from app.services import pdf_qa

router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@router.post("")
def ask_question(
    body: AskRequest,
    _user: Annotated[dict, Depends(get_current_user)],
):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question to search.")

    if not pdf_qa.database_ready():
        raise HTTPException(
            status_code=400,
            detail="Document database is not ready. Upload and process PDFs first.",
        )

    if not pdf_qa.get_openai_api_key():
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")

    answer, metadata = pdf_qa.process_answer(question)
    return {
        "question": question,
        "answer": answer,
        "sources": pdf_qa.serialize_sources(metadata),
    }
