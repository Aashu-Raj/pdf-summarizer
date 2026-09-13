from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.deps import get_current_user
from app.services import pdf_qa

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def list_documents(_user: Annotated[dict, Depends(get_current_user)]):
    return {
        "files": pdf_qa.list_docs(),
        "database_ready": pdf_qa.database_ready(),
        "doc_count": pdf_qa.docs_count(),
    }


@router.post("/process")
async def process_documents(
    _user: Annotated[dict, Depends(get_current_user)],
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    payload: list[tuple[str, bytes]] = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed: {f.filename}")
        content = await f.read()
        payload.append((f.filename, content))

    if not pdf_qa.get_openai_api_key():
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")

    success, message = pdf_qa.process_uploaded_documents(payload)
    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "message": message,
        "database_ready": pdf_qa.database_ready(),
        "doc_count": pdf_qa.docs_count(),
        "files": pdf_qa.list_docs(),
    }


@router.delete("")
def clear_documents(_user: Annotated[dict, Depends(get_current_user)]):
    success, message = pdf_qa.clear_database()
    return {
        "message": message,
        "database_ready": False,
        "doc_count": 0,
        "files": [],
        "success": success,
    }
