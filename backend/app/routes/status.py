from typing import Annotated

from fastapi import APIRouter, Depends

from app.deps import get_current_user
from app.services import pdf_qa

router = APIRouter(prefix="/status", tags=["status"])


@router.get("")
def system_status(_user: Annotated[dict, Depends(get_current_user)]):
    return {
        "openai_connected": bool(pdf_qa.get_openai_api_key()),
        "database_ready": pdf_qa.database_ready(),
        "doc_count": pdf_qa.docs_count(),
        "files": pdf_qa.list_docs(),
    }
