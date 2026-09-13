"""PDF processing and QA — same LangChain/Chroma/OpenAI pipeline as the Streamlit app."""

from __future__ import annotations

import os
import shutil
from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DOCS_DIRECTORY, OPENAI_API_KEY, PERSIST_DIRECTORY

_qa_chain: RetrievalQA | None = None


def get_openai_api_key() -> str:
    return OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")


def database_ready() -> bool:
    return os.path.exists(PERSIST_DIRECTORY)


def docs_count() -> int:
    if not os.path.exists(DOCS_DIRECTORY):
        return 0
    return len([f for f in os.listdir(DOCS_DIRECTORY) if f.endswith(".pdf")])


def list_docs() -> list[dict[str, Any]]:
    if not os.path.exists(DOCS_DIRECTORY):
        return []
    files = []
    for name in sorted(os.listdir(DOCS_DIRECTORY)):
        if not name.endswith(".pdf"):
            continue
        path = os.path.join(DOCS_DIRECTORY, name)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        files.append({"name": name, "size_mb": round(size_mb, 2)})
    return files


def invalidate_qa_cache() -> None:
    global _qa_chain
    _qa_chain = None


def llm_pipeline() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=1024,
        api_key=get_openai_api_key(),
    )


def qa_llm() -> RetrievalQA | None:
    global _qa_chain
    if _qa_chain is not None:
        return _qa_chain

    if not database_ready():
        return None

    embeddings = OpenAIEmbeddings(api_key=get_openai_api_key())
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm_pipeline(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return _qa_chain


def process_answer(instruction: str) -> tuple[str, dict[str, Any]]:
    qa = qa_llm()
    if qa is None:
        return "Error: Could not initialize QA system. Upload and process documents first.", {}

    try:
        generated_text = qa(instruction)
        answer = generated_text["result"]
        return answer, generated_text
    except Exception as e:
        return f"Error processing question: {str(e)}", {}


def save_uploaded_files(files: list[tuple[str, bytes]]) -> list[str]:
    """Save (filename, content) pairs into docs/ and return saved paths."""
    os.makedirs(DOCS_DIRECTORY, exist_ok=True)
    saved: list[str] = []
    for filename, content in files:
        safe_name = os.path.basename(filename)
        if not safe_name.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(DOCS_DIRECTORY, safe_name)
        with open(file_path, "wb") as f:
            f.write(content)
        saved.append(file_path)
    return saved


def process_uploaded_documents(files: list[tuple[str, bytes]]) -> tuple[bool, str]:
    """Process uploaded PDFs and rebuild the vector database. Returns (success, message)."""
    if not files:
        return False, "No files uploaded."

    saved_files = save_uploaded_files(files)
    if not saved_files:
        return False, "Failed to save uploaded files. Only PDF files are accepted."

    documents = []
    for file_path in saved_files:
        try:
            loader = PDFMinerLoader(file_path)
            file_documents = loader.load()
            documents.extend(file_documents)
        except Exception as e:
            return False, f"Error processing {os.path.basename(file_path)}: {str(e)}"

    if not documents:
        return False, "No content extracted from uploaded PDFs."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    if not texts:
        return False, "No texts found after splitting documents."

    embeddings = OpenAIEmbeddings(api_key=get_openai_api_key())

    try:
        invalidate_qa_cache()
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )

        return (
            True,
            f"Successfully processed {len(texts)} text chunks from {len(documents)} pages "
            f"across {len(saved_files)} document(s).",
        )
    except Exception as e:
        return False, f"Error while creating embeddings: {e}"


def clear_database() -> tuple[bool, str]:
    invalidate_qa_cache()
    messages = []
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        messages.append("Database cleared.")
    if os.path.exists(DOCS_DIRECTORY):
        shutil.rmtree(DOCS_DIRECTORY)
        messages.append("Uploaded files cleared.")
    if not messages:
        return True, "Nothing to clear."
    return True, " ".join(messages)


def serialize_sources(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    sources = []
    docs = metadata.get("source_documents") or []
    for i, doc in enumerate(docs):
        content = getattr(doc, "page_content", "") or ""
        preview = content[:500] + ("..." if len(content) > 500 else "")
        meta = getattr(doc, "metadata", None) or {}
        source_file = meta.get("source", "Unknown")
        sources.append(
            {
                "index": i + 1,
                "preview": preview,
                "source": os.path.basename(source_file) if source_file else "Unknown",
            }
        )
    return sources
