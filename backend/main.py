import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend/ is on path so `app` package resolves when running uvicorn
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

ROOT = BACKEND_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CORS_ORIGINS  # noqa: E402
from app.routes import admin, ask, auth, documents, status  # noqa: E402

import auth_db  # noqa: E402

app = FastAPI(title="PDF Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    auth_db.init_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.include_router(auth.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(ask.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(status.router, prefix="/api")
