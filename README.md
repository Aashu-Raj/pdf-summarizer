# PDF Summarizer

AI-powered PDF search and Q&A. Upload PDFs, index them with OpenAI embeddings + ChromaDB, and ask questions.

**Frontend:** React + Vite + shadcn/ui  
**Backend:** FastAPI (MySQL auth, LangChain, Chroma, OpenAI)

## Features

- PDF upload (multiple files)
- Process / clear vector database
- Ask questions with source references
- Login / register (MySQL users)
- Admin panel (approve users, roles, delete)

## Setup

### Prerequisites

- Python 3.10+
- Node.js 20+
- MySQL
- OpenAI API key

### 1. Environment

Create MySQL database:

```sql
CREATE DATABASE pdf_summarizer;
```

Create `.env` in the project root:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database_name
JWT_SECRET=change-me-in-production
```

### 2. Install

```bash
python3 -m venv myenv
source myenv/bin/activate   # Windows: myenv\Scripts\activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

### 3. Run

Use the helper scripts (two terminals), or run manually:

```bash
# Terminal 1 — API
./scripts/dev-backend.sh

# Terminal 2 — React
./scripts/dev-frontend.sh
```

Manual:

```bash
# backend
cd backend && uvicorn main:app --reload --port 8000

# frontend
cd frontend && npm run dev
```

Open http://localhost:5173 (Vite proxies `/api` → backend `:8000`).

On first API start, `init_db()` creates the `users` table and seeds admin (`admin` / `admin123`). Change that password after first login.

## Usage

1. Login or register
2. Upload PDFs → **Process Documents**
3. Ask a question → **Search**
4. Admins see the sidebar admin panel for approvals

## Technical stack

- React + shadcn/ui + Tailwind
- FastAPI
- MySQL (`auth_db.py`)
- LangChain + ChromaDB + OpenAI GPT-3.5-turbo / embeddings
- PDFMiner

## Project layout

```
pdf-summarizer/
├── frontend/            # React + shadcn UI
├── backend/             # FastAPI API
│   ├── main.py
│   └── app/
│       ├── routes/      # auth, documents, ask, admin, status
│       └── services/    # PDF / QA pipeline
├── scripts/             # dev-backend.sh / dev-frontend.sh
├── auth_db.py           # MySQL user helpers
└── requirements.txt
```
