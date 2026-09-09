# PDF Summarizer

A powerful AI-powered PDF document search and question-answering application built with Streamlit and OpenAI. Upload PDF documents and ask questions to get intelligent answers based on your document content.

## Features

- **PDF Document Upload**: Upload multiple PDF files at once
- **AI-Powered Search**: Ask questions about your documents using OpenAI GPT models
- **Source References**: Get answers with source document references
- **Interactive UI**: Clean and seamless Streamlit interface
- **Real-time Processing**: Live document processing with progress indicators
- **User Authentication**: Login/register with MySQL-backed user accounts and admin panel

## How It Works

1. **Upload PDFs**: Select and upload your PDF documents
2. **Process Documents**: The app extracts text, creates embeddings, and builds a searchable vector database
3. **Ask Questions**: Query your documents using natural language
4. **Get Answers**: Receive AI-powered answers with source document references

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- MySQL Server installed and running

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Aashu-Raj/pdf-summarizer.git
cd pdf-summarizer
```

2. Create a virtual environment:
```bash
python -m venv myenv
# On Windows
myenv\Scripts\activate
# On macOS/Linux
source myenv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create the MySQL database:
```sql
CREATE DATABASE pdf_summarizer;
```

5. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=pdf_summarizer
```

6. Run the application:
```bash
streamlit run main.py
```

On first run, `init_db()` creates the `users` table and seeds a default admin account (`admin` / `admin123`). Change that password after first login.

## Usage

1. **Start the App**: Run `streamlit run main.py` and open your browser to the provided URL
2. **Login or Register**: Authenticate before using the tool
3. **Upload Documents**: Use the file uploader to select PDF files
4. **Process Documents**: Click "Process Documents" to create the searchable database
5. **Ask Questions**: Enter your questions in the text area and click "Search"
6. **View Results**: Get AI-powered answers with source references

## Configuration

### Model Settings

- **Model**: GPT-3.5-turbo
- **Temperature**: 0.2
- **Max Tokens**: 1024

### Document Processing

- **Chunk Size**: 1000 characters per text chunk
- **Chunk Overlap**: 100 characters overlap between chunks
- **Retrieval**: Top 3 most relevant chunks per query

## Technical Stack

- **Frontend**: Streamlit
- **AI Models**: OpenAI GPT-3.5-turbo
- **User Database**: MySQL
- **Vector Database**: ChromaDB
- **Document Processing**: LangChain
- **PDF Processing**: PDFMiner
- **Embeddings**: OpenAI Embeddings

## File Structure

```
pdf-summarizer/
├── main.py              # Entry point: Streamlit application
├── auth.py              # Login / registration UI
├── auth_db.py           # MySQL user database helpers
├── admin.py             # Admin user management panel
├── requirements.txt     # List of Python dependencies
├── .env                 # Environment variables (create this)
├── db/                  # Vector database storage (auto-created when running the app)
├── docs/                # Directory for uploaded PDF files (auto-created on upload)
└── README.md            # Project documentation
```

## API Key Setup

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root
3. Add your API key: `OPENAI_API_KEY=sk-your-actual-api-key-here`
4. Add MySQL connection settings (`MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`)
