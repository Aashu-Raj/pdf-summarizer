# PDF Summarizer

A powerful AI-powered PDF document search and question-answering application built with Streamlit and OpenAI. Upload PDF documents and ask questions to get intelligent answers based on your document content.

## Features

- **PDF Document Upload**: Upload multiple PDF files at once
- **AI-Powered Search**: Ask questions about your documents using OpenAI GPT models
- **Vector Database**: Uses ChromaDB for efficient document indexing and retrieval
- **Source References**: Get answers with source document references
- **Multiple Model Support**: Choose between GPT-3.5-turbo, GPT-4, and GPT-4-turbo-preview
- **Interactive UI**: Clean and intuitive Streamlit interface
- **Real-time Processing**: Live document processing with progress indicators

## How It Works

1. **Upload PDFs**: Select and upload your PDF documents
2. **Process Documents**: The app extracts text, creates embeddings, and builds a searchable vector database
3. **Ask Questions**: Query your documents using natural language
4. **Get Answers**: Receive AI-powered answers with source document references

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

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

4. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

5. Run the application:
```bash
streamlit run main.py
```

## Usage

1. **Start the App**: Run `streamlit run main.py` and open your browser to the provided URL
2. **Upload Documents**: Use the file uploader to select PDF files
3. **Process Documents**: Click "Process Documents" to create the searchable database
4. **Ask Questions**: Enter your questions in the text area and click "Search"
5. **View Results**: Get AI-powered answers with source references

## Configuration

### Model Settings

- **Model Selection**: Choose between GPT-3.5-turbo, GPT-4, or GPT-4-turbo-preview
- **Temperature**: Adjust creativity level (0.0 = focused, 1.0 = creative)
- **Max Tokens**: Control response length

### Document Processing

- **Chunk Size**: 1000 characters per text chunk
- **Chunk Overlap**: 100 characters overlap between chunks
- **Retrieval**: Top 3 most relevant chunks per query

## Technical Stack

- **Frontend**: Streamlit
- **AI Models**: OpenAI GPT-3.5-turbo/GPT-4
- **Vector Database**: ChromaDB
- **Document Processing**: LangChain
- **PDF Processing**: PDFMiner
- **Embeddings**: OpenAI Embeddings

## File Structure

```
pdf-summarizer/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── db/                  # Vector database storage
├── docs/                # Uploaded PDF files
└── README.md            # This file
```

## API Key Setup

1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root
3. Add your API key: `OPENAI_API_KEY=sk-your-actual-api-key-here`

