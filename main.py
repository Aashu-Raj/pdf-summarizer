import streamlit as st
import openai
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import tempfile
import shutil

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

persist_directory = "db"

# Define CHROMA_SETTINGS directly
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory="db",
    anonymized_telemetry=False
)

@st.cache_resource
def llm_pipeline():
    """Initialize the OpenAI LLM pipeline"""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=1024,
        api_key=openai.api_key
    )
    return llm

@st.cache_resource
def qa_llm():
    """Initialize the QA system with vector database"""
    llm = llm_pipeline()
    
    # Use OpenAI embeddings for better integration
    embeddings = OpenAIEmbeddings(api_key=openai.api_key)
    
    # Load existing vector database if it exists
    if os.path.exists(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        st.error("No document database found. Please upload and process documents first.")
        return None
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    """Process user question and return answer with metadata"""
    qa = qa_llm()
    if qa is None:
        return "Error: Could not initialize QA system", {}
    
    try:
        generated_text = qa(instruction)
        answer = generated_text['result']
        return answer, generated_text
    except Exception as e:
        return f"Error processing question: {str(e)}", {}

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory and return the path"""
    if not uploaded_files:
        return None
    
    # Create docs directory if it doesn't exist
    docs_directory = "docs"
    if not os.path.exists(docs_directory):
        os.makedirs(docs_directory)
    
    saved_files = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to docs directory
        file_path = os.path.join(docs_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
        st.success(f"✅ Saved: {uploaded_file.name}")
    
    return saved_files

def process_uploaded_documents(uploaded_files):
    """Process uploaded PDF documents and create vector database"""
    if not uploaded_files:
        st.error("No files uploaded.")
        return False
    
    # Save uploaded files
    saved_files = save_uploaded_files(uploaded_files)
    if not saved_files:
        st.error("Failed to save uploaded files.")
        return False
    
    documents = []
    
    # Process each uploaded file
    for file_path in saved_files:
        try:
            st.info(f"📄 Processing: {os.path.basename(file_path)}")
            loader = PDFMinerLoader(file_path)
            file_documents = loader.load()
            documents.extend(file_documents)
            st.success(f"✅ Loaded: {os.path.basename(file_path)} ({len(file_documents)} pages)")
        except Exception as e:
            st.error(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")

    if not documents:
        st.error("No content extracted from uploaded PDFs.")
        return False

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    if not texts:
        st.error("No texts found after splitting documents.")
        return False

    # Create embeddings and vector database
    embeddings = OpenAIEmbeddings(api_key=openai.api_key)
    
    try:
        # Remove existing database if it exists
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        
        # Create new database
        db = Chroma.from_documents(
            texts, 
            embeddings, 
            persist_directory=persist_directory
        )
        
        st.success(f"🎉 Successfully processed {len(texts)} text chunks from {len(documents)} pages across {len(saved_files)} document(s)!")
        return True
    except Exception as e:
        st.error(f"Error while creating embeddings: {e}")
        return False

def clear_database():
    """Clear the existing vector database"""
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        st.success("🗑️ Database cleared successfully!")
    
    if os.path.exists("docs"):
        shutil.rmtree("docs")
        st.success("📁 Uploaded files cleared!")

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title('🔍 Search your PDF with OpenAI')

# Check if API key is loaded
if not openai.api_key:
    st.error("❌ OpenAI API Key not found!")
    st.markdown("""
    **Please ensure your .env file contains:**
    ```
    OPENAI_API_KEY=sk-your-actual-api-key-here
    ```
    """)
    
    # Fallback: Ask user for API key
    api_key = st.text_input("Or enter your OpenAI API Key manually:", type="password")
    if api_key:
        openai.api_key = api_key
        st.success("✅ API Key set successfully!")
        st.rerun()
    else:
        st.stop()
else:
    st.success("✅ OpenAI API Key loaded successfully!")

# App description
with st.expander('📖 About the App'):
    st.markdown("""
    This is a **Generative AI powered Question and Answering App** that uses OpenAI's GPT models
    to answer questions based on your PDF documents.
    
    **How to use:**
    1. 📤 **Upload** your PDF files using the file uploader below
    2. 🔄 Click **'Process Documents'** to analyze and index your documents
    3. ❓ **Ask questions** about your documents
    4. 🎯 Get **AI-powered answers** with source references
    
    **Features:**
    - Direct PDF file upload (no need to create folders)
    - Supports multiple PDF files at once
    - Semantic search through document content
    - Source document references with page content
    - Powered by OpenAI GPT-3.5-turbo
    """)

# Document upload and processing section
st.subheader("📚 Document Upload & Processing")

# File uploader
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    # help="Upload one or more PDF files to search through"
)

# Show uploaded files
if uploaded_files:
    st.write("📁 **Uploaded Files:**")
    for i, uploaded_file in enumerate(uploaded_files, 1):
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        st.write(f"{i}. {uploaded_file.name} ({file_size:.2f} MB)")

# Processing buttons
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    process_btn = st.button(
        "🔄 Process Documents", 
        type="primary",
        disabled=not uploaded_files,
        help="Process uploaded PDFs and create searchable database"
    )

with col2:
    if os.path.exists(persist_directory) or os.path.exists("docs"):
        clear_btn = st.button(
            "🗑️ Clear All Data",
            help="Remove all uploaded files and database"
        )
        if clear_btn:
            clear_database()
            st.rerun()

with col3:
    # Show current status
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} file(s) ready")
    else:
        st.info("📤 Upload PDFs")

# Process documents when button is clicked
if process_btn and uploaded_files:
    with st.spinner("🤖 Processing documents... This may take a while depending on the number and size of PDFs."):
        success = process_uploaded_documents(uploaded_files)
        if success:
            # Clear the cache to reload the QA system with new documents
            st.cache_resource.clear()

# Check if database exists and show status
if os.path.exists(persist_directory):
    st.success("✅ Document database is ready! You can now ask questions.")
    
    # Question answering section
    st.subheader("💬 Ask Questions")
    
    # Create two columns for the question area
    question_col, settings_col = st.columns([3, 1])
    
    with question_col:
        question = st.text_area(
            "Enter your question about the documents:", 
            height=100,
            placeholder="Example: What are the main topics discussed in these documents?"
        )
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("🤖 Searching for answers..."):
                st.info("**📝 Your Question:** " + question)
                
                answer, metadata = process_answer(question)
                
                # Display answer
                st.subheader("🎯 Answer")
                st.markdown(answer)
                
                # Display source documents if available
                if 'source_documents' in metadata and metadata['source_documents']:
                    with st.expander("📄 Source Documents", expanded=False):
                        for i, doc in enumerate(metadata['source_documents']):
                            st.markdown(f"**📑 Source {i+1}:**")
                            
                            # Show document content (first 500 characters)
                            content_preview = doc.page_content[:500]
                            if len(doc.page_content) > 500:
                                content_preview += "..."
                            
                            st.markdown(f"``````")
                            
                            # Show metadata if available
                            if hasattr(doc, 'metadata') and doc.metadata:
                                source_file = doc.metadata.get('source', 'Unknown')
                                st.caption(f"📁 *Source: {os.path.basename(source_file)}*")
                            
                            if i < len(metadata['source_documents']) - 1:
                                st.divider()
                else:
                    st.info("ℹ️ No source documents found for this query.")
        else:
            st.warning("⚠️ Please enter a question to search.")
            
else:
    if uploaded_files:
        st.info("📋 Files uploaded! Click **'Process Documents'** to create the searchable database.")
    else:
        st.warning("📤 Please upload PDF files first to get started.")

# Sidebar with system information
with st.sidebar:
    st.header("🛠️ System Status")
    
    # Show API key status
    if openai.api_key:
        st.success("✅ OpenAI API: Connected")
    else:
        st.error("❌ OpenAI API: Not Connected")
    
    # Show database status
    if os.path.exists(persist_directory):
        st.success("✅ Vector Database: Ready")
        
        # Show number of documents if possible
        if os.path.exists("docs"):
            doc_count = len([f for f in os.listdir("docs") if f.endswith('.pdf')])
            if doc_count > 0:
                st.info(f"📄 Documents: {doc_count} PDF(s) processed")
    else:
        st.warning("⚠️ Vector Database: Not Created")
    
    st.divider()
    
    st.header("💡 Tips for Better Results")
    st.markdown("""
    **Asking Questions:**
    - Be specific and detailed
    - Reference key terms from your documents  
    - Try rephrasing if needed
    - Ask about specific topics or concepts
    
    **Document Tips:**
    - Ensure PDFs contain selectable text
    - Avoid scanned images without OCR
    - Upload related documents together
    - Larger documents take longer to process
    """)
    
    st.divider()
    
    st.header("🔧 Technical Details")
    st.markdown("""
    **Powered by:**
    - 🤖 OpenAI GPT-3.5-turbo
    - 🦜 LangChain Framework  
    - 🔍 ChromaDB Vector Search
    - 📱 Streamlit Interface
    - 🐍 Python Backend
    """)
    
    # Show upload limits
    st.info("""
    **Upload Limits:**
    - File types: PDF only
    - Multiple files: Yes
    - Max file size: 200MB per file
    - Processing time: ~1-2 min per MB
    """)
