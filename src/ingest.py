import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"

def build_vector_store():
    # Defensive check: Ensure data directory exists and has files
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("❌ Error: Drop at least one PDF file inside the 'data/' folder first!")
        return None

    all_docs = []
    print("⏳ Scanning folder for documents...")
    
    # Process all PDFs inside the data folder
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, file)
            print(f"📖 Processing: {file}")
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())

    if not all_docs:
        print("❌ No readable text files or PDFs extracted.")
        return None

    # Slice text chunks to fit inside context window constraints comfortably
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"✂️ Sliced documents into {len(chunks)} structural chunks.")

    print("🚀 Initializing Vector Database via Local Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Generate and build the Chroma database locally
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        client_settings=Settings(anonymized_telemetry=False)
    )
    print(f"✅ Success! Local vector database fully built inside '{CHROMA_DIR}'")
    return vector_db

if __name__ == "__main__":
    build_vector_store()