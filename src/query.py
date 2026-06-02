from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from chromadb.config import Settings

CHROMA_DIR = "./chroma_db"

def execute_rag_query(user_query: str):
    # Initialize the local embedding model engine
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Load the local Chroma DB directory mapping
    db = Chroma(
        persist_directory=CHROMA_DIR, 
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False)
    )
    
    # Fetch top 4 most matching text slices from the documents
    retrieved_chunks = db.similarity_search(user_query, k=4)
    
    if not retrieved_chunks:
        return "No corresponding context found inside records.", []

    contexts = []
    citations = []
    
    # Process background content chunks and generate grounding citations
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        contexts.append(chunk.page_content)
        source_name = chunk.metadata.get("source", "Unknown Document")
        base_name = source_name.split("/")[-1].split("\\")[-1]
        page_num = chunk.metadata.get("page", 0) + 1
        citations.append(f"📄 **[{idx}]** {base_name} — **Page {page_num}**")

    compiled_context = "\n\n---\n\n".join(contexts)

    # Prompt forcing strict reliance on provided files to prevent hallucinations
    system_prompt = f"""You are a senior academic research assistant. 
Answer the user's research query using ONLY the verified source context chunks provided below. 
You must align your statements back to references via source identifiers like [1], [2], etc.
If the provided text elements do not contain information sufficient to form a clean answer, 
state explicitly that the background data doesn't mention it. Do not hypothesize.

CONTEXT HIGHLIGHTS:
{compiled_context}

QUESTION:
{user_query}

INTEGRATED SCIENTIFIC ANSWER:"""

    # Boot up local lightweight LLM brain
    llm = ChatOllama(model="llama3.2:1b", temperature=0.1)
    response = llm.invoke(system_prompt)
    
    return response.content, citations