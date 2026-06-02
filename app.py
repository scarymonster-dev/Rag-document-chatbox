import streamlit as st
import os
from src.ingest import build_vector_store
from src.query import execute_rag_query

# Configure Streamlit Browser Shell Metadata Parameters
st.set_page_config(page_title="InsightHub AI", page_icon="🔬", layout="wide")

# Inject Custom CSS Styling Overrides for Premium Dark UI Look
st.markdown("""
    <style>
        .main-header { font-size:2.6rem !important; font-weight: 800; color: #4F46E5; background: linear-gradient(45deg, #4F46E5, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .stat-card { background-color: #1E293B; border-radius: 12px; padding: 20px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .citation-box { background-color: #0F172A; padding: 12px; border-radius: 8px; border-left: 4px solid #06B6D4; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔬 InsightHub AI</h1>', unsafe_allow_html=True)
st.markdown("##### *Next-Gen Enterprise Knowledge Discovery & Advanced Local RAG Pipeline*")

# Top System Realtime Metrics Overview Layout
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-card">🟢 <b>System Core Status</b><br><span style="font-size:1.5rem;color:#10B981;">Local Ollama Operational</span></div>', unsafe_allow_html=True)
with col2:
    db_exists = os.path.exists("./chroma_db")
    db_status = "Connected" if db_exists else "Not Initialized"
    db_color = "#3B82F6" if db_exists else "#EF4444"
    st.markdown(f'<div class="stat-card">📦 <b>Vector Workspace Space</b><br><span style="font-size:1.5rem;color:{db_color};">{db_status}</span></div>', unsafe_allow_html=True)
with col3:
    pdf_count = len([f for f in os.listdir("./data") if f.endswith(".pdf")]) if os.path.exists("./data") else 0
    st.markdown(f'<div class="stat-card">📚 <b>Indexed Knowledge Library</b><br><span style="font-size:1.5rem;color:#F59E0B;">{pdf_count} Active PDF(s)</span></div>', unsafe_allow_html=True)

st.write("")
st.divider()

# Left Sidebar Controls Management Frame
with st.sidebar:
    st.markdown("### 🛠️ Control Center")
    st.write("Manage your background research assets locally.")
    
    if st.button("🔄 Sync & Rebuild Database", use_container_width=True, type="primary"):
        with st.spinner("Processing PDFs into Local Vector Space..."):
            db = build_vector_store()
            if db:
                st.success("Vector Space mapping updated!")
                st.rerun()
            else:
                st.error("Ingestion failed. Ensure your 'data/' directory contains PDF items.")

# Session Memory Initializations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Keep Chat Messages rendering inside current view session
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])
        if "citations" in chat and chat["citations"]:
            with st.expander("🔍 Verified Source Grounding & Citations"):
                for cite in chat["citations"]:
                    st.markdown(f'<div class="citation-box">{cite}</div>', unsafe_allow_html=True)

# User Chat Action Routing Pipeline
if user_input := st.chat_input("Ask an analytical question about your background files..."):
    
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents locally using Llama 3.2..."):
            if not os.path.exists("./chroma_db"):
                st.warning("⚠️ Database registry empty. Click 'Sync & Rebuild Database' in the sidebar panel to index your files first.")
            else:
                answer, citations = execute_rag_query(user_input)
                st.markdown(answer)
                
                if citations:
                    with st.expander("🔍 Verified Source Grounding & Citations"):
                        for cite in citations:
                            st.markdown(f'<div class="citation-box">{cite}</div>', unsafe_allow_html=True)
                            
                # Push elements to long term layout tracking list
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": answer,
                    "citations": citations
                })