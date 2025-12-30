import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from rag_engine import RAGDemo
import numpy as np

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="RAG Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e1e2e;
        font-weight: 700;
    }
    
    /* Cards */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 2rem;
    }

    /* Modern metric containers */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Custom divider */
    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid #e9ecef;
    }
    
    /* Success/Info Alerts */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key_env = os.getenv("GOOGLE_API_KEY", "")
    api_key = api_key_env
    # api_key = st.text_input("Google API Key", value=api_key_env, type="password")
    
    if not api_key:
        st.warning("⚠️ No Google API Key found in environment variables.")
        st.info("Please set the `GOOGLE_API_KEY` in your `.env` file.")
        st.stop()
    else:
        st.success("✅ API Key loaded from environment")
        
    st.divider()
    st.markdown("### About")
    st.info(
        "This app demonstrates the **RAG (Retrieval-Augmented Generation)** pipeline.\n\n"
        "1. **Ingest**: Text is split into chunks and embedded.\n"
        "2. **Store**: Vectors are stored in a FAISS index.\n"
        "3. **Retrieve**: Queries are embedded and matched against the index."
    )

# --- Session State ---
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGDemo(api_key=api_key)

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False

# --- Main Content ---
st.title("📚 RAG Pipeline Visualizer")
st.markdown("Interactive demonstration of how text is transformed into vectors and retrieved.")

# Create tabs for clear separation of concerns
tab_ingest, tab_retrieve = st.tabs(["📤 1. Ingestion Phase", "🔍 2. Retrieval Phase"])

# ========================
# INGESTION TAB
# ========================
with tab_ingest:
    st.header("Document Ingestion")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Upload Source Text")
        uploaded_file = st.file_uploader("Choose a text file", type="txt")
        text_data = ""
        if uploaded_file is not None:
            uploaded_file.seek(0)
            text_data = uploaded_file.read().decode("utf-8")
        elif os.path.exists("constitution.txt"):
            with open("constitution.txt", "r", encoding="utf-8") as f:
                text_data = f.read()
            st.info("ℹ️ Using default file: **constitution.txt**")

        st.text_area("File Preview", text_data[:500] + ("..." if len(text_data) > 500 else ""), height=150)
            
    with col2:
        st.markdown("#### Processing Parameters")
        chunk_size = st.slider(
            "Chunk Size (characters)", 
            min_value=50, 
            max_value=1000, 
            value=200, 
            step=50,
            help="Determines the length of each text segment."
        )
        
        if st.button("🚀 Process Document", use_container_width=True, type="primary"):
            if not text_data:
                st.error("Please upload a file first.")
            else:
                with st.spinner("Chunking, Embedding, and Indexing..."):
                    try:
                        # Re-initialize engine with current key
                        st.session_state.rag_engine = RAGDemo(api_key=api_key)
                        
                        # Debug info
                        # st.write(f"Debug: Text length: {len(text_data)}")
                        
                        count = st.session_state.rag_engine.ingest(text_data, chunk_size)
                        st.session_state.processed_docs = True
                        st.success(f"Successfully processed **{count} chunks**!")
                    except Exception as e:
                        import traceback
                        err_trace = traceback.format_exc()
                        st.error(f"Error during processing: {str(e)}")
                        with st.expander("Details"):
                            st.code(err_trace)
                            st.write(f"API Key present: {bool(api_key)}")


    # Visualization of what just happened
    if st.session_state.processed_docs and st.session_state.rag_engine.chunks:
        st.divider()
        st.subheader("🔍 Inside the Vector Database")
        
        # Display sample chunks
        st.markdown(f"**Total Chunks:** `{len(st.session_state.rag_engine.chunks)}` | **Embedding Dimensions:** `{st.session_state.rag_engine.dimension}`")
        
        with st.expander("View Generated Chunks & Embeddings (First 5)", expanded=True):
            for i in range(min(5, len(st.session_state.rag_engine.chunks))):
                cols = st.columns([3, 1])
                cols[0].code(st.session_state.rag_engine.chunks[i], language="text")
                # Show a tiny heatmap or stats for the embedding
                embedding_sample = st.session_state.rag_engine.embeddings[i][:10] # just first 10 dims
                cols[1].caption(f"Vector ID: {i}")
                cols[1].write(f"Vec: {np.round(embedding_sample, 3)}...")

# ========================
# RETRIEVAL TAB
# ========================
with tab_retrieve:
    st.header("Semantic Retrieval")
    
    if not st.session_state.processed_docs:
        st.warning("⚠️ You must ingest a document in the 'Ingestion Phase' tab first.")
    else:
        query = st.text_input("Enter your query:", placeholder="e.g., What is the main topic of the document?")
        
        if st.button("🔎 Search", type="primary"):
            if not query:
                st.warning("Please enter a query.")
            else:
                with st.spinner("Embedding query and scanning index..."):
                    try:
                        results, query_vec = st.session_state.rag_engine.search(query, top_k=3)
                        
                        # --- Visualization ---
                        st.divider()
                        
                        # Results Viz
                        st.subheader("Top Matches")
                        st.markdown("The system found these chunks with the **smallest Euclidean distance** (closest similarity).")
                        
                        # Display Results Loop already follows here...
                        
                        for idx, res in enumerate(results):
                            with st.container():
                                st.markdown(f"### Result #{idx+1}")
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    st.info(res['text'], icon="📄")
                                with c2:
                                    st.metric("Distance Score", f"{res['distance']:.4f}", delta_color="inverse")
                                st.progress(max(0.0, 1.0 - res['distance']), text="Similarity Confidence (Approx)")
                                
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")

