import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.embedder import get_embeddings
from app.retrieval.vectorstore import create_vectorstore
from app.retrieval.retriever import get_retriever
from app.llm.claude import get_llm
from app.chains.rag_chain import build_rag_chain

# Cache the data loading and processing
@st.cache_resource
def initialize_rag():
    # Load
    docs = load_pdf(r"D:\AI-ML\My Projects\GenAi-RAG\data\raw\SaiCharan_Pittala_Salesforce_Admin_Resume.pdf")
    
    # Split
    chunks = split_documents(docs)
    
    # Embed
    embeddings = get_embeddings()
    
    # Store
    vectorstore = create_vectorstore(chunks, embeddings)
    
    # Retrieve
    retriever = get_retriever(vectorstore)
    
    # LLM
    llm = get_llm()
    
    # Chain
    rag = build_rag_chain(llm, retriever)
    
    return rag

# Page config
st.set_page_config(page_title="RAG Resume Q&A", page_icon="🤖")

st.title("🤖 Sai Charan's RAG")
st.markdown("Ask questions about the resume using AI-powered retrieval!")

# Initialize RAG
try:
    rag = initialize_rag()
    st.success("✅ RAG system initialized successfully!")
except Exception as e:
    st.error(f"❌ Error initializing RAG: {e}")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about the resume..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.invoke(prompt)
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})