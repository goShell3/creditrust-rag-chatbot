import os
import streamlit as st
from src.rag_pipeline import answer_question


st.set_page_config(page_title="CrediTrust RAG Assistant", layout="centered")
st.title("📊 CrediTrust Customer Complaint Assistant")
st.write("Ask questions about customer complaints. Powered by Retrieval-Augmented Generation.")

# Session state to hold conversation
if "history" not in st.session_state:
    st.session_state.history = []

# Input area
with st.form("qa_form"):
    question = st.text_input("Your question:", placeholder="e.g., What complaints are common for money transfers?")
    submitted = st.form_submit_button("Ask")
    clear = st.form_submit_button("Clear")

# Clear history
if clear:
    st.session_state.history = []
    st.experimental_rerun()

# Handle submission
if submitted and question:
    with st.spinner("Retrieving and generating answer..."):
        result = answer_question(question)
        st.session_state.history.append({
            "question": result["question"],
            "answer": result["response"],
            "sources": result["sources"]
        })

# Display history
for qa in reversed(st.session_state.history):
    st.markdown("### ❓ Question")
    st.markdown(f"> {qa['question']}")
    st.markdown("### 🤖 Answer")
    st.markdown(qa['answer'])

    with st.expander("🔍 View Source Chunks"):
        for i, src in enumerate(qa['sources']):
            st.markdown(f"**Chunk {i+1}:**\n{src}")

# Optional Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + LangChain + ChromaDB + Hugging Face")
