"""
Capstone Streamlit UI for the Agentic AI Course Assistant.
Uses @st.cache_resource for all expensive initializations.
Windows encoding handled with utf-8.
"""
import streamlit as st
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Agentic AI Course Assistant",
    page_icon="🎓",
    layout="wide",
)

# Ensure required API keys are available
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ Please set the GOOGLE_API_KEY environment variable in your .env file.")
    st.stop()

# Import after env vars are loaded so LLMs pick them up
from agent import ask, init_vectorstore


@st.cache_resource
def setup_resources():
    """Cached initialization of ChromaDB, embeddings, and LangGraph — runs only once."""
    vs = init_vectorstore()
    return vs


# Initialize resources
setup_resources()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Course Assistant")
    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "This is a **Course Assistant** for Dr. Kanthi Kiran Sirra's "
        "**Agentic AI Hands-On Course** (2026). It uses LangGraph, "
        "ChromaDB, and Streamlit to answer questions about the course."
    )

    st.markdown("### Topics Covered")
    st.markdown(
        """
        - Agentic AI Overview
        - LangGraph Framework
        - ChromaDB Vector Database
        - RAG Pipeline
        - StateGraph & TypedDict Design
        - Semantic Search (SentenceTransformers)
        - Capstone Project Requirements
        - Red-Teaming & AI Safety
        - Memory & Persistence
        - RAGAS Evaluation Metrics
        - Graph Node Architecture
        - Streamlit Deployment
        """
    )

    st.markdown("### Tool Available")
    st.markdown("🕐 **Project Deadline Calculator** — Ask about the deadline!")

    st.markdown("---")

    # New Conversation button
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with LangGraph + ChromaDB + Streamlit")
    st.caption("Model: Gemma 3 27B (Google AI)")

# ── Session State ────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main Chat Area ───────────────────────────────────────
st.title("🎓 Agentic AI Course Assistant")
st.markdown("Ask me questions about the course, project guidelines, LangGraph, or Capstone deadlines!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask(prompt, thread_id=st.session_state.thread_id)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
