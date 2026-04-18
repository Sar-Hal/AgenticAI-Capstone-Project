import streamlit as st
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Course Assistant", layout="wide")

# Ensure required API keys are available
if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Import after env vars are loaded so LLMs pick them up
from agent import ask, init_vectorstore

@st.cache_resource
def setup_resources():
    """Cached initialization of ChromaDB and any heavy resources."""
    vs = init_vectorstore()
    return vs

# Initialize resources
setup_resources()

# Session State for thread_id and chat history UI
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
