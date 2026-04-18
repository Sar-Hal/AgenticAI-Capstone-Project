"""
Agent module: vectorstore initialization and ask() helper function.
Uses SentenceTransformer('all-MiniLM-L6-v2') for embeddings and ChromaDB for storage.
"""
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from graph import graph
from nodes import set_vectorstore, load_topic_map


def init_vectorstore():
    """
    Initializes ChromaDB vectorstore from knowledge_base/ directory.
    Loads .txt files, splits them, and stores with metadata including topic.
    Uses all-MiniLM-L6-v2 for embeddings (384-dimensional, 256-token limit).
    """
    persist_dir = "./chroma_db"
    kb_dir = "knowledge_base"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # If vectorstore already exists, just load it
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        set_vectorstore(vs)
        load_topic_map()
        return vs

    # Otherwise build from knowledge_base/
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)

    # Load each .txt file with its topic metadata from JSON sidecar
    all_docs = []
    for fname in sorted(os.listdir(kb_dir)):
        if not fname.endswith(".txt"):
            continue
        txt_path = os.path.join(kb_dir, fname)
        doc_id = fname.replace(".txt", "")
        meta_path = os.path.join(kb_dir, f"{doc_id}_meta.json")

        # Load topic from sidecar
        topic = "General"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                topic = meta.get("topic", "General")
            except Exception:
                pass

        # Load text
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue

        all_docs.append(
            Document(
                page_content=text,
                metadata={"source": fname, "topic": topic, "doc_id": doc_id},
            )
        )

    if not all_docs:
        all_docs = [
            Document(
                page_content="Empty Knowledge Base",
                metadata={"source": "dummy", "topic": "General", "doc_id": "dummy"},
            )
        ]

    # Split documents (100-500 words target, staying under 256-token embedding limit)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)

    vs = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    set_vectorstore(vs)
    load_topic_map()
    print(f"Vectorstore built with {len(splits)} chunks from {len(all_docs)} documents.")
    return vs


def ask(query: str, thread_id: str = "default_thread") -> str:
    """
    Helper function to run the LangGraph agent.
    Calls graph.stream() and returns the final answer.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": query,
        "eval_retries": 0,
    }

    # Stream through the graph
    for output in graph.stream(initial_state, config=config):
        pass

    # Get final state
    state = graph.get_state(config)
    final_state = state.values
    return final_state.get("answer", "I'm sorry, I couldn't generate a response.")
