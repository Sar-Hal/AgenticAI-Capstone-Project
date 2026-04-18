import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from graph import graph
from nodes import set_vectorstore

def init_vectorstore():
    """Initializes ChromaDB vectorstore from knowledge_base directory."""
    persist_dir = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # If the vectorstore already exists, load it
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        # Otherwise create from knowledge_base directory
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        if not os.path.exists("knowledge_base"):
            os.makedirs("knowledge_base", exist_ok=True)
            
        loader = DirectoryLoader("knowledge_base", glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()
        
        if not docs:
            # Add a dummy document to initialize schema if empty
            from langchain_core.documents import Document
            docs = [Document(page_content="Empty Knowledge Base", metadata={"source": "dummy"})]
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        vs = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=persist_dir
        )
        
    set_vectorstore(vs)
    return vs

def ask(query: str, thread_id: str = "default_thread"):
    """Helper function to run the LangGraph agent."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize state if not present (handled automatically by MemorySaver)
    initial_state = {
        "question": query,
        "eval_retries": 0, # Reset retries for new turn
    }
    
    # We yield output from the graph for streaming/displaying progress
    for output in graph.stream(initial_state, config=config):
        # We can yield or print the intermediate steps if needed
        pass
    
    # Get final state
    state = graph.get_state(config)
    final_state = state.values
    return final_state["answer"]
