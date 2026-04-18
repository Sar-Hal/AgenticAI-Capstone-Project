import os

os.makedirs('knowledge_base', exist_ok=True)

chunks = [
    "Agentic AI is a subfield of artificial intelligence that focuses on building systems capable of autonomous goal-directed behavior. These agents can perceive their environment, make decisions, and take actions to achieve specific objectives.",
    "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
    "ChromaDB is the open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs. It stores embeddings and their metadata, allowing for efficient similarity search.",
    "Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps.",
    "RAG (Retrieval-Augmented Generation) is an AI framework that retrieves data from external knowledge bases to ground large language models (LLMs) on the most accurate, up-to-date information. It helps reduce hallucinations.",
    "A StateGraph in LangGraph is a graph where the nodes are functions that update a shared state, and the edges define the control flow. The state is typically represented as a TypedDict.",
    "Semantic search with SentenceTransformers uses models like 'all-MiniLM-L6-v2' to convert text into dense vector representations. These vectors can then be compared using cosine similarity to find relevant information.",
    "The Capstone Project for the Agentic AI course requires building a production-grade Course Assistant. It must include state-first design, an 8-node LangGraph architecture, ChromaDB for retrieval, and a Streamlit UI.",
    "Red-teaming in AI involves systematically testing models or agents to identify vulnerabilities, biases, or unexpected behaviors. For the Capstone, the agent must not hallucinate answers to out-of-scope questions and should provide a helpline number.",
    "Memory persistence in LangGraph allows agents to remember previous turns in a conversation. By using a MemorySaver with a thread_id, the graph state can be stored and retrieved across multiple interactions."
]

for i, chunk in enumerate(chunks, 1):
    with open(f'knowledge_base/chunk_{i}.txt', 'w', encoding='utf-8') as f:
        f.write(chunk)

print("Generated 10 chunks in knowledge_base/")
