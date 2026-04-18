# Agentic AI Course Assistant — Capstone Project

**Course**: Agentic AI Hands-On Course 2026  
**Instructor**: Dr. Kanthi Kiran Sirra | Sr. AI Engineer  
**Deadline**: April 21, 2026 | 11:59 PM

---

## Overview

A production-grade Course Assistant built using **LangGraph**, **ChromaDB**, and **Streamlit**. The agent answers questions about the Agentic AI course using a RAG pipeline, calculates project deadlines, maintains multi-turn conversation memory, and includes red-teaming safety defenses.

## Architecture

```
User question → [memory_node] → [router_node] → [retrieval/tool/skip_node]
            → [answer_node] → [eval_node] → (retry?) → [save_node] → END
```

### 8-Node LangGraph Pipeline
| Node | Purpose |
|------|---------|
| `memory_node` | Sliding window (last 6 messages) + name extraction |
| `router_node` | LLM-based routing: retrieve / tool / skip |
| `retrieval_node` | ChromaDB top-3 similarity search with [Topic] labels |
| `skip_node` | Clears stale context for non-RAG queries |
| `tool_node` | Project Deadline Calculator (April 21, 11:59 PM) |
| `answer_node` | Grounded generation with safety prompt + helpline fallback |
| `eval_node` | LLM-based faithfulness scoring (threshold 0.7) |
| `save_node` | Appends response to conversation history |

## Tech Stack

- **LLM**: Gemma 3 27B (via Google AI)
- **Embeddings**: SentenceTransformer `all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB (persistent)
- **Framework**: LangGraph with MemorySaver
- **UI**: Streamlit with `@st.cache_resource`

## Project Structure

```
├── state.py              # CapstoneState TypedDict
├── tools.py              # Deadline Calculator tool
├── nodes.py              # All 8 node functions
├── graph.py              # StateGraph assembly & compilation
├── agent.py              # ask() helper + vectorstore init
├── capstone_streamlit.py # Streamlit UI with sidebar
├── day13_capstone.ipynb  # Complete capstone notebook
├── test_agent.py         # 12 test questions + memory test
├── eval_ragas.py         # RAGAS baseline evaluation (5 Q/A pairs)
├── generate_chunks.py    # Knowledge base generator
├── knowledge_base/       # 12 domain documents with metadata
├── requirements.txt      # Dependencies
└── Context/              # Course reference documents
```

## Setup & Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` with your Google API key: `GOOGLE_API_KEY=your_key_here`
4. Generate knowledge base: `python generate_chunks.py`
5. Run the app: `streamlit run capstone_streamlit.py`

## Safety & Red-Teaming

The agent defends against 5 adversarial categories:
- **Out-of-scope**: Admits ignorance + provides helpline (+91 40 1234 5678)
- **False premise**: Corrects assumptions without fabricating
- **Prompt injection**: Refuses to reveal system prompt
- **Hallucination bait**: Never invents information
- **Emotional queries**: Empathetic redirect to appropriate resources
