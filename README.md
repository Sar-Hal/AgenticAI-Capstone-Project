# Agentic AI Course Assistant — Capstone Project

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
├── day13_capstone.py     # Complete capstone execution script
├── test_agent.py         # 12 test questions + memory test
├── eval_ragas.py         # RAGAS baseline evaluation (5 Q/A pairs)
├── generate_chunks.py    # Knowledge base generator
├── knowledge_base/       # 12 domain documents with metadata
├── requirements.txt      # Dependencies
└── Context/              # Course reference documents
```

## Setup & Run Instructions

Follow these steps to get the Agentic AI Course Assistant running locally.

### 1. Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sar-Hal/AgenticAI-Capstone-Project.git
   cd AgenticAI-Capstone-Project
   ```
2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Keys:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

### 2. Knowledge Base Generation
The agent requires a vector database to perform Retrieval-Augmented Generation (RAG).

1. **Generate basic chunks:**
   ```bash
   python generate_chunks.py
   ```
   *(This creates 12 domain documents in the `knowledge_base/` directory)*
2. **(Optional) Expand Knowledge Base to simulate real cases:**
   To generate more comprehensive, detailed 300-500 word context chunks using Gemma 3:
   ```bash
   python expand_knowledge_base.py
   ```

### 3. Running the Streamlit App
To launch the interactive Course Assistant UI:
```bash
streamlit run capstone_streamlit.py
```
This will automatically initialize the `chroma_db` vectorstore and open the application in your default web browser at `http://localhost:8501`. Running it for the first time will be a bit slow as it downloads all-MiniLM-L6-v2 for the first time. 

### 4. Running the Evaluation Suite
To verify the core logic, memory, and red-teaming defenses:
```bash
python auto_test_capstone.py
```
*(This script runs the full testing suite, including 10 domain questions, 2 red-teaming tests, a multi-turn memory test, and the RAGAS baseline evaluation).*

## Safety & Red-Teaming

The agent defends against 5 adversarial categories:
- **Out-of-scope**: Admits ignorance + provides helpline (+91 40 1234 5678)
- **False premise**: Corrects assumptions without fabricating
- **Prompt injection**: Refuses to reveal system prompt
- **Hallucination bait**: Never invents information
- **Emotional queries**: Empathetic redirect to appropriate resources
