from state import CapstoneState
from tools import calculate_project_deadline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json

# Ensure you have set GOOGLE_API_KEY in your environment or .env
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# We will define a global accessor for vectorstore to be set by the app
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    return _vectorstore

def set_vectorstore(vs):
    global _vectorstore
    _vectorstore = vs

def memory_node(state: CapstoneState):
    """Slides window to last 6 messages."""
    msgs = state.get("messages", [])
    if len(msgs) > 6:
        msgs = msgs[-6:]
    return {"messages": msgs}

def router_node(state: CapstoneState):
    """Routes to retrieve, tool, or skip."""
    question = state["question"]
    prompt = f"""
    You are a routing assistant. Classify the user question into one of three categories:
    1. 'tool' - if the question asks about project deadlines or time remaining.
    2. 'retrieve' - if the question asks about course concepts, capstone guidelines, LangGraph, RAG, etc.
    3. 'skip' - if the question is a general greeting or unrelated chatter.
    
    Respond with ONLY the category word: 'tool', 'retrieve', or 'skip'.
    
    Question: {question}
    """
    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    if route not in ['tool', 'retrieve', 'skip']:
        route = 'skip'
    return {"route": route}

def retrieval_node(state: CapstoneState):
    """Retrieves top 3 chunks from ChromaDB."""
    vs = get_vectorstore()
    if not vs:
        return {"retrieved": "Error: Vectorstore not initialized.", "sources": []}
    
    docs = vs.similarity_search(state["question"], k=3)
    retrieved_text = "\n\n".join([d.page_content for d in docs])
    sources = [d.metadata.get("source", "unknown") for d in docs]
    
    return {"retrieved": retrieved_text, "sources": sources}

def skip_node(state: CapstoneState):
    """Crucial requirement: Explicitly clear state for non-RAG queries."""
    return {"retrieved": "", "sources": []}

def tool_node(state: CapstoneState):
    """Calls the project deadline calculator."""
    result = calculate_project_deadline.invoke({})
    return {"tool_result": result}

def answer_node(state: CapstoneState):
    """Strictly grounded system prompt for generating the answer."""
    context = state.get("retrieved", "")
    tool_res = state.get("tool_result", "")
    question = state["question"]
    
    # Red-teaming & Safety requirement
    system_prompt = f"""
    You are a Course Assistant. Answer ONLY using the provided context or tool result.
    If the context is empty, or if the user's query is out-of-scope/unrelated to the course,
    you MUST NOT hallucinate an answer. Instead, admit ignorance and provide the Helpline Number: +91 40 1234 5678.
    
    Context: {context}
    Tool Result: {tool_res}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

def eval_node(state: CapstoneState):
    """LLM-based faithfulness scoring."""
    answer = state.get("answer", "")
    context = state.get("retrieved", "") + " " + state.get("tool_result", "")
    
    if not context.strip():
        # If no context, we can't really judge faithfulness to context, but if the agent gave the helpline, it's faithful to instructions.
        if "+91 40 1234 5678" in answer:
            return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0) + 1}
        return {"faithfulness": 0.0, "eval_retries": state.get("eval_retries", 0) + 1}

    prompt = f"""
    Evaluate if the following Answer is faithful to the Context. 
    Score it 1.0 if it only contains information present in the Context.
    Score it 0.0 if it hallucinates information not present in the Context.
    Output ONLY the float number.
    
    Context: {context}
    Answer: {answer}
    """
    
    try:
        response = eval_llm.invoke(prompt)
        score = float(response.content.strip())
    except:
        score = 0.0
        
    retries = state.get("eval_retries", 0) + 1
    return {"faithfulness": score, "eval_retries": retries}

def save_node(state: CapstoneState):
    """Appends to history."""
    return {"messages": [AIMessage(content=state["answer"])]}
