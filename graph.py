"""
Graph assembly for the Agentic AI Course Assistant.
Builds the 8-node StateGraph with conditional edges and MemorySaver persistence.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import CapstoneState
from nodes import (
    memory_node, router_node, retrieval_node,
    skip_node, tool_node, answer_node, eval_node, save_node
)

# ── Routing Decision Functions (standalone for testability) ──

def route_decision(state: CapstoneState) -> str:
    """Reads state['route'] and returns the next node name."""
    return state["route"]


def eval_decision(state: CapstoneState) -> str:
    """
    Reads faithfulness and eval_retries to decide:
    - 'save_node' if faithfulness >= 0.7 (PASS)
    - 'answer_node' if retries < 3 (RETRY)
    - 'save_node' if max retries reached (GIVE UP)
    """
    if state["faithfulness"] >= 0.7:
        return "save_node"
    if state.get("eval_retries", 0) < 3:
        return "answer_node"
    return "save_node"


def build_graph():
    """Assembles the StateGraph with all 8 nodes and compiles with MemorySaver."""
    builder = StateGraph(CapstoneState)

    # ── Add all 8 nodes ──
    builder.add_node("memory_node", memory_node)
    builder.add_node("router_node", router_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("skip_node", skip_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("answer_node", answer_node)
    builder.add_node("eval_node", eval_node)
    builder.add_node("save_node", save_node)

    # ── Fixed edges ──
    builder.add_edge(START, "memory_node")
    builder.add_edge("memory_node", "router_node")
    builder.add_edge("retrieval_node", "answer_node")
    builder.add_edge("tool_node", "answer_node")
    builder.add_edge("skip_node", "answer_node")
    builder.add_edge("answer_node", "eval_node")
    builder.add_edge("save_node", END)

    # ── Conditional edge after router ──
    builder.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "retrieve": "retrieval_node",
            "tool": "tool_node",
            "skip": "skip_node",
        }
    )

    # ── Conditional edge after eval ──
    builder.add_conditional_edges(
        "eval_node",
        eval_decision,
        {
            "answer_node": "answer_node",
            "save_node": "save_node",
        }
    )

    # ── Compile with MemorySaver for thread-based persistence ──
    memory = MemorySaver()
    compiled_graph = builder.compile(checkpointer=memory)

    print("Graph compiled successfully")
    return compiled_graph


# Expose global compiled graph instance
graph = build_graph()
