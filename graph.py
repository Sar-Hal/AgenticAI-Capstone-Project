from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import CapstoneState
from nodes import (
    memory_node, router_node, retrieval_node, 
    skip_node, tool_node, answer_node, eval_node, save_node
)

def route_after_router(state: CapstoneState):
    return state["route"]

def route_after_eval(state: CapstoneState):
    if state["faithfulness"] >= 0.7:
        return "save_node"
    if state.get("eval_retries", 0) < 3:
        return "answer_node"
    return "save_node"

def build_graph():
    builder = StateGraph(CapstoneState)

    # Add Nodes
    builder.add_node("memory_node", memory_node)
    builder.add_node("router_node", router_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("skip_node", skip_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("answer_node", answer_node)
    builder.add_node("eval_node", eval_node)
    builder.add_node("save_node", save_node)

    # Edges
    builder.add_edge(START, "memory_node")
    builder.add_edge("memory_node", "router_node")

    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "retrieve": "retrieval_node",
            "tool": "tool_node",
            "skip": "skip_node",
        }
    )

    builder.add_edge("retrieval_node", "answer_node")
    builder.add_edge("tool_node", "answer_node")
    builder.add_edge("skip_node", "answer_node")

    builder.add_edge("answer_node", "eval_node")

    builder.add_conditional_edges(
        "eval_node",
        route_after_eval,
        {
            "answer_node": "answer_node",
            "save_node": "save_node"
        }
    )

    builder.add_edge("save_node", END)

    # Persistence
    memory = MemorySaver()
    compiled_graph = builder.compile(checkpointer=memory)
    
    return compiled_graph

# Expose global instance
graph = build_graph()
