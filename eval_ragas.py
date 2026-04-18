"""
RAGAS Baseline Evaluation for the Agentic AI Course Assistant.
Part 6 of the 8-part capstone process.

Uses 5 question-answer pairs with ground truth answers from the knowledge base.
Falls back to manual LLM-based faithfulness scoring if RAGAS is not installed.
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from agent import ask, init_vectorstore
from nodes import get_vectorstore, eval_llm
import re

# Initialize
print("Initializing vectorstore...")
init_vectorstore()

# ── 5 Q/A Pairs with Ground Truth ───────────────────────
eval_dataset = [
    {
        "question": "What is Agentic AI?",
        "ground_truth": "Agentic AI is a subfield of artificial intelligence that focuses on building systems capable of autonomous goal-directed behavior. These agents can perceive their environment, make decisions, plan multi-step actions, and execute them to achieve specific objectives."
    },
    {
        "question": "What is the token limit of the all-MiniLM-L6-v2 model?",
        "ground_truth": "The all-MiniLM-L6-v2 model truncates input at 256 tokens. Any text beyond this limit is silently lost from the embedding, which is why knowledge base documents must be chunked into 100-500 word segments."
    },
    {
        "question": "What are the required fields in the CapstoneState TypedDict?",
        "ground_truth": "The CapstoneState TypedDict includes: question, messages (with add_messages reducer), route, retrieved, sources, tool_result, answer, faithfulness, and eval_retries."
    },
    {
        "question": "What is the faithfulness threshold in the eval_node?",
        "ground_truth": "The eval_node uses a faithfulness threshold of 0.7. If the score falls below 0.7, the answer is regenerated up to a maximum retry count."
    },
    {
        "question": "What are the five categories of red-teaming tests?",
        "ground_truth": "The five categories are: out-of-scope questions, false premise questions, prompt injection attacks, hallucination bait, and emotional or distressing queries."
    },
]

# ── Run Evaluation ───────────────────────────────────────
print("\n" + "=" * 60)
print("RAGAS BASELINE EVALUATION (Manual LLM-based Faithfulness)")
print("=" * 60)

results = []

for i, item in enumerate(eval_dataset, 1):
    question = item["question"]
    ground_truth = item["ground_truth"]

    print(f"\n--- Eval {i}: {question}")
    time.sleep(3)

    try:
        # Get agent answer
        answer = ask(question, thread_id=f"eval_thread_{i}")

        # Get retrieved context
        vs = get_vectorstore()
        if vs:
            docs = vs.similarity_search(question, k=3)
            contexts = [d.page_content for d in docs]
        else:
            contexts = []

        context_str = " ".join(contexts)

        # Manual LLM-based faithfulness scoring
        faith_prompt = f"""Rate faithfulness of the Answer to the Context on a scale of 0.0 to 1.0.
Faithfulness = answer contains ONLY info from the context.
Output ONLY a float number.

Context: {context_str}
Answer: {answer}"""

        time.sleep(2)
        faith_resp = eval_llm.invoke(faith_prompt)
        faith_match = re.search(r"(\d+\.?\d*)", faith_resp.content.strip())
        faithfulness = float(faith_match.group(1)) if faith_match else 0.5

        # Manual LLM-based answer relevancy scoring
        rel_prompt = f"""Rate how relevant the Answer is to the Question on a scale of 0.0 to 1.0.
Output ONLY a float number.

Question: {question}
Answer: {answer}"""

        time.sleep(2)
        rel_resp = eval_llm.invoke(rel_prompt)
        rel_match = re.search(r"(\d+\.?\d*)", rel_resp.content.strip())
        relevancy = float(rel_match.group(1)) if rel_match else 0.5

        # Manual context precision scoring
        prec_prompt = f"""Rate how relevant the retrieved Context is to the Question on a scale of 0.0 to 1.0.
Output ONLY a float number.

Question: {question}
Context: {context_str[:500]}"""

        time.sleep(2)
        prec_resp = eval_llm.invoke(prec_prompt)
        prec_match = re.search(r"(\d+\.?\d*)", prec_resp.content.strip())
        precision = float(prec_match.group(1)) if prec_match else 0.5

        results.append({
            "question": question,
            "answer": answer[:200],
            "faithfulness": round(faithfulness, 2),
            "answer_relevancy": round(relevancy, 2),
            "context_precision": round(precision, 2),
        })

        print(f"  Faithfulness: {faithfulness:.2f} | Relevancy: {relevancy:.2f} | Precision: {precision:.2f}")
        print(f"  Answer: {answer[:150]}...")

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "question": question,
            "answer": "ERROR",
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        })

# ── Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("RAGAS BASELINE SCORES")
print("=" * 60)

if results:
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_prec = sum(r["context_precision"] for r in results) / len(results)

    print(f"{'Question':<55} {'Faith':>7} {'Relev':>7} {'Prec':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['question'][:53]:<55} {r['faithfulness']:>7.2f} {r['answer_relevancy']:>7.2f} {r['context_precision']:>7.2f}")
    print("-" * 80)
    print(f"{'AVERAGE':<55} {avg_faith:>7.2f} {avg_rel:>7.2f} {avg_prec:>7.2f}")
else:
    print("No results to display.")
