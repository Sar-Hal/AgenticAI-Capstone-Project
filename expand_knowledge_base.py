import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Ensure API key is present
if not os.environ.get('GOOGLE_API_KEY'):
    print("Error: GOOGLE_API_KEY not found in environment.")
    exit(1)

# Initialize Gemma 3 27B
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0.2)

kb_dir = "knowledge_base"

# Prompt template
system_prompt = """You are an expert AI instructor writing course material for the 'Agentic AI Hands-On Course'. 
Your task is to expand the provided brief summary into a comprehensive, highly detailed, and technical explanation suitable for a RAG knowledge base.

Guidelines:
- Expand the content to be around 300-400 words.
- Provide clear definitions, technical details, examples, and best practices.
- Maintain a professional, educational tone.
- DO NOT include conversational filler like "Here is the expanded text..." or "In conclusion...". Start directly with the content.
- Keep the focus strictly on the provided topic. Do not hallucinate outside information that contradicts the original summary.

Topic: {topic}
Original Summary: {summary}
"""

def expand_document(filepath, meta_path):
    # Load topic
    topic = "General"
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as mf:
            meta = json.load(mf)
            topic = meta.get('topic', 'General')
            
    # Load original text
    with open(filepath, 'r', encoding='utf-8') as f:
        original_text = f.read().strip()
        
    print(f"Expanding [{topic}]...")
    
    prompt = system_prompt.format(topic=topic, summary=original_text)
    
    # Rate limit handling for Gemma 3 (30 req/min)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            expanded_text = response.content.strip()
            
            # Save expanded text
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(expanded_text)
            
            print(f"  -> Success! Expanded to {len(expanded_text.split())} words.")
            time.sleep(3) # Respect rate limits
            break
        except Exception as e:
            print(f"  -> Error on attempt {attempt+1}: {e}")
            time.sleep(5)
            if attempt == max_retries - 1:
                print(f"  -> Failed to expand {filepath}")

def main():
    print("Starting Knowledge Base Expansion using Gemma 3 27B...")
    
    if not os.path.exists(kb_dir):
        print(f"Directory {kb_dir} not found.")
        return
        
    files = sorted([f for f in os.listdir(kb_dir) if f.endswith('.txt')])
    
    for fname in files:
        txt_path = os.path.join(kb_dir, fname)
        doc_id = fname.replace('.txt', '')
        meta_path = os.path.join(kb_dir, f"{doc_id}_meta.json")
        
        expand_document(txt_path, meta_path)
        
    print("\nExpansion complete! To apply these changes, the ChromaDB vectorstore must be rebuilt.")
    print("You can rebuild it by deleting the 'chroma_db' folder and letting the app recreate it.")

if __name__ == "__main__":
    main()
