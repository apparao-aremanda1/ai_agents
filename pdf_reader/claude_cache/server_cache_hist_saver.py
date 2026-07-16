'''
This code also sends the pdf file into claude cache.
Also this code sends the previous questions also. Some times user sends the followup on previous questions like "What happened after that?".
Actually adding upto 200 previous questions will not add much costly. If it crosses then it can be expensive.

'''

import os
import time
import fitz  # PyMuPDF
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import anthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # <--- Import Checkpointer


# --- 1. DEFINE THE STATE ---
class GraphState(TypedDict):
    question: str
    chat_history: List[
        Dict[str, str]]  # <--- Stores [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    answer: str
    fallback_needed: bool
    execution_time: float


# --- 2. DEFINE THE NODES ---
def check_or_create_cache_node(state: GraphState):
    """Checks if the text cache exists. If not, creates it from the PDF."""
    print("\n[NODE: System Init] Checking for local text cache...")
    start_time = time.time()

    cache_filename = "../book_text_cache.txt"
    pdf_filename = "../sample.pdf"

    if os.path.exists(cache_filename):
        print(f"   -> Cache file '{cache_filename}' already exists. Proceeding.")
    else:
        print(f"   -> Cache not found. Extracting text from '{pdf_filename}'...")
        if not os.path.exists(pdf_filename):
            raise FileNotFoundError(f"ERROR: Cannot find '{pdf_filename}'. Please check the file path.")

        pdf_document = fitz.open(pdf_filename)
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            full_text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
            full_text += page.get_text("text")

        with open(cache_filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        pdf_document.close()
        print(f"   -> Extraction complete! Saved to '{cache_filename}'.")

    execution_time = round(time.time() - start_time, 2)
    return {"execution_time": state.get("execution_time", 0.0) + execution_time}


def query_cached_book_node(state: GraphState):
    print(f"\n[NODE: Cache Router] Analyzing question with history: '{state['question']}'")
    start_time = time.time()

    with open("../book_text_cache.txt", "r", encoding="utf-8") as f:
        book_text = f.read()

    # Get existing history or initialize it
    history = state.get("chat_history", [])

    # Format the past history so Claude understands the continuity
    formatted_history = ""
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {turn['content']}\n"

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": f"You are an expert assistant. Read the following book:\n{book_text}",
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"""Here is our conversation history so far:
                {formatted_history}

                Answer the following follow-up question based ONLY on the provided book text.
                If the question relies on previous turns, use the history context to understand it.
                If the answer is NOT explicitly in the book, reply with the exact word: FALLBACK.

                Question: {state['question']}"""
            }
        ]
    )

    raw_answer = response.content[0].text.strip()
    execution_time = round(time.time() - start_time, 2)
    total_time = state.get("execution_time", 0.0) + execution_time

    if raw_answer == "FALLBACK":
        print(f"   -> [ROUTING] Answer not in book. Triggering web failsafe.")
        return {"fallback_needed": True, "execution_time": total_time}
    else:
        print(f"   -> [ROUTING] Answer found in book cache!")
        # Update history with this successful turn
        updated_history = history + [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": raw_answer}
        ]
        return {"answer": raw_answer, "fallback_needed": False, "execution_time": total_time,
                "chat_history": updated_history}


def web_search_node(state: GraphState):
    print("\n[NODE: Web Search] Scraping the live internet...")
    start_time = time.time()

    search_tool = DuckDuckGoSearchRun()
    web_results = search_tool.invoke(state['question'])

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Answer the user's question using ONLY these web search results.\n\nResults: {web_results}\n\nQuestion: {state['question']}"
            }
        ]
    )

    raw_answer = response.content[0].text
    execution_time = round(time.time() - start_time, 2)
    total_time = state.get("execution_time", 0.0) + execution_time

    # Update history even if it came from the web
    history = state.get("chat_history", [])
    updated_history = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": raw_answer}
    ]

    return {"answer": raw_answer, "execution_time": total_time, "chat_history": updated_history}


def should_we_fallback(state: GraphState):
    if state["fallback_needed"]:
        return "search_web"
    return "end_process"


# --- 3. BUILD THE GRAPH WITH MEMORY ---
def build_agent():
    load_dotenv()
    workflow = StateGraph(GraphState)

    workflow.add_node("Init_Cache", check_or_create_cache_node)
    workflow.add_node("Query_Book", query_cached_book_node)
    workflow.add_node("Web_Search", web_search_node)

    workflow.set_entry_point("Init_Cache")
    workflow.add_edge("Init_Cache", "Query_Book")

    workflow.add_conditional_edges(
        "Query_Book",
        should_we_fallback,
        {
            "search_web": "Web_Search",
            "end_process": END
        }
    )
    workflow.add_edge("Web_Search", END)

    # Create an in-memory checkpointer
    memory = MemorySaver()

    # Compile the graph while attaching the memory checkpointer
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    app = build_agent()

    # Define a thread configuration. This ID acts as the session cookie for this specific user chat.
    config = {"configurable": {"thread_id": "session_user_123"}}

    # --- TURN 1 ---
    print("=== TURN 1 ===")
    state_turn_1 = {"question": "Who is the protagonist of the book, and what is his main goal?"}
    result_1 = app.invoke(state_turn_1, config=config)  # Pass the config here
    print(f"Answer: {result_1['answer']}")

    print("-" * 50)

    # --- TURN 2 (Context Dependent) ---
    print("=== TURN 2 ===")
    # Notice this question relies entirely on knowing who "he" is from Turn 1!
    state_turn_2 = {"question": "What he did to pursue his goal?"}
    result_2 = app.invoke(state_turn_2, config=config)  # Same exact config thread_id
    print(f"Answer: {result_2['answer']}")

    print("=== TURN 3 ===")
    # Notice this question relies entirely on knowing who "he" is from Turn 1!
    state_turn_3 = {"question": "What he did next?"}
    result_3 = app.invoke(state_turn_3, config=config)  # Same exact config thread_id
    print(f"Answer: {result_3['answer']}")

    print("=== TURN 4 ===")
    # Notice this question relies entirely on knowing who "he" is from Turn 1!
    state_turn_4 = {"question": "Why the nifty is raising today?"}
    result_4 = app.invoke(state_turn_4, config=config)  # Same exact config thread_id
    print(f"Answer: {result_4['answer']}")
