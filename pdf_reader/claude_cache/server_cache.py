'''
This code sends the pdf file to Claude server and it will it in claude cache.
It will keep the cache for 5 mins. If there is no question in these 5 mins then it will remove it from cache.
The maximum tokans it can hold is 200k token around 560 pdf pages.
The minimum tokens it will accept is:
claude 3 haiku -  2048 tokens
claude 3.5 Sonet/Opus - 1024

The cost of questions with this approach is like this.
Method	        Final Cost	        Avg / Turn
● Standard	        $2.700	        $0.1800
● Caching	        $0.477	        $0.0318
● Vector RAG	    $0.090	        $0.0060
'''

import os
import time
from typing import TypedDict

import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END


# --- 1. DEFINE THE STATE ---
class GraphState(TypedDict):
    question: str
    answer: str
    fallback_needed: bool
    execution_time: float


# --- 2. DEFINE THE NODES ---

def check_or_create_cache_node(state: GraphState):
    """Checks if the text cache exists. If not, creates it from the PDF."""
    print("\n[NODE: System Init] Checking for local text cache...")
    start_time = time.time()

    # Define file paths (adjust these if your files are in a different folder)
    cache_filename = "../book_text_cache.txt"
    pdf_filename = "../sample.pdf"

    if os.path.exists(cache_filename):
        print(f"   -> Cache file '{cache_filename}' already exists. Proceeding.")
    else:
        print(f"   -> Cache not found. Extracting text from '{pdf_filename}'...")

        if not os.path.exists(pdf_filename):
            raise FileNotFoundError(f"ERROR: Cannot find '{pdf_filename}'. Please check the file path.")

        # Open PDF and extract text using PyMuPDF
        pdf_document = fitz.open(pdf_filename)
        full_text = ""

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            full_text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
            full_text += page.get_text("text")

        # Save to cache
        with open(cache_filename, "w", encoding="utf-8") as f:
            f.write(full_text)

        pdf_document.close()
        print(f"   -> Extraction complete! Saved to '{cache_filename}'.")

    execution_time = round(time.time() - start_time, 2)
    # Safely get current time (defaults to 0 if it's the first step)
    total_time = state.get("execution_time", 0.0) + execution_time

    return {"execution_time": total_time}


def query_cached_book_node(state: GraphState):
    print(f"\n[NODE: Cache Router] Checking server-side book cache for: '{state['question']}'")
    start_time = time.time()

    # Load the local text file
    with open("../book_text_cache.txt", "r", encoding="utf-8") as f:
        book_text = f.read()

    client = anthropic.Anthropic()

    # Using the valid, currently active Haiku model
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": f"You are an expert assistant. Read the following book:\n{book_text}",
                "cache_control": {"type": "ephemeral"}  # Locks text into Anthropic's RAM
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"""Answer the following question based ONLY on the provided book text. 
                If the answer is NOT explicitly in the book, reply with the exact word: FALLBACK.

                Question: {state['question']}"""
            }
        ]
    )

    raw_answer = response.content[0].text.strip()
    execution_time = round(time.time() - start_time, 2)

    if raw_answer == "FALLBACK":
        print(f"   -> [ROUTING] Answer not in book. Triggering web failsafe. (Took {execution_time}s)")
        return {"fallback_needed": True, "execution_time": state.get("execution_time", 0.0) + execution_time}
    else:
        print(f"   -> [ROUTING] Answer found in book cache! (Took {execution_time}s)")
        return {"answer": raw_answer, "fallback_needed": False,
                "execution_time": state.get("execution_time", 0.0) + execution_time}


def web_search_node(state: GraphState):
    print("\n[NODE: Web Search] Scraping the live internet...")
    start_time = time.time()

    search_tool = DuckDuckGoSearchRun()
    web_results = search_tool.invoke(state['question'])

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Answer the user's question using ONLY these web search results.\n\nResults: {web_results}\n\nQuestion: {state['question']}"
            }
        ]
    )

    execution_time = round(time.time() - start_time, 2)
    print(f"   -> [ROUTING] Web search complete. (Took {execution_time}s)")

    total_time = state.get("execution_time", 0.0) + execution_time
    return {"answer": response.content[0].text, "execution_time": total_time}


# --- 3. DEFINE THE ROUTING LOGIC ---
def should_we_fallback(state: GraphState):
    if state["fallback_needed"]:
        return "search_web"
    return "end_process"


# --- 4. BUILD THE GRAPH ---
def build_agent():
    load_dotenv()
    workflow = StateGraph(GraphState)

    # 1. Add all three nodes
    workflow.add_node("Init_Cache", check_or_create_cache_node)
    workflow.add_node("Query_Book", query_cached_book_node)
    workflow.add_node("Web_Search", web_search_node)

    # 2. Set Entry Point to the new Init node
    workflow.set_entry_point("Init_Cache")

    # 3. Once Init is done, go unconditionally to Query_Book
    workflow.add_edge("Init_Cache", "Query_Book")

    # 4. Conditional logic (Book -> Web OR End)
    workflow.add_conditional_edges(
        "Query_Book",
        should_we_fallback,
        {
            "search_web": "Web_Search",
            "end_process": END
        }
    )

    # 5. Web Search always ends the process
    workflow.add_edge("Web_Search", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_agent()

    # If you need to test the extraction, simply delete your existing 'book_text_cache.txt' file
    # before running this code!

    state = {"question": "How how the boy got the money again to reach pyramids"}

    print("=== STARTING SMART ROUTER ===")
    final_state = app.invoke(state)

    print("\n=== FINAL OUTPUT ===")
    print(final_state.get("answer", "No answer generated."))
    print(f"\n[Total System Execution Time: {round(final_state.get('execution_time', 0.0), 2)} seconds]")
