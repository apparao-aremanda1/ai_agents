import os
import time
import fitz  # PyMuPDF
import logging
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import anthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma  # <-- Modern, clean import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver

# --- NEW POSTGRES IMPORTS ---
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from langgraph.graph import StateGraph, END


# (Include your other existing imports for FAISS, DuckDuckGo, etc. here)

class GraphState(TypedDict):
    question: str
    standalone_query: str
    chat_history: List[Dict[str, str]]
    answer: str
    fallback_needed: bool
    execution_time: float


EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DIR = "./chroma_linux_db"


def check_or_create_vector_db():
    if os.path.exists(CHROMA_DIR):
        logging.info("[System Setup] Local Chroma DB detected. Ready for querying.")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDING_MODEL)

    logging.info("[System Setup] Vector DB not found. Building Chroma database...")
    cache_filename = "book_text_cache.txt"
    pdf_filename = "sample.pdf"

    if not os.path.exists(cache_filename):
        logging.info(f"[System Setup] Cache not found. Extracting text from '{pdf_filename}'...")
        pdf_document = fitz.open(pdf_filename)
        full_text = ""
        for page_num in range(len(pdf_document)):
            full_text += pdf_document[page_num].get_text("text")
        with open(cache_filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        pdf_document.close()

    with open(cache_filename, "r", encoding="utf-8") as f:
        book_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    docs = text_splitter.create_documents([book_text])

    vector_db = Chroma.from_documents(docs, EMBEDDING_MODEL, persist_directory=CHROMA_DIR)
    logging.info(f"[System Setup] Successfully embedded and saved {len(docs)} text chunks to Chroma.")
    return vector_db


def contextualize_query_node(state: GraphState):
    logging.info(f"\n[NODE: Contextualizer] Rewriting query for database search...")
    start_time = time.time()

    history = state.get("chat_history", [])

    if not history:
        rewritten = f"{state['question']} in the book The Alchemist"
        logging.info(f"   -> Turn 1 Baseline Query: '{rewritten}'")
        return {"standalone_query": rewritten, "execution_time": state.get("execution_time", 0.0)}

    formatted_history = "".join(
        [f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}\n" for t in history])

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system="""You are an expert search query rewriter.
        Your task is to rewrite the latest user question into a highly specific, standalone search query using the Chat History.

        CRITICAL RULES:
        1. Replace vague pronouns ("he", "it") and general nouns ("the boy") with specific context from the history.
        2. Always append the phrase "in the book The Alchemist" to the very end of the query.
        3. Do NOT answer the question. Only output the rewritten search string.""",
        messages=[
            {
                "role": "user",
                "content": f"Chat History:\n{formatted_history}\n\nLatest Question: {state['question']}\n\nStandalone Question:"
            }
        ]
    )

    rewritten_query = response.content[0].text.strip()
    logging.info(f"   -> Rewrote '{state['question']}' to: '{rewritten_query}'")

    execution_time = round(time.time() - start_time, 2)
    return {"standalone_query": rewritten_query, "execution_time": state.get("execution_time", 0.0) + execution_time}


def query_vector_rag_node(state: GraphState):
    logging.info(f"\n[NODE: Vector RAG Router] Searching Chroma vector space...")
    start_time = time.time()

    vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDING_MODEL)
    relevant_docs = vector_db.similarity_search(state['standalone_query'], k=3)

    retrieved_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    logging.info(f"   -> Retrieved {len(relevant_docs)} context chunks")

    history = state.get("chat_history", [])
    formatted_history = "".join(
        [f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}\n" for t in history])

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""You are a specialized document assistant.

                Here is the verified context extracted from the source material:
                {retrieved_context}

                Here is our conversation history:
                {formatted_history}

                Answer the following question using ONLY the context block provided above.
                If the true answer cannot be verified directly from the provided context, reply with the single word: FALLBACK.

                Question: {state['question']}"""
            }
        ]
    )

    raw_answer = response.content[0].text.strip()
    execution_time = round(time.time() - start_time, 2)
    total_time = state.get("execution_time", 0.0) + execution_time

    if "FALLBACK" in raw_answer.upper()[:20]:
        logging.info("   -> [ROUTING] Context insufficient. Routing to Web Search.")
        return {"fallback_needed": True, "answer": raw_answer, "execution_time": total_time}
    else:
        logging.info("   -> [ROUTING] Query satisfied via local vector chunks.")
        updated_history = history + [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": raw_answer}
        ]
        return {"answer": raw_answer, "fallback_needed": False, "execution_time": total_time,
                "chat_history": updated_history}


def web_search_node(state: GraphState):
    logging.info("\n[NODE: Web Search] Scraping internet failsafe...")
    start_time = time.time()

    search_tool = DuckDuckGoSearchRun()
    web_results = search_tool.invoke(state['standalone_query'])

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Answer using ONLY these web results.\n\nResults: {web_results}\n\nQuestion: {state['question']}"
            }
        ]
    )

    raw_answer = response.content[0].text
    execution_time = round(time.time() - start_time, 2)
    total_time = state.get("execution_time", 0.0) + execution_time

    history = state.get("chat_history", [])
    updated_history = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": raw_answer}
    ]
    return {"answer": raw_answer, "execution_time": total_time, "chat_history": updated_history}


def build_workflow():
    """Builds the graph structure, but does NOT compile it yet."""
    load_dotenv()
    workflow = StateGraph(GraphState)

    workflow.add_node("Contextualize", contextualize_query_node)
    workflow.add_node("Query_Vector_RAG", query_vector_rag_node)
    workflow.add_node("Web_Search", web_search_node)

    workflow.set_entry_point("Contextualize")
    workflow.add_edge("Contextualize", "Query_Vector_RAG")

    workflow.add_conditional_edges(
        "Query_Vector_RAG",
        should_we_fallback,
        {"search_web": "Web_Search", "end_process": END}
    )
    workflow.add_edge("Web_Search", END)

    return workflow


def should_we_fallback(state: GraphState):
    if state["fallback_needed"]:
        return "search_web"
    return "end_process"


def get_chat_history_from_pg(app, session_id: str):
    """Pulls the chat history from Postgres without running the graph."""
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = app.get_state(config)

    if state_snapshot and state_snapshot.values:
        return state_snapshot.values.get("chat_history", [])

    return []


if __name__ == "__main__":
    # 1. Define your Postgres Database URI
    # Format: postgresql://username:password@hostname:port/database_name
    DB_URI = "postgresql://postgres:mysecretpassword@localhost:5432/langgraph_db"

    # 2. Build the uncompiled graph
    workflow = build_workflow()

    # 3. Create a Connection Pool to manage Postgres connections efficiently
    with ConnectionPool(
            conninfo=DB_URI,
            max_size=20,  # Handles up to 20 concurrent thread writes
            kwargs={"autocommit": True}
    ) as pool:
        # 4. Initialize the Postgres Checkpointer
        checkpointer = PostgresSaver(pool)

        # 5. Automatically create the LangGraph state tables if they don't exist
        checkpointer.setup()

        # 6. Compile the graph with the Postgres checkpointer
        app = workflow.compile(checkpointer=checkpointer)

        # --- EXECUTE YOUR AGENT ---
        config = {"configurable": {"thread_id": "pg_session_001"}}

        print("\n=== RUNNING TURN 1 ===")
        state_1 = {"question": "How did he get his money back from the crystal shop?"}
        final_state = app.invoke(state_1, config=config)
        print(f"\nFinal Answer: {final_state.get('answer')}")

        print("\n=== RUNNING TURN 2 ===")
        state_2 = {"question": "What is the boy name?"}
        final_state_2 = app.invoke(state_2, config=config)
        print(f"\nFinal Answer: {final_state_2.get('answer')}")
