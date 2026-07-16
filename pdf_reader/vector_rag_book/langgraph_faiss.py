import os
import time
import fitz  # PyMuPDF
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import anthropic
import logging

# --- 1. SILENCE THIRD-PARTY NOISE ---
logging.basicConfig(
    level=logging.INFO, filename='output.log',
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

# Muzzle the noisy background libraries
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# --- 2. DEFINE THE STATE ---
class GraphState(TypedDict):
    question: str
    standalone_query: str
    chat_history: List[Dict[str, str]]
    answer: str
    fallback_needed: bool
    execution_time: float


# --- 3. LOCAL FAISS VECTOR STORAGE SETUP ---
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
FAISS_DIR = "./faiss_linux_db"


def check_or_create_vector_db():
    """If database doesn't exist, chunk the text file and embed it into FAISS."""
    if os.path.exists(FAISS_DIR):
        logging.info("[System Setup] Local FAISS DB detected. Ready for querying.")
        # FAISS requires explicit permission to load local pickle files
        return FAISS.load_local(FAISS_DIR, EMBEDDING_MODEL, allow_dangerous_deserialization=True)

    logging.info("[System Setup] Vector DB not found. Building FAISS database...")
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

    # Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    docs = text_splitter.create_documents([book_text])

    # Generate vectors and save them locally using FAISS
    vector_db = FAISS.from_documents(docs, EMBEDDING_MODEL)
    vector_db.save_local(FAISS_DIR)
    logging.info(f"[System Setup] Successfully embedded and saved {len(docs)} text chunks to FAISS.")
    return vector_db


# --- 4. DEFINE THE GRAPH NODES ---

def contextualize_query_node(state: GraphState):
    logging.info("[NODE: Contextualizer] Rewriting query for database search...")
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
    logging.info("[NODE: Vector RAG Router] Searching FAISS vector space...")
    start_time = time.time()

    vector_db = FAISS.load_local(FAISS_DIR, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
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
    logging.info("[NODE: Web Search] Scraping internet failsafe...")
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


def should_we_fallback(state: GraphState):
    if state["fallback_needed"]:
        return "search_web"
    return "end_process"


# --- 5. BUILD THE GRAPH ---
def build_agent():
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

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    check_or_create_vector_db()

    app = build_agent()
    config = {"configurable": {"thread_id": "faiss_linux_session_001"}}

    logging.info("\n" + "=" * 50)
    logging.info("=== RUNNING TURN 1 ===")
    state_1 = {"question": "How did he get his money back from the crystal shop?"}
    final_state = app.invoke(state_1, config=config)
    logging.info(f"\nFinal Answer: {final_state.get('answer')}")

    logging.info("\n" + "=" * 50)
    logging.info("=== RUNNING TURN 2 ===")
    state_2 = {"question": "What is the boy name?"}
    final_state_2 = app.invoke(state_2, config=config)
    logging.info(f"\nFinal Answer: {final_state_2.get('answer')}")
