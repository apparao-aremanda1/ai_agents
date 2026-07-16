'''
This code was running ubuntu box. not in Windows.

Chroma DB is simple and easy to install and can support 10 million vectors.
FAISS, it will support billions or vectors.

chromadb will be for small and medium sized applications and FAISS is for huge applications.
'''
import os
import time
import logging
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
import anthropic
from langchain_chroma import Chroma  # <-- Modern, clean import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pytesseract
from pdf2image import convert_from_path
from langchain_text_splitters import CharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, filename='output.log',
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

# 2. SILENCE THE NOISY THIRD-PARTY LIBRARIES HERE:
logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

# Highly recommended extra silencers for AI/Web applications:
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# --- 1. DEFINE THE STATE ---
class GraphState(TypedDict):
    question: str
    standalone_query: str
    chat_history: List[Dict[str, str]]
    answer: str
    fallback_needed: bool
    execution_time: float


# --- 2. LOCAL CHROMA VECTOR STORAGE SETUP ---
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DIR = "./chroma_linux_db"


def check_or_create_vector_db():
    if os.path.exists(CHROMA_DIR):
        logging.info("[System Setup] Local Chroma DB detected. Ready for querying.")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDING_MODEL)

    logging.info("[System Setup] Vector DB not found. Building Chroma database...")
    cache_filename = "book_text_cache.txt"
    pdf_filename = "Hospital_bills.pdf"

    if not os.path.exists(cache_filename):
        logging.info(f"[System Setup] Extracting text from scanned '{pdf_filename}' using OCR...")
        pages = convert_from_path(pdf_filename)
        full_text = ""
        for page_num, page_image in enumerate(pages):
            # OCR reads the image and converts it to text
            page_text = pytesseract.image_to_string(page_image)
            # Tagging the page number helps Claude understand bill separation
            full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}"
        with open(cache_filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        pages.close()

    with open(cache_filename, "r", encoding="utf-8") as f:
        book_text = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n--- PAGE",
        chunk_size=4000,
        chunk_overlap=0  # No overlap needed if we split cleanly by page
    )
    docs = text_splitter.create_documents([book_text])

    vector_db = Chroma.from_documents(docs, EMBEDDING_MODEL, persist_directory=CHROMA_DIR)
    logging.info(f"[System Setup] Successfully embedded and saved {len(docs)} text chunks to Chroma.")
    return vector_db


# --- 3. DEFINE THE GRAPH NODES ---

def contextualize_query_node(state: GraphState):
    logging.info(f"\n[NODE: Contextualizer] Rewriting query for database search...")
    start_time = time.time()

    history = state.get("chat_history", [])

    if not history:
        rewritten = f"{state['question']} in the scanned hospital bills"
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
        2. Always append the phrase "In the bills " to the very end of the query.
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
                "content": f"""You are a specialized medical billing assistant.

                    Here is the verified context extracted from the scanned hospital bills:
                    {retrieved_context}

                    Here is our conversation history:
                    {formatted_history}

                    Answer the following question using ONLY the context block provided above.

                    CRITICAL INSTRUCTION:
                    If the true answer cannot be verified from the context, do NOT invent an answer. 
                    Simply state: "I cannot find the answer to that in the provided bills."
                    If a question has multiple parts, answer the parts you can, and clearly state which parts are missing from the documents.

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


# --- 4. BUILD THE GRAPH ---
def build_agent():
    load_dotenv()
    workflow = StateGraph(GraphState)

    workflow.add_node("Contextualize", contextualize_query_node)
    workflow.add_node("Query_Vector_RAG", query_vector_rag_node)

    workflow.set_entry_point("Contextualize")
    workflow.add_edge("Contextualize", "Query_Vector_RAG")
    workflow.add_edge("Query_Vector_RAG", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    check_or_create_vector_db()

    app = build_agent()
    config = {"configurable": {"thread_id": "linux_session_001"}}

    logging.info("\n=== RUNNING TURN 1 ===")
    state_1 = {"question": "When was the first blood test happened?"}
    final_state = app.invoke(state_1, config=config)
    logging.info(f"\nFinal Answer: {final_state.get('answer')}")
    logging.info("\n" + "=" * 50)

    logging.info("\n=== RUNNING TURN 2 ===")
    state_2 = {"question": "What is the oldest bill and what is latest one?"}
    final_state_2 = app.invoke(state_2, config=config)
    logging.info(f"\nFinal Answer: {final_state_2.get('answer')}")
    logging.info("\n" + "=" * 50)

    logging.info("\n=== RUNNING TURN 3 ===")
    state_3 = {"question": "What did you understand from Konnect diagnostics?"}
    final_state_3 = app.invoke(state_3, config=config)
    logging.info(f"\nFinal Answer: {final_state_3.get('answer')}")
    logging.info("\n" + "=" * 50)

    logging.info("\n=== RUNNING TURN 4 ===")
    state_4 = {"question": "What is the hospital final and what the discharge summary?"}
    final_state_4 = app.invoke(state_4, config=config)
    logging.info(f"\nFinal Answer: {final_state_4.get('answer')}")
    logging.info("\n" + "=" * 50)

    logging.info("\n=== RUNNING TURN 5 ===")
    state_5 = {"question": "Did you find anything odd in the reports?"}
    final_state_5 = app.invoke(state_5, config=config)
    logging.info(f"\nFinal Answer: {final_state_5.get('answer')}")
    logging.info("\n" + "=" * 50)
