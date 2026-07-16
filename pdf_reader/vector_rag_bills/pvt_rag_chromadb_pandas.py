import json
import logging
import os
import time
from typing import TypedDict, List, Dict

import anthropic
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from langchain_chroma import Chroma  # <-- Modern, clean import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from pdf2image import convert_from_path

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DIR = "./chroma_linux_db"
load_dotenv()

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


class GraphState(TypedDict):
    question: str
    intent: str # <-- NEW: Tracks "bill_query", "greeting", or "off_topic"
    standalone_query: str
    chat_history: List[Dict[str, str]]
    answer: str
    fallback_needed: bool
    execution_time: float


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


def gatekeeper_node(state: GraphState):
    logging.info("[NODE: Gatekeeper] Classifying user intent...")
    logging.info(f"   -> User Query: '{state['question']}'")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        # !!! UPDATE THIS SYSTEM PROMPT HERE !!!
        system="""You are a strict intent classifier for a hospital billing system.
        Analyze the user's input and classify it into exactly ONE of these categories:

        1. BILL_QUERY (Questions about bills, amounts, dates, clinics, timelines, reports, or diagnostics)
        2. GREETING (Hello, hi, thank you, goodbye)
        3. OFF_TOPIC (Poems, coding, general knowledge, recipes, or general medical advice)

        CRITICAL EXAMPLES FOR BILL_QUERY:
        - "What is the oldest bill and what is latest one?" -> BILL_QUERY
        - "How many bills are there?" -> BILL_QUERY
        - "What did Konnect diagnostics say?" -> BILL_QUERY
        - "Total amount spent?" -> BILL_QUERY

        Reply with ONLY the category name (BILL_QUERY, GREETING, or OFF_TOPIC). No explanation.""",
        messages=[{"role": "user", "content": state['question']}]
    )

    user_intent = response.content[0].text.strip()
    logging.info(f"   -> Intent detected: {user_intent}")

    return {"intent": user_intent}


def guardrail_node(state: GraphState):
    logging.info("[NODE: Guardrail] Handling off-topic or general chat...")

    if state["intent"] == "GREETING":
        answer = "Hello! I am your hospital billing assistant. How can I help you with your bills today?"
    else:
        answer = "I'm sorry, I am specifically designed to help you analyze your hospital bills and reports. I cannot " \
                 "answer general questions, write content, or provide medical advice. "

    return {"answer": answer}


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

    logging.info("   -> [ROUTING] Returning AI response based on document context.")

    updated_history = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": raw_answer}
    ]

    return {"answer": raw_answer, "fallback_needed": False, "execution_time": total_time,
            "chat_history": updated_history}


def structured_data_query_node(state: GraphState):
    logging.info("[NODE: Structured Data] Analyzing real Pandas DataFrame...")

    # 1. Load the REAL database extracted from your PDF
    df = pd.read_csv("bills_database.csv")

    # 2. Convert to Markdown for Claude to read
    markdown_table = df.to_markdown(index=False)

    # 3. Ask Claude to do the math based on the real data
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system="""You are a strict financial data analyst. 
        You will be provided with a Markdown table representing a patient's real hospital bills.
        Answer the user's question by performing calculations on the provided table.
        Do NOT invent data. If the answer is not in the table, say so.""",
        messages=[
            {
                "role": "user",
                "content": f"Here is the database table:\n\n{markdown_table}\n\nQuestion: {state['question']}"
            }
        ]
    )

    return {"answer": response.content[0].text.strip()}


def route_intent(state: GraphState):
    if state["intent"] == "BILL_QUERY":
        return "search_database"
    else:
        return "block_query"


# --- BUILD THE GRAPH ---
def build_agent():
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("Gatekeeper", gatekeeper_node)
    workflow.add_node("Guardrail", guardrail_node)
    workflow.add_node("Contextualize", contextualize_query_node)
    workflow.add_node("Query_Vector_RAG", query_vector_rag_node)

    # 1. The graph now starts at the Gatekeeper
    workflow.set_entry_point("Gatekeeper")

    # 2. The Gatekeeper decides where to go
    workflow.add_conditional_edges(
        "Gatekeeper",
        route_intent,
        {
            "search_database": "Contextualize",  # Go to RAG
            "block_query": "Guardrail"  # Go to Refusal
        }
    )

    # 3. Standard RAG flow
    workflow.add_edge("Contextualize", "Query_Vector_RAG")
    workflow.add_edge("Query_Vector_RAG", END)

    # 4. Guardrail flow just ends immediately
    workflow.add_edge("Guardrail", END)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def extract_pdf_to_pandas(pdf_filename="Hospital_bills.pdf", csv_filename="bills_database.csv"):
    """Reads a scanned PDF, uses Claude to extract structured data, and saves a CSV."""

    # Skip extraction if we already built the database
    if os.path.exists(csv_filename):
        logging.info("[System Setup] Pandas database already exists. Skipping OCR extraction.")
        return pd.read_csv(csv_filename)

    logging.info(f"[System Setup] Starting OCR on '{pdf_filename}'...")

    # 1. Run OCR on all pages
    pages = convert_from_path(pdf_filename)
    raw_ocr_text = ""
    for page_image in pages:
        raw_ocr_text += pytesseract.image_to_string(page_image) + "\n"

    logging.info("[System Setup] OCR complete. Asking Claude to structure the data...")

    # 2. Force Claude to convert the messy text into strict JSON
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2000,
        system="""You are an expert data extraction bot. 
        Read the provided OCR text from hospital bills.
        Extract the data into a strict JSON array of objects.
        Do NOT output any markdown, conversational text, or explanations. Only output raw JSON.

        Required keys for each object:
        - "bill_no" (string, use "UNKNOWN" if missing)
        - "date" (string YYYY-MM-DD)
        - "clinic_name" (string)
        - "amount_inr" (float, numbers only, no currency symbols)""",
        messages=[
            {
                "role": "user",
                "content": f"Extract data from this text:\n\n{raw_ocr_text}"
            }
        ]
    )

    # 3. Parse the JSON and create the Pandas DataFrame
    try:
        raw_json_string = response.content[0].text.strip()
        # Clean up in case Claude added markdown backticks
        if raw_json_string.startswith("```json"):
            raw_json_string = raw_json_string[7:-3]

        extracted_data = json.loads(raw_json_string)
        df = pd.DataFrame(extracted_data)

        # 4. Save to CSV so we never have to run OCR/Extraction on this file again
        df.to_csv(csv_filename, index=False)
        logging.info(f"[System Setup] Successfully saved {len(df)} bills to {csv_filename}!")
        return df

    except Exception as e:
        logging.error(f"Failed to parse Claude's JSON: {e}")
        logging.error(f"Raw output was: {response.content[0].text}")
        return pd.DataFrame()


def structured_data_query_node(state: GraphState):
    logging.info("[NODE: Structured Data] Analyzing real Pandas DataFrame...")

    # 1. Load the REAL database extracted from your PDF
    df = pd.read_csv("bills_database.csv")

    # 2. Convert to Markdown for Claude to read
    markdown_table = df.to_markdown(index=False)

    # 3. Ask Claude to do the math based on the real data
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system="""You are a strict financial data analyst. 
        You will be provided with a Markdown table representing a patient's real hospital bills.
        Answer the user's question by performing calculations on the provided table.
        Do NOT invent data. If the answer is not in the table, say so.""",
        messages=[
            {
                "role": "user",
                "content": f"Here is the database table:\n\n{markdown_table}\n\nQuestion: {state['question']}"
            }
        ]
    )

    return {"answer": response.content[0].text.strip()}


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


if __name__ == "__main__":
    # ---------------------------------------------------------
    # STEP 1: SYSTEM INITIALIZATION & DATA PIPELINES
    # ---------------------------------------------------------

    # A. Build the Vector Database (For Text/Doctor Notes)
    check_or_create_vector_db()

    # B. Build the Pandas/CSV Database (For Math/Financials)
    # This runs OCR once, creates bills_database.csv, and never runs OCR again.
    extract_pdf_to_pandas(pdf_filename="Hospital_bills.pdf", csv_filename="bills_database.csv")

    # ---------------------------------------------------------
    # STEP 2: START THE LANGGRAPH AGENT
    # ---------------------------------------------------------

    app = build_agent()
    config = {"configurable": {"thread_id": "linux_session_001"}}

    # --- EXECUTE YOUR TEST TURNS BELOW ---
    logging.info("\n=== RUNNING TURN 1 ===")
    state_1 = {"question": "Helllo... "}
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

    logging.info("\n=== RUNNING TURN 6 ===")
    state_6 = {"question": "What is total amount and how many bill are there?"}
    final_state = app.invoke(state_6, config=config)
    logging.info(f"\nFinal Answer: {final_state.get('answer')}")
    logging.info("\n" + "=" * 50)

    logging.info("\n=== RUNNING TURN 7 ===")
    state_7 = {"question": "What is Maximum Bill and Minimum Bill. What is the difference between these two?"}
    final_state = app.invoke(state_7, config=config)
    logging.info(f"\nFinal Answer: {final_state.get('answer')}")
    logging.info("\n" + "=" * 50)

    logging.info("\n=== RUNNING TURN 8 ===")
    state_8 = {"question": "On which date Pragati Clinic was contacted."}
    final_state = app.invoke(state_8, config=config)
    logging.info(f"\nFinal Answer: {final_state.get('answer')}")
    logging.info("\n" + "=" * 50)