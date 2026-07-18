import json
import logging
import os
from typing import Annotated, TypedDict

import pandas as pd
import pytesseract
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pdf2image import convert_from_path

load_dotenv()

# Global Logging
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

BASE_DIR = r"D:\ai_agents\pdf_reader"
BILLS_DIR = os.path.join(BASE_DIR, "bills")
# Match supervisor_worker.py so the vector store is written where the supervisor reads it.
DB_DIR = os.path.join(BASE_DIR, "chroma_windows_db")
os.makedirs(BILLS_DIR, exist_ok=True)

# Windows OCR toolchain locations. Adjust if you installed these elsewhere.
TESSERACT_CMD = r"D:\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"D:\poppler-26.02.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logging.info("Booting up local embedding model (One-time setup)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="hospital_bills",
    persist_directory=DB_DIR,
    embedding_function=embeddings
)


# ==========================================
# PART 1: DATA PIPELINE (PDF -> CSV & VECTOR)
# ==========================================
def setup_real_database(pdf_filename="Hospital_bills.pdf"):
    """Reads a scanned PDF via OCR and processes pages statefully to combine multi-page bills and receipts."""
    # The PDF ships next to this script (vector_rag_bills/); fall back to BILLS_DIR.
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), pdf_filename),
        os.path.join(BILLS_DIR, pdf_filename),
    ]
    pdf_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if pdf_path is None:
        logging.error(f"Cannot find {pdf_filename}. Looked in: {candidate_paths}")
        return

    logging.info("--- 📄 STARTING STATE-AWARE OCR PROCESSING ---")

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(collection_name="hospital_bills", persist_directory=DB_DIR, embedding_function=embeddings)

    # This list will hold our final, deduplicated, and merged records
    csv_data = []

    try:
        logging.info("Converting PDF pages to images...")
        pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        logging.info(f"Successfully loaded {len(pages)} pages.")

        for i, page_image in enumerate(pages):
            logging.info(f"Running OCR on Page {i + 1}...")
            text = pytesseract.image_to_string(page_image)

            if not text.strip():
                continue

            # 1. Always index the full text into the Vector store for semantic context search
            vector_store.add_texts(texts=[text], metadatas=[{"page": i + 1}])

            # 2. Format the currently extracted database records as a string so Claude has memory
            current_database_snapshot = json.dumps(csv_data, indent=2) if csv_data else "No bills extracted yet."

            # 3. State-aware prompt
            prompt = f"""You are a medical billing data architect. Your job is to extract structured bill data from OCR text.

            CRITICAL CONTEXT:
            A single bill might span multiple consecutive pages, or a page might be a payment receipt for a bill listed on a previous page. 
            Review the "Current Extracted Bills" below before deciding what to do with this new page.

            Current Extracted Bills So Far:
            {current_database_snapshot}

            New Page Text (Page {i + 1}):
            {text}

            INSTRUCTIONS:
            1. If this page is a MEDICAL REPORT or DISCHARGE SUMMARY, output exactly: "REPORT"
            2. If this page is a brand NEW bill entirely, return a JSON object with a "status": "NEW" along with details:
               {{"status": "NEW", "bill_no": "...", "date": "YYYY-MM-DD", "clinic_name": "...", "amount_inr": 0.00, "patient_name": "..."}}
            3. If this page is a CONTINUATION or a PAYMENT RECEIPT for a bill already present in the "Current Extracted Bills So Far", do NOT create a new record. Instead, return a JSON object with "status": "MERGE", specifying which "bill_no" it belongs to, and update the fields if necessary (e.g., matching final payment totals):
               {{"status": "MERGE", "target_bill_no": "EXISTING_BILL_NO_HERE", "updated_amount_inr": FINAL_TOTAL_AMOUNT}}

            Output ONLY valid JSON or the word "REPORT" without any conversational filler or markdown wrappers."""

            response = llm.invoke([HumanMessage(content=prompt)])
            output = response.content.strip()

            # Strip markdown wrappers if the LLM includes them
            if output.startswith("```json"):
                output = output.split("```json")[1].split("```")[0].strip()
            elif output.startswith("```"):
                output = output.split("```")[1].strip()

            if "REPORT" in output.upper():
                logging.info(f"   -> Page {i + 1}: Detected Medical Report. (Skipped CSV)")
                continue

            try:
                decision = json.loads(output)

                if decision.get("status") == "NEW":
                    # Remove status key before saving to database
                    record = {k: v for k, v in decision.items() if k != "status"}
                    csv_data.append(record)
                    logging.info(f"   -> Page {i + 1}: Detected NEW Bill! Extracted Rs. {record.get('amount_inr')}")

                elif decision.get("status") == "MERGE":
                    target = decision.get("target_bill_no")
                    new_total = decision.get("updated_amount_inr")

                    # Look back through our records and find the matching bill to update its value
                    updated = False
                    for record in csv_data:
                        if str(record.get("bill_no")).lower() == str(target).lower():
                            record["amount_inr"] = new_total
                            updated = True
                            logging.info(
                                f"   -> Page {i + 1}: Successfully MERGED receipt/continuation into Bill No {target}. Updated Total: Rs. {new_total}")
                            break

                    if not updated:
                        logging.warning(
                            f"   -> Page {i + 1}: Model requested MERGE for '{target}', but that Bill No wasn't found. Saving as a new entry instead.")
                        # Fallback: save as new if lookback failed
                        record = {k: v for k, v in decision.items() if
                                  k not in ["status", "target_bill_no", "updated_amount_inr"]}
                        record["bill_no"] = target
                        record["amount_inr"] = new_total
                        csv_data.append(record)

            except Exception as parse_error:
                logging.warning(f"   -> Failed to process data structure on page {i + 1}: {str(parse_error)}")

    except Exception as e:
        logging.error(f"Critical error during stateful OCR processing: {str(e)}")
        return

    # Write the clean, merged list to the CSV database
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(BILLS_DIR, "bills_database.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"✅ Successfully built deduplicated bills_database.csv with {len(df)} reconciled records!")
    else:
        logging.warning("No financial bills were detected during the pass.")


# ==========================================
# PART 2: THE ACTUAL TOOLS
# ==========================================

@tool
def search_vector_db(query: str) -> str:
    """
        Use this tool to search for medical diagnostics, text reports, or discharge summaries.
        query: The search string (e.g., 'Discharge Summary', 'Doctor Notes', 'Apparao')
        num_pages: The number of pages to retrieve. Default is 10 to ensure wide coverage.
        """
    num_pages = 10
    logging.info(f"[TOOL EXECUTING] Searching Vector DB for: '{query}' (Scanning top {num_pages} pages)")

    # 2. INCREASE THE SEARCH RADIUS
    results = vector_store.similarity_search(query, k=num_pages)

    if not results:
        return "I cannot find any documents matching this query."

    # 3. RETURN CLEAN CONTEXT
    return "\n\n".join([f"--- Page {doc.metadata.get('page', 'Unknown')} ---\n{doc.page_content}" for doc in results])


@tool
def calculate_csv_totals(query_type: str, patient_name: str = None) -> str:
    """
    Use this tool to perform math or list records from the CSV database.
    query_type MUST be one of: 'total_sum', 'bill_count', 'oldest_date', 'newest_date', 'list_bills'.
    Use patient_name to filter rows for a specific patient if requested.
    """
    logging.info(f"[TOOL EXECUTING] Database Query: type='{query_type}', filter='{patient_name}'")
    csv_path = os.path.join(BILLS_DIR, "bills_database.csv")

    if not os.path.exists(csv_path):
        return "Error: The CSV database has not been built yet."

    df = pd.read_csv(csv_path)

    # 1. Safely handle the patient_name column in case OCR missed it on some pages
    if "patient_name" not in df.columns:
        df["patient_name"] = "Unknown"
    else:
        df["patient_name"] = df["patient_name"].fillna("Unknown")

    # 2. Apply patient filter if the LLM provided one
    if patient_name:
        # Normalize strings: lowercase everything and remove all whitespace using regex
        # This makes "Appa Rao", "APPA RAO", and "Apparao" all evaluate as "apparao"
        normalized_db_names = df["patient_name"].astype(str).str.lower().str.replace(r"\s+", "", regex=True)
        normalized_search = patient_name.lower().replace(" ", "")

        # Apply the filter using the normalized strings
        df = df[normalized_db_names.str.contains(normalized_search)]

        if df.empty:
            return f"No bills found matching patient name: '{patient_name}'"

    try:
        if query_type == "total_sum":
            total = df["amount_inr"].sum()
            filter_str = f" for {patient_name}" if patient_name else ""
            return f"The total sum of bills{filter_str} is Rs. {total:,.2f}"

        elif query_type == "bill_count":
            filter_str = f" for {patient_name}" if patient_name else ""
            return f"There are a total of {len(df)} bills on record{filter_str}."

        elif query_type == "oldest_date":
            oldest = pd.to_datetime(df['date']).min().strftime('%Y-%m-%d')
            return f"The oldest bill on record is from {oldest}."

        elif query_type == "newest_date":
            newest = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
            return f"The most recent bill on record is from {newest}."

        # --- NEW IMPLEMENTATION: LIST BILLS ---
        elif query_type == "list_bills":
            # Select and reorder columns for a clean presentation
            columns_to_show = ["bill_no", "date", "clinic_name", "amount_inr", "patient_name"]
            available_cols = [col for col in columns_to_show if col in df.columns]

            # Convert the Pandas dataframe into a Markdown table string
            markdown_table = df[available_cols].to_markdown(index=False)

            # Return the table directly to the LLM
            return f"Here are the requested bill records:\n\n{markdown_table}"

        else:
            return "Error: Unrecognized database operation requested."

    except Exception as e:
        return f"Database processing failed: {str(e)}"


tools_list = [search_vector_db, calculate_csv_totals]


# Using the native LangGraph Messages State (automatically tracks history)
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

# ==========================================
# PART 3: THE LANGGRAPH RE-ACT AGENT
# ==========================================
def agent_node(state: GraphState):
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    llm_with_tools = llm.bind_tools(tools_list)

    # System prompt provides persona and limits
    sys_msg = SystemMessage(
        content="You are a hospital billing AI. Use tools to answer questions. If the user asks for a poem, code, or something off-topic, politely refuse.")
    messages_to_pass = [sys_msg] + state["messages"]

    response = llm_with_tools.invoke(messages_to_pass)
    return {"messages": [response]}


def build_agent():
    workflow = StateGraph(GraphState)

    workflow.add_node("Agent", agent_node)

    # LangGraph's prebuilt ToolNode automatically executes whichever tool Claude selects
    workflow.add_node("Tools", ToolNode(tools_list))

    workflow.set_entry_point("Agent")

    # tools_condition checks if Claude returned a ToolCall.
    # If yes -> goes to 'Tools'. If no (e.g. conversational answer) -> goes to END.
    workflow.add_conditional_edges("Agent", tools_condition, {"tools": "Tools", "__end__": END})

    # After the tool runs, loop BACK to the Agent so Claude can read the result and format a nice answer!
    workflow.add_edge("Tools", "Agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Run this ONCE to process the real PDF into the CSV and Vector DB
    # (Comment this line out after running it the first time so it doesn't re-process the PDF every time)
    setup_real_database("Hospital_bills.pdf")

    # 2. Boot up the Agent
    app = build_agent()
    config = {"configurable": {"thread_id": "real_data_test_1"}}

    logging.info("\n--- TEST: CHROMA VECTOR SEARCH ---")
    inputs = {"messages": [HumanMessage(content="What is the discharge summary status for the patient Apparao?")]}
    for event in app.stream(inputs, config=config, stream_mode="values"):
        last_message = event["messages"][-1]
        last_message.pretty_print()

    inputs = {"messages": [HumanMessage(content="What is the problem in these reports?")]}
    for event in app.stream(inputs, config=config, stream_mode="values"):
        last_message = event["messages"][-1]
        last_message.pretty_print()

    logging.info("\n--- TEST: PANDAS MATH TOOL ---")
    inputs2 = {"messages": [HumanMessage(content="Can you list all bills with the amount, bill number and date for the Patient Apparao?")]}
    for event in app.stream(inputs2, config=config, stream_mode="values"):
        last_message = event["messages"][-1]
        speaker = last_message.__class__.__name__  # Returns 'HumanMessage' or 'AIMessage'
        logging.info(f"\n--- {speaker.upper()} ---")

        # 2. Log the actual text content
        if last_message.content:
            logging.info(last_message.content)

        # 3. Log any tool calls if Claude decided to use one
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool in last_message.tool_calls:
                logging.info(f"[TOOL TRIGGERED] -> {tool['name']} | Args: {tool['args']}")
