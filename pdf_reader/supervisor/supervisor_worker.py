import logging
import os
from typing import Annotated, TypedDict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
# OCR & Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
# LangGraph & LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

load_dotenv()

# ==========================================
# 1. SETUP & GLOBAL DEFINITIONS
# ==========================================
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
DB_DIR = os.path.join(BASE_DIR, "chroma_windows_db")
os.makedirs(BILLS_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(collection_name="hospital_bills", persist_directory=DB_DIR, embedding_function=embeddings)


# ==========================================
# 2. SHARED TOOLS (Billing & Research)
# ==========================================
@tool
def calculate_csv_totals(query_type: str, patient_name: str = None) -> str:
    """Use for math or list records from the CSV. query_type: total_sum, bill_count, oldest_date, newest_date, list_bills."""
    csv_path = os.path.join(BILLS_DIR, "bills_database.csv")
    if not os.path.exists(csv_path): return "Error: Database not built."
    df = pd.read_csv(csv_path)
    if "patient_name" not in df.columns: df["patient_name"] = "Unknown"

    if patient_name:
        norm_db = df["patient_name"].astype(str).str.lower().str.replace(r"\s+", "", regex=True)
        norm_search = patient_name.lower().replace(" ", "")
        df = df[norm_db.str.contains(norm_search)]
        if df.empty: return f"No bills for '{patient_name}'"

    if query_type == "total_sum": return f"Total: Rs. {df['amount_inr'].sum():,.2f}"
    if query_type == "list_bills": return df.to_markdown(index=False)
    return "Operation completed."


# ==========================================
# 3. CRAG WORKER NODES
# ==========================================
class CRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: List[str]
    retries: int


def retrieve_node(state: CRAGState):
    results = vector_store.similarity_search(state["question"], k=5)
    return {"documents": [doc.page_content for doc in results]}


class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


def grade_documents_node(state: CRAGState):
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    grader = llm.with_structured_output(Grade)
    filtered = []
    for d in state["documents"]:
        score = grader.invoke([{"role": "user", "content": f"Doc: {d}\nQuestion: {state['question']}"}])
        if score.binary_score == "yes": filtered.append(d)
    return {"documents": filtered}


def decide_to_generate(state: CRAGState):
    if not state["documents"]:
        return "rewrite" if state.get("retries", 0) < 2 else "fallback"
    return "generate"


def rewrite_node(state: CRAGState):
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    res = llm.invoke(f"Rewrite this for vector search: {state['question']}")
    return {"question": res.content, "retries": state.get("retries", 0) + 1}


def generate_node(state: CRAGState):
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    res = llm.invoke(f"Answer using these docs: {state['documents']}. Question: {state['question']}")
    return {"messages": [res]}


def fallback_node(state: CRAGState):
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    res = llm.invoke(f"Answer this using general knowledge: {state['question']}")
    return {"messages": [res]}


# ==========================================
# 4. AGENT FACTORIES
# ==========================================
def build_billing_agent():
    workflow = StateGraph(TypedDict("State", {"messages": Annotated[list, add_messages]}))
    workflow.add_node("Agent", lambda s: {"messages": [
        ChatAnthropic(model="claude-haiku-4-5").bind_tools([calculate_csv_totals]).invoke(s["messages"])]})
    workflow.add_node("tools", ToolNode([calculate_csv_totals]))
    workflow.add_edge(START, "Agent")
    workflow.add_conditional_edges("Agent", tools_condition)
    workflow.add_edge("tools", "Agent")
    return workflow.compile(checkpointer=MemorySaver())


def build_crag_agent():
    workflow = StateGraph(CRAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("fallback", fallback_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", decide_to_generate,
                                   {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)
    return workflow.compile(checkpointer=MemorySaver())


# ==========================================
# 5. SUPERVISOR
# ==========================================
@tool
def research_worker(query: str) -> str:
    """
    Use this tool to research discharge summaries, medical diagnostics,
    or general medical questions. The agent will retrieve and grade
    documents until it finds a relevant answer.
    """
    return build_crag_agent().invoke({"question": query, "messages": [HumanMessage(content=query)]},
                                     {"configurable": {"thread_id": "res_1"}})["messages"][-1].content


@tool
def billing_worker(query: str) -> str:
    """
    Use this tool to perform math or list records from the CSV database.
    query_type MUST be one of: 'total_sum', 'bill_count', 'oldest_date', 'newest_date', 'list_bills'.
    Use patient_name to filter rows for a specific patient if requested.
    """
    return build_billing_agent().invoke({"messages": [HumanMessage(content=query)]},
                                        {"configurable": {"thread_id": "bill_1"}})["messages"][-1].content


def supervisor_node(state):
    llm = ChatAnthropic(model="claude-haiku-4-5").bind_tools([research_worker, billing_worker])
    return {"messages": [llm.invoke(state["messages"])]}


def build_master_graph():
    workflow = StateGraph(TypedDict("State", {"messages": Annotated[list, add_messages]}))
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Workers", ToolNode([research_worker, billing_worker]))
    workflow.set_entry_point("Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        tools_condition,
        {"tools": "Workers", "__end__": END}
    )
    workflow.add_edge("Workers", "Supervisor")
    return workflow.compile(checkpointer=MemorySaver())

# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    app = build_master_graph()
    config = {"configurable": {"thread_id": "main_session"}}

    user_input = "Calculate the total for Apparao and then explain the difference between CT and Angiography."
    for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
        if "Supervisor" in event:
            # 2. Grab the latest message in the Supervisor's history
            last_message = event["Supervisor"]["messages"][-1]

            # 3. Only print if it's an AIMessage (The final answer) and not a Tool call
            if last_message.__class__.__name__ == "AIMessage" and last_message.content:
                logging.info(last_message.content)

    logging.info("\n")
