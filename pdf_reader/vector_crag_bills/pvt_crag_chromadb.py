import logging
import os
from typing import Annotated, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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

logging.info("Booting up local embedding model (One-time setup)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

BASE_DIR = r"D:\ai_agents\pdf_reader"
BILLS_DIR = os.path.join(BASE_DIR, "bills")
DB_DIR = os.path.join(BASE_DIR, "chroma_windows_db")
os.makedirs(BILLS_DIR, exist_ok=True)

vector_store = Chroma(
    collection_name="hospital_bills",
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: List[str] # Stores the chunks from Chroma
    retries: int


def retrieve_node(state: GraphState):
    logging.info("--- RETRIEVING DOCUMENTS ---")
    query = state["question"]

    # Using your existing global vector_store
    results = vector_store.similarity_search(query, k=5)

    # Extract the text and save it to the state
    docs = [doc.page_content for doc in results]

    return {"documents": docs}


def generate_node(state: GraphState):
    logging.info("--- GENERATING FINAL ANSWER ---")
    question = state["question"]
    documents = state["documents"]

    # Combine the surviving documents into a single text block
    context = "\n\n".join(documents)

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    # Force the LLM to use the context we verified
    system_prompt = (
        "You are a helpful hospital billing assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If the answer is not in the context, just say that you don't know. "
        "Context:\n\n"
        f"{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    # Generate the answer
    response = llm.invoke(messages)

    # Append Claude's response to the chat history
    return {"messages": [response]}


# Define the strict output schema for the Grader
class Grade(BaseModel):
    binary_score: str = Field(
        description="Relevance score 'yes' or 'no'"
    )


def rewrite_node(state: GraphState):
    logging.info("--- REWRITING SEARCH QUERY ---")
    question = state["question"]
    current_retries = state.get("retries", 0)

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    system_prompt = (
        "You are an expert at optimizing user questions for vector database retrieval. "
        "Look at the input question and extract the underlying semantic intent and keywords. "
        "Output ONLY the rewritten question without any preamble, quotes, or explanation."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Original question: {question}")
    ]

    # Claude generates a better search query
    response = llm.invoke(messages)
    better_question = response.content.strip()

    logging.info(f"   -> Original: '{question}'")
    logging.info(f"   -> Rewritten: '{better_question}'")

    # Overwrite the old question in the state with the new one.
    # Because this node loops back to 'retrieve', the vector search will automatically use this new string!
    return {"question": better_question, "retries": current_retries + 1}


def grade_documents_node(state: GraphState):
    logging.info("--- GRADING DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]

    # Initialize Claude and force it to output the Grade schema
    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
    structured_llm_grader = llm.with_structured_output(Grade)

    # The grading prompt
    system = """You are a strict grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'.
    If the document is completely unrelated, grade it as 'no'."""

    # Grade each document
    filtered_docs = []
    for d in documents:
        prompt = f"Retrieved document: \n\n {d} \n\n User question: {question}"
        score = structured_llm_grader.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])

        # Only keep the documents that passed the test!
        if score.binary_score == "yes":
            logging.info("   -> Document Relevant.")
            filtered_docs.append(d)
        else:
            logging.info("   -> Document Irrelevant. Discarding.")

    return {"documents": filtered_docs}


def decide_to_generate(state: GraphState):
    logging.info("--- ASSESSING GRADED DOCUMENTS ---")
    filtered_documents = state["documents"]
    current_retries = state.get("retries", 0)

    # If the list is empty, it means all documents failed the test
    if not filtered_documents:
        if current_retries >= 2:
            print("   -> Max retries reached. Database doesn't have the answer. Routing to General Knowledge Fallback.")
            return "fallback"

        print(f"   -> All documents failed (Attempt {current_retries + 1}/2). Routing to Rewrite.")
        return "rewrite"

    # If we have good documents, proceed to generate the final answer
    logging.info("   -> Good documents found. Routing to Generate.")
    return "generate"


def fallback_node(state: GraphState):
    print("--- GENERATING FROM GENERAL KNOWLEDGE ---")
    question = state["question"]

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    # We tell Claude the database failed, so just answer it normally
    system_prompt = "You are a helpful medical assistant. The user asked a question that was not in their personal medical records. Answer their question directly based on your general medical knowledge."

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return {"messages": [response]}


def build_agent():

    workflow = StateGraph(GraphState)
    # Add all your nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node) # Your standard answering node
    workflow.add_node("rewrite_query", rewrite_node) # A node that uses LLM to rephrase the question
    workflow.add_node("fallback", fallback_node)

    # Build the execution flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # THE CRAG ROUTER:
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "fallback": "fallback"  # Add the escape hatch!
        }
    )
    # If it rewrote the query, loop BACK to the beginning to search again!
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

if __name__ == "__main__":
    app = build_agent()
    config = {"configurable": {"thread_id": "1"}}

    user_query = "How much Spent People Hospital as inpatient? "

    # Initialize the state with the user's question
    initial_state = {
        "question": user_query,
        "messages": [HumanMessage(content=user_query)]
    }

    logging.info(f"\nUser: {user_query}\n")

    # Stream the graph execution
    for event in app.stream(initial_state, config=config):
        # This will logging.info the node names as they execute so you can watch the CRAG loop in action
        for key, value in event.items():
            logging.info(f"Finished executing node: {key}")

    # logging.info the final output from the AI
    final_state = app.get_state(config).values
    if final_state.get("messages"):
        logging.info(f"\nAgent: {final_state['messages'][-1].content}")
