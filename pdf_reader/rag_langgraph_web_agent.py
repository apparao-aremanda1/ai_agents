import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END


# --- 1. DEFINE THE STATE (MEMORY) ---
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    evaluation: str
    search_source: str  # Tracks if we are reading the "pdf" or the "web"


# --- 2. DEFINE THE NODES (ACTIONS) ---
def retrieve_node(state: GraphState):
    print(f"\n[NODE: Retrieve_PDF] Searching LOCAL database for: '{state['question']}'")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = vectorstore.similarity_search(state['question'], k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": context, "search_source": "pdf"}


def web_search_node(state: GraphState):
    print(f"\n[NODE: Web_Search] Local DB failed. Searching the INTERNET for: '{state['question']}'")
    search_tool = DuckDuckGoSearchRun()

    # Actually run a live web search!
    web_results = search_tool.invoke(state['question'])

    return {"context": web_results, "search_source": "web"}


def generate_node(state: GraphState):
    print(f"[NODE: Generate] Writing answer based on {state['search_source']} chunks...")
    llm = ChatAnthropic(model="claude-opus-4-8")

    prompt = f"""You are an assistant. Answer the question using ONLY the provided context. 
    If the answer is not in the context, explicitly say 'I cannot answer this based on the context.'

    Question: {state['question']}
    Context: {state['context']}
    """
    response = llm.invoke(prompt)
    return {"answer": response.content}


def evaluate_node(state: GraphState):
    print("[NODE: Evaluate] Auditing the answer...")
    answer = state['answer'].lower()

    if "cannot answer" in answer or "don't know" in answer or "not in the context" in answer:
        print(f"   -> AUDIT FAILED: The {state['search_source']} chunks were useless.")
        return {"evaluation": "FAIL"}
    else:
        print(f"   -> AUDIT PASSED: A valid answer was generated from the {state['search_source']}!")
        return {"evaluation": "PASS"}


# --- 3. DEFINE THE ROUTING LOGIC (EDGES) ---
def route_after_evaluation(state: GraphState):
    # If we found a good answer, we are done!
    if state["evaluation"] == "PASS":
        return "end_process"

    # If the audit failed, let's check WHERE we just searched
    if state["search_source"] == "pdf":
        print("   -> ROUTER: PDF didn't have the answer. Diverting to Web Search...")
        return "search_web"
    else:
        # If we already searched the web and STILL failed, we have to give up.
        print("\n[ROUTER] Both PDF and Web Search failed. Ending process.")
        return "end_process"


# --- 4. BUILD THE GRAPH ---
def build_agent():
    load_dotenv()
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("Retrieve_PDF", retrieve_node)
    workflow.add_node("Web_Search", web_search_node)
    workflow.add_node("Generate", generate_node)
    workflow.add_node("Evaluate", evaluate_node)

    # Add Edges
    workflow.set_entry_point("Retrieve_PDF")

    # Both search nodes go to Generate
    workflow.add_edge("Retrieve_PDF", "Generate")
    workflow.add_edge("Web_Search", "Generate")

    # Generate always goes to Evaluate
    workflow.add_edge("Generate", "Evaluate")

    # The Conditional Router
    workflow.add_conditional_edges(
        "Evaluate",
        route_after_evaluation,
        {
            "end_process": END,
            "search_web": "Web_Search"
        }
    )

    return workflow.compile()


if __name__ == "__main__":
    app = build_agent()

    # We ask a question that is DEFINITELY not in "The Alchemist" PDF
    initial_state = {
        "question": "What is the final conclusion in 'The Alchemist' ?"
    }

    print("=== STARTING HYBRID AGENTIC RAG SYSTEM ===")
    final_state = app.invoke(initial_state)

    print("\n=== FINAL OUTPUT ===")
    print(final_state["answer"])
