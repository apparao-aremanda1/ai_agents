import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END


# --- 1. DEFINE THE STATE (MEMORY) ---
class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    evaluation: str
    loop_count: int


# --- 2. DEFINE THE NODES (ACTIONS) ---
def retrieve_node(state: GraphState):
    print(f"\n[NODE: Retrieve] Searching database for: '{state['question']}'")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Load your existing FAISS database
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # K=3 means pull the top 3 chunks
    docs = vectorstore.similarity_search(state['question'], k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": context}


def generate_node(state: GraphState):
    print("[NODE: Generate] Writing answer based on found chunks...")
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

    # Simple logic: If the AI admitted it couldn't find the answer, we FAIL it.
    if "cannot answer" in answer or "don't know" in answer or "not in the context" in answer:
        print("   -> AUDIT FAILED: The database chunks were useless.")
        return {"evaluation": "FAIL"}
    else:
        print("   -> AUDIT PASSED: A valid answer was generated!")
        return {"evaluation": "PASS"}


def rewrite_node(state: GraphState):
    print("[NODE: Rewrite] Rephrasing the question to try a different search...")
    llm = ChatAnthropic(model="claude-opus-4-8")

    prompt = f"""The following question failed to find results in our vector database. 
    Rewrite the question using different keywords or synonyms so we can try searching again.
    Just provide the new question, nothing else.

    Original Question: {state['question']}
    """
    response = llm.invoke(prompt)
    new_question = response.content.strip()

    # Increment the loop counter
    new_count = state.get('loop_count', 0) + 1
    print(f"   -> NEW QUESTION: '{new_question}' (Attempt {new_count + 1}/3)")

    return {"question": new_question, "loop_count": new_count}


# --- 3. DEFINE THE ROUTING LOGIC (EDGES) ---
def should_we_loop(state: GraphState):
    if state["evaluation"] == "PASS":
        return "end_process"

    if state["loop_count"] >= 2:  # Stop after 3 total attempts (0, 1, 2)
        print("\n[ROUTER] Max retries reached. Halting loop to save API credits.")
        return "end_process"

    return "rewrite_question"


# --- 4. BUILD THE GRAPH ---
def build_agent():
    load_dotenv()

    # Initialize the Graph
    workflow = StateGraph(GraphState)

    # Add our 4 Nodes
    workflow.add_node("Retrieve", retrieve_node)
    workflow.add_node("Generate", generate_node)
    workflow.add_node("Evaluate", evaluate_node)
    workflow.add_node("Rewrite", rewrite_node)

    # Connect the basic pipeline
    workflow.set_entry_point("Retrieve")
    workflow.add_edge("Retrieve", "Generate")
    workflow.add_edge("Generate", "Evaluate")

    # Add the Conditional Edge (The Loop)
    workflow.add_conditional_edges(
        "Evaluate",
        should_we_loop,
        {
            "end_process": END,
            "rewrite_question": "Rewrite"
        }
    )

    # Connect Rewrite back to Retrieve to close the loop
    workflow.add_edge("Rewrite", "Retrieve")

    # Compile the state machine
    return workflow.compile()


if __name__ == "__main__":
    app = build_agent()

    # We ask a highly specific question that might require a rewrite to find
    initial_state = {
        "question": "Did Santiago find the boy's sheep?",
        "loop_count": 0
    }

    print("=== STARTING AGENTIC RAG SYSTEM ===")
    final_state = app.invoke(initial_state)

    print("\n=== FINAL OUTPUT ===")
    print(final_state["answer"])
