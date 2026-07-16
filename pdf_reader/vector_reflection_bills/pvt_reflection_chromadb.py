import os
import logging
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

# LangGraph & LangChain Imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

load_dotenv()

# ==========================================
# 1. SETUP & STATE DEFINITION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')


# We need to track the draft, the list of critiques, and how many times we've looped
class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str
    critiques: List[str]
    retries: int


# Strict schema for the Critic to follow
class CritiqueSchema(BaseModel):
    is_perfect: bool = Field(
        description="True if the draft is flawless, factually accurate, and ready for the user. False otherwise.")
    feedback: str = Field(
        description="Harsh, specific critique on what needs to be fixed. Leave empty if is_perfect is True.")


# ==========================================
# 2. THE GENERATOR NODE (The Writer)
# ==========================================
def generate_node(state: ReflectionState):
    logging.info("--- ✍️ GENERATING DRAFT ---")

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.2)  # Slight temp for writing flexibility

    # If we have critiques, tell the AI to revise. If not, write the first draft.
    critiques = state.get("critiques", [])

    if critiques:
        sys_prompt = f"""You are an expert medical writer. 
        Your previous draft was rejected by the senior medical reviewer.

        CRITIQUE TO ADDRESS:
        {critiques[-1]}

        Rewrite your response to perfectly address these criticisms. Output ONLY the revised draft."""
    else:
        sys_prompt = """You are an expert medical writer. 
        Answer the user's question clearly, accurately, and professionally. Output ONLY the draft."""

    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm.invoke(messages)

    logging.info(f"   -> Draft produced (Length: {len(response.content)} chars)")

    # Return the new draft, and initialize retries if this is the first pass
    return {
        "draft": response.content,
        "retries": state.get("retries", 0)
    }


# ==========================================
# 3. THE CRITIC NODE (The Reviewer)
# ==========================================
def reflect_node(state: ReflectionState):
    logging.info("--- 🧐 CRITIQUING DRAFT ---")

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)  # Temp 0 for strict evaluation
    critic = llm.with_structured_output(CritiqueSchema)

    sys_prompt = """You are a ruthless Senior Medical Editor. 
    Review the provided draft answering the user's original prompt.

    Check for:
    1. Medical accuracy and safety.
    2. Missing critical context.
    3. Formatting, tone, and clarity.

    If it is perfect, set is_perfect to True. If it has flaws, set is_perfect to False and provide a harsh, actionable critique."""

    user_query = state["messages"][0].content
    draft = state["draft"]

    evaluation = critic.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"User Query: {user_query}\n\nCurrent Draft:\n{draft}")
    ])

    if evaluation.is_perfect:
        logging.info("   -> Critic Score: PERFECT")
        return {"critiques": ["PERFECT"]}
    else:
        logging.info(f"   -> Critic Score: REJECTED. Feedback: {evaluation.feedback}")
        # Append the new critique to the list and increment the retry counter
        current_critiques = state.get("critiques", [])
        current_critiques.append(evaluation.feedback)

        return {
            "critiques": current_critiques,
            "retries": state.get("retries", 0) + 1
        }


# ==========================================
# 4. THE ROUTER (The Decider)
# ==========================================
def route_reflection(state: ReflectionState):
    # Check the latest critique
    latest_critique = state["critiques"][-1] if state.get("critiques") else ""
    current_retries = state.get("retries", 0)

    if latest_critique == "PERFECT":
        logging.info("--- ✅ DRAFT APPROVED. ENDING LOOP. ---")
        return "end"

    if current_retries >= 3:
        logging.info("--- ⚠️ MAX RETRIES REACHED. FORCING END. ---")
        return "end"

    logging.info(f"--- 🔄 DRAFT FAILED. SENDING BACK TO WRITER (Attempt {current_retries}/3) ---")
    return "generate"


# ==========================================
# 5. BUILD THE GRAPH
# ==========================================
def build_reflection_agent():
    workflow = StateGraph(ReflectionState)

    workflow.add_node("generate", generate_node)
    workflow.add_node("reflect", reflect_node)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        route_reflection,
        {
            "generate": "generate",
            "end": END
        }
    )

    return workflow.compile()


# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    app = build_reflection_agent()

    # We'll ask a tricky question that LLMs often answer poorly on the first try
    user_input = "Explain how paracetamol works to a 5-year-old, but make sure to include the exact chemical mechanism."

    print("\n" + "=" * 50)
    print(f"USER: {user_input}")
    print("=" * 50 + "\n")

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "critiques": [],
        "retries": 0
    }

    # Run the graph and capture the final state
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("FINAL APPROVED ANSWER:")
    print("=" * 50)
    print(final_state["draft"])
