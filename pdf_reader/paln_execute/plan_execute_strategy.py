import operator
import os
from typing import List, Tuple, Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic  # <-- SWITCHED TO ANTHROPIC
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


# Make sure your ANTHROPIC_API_KEY is set in your environment variables
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# ---------------------------------------------------------
# 1. STATE & SCHEMAS (The Deterministic Guardrails)
# ---------------------------------------------------------
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    # operator.add ensures we append new steps to the history rather than overwriting
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str


class Plan(BaseModel):
    """The Planner node will strictly output this schema."""
    steps: List[str] = Field(description="Strict, step-by-step execution plan.")


class FinalResponse(BaseModel):
    """The Re-Planner node will output this when the goal is complete."""
    response: str = Field(description="Final summary of actions taken.")


class Act(BaseModel):
    """The Re-Planner will output EITHER a new plan OR a final response."""
    action: Plan | FinalResponse = Field(
        description="Action to perform: Update the plan, or provide a final response."
    )


# ---------------------------------------------------------
# 2. DEFINING THE TOOLS
# ---------------------------------------------------------
@tool
def fetch_live_positions(asset: str) -> str:
    """Fetch current live trading positions for a specific asset."""
    # Mocking an API response
    return f"Live positions for {asset}: Short Nifty 24500 CE (SQUARED OFF BY BROKER), Long Nifty 24700 CE (ACTIVE - ORPHAN LEG)."


@tool
def execute_square_off(position_id: str) -> str:
    """Execute a market order to close a specific position."""
    # Mocking trade execution
    return f"SUCCESS: Market order executed. Position {position_id} squared off."


tools = [fetch_live_positions, execute_square_off]

# ---------------------------------------------------------
# 3. DEFINING THE NODES
# ---------------------------------------------------------
# Claude 3.5 Sonnet is highly recommended here for its speed and reasoning capabilities
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)  # <-- SWITCHED TO CLAUDE


def planner_node(state: PlanExecuteState):
    """Generates the initial blueprint."""
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Principal AI Agent Architect managing financial systems. "
                   "For the given objective, generate a strict step-by-step plan. "
                   "Do NOT execute the steps. Just write the blueprint."),
        ("user", "{input}")
    ])

    # Claude fully supports with_structured_output in LangChain
    planner = planner_prompt | llm.with_structured_output(Plan)
    result = planner.invoke({"input": state["input"]})
    return {"plan": result.steps}


def executor_node(state: PlanExecuteState):
    """Takes the FIRST step from the plan and executes it."""
    current_task = state["plan"][0]

    # The executor runs a micro-agent to accomplish just this one task
    executor_agent = create_react_agent(llm, tools)
    agent_response = executor_agent.invoke({"messages": [("user", current_task)]})
    result_string = agent_response["messages"][-1].content

    return {"past_steps": [(current_task, result_string)]}


def replanner_node(state: PlanExecuteState):
    """Evaluates the result and updates the plan or ends the process."""
    replanner_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Supervisor Agent. Evaluate the original objective, the completed steps, and the remaining plan."
         "If the objective is fully resolved, output a FinalResponse."
         "If more steps are needed, output a new, updated Plan."),
        ("user", "Objective: {input}\n\n"
                 "Completed Steps: {past_steps}\n\n"
                 "Remaining Plan: {plan}")
    ])

    replanner = replanner_prompt | llm.with_structured_output(Act)
    result = replanner.invoke(state)

    if isinstance(result.action, FinalResponse):
        return {"response": result.action.response}
    else:
        return {"plan": result.action.steps}


# ---------------------------------------------------------
# 4. COMPILING THE LANGGRAPH STATE MACHINE
# ---------------------------------------------------------
def should_end(state: PlanExecuteState) -> str:
    if "response" in state and state["response"]:
        return "true"
    return "false"


def create_app(checkpointer=None):
    """Factory function to build and compile the LangGraph application."""
    workflow = StateGraph(PlanExecuteState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replanner")

    workflow.add_conditional_edges(
        "replanner",
        should_end,
        {
            "true": END,
            "false": "executor"
        }
    )

    # Compile the workflow into an executable app
    app = workflow.compile(checkpointer=checkpointer)

    return app


# Global initialization (Clean and concise)
app = create_app()


# ---------------------------------------------------------
# 5. RUNNING THE SYSTEM
# ---------------------------------------------------------
if __name__ == "__main__":
    objective = "Audit the live Nifty portfolio. If the short leg is closed but the long leg is active, " \
                "square off the active long leg to neutralize risk."

    initial_state = {"input": objective}

    for event in app.stream(initial_state, config={"recursion_limit": 10}):
        for node_name, node_state in event.items():
            print(f"--- STATE UPDATE FROM: {node_name.upper()} ---")

            if "plan" in node_state:
                print(f"Current Plan: {node_state['plan']}\n")
            if "past_steps" in node_state:
                print(f"Just Completed: {node_state['past_steps'][-1]}\n")
            if "response" in node_state:
                print(f"FINAL OUTPUT: {node_state['response']}\n")