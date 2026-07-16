import os
import ast
import time
import pandas as pd
import anthropic
from dotenv import load_dotenv

# Import your compiled LangGraph app
# Ensure the filename matches your actual main script (e.g., pvt_rag_chromadb_pandas)
from vector_rag_bills.pvt_rag_chromadb_pandas import build_agent

load_dotenv()


# --- 1. GENERATE SYNTHETIC QUESTIONS ---
def generate_synthetic_questions(num_questions=10):
    print(f"🤖 Step 1: Generating {num_questions} synthetic test questions...")

    try:
        df = pd.read_csv("bills_database.csv")
        database_context = df.to_markdown(index=False)
    except FileNotFoundError:
        print("❌ Error: bills_database.csv not found. Please ensure it exists in the directory.")
        return []

    prompt = f"""You are a patient who is trying to understand their hospital bills.
    Here is the data from your bills:

    {database_context}

    Generate exactly {num_questions} highly realistic questions you would ask an AI assistant about these bills.

    Include a mix of:
    - Simple lookups (e.g., dates, amounts, clinic names)
    - Math questions (e.g., totals, differences)
    - Tricky/Missing info questions (to test the system's ability to say 'I don't know')

    Output ONLY a clean, valid Python list of strings. Do not include markdown formatting, markdown blocks (```python), or explanations.
    Example format: ["Question 1", "Question 2", "Question 3"]
    """

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.content[0].text.strip()

    try:
        # Safely parse the string into a Python list
        question_list = ast.literal_eval(raw_output)
        print(f"✅ Successfully generated {len(question_list)} questions.\n")
        return question_list
    except Exception as e:
        print(f"❌ Error parsing generated questions: {e}")
        print(f"Raw Output was:\n{raw_output}")
        return []


# --- 2. THE BLIND JUDGE ---
def judge_answer(question, actual_answer):
    """Evaluates an answer based on fulfillment and relevance without knowing the ground truth."""
    client = anthropic.Anthropic()

    prompt = f"""You are a strict QA Auditor for a hospital billing AI system.
    Evaluate the AI's output based ONLY on the Question and the Actual AI Output.

    Question: {question}
    Actual AI Output: {actual_answer}

    Analyze the AI Output and classify it into exactly ONE of these categories:

    1. ANSWERED (The AI provided a confident, detailed response with data or facts that addresses the question)
    2. MISSING_DATA (The AI politely stated it could not find the answer in the provided documents)
    3. BLOCKED (The AI correctly refused to answer because the query was off-topic)
    4. EVASIVE (The AI gave a vague, unhelpful, or rambling answer that doesn't answer the question)

    Reply with ONLY the exact category name. No explanation."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip().upper()
    except Exception as e:
        return "ERROR_IN_JUDGE"


# --- 3. MAIN EVALUATION PIPELINE ---
def run_automated_pipeline():
    # Load your LangGraph application
    app = build_agent()

    # Generate the questions dynamically (Change the number here to test more/less)
    test_questions = generate_synthetic_questions(num_questions=15)

    if not test_questions:
        print("Pipeline aborted due to question generation failure.")
        return

    results = []
    success_count = 0

    print(f"🚀 Step 2: Running Evaluation on {len(test_questions)} questions...\n")

    for i, question in enumerate(test_questions):
        print(f"Test {i + 1}/{len(test_questions)}: {question}")

        # Use a unique thread ID for every test so context doesn't bleed between questions
        config = {"configurable": {"thread_id": f"auto_eval_{int(time.time())}_{i}"}}
        state = {"question": question}

        # Execute the RAG Graph
        try:
            final_state = app.invoke(state, config=config)
            actual_answer = final_state.get("answer", "NO ANSWER RETURNED")
        except Exception as e:
            actual_answer = f"SYSTEM CRASH: {str(e)}"

        # Grade the output
        grade = judge_answer(question, actual_answer)

        # Track metrics (We consider both Answered and properly Blocked/Missing Data as successes in this context)
        if grade in ["ANSWERED", "MISSING_DATA", "BLOCKED"]:
            success_count += 1

        results.append({
            "Question": question,
            "Actual_Answer": actual_answer,
            "Grade": grade
        })
        print(f"   -> Grade: {grade}\n" + "-" * 50)

    # --- 4. CALCULATE METRICS & SAVE ---
    print(f"\n📊 STEP 3: EVALUATION COMPLETE 📊")
    print(f"Total Questions Evaluated: {len(test_questions)}")
    print(f"Valid Responses (Answered/Missing/Blocked): {success_count}")

    df_report = pd.DataFrame(results)

    # Calculate grade distribution
    distribution = df_report['Grade'].value_counts().to_dict()
    print("\nResult Distribution:")
    for key, val in distribution.items():
        print(f" - {key}: {val}")

    # Save to disk
    timestamp = int(time.time())
    report_filename = f"evaluation_report_{timestamp}.csv"
    df_report.to_csv(report_filename, index=False)
    print(f"\n📝 Detailed report saved cleanly to {report_filename}")


if __name__ == "__main__":
    run_automated_pipeline()
