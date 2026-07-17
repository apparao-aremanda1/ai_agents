from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# This automatically looks for your .env file and loads the ANTHROPIC_API_KEY
load_dotenv()


def test_ai_connection():
    print("Connecting to Anthropic...")

    # We are using Opus 4.8, the latest 2026 model for coding and logic.
    llm = ChatAnthropic(
        model="claude-opus-4-8"
    )

    prompt = "Hi Claude, I am a senior Python developer. Reply with a short, one-sentence greeting."

    try:
        response = llm.invoke(prompt)
        print("\n--- SUCCESS! Response from Claude ---")
        print(response.content)
        print("-------------------------------------")

    except Exception as e:
        print(f"\nError connecting to API: {e}")


if __name__ == "__main__":
    test_ai_connection()
