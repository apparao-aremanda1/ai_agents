import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def run_rag_pipeline():
    # 1. Load the API Key from your .env file
    load_dotenv()

    print("1. Waking up the local FAISS Database...")
    # Initialize the EXACT same embedding model we used yesterday
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the database from your local folder
    # Note: allow_dangerous_deserialization=True is required by LangChain for local FAISS files
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Convert the database into a "retriever" (a search engine). 'k=3' means find the top 3 chunks.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("2. Connecting to Claude...")
    # Initialize Claude using the model string that worked for your API key
    llm = ChatAnthropic(model="claude-opus-4-8")

    print("3. Building the Brain (RAG Chain)...")
    # Define the strict instructions for Claude
    system_prompt = (
        "You are an intelligent, professional assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer based on the context, just say that you don't know. "
        "Do not hallucinate. \n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Connect the prompt to Claude
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Combine the database search (retriever) with the Claude chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n--- RAG PIPELINE READY ---\n")

    # 4. Ask the exact same question that failed yesterday!
    question = "What is the main topic or summary of this document?"
    print(f"Question: '{question}'\n")
    print("Thinking...\n")

    # Execute the entire pipeline!
    response = rag_chain.invoke({"input": question})

    print("--- CLAUDE'S INTELLIGENT ANSWER ---")
    # The final answer is stored in the "answer" key of the dictionary
    print(response["answer"])
    print("-" * 40)


if __name__ == "__main__":
    run_rag_pipeline()
