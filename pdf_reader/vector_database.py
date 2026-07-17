from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vector_database():
    print("1. Loading and chunking sample.pdf...")
    # Step 1: Load and Split (What we did yesterday)
    loader = PyPDFLoader("sample.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"   Success! Created {len(chunks)} chunks.")

    print("\n2. Initializing the Embedding Model...")
    print("   (If this is your first time, it may take a few seconds to download the free model)")
    # Step 2: Initialize Embeddings
    # This model turns English sentences into lists of 384 numbers for mathematical comparison
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("\n3. Converting text to embeddings and building FAISS database...")
    # Step 3: Build the Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("\n4. Saving database locally to 'faiss_index' folder...")
    # Step 4: Save it so we don't have to rebuild it every time we ask a question
    vectorstore.save_local("faiss_index")

    print("\n--- SUCCESS! Vector Database created and saved! ---")

    # --- QUICK TEST ---
    # Let's see if the database actually works by running a similarity search
    print("\n--- Running a quick test search ---")
    test_query = "What is the main topic or summary of this document?"
    print(f"Query: '{test_query}'")

    # We ask FAISS to find the top 2 chunks that mathematically match our query
    results = vectorstore.similarity_search(test_query, k=2)

    print("\nTop match found in the database:")
    print("-" * 40)
    # Print the first 250 characters of the best matching chunk
    print(results[0].page_content[:250] + "...")
    print("-" * 40)


if __name__ == "__main__":
    build_vector_database()