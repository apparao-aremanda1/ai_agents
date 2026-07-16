import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_pdf(file_path):
    print(f"Loading PDF: {file_path}...")

    # 1. Load the PDF document
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")

    # 2. Chop the document into chunks
    # We overlap chunks by 200 characters so we don't accidentally cut a sentence in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(pages)
    print(f"Successfully chopped the PDF into {len(chunks)} chunks!")

    # Print the first chunk to prove it worked
    if chunks:
        print("\n--- Here is Chunk #1 ---")
        print(chunks[0].page_content)
        print("------------------------\n")


if __name__ == "__main__":
    # The script looks for a file named 'sample.pdf' in your project root
    # Since this script is inside the 'utils' folder, we need to point to the folder above it
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_file = os.path.join(project_root, "sample.pdf")

    if os.path.exists(pdf_file):
        ingest_pdf(pdf_file)
    else:
        print(f"Error: Could not find 'sample.pdf' at {pdf_file}")
        print("Please drag a PDF into your project folder and rename it to 'sample.pdf'.")