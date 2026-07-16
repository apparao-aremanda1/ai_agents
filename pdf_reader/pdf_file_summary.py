import os
import time
import fitz  # PyMuPDF


def build_local_text_cache(pdf_filename="sample.pdf", cache_filename="book_text_cache.txt"):
    print(f"=== Starting PDF Extraction ===")
    start_time = time.time()

    # 1. Check if the PDF actually exists
    if not os.path.exists(pdf_filename):
        print(f"[ERROR] Could not find '{pdf_filename}'. Please check the file path.")
        return

    print(f"[1/3] Opening '{pdf_filename}'...")

    # 2. Open the PDF using PyMuPDF
    pdf_document = fitz.open(pdf_filename)
    total_pages = len(pdf_document)
    full_text = ""

    print(f"[2/3] Extracting text from {total_pages} pages...")

    # 3. Loop through every page and extract the text
    for page_num in range(total_pages):
        page = pdf_document[page_num]

        # PyMuPDF's "text" extraction maintains paragraphs and reading order much better
        page_text = page.get_text("text")

        # Append to our massive string, adding a clear page break indicator
        full_text += f"\n\n--- PAGE {page_num + 1} ---\n\n"
        full_text += page_text

    # 4. Save the extracted text to your cache file
    print(f"[3/3] Saving extracted text to '{cache_filename}'...")
    with open(cache_filename, "w", encoding="utf-8") as f:
        f.write(full_text)

    # Cleanup
    pdf_document.close()

    execution_time = round(time.time() - start_time, 2)
    file_size_kb = round(os.path.getsize(cache_filename) / 1024, 2)

    print(f"\n=== Extraction Complete ===")
    print(f"File Saved: {cache_filename}")
    print(f"Cache Size: {file_size_kb} KB")
    print(f"Total Time: {execution_time} seconds")


if __name__ == "__main__":
    # Ensure your PDF is named 'sample.pdf' and is in the same folder as this script
    build_local_text_cache(pdf_filename="sample.pdf", cache_filename="book_text_cache.txt")
