sudo apt update
sudo apt install build-essential python3-dev python3-venv

# 2. Create and activate a fresh virtual environment
python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pymupdf langchain-community langchain-huggingface sentence-transformers langgraph

pip install faiss-cpu
pip install -U ddgs
pip install anthropic python-dotenv
