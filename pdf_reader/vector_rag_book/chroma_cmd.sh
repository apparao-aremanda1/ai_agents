# 1. Install standard build tools (usually already on Ubuntu)
sudo apt update
sudo apt install build-essential python3-dev python3-venv

# 2. Create and activate a fresh virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install the LATEST versions of our stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pymupdf langchain-community langchain-huggingface sentence-transformers langgraph chromadb langchain-chroma

pip install -U ddgs
pip install anthropic python-dotenv


# For SQLLite DB
pip install langgraph-checkpoint-sqlite

# For Postgresql DB
pip install langgraph-checkpoint-postgres psycopg psycopg-pool
pip install psycopg-binary

#for server installation
sudo apt update
sudo apt install postgresql postgresql-contrib -y

sudo systemctl start postgresql
sudo systemctl enable postgresql

# 1. Create the database
sudo -u postgres psql -c "CREATE DATABASE langgraph_db;"
# 2. Set the password for the 'postgres' user
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'mysecretpassword';"

