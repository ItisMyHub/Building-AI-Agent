## Project Overview

This project demonstrates two AI agent architectures, accessible through a single web interface:

1.  **Showcase 1: RAG Agent:** A Retrieval-Augmented Generation agent that answers questions based on a local corpus of markdown documents.
2.  **Showcase 2: Live SQL Agent:** An agent that uses LangChain to translate natural language questions into SQL queries, providing live answers from a database.

## Following prerequisties needed o get the application running locally.

Follow these steps to get the application running locally.
*   Python 3.9+
*   An Ollama server running locally. [ollama.com](https://ollama.com/).

### 2. Initial Setup

Set up a virtual environment and install the required Python libraries. The insturction is only for macbooks

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies from the requirements file
pip install -r requirements.txt
```

### 3. Download the AI Model

This project uses `llama3.2` by default. 

```bash
ollama pull llama3.2
```

### 4. Build the Search Index

The RAG agent for Showcase 1 requires a search index. Run the following command from the project's root directory to build it.

```bash
python build_all_indexes.py
```

### 5. Run the Web Application

Start the Flask web server.

```bash
python web/app.py
```

Now, with your web browser, navigate to **http://127.0.0.1:5000** to use the application.

---