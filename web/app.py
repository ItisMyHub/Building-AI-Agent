# This is the main Flask application file for my project.
# It serves the HTML interface and acts as the controller that decides which backend script to call based on the user's actions.
import os
import sys
import json
import subprocess
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify
import time

# Initializing Flask app here, telling it where to find the templates and static files.
app = Flask(__name__, template_folder="templates", static_folder="static")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(ROOT_DIR, "mini_rag_answer.py")

# Configure directories for each showcase
SHOWCASE_DOCS_DIR = {
    "1": "docs",
    }

def run_query_json(query: str, mode: str, docs_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    This is a helper function to run 'mini_rag_answer.py' script.
    It runs the script as a separate process and captures its JSON output.
    This is how the web app "talks" to the RAG agent for Showcase 1.
    """
    env = os.environ.copy()
    env["RAG_JSON"] = "1"
    env["RAG_QUIET"] = "1"
    env["RAG_OUTPUT_MODE"] = mode

    llm_model = os.environ.get("RAG_LLM_MODEL", "llama3.2")
    env["RAG_LLM_MODEL"] = llm_model
    env["RAG_SYNTH_MODEL"] = os.environ.get("RAG_SYNTH_MODEL", llm_model)

    if docs_dir:
        env["RAG_DOCS_DIR"] = docs_dir
        print(f"Setting RAG_DOCS_DIR to {docs_dir}")

    cmd = [sys.executable, SCRIPT_PATH, query]
    try:
        print(f"Executing: {' '.join(cmd)} with RAG_DOCS_DIR={docs_dir}")
        proc = subprocess.run(cmd, cwd=ROOT_DIR, env=env, capture_output=True, text=True, timeout=180)
        stdout = proc.stdout.strip()
        
        if not stdout and proc.stderr:
            print(f"Error output from RAG script: {proc.stderr}")
            return {"error": proc.stderr}
            
        if not stdout:
            return {"error": "Empty response from backend."}
            
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {je}")
            print(f"Raw output: {stdout[:200]}...")
            return {"error": f"Invalid JSON response: {je}"}
            
    except subprocess.TimeoutExpired:
        return {"error": "The request timed out. The local LLM may be busy or slow."}
    except Exception as e:
        print(f"Exception running RAG script: {str(e)}")
        return {"error": f"Failed to execute backend: {e}"}

EXAMPLE_QUERIES_S1 = [
    "What main items included in week2 shopping list?",
    "What is blockchain and how does its consensus mechanism work?",
    "Give a short overview of the human microbiome",
    "What are the main themes in AI ethics?",
    "Summarize major milestones in the history of the internet",
]

EXAMPLE_QUERIES_S2 = [
    "List the orders for customer ID 5 in 2024.",
    "Which products did customer 12 purchase last month?",
    "Show total spend by each customer in Q1 2024.",
    "What are the top 5 selling products this year?",
    "Find orders containing the product named 'Wireless Mouse'.",
    "How many orders did customer ID 8 place in March?",
]

# This is the main route for running the application. It handles the initial page load and the form submissions for both showcases.
@app.route("/", methods=["GET", "POST"])
def index():
    sys.path.append(ROOT_DIR)
    from live_sql_agent import query_live_database
    
    result: Dict[str, Any] = {}
    query = ""
    showcase_id = str(request.form.get("showcase_id") or request.args.get("showcase_id") or "1")
    
    print(f"Request for showcase {showcase_id}")

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            # --- Main routing logic ---
            if showcase_id == "1":
                # Showcase 1 works the same as before (document retrieval)
                print(f"Processing S1 query '{query}' using mini_rag_answer.py")
                docs_dir = SHOWCASE_DOCS_DIR.get("1")
                result = run_query_json(query=query, mode="none", docs_dir=docs_dir)
                if result.get("abstained", False) or not result.get("relevant_docs"):
                    result["llm_answer"] = None
            
            elif showcase_id == "2":
                # If it's Showcase 2 it directly calls the python function from the LIVE SQL AGENT.
                print(f"Processing S2 query '{query}' using live_sql_agent.py")
                sql_result = query_live_database(query)
                # We build a 'result' dictionary that looks like what the template expects
                result = {
                    "query": query,
                    "llm_answer": sql_result.get("answer", "Sorry, I could not process that request."),
                    # This flag allows html to display the answer immediately after the query.
                    "is_direct_answer": True 
                }

    return render_template(
        "index.html",
        query=query,
        examples_s1=EXAMPLE_QUERIES_S1,
        examples_s2=EXAMPLE_QUERIES_S2,
        result=result,
        showcase_id=showcase_id,
    )

@app.route("/generate", methods=["POST"])
def generate():
    print("\n=== GENERATE REQUEST ===")
    query = request.form.get("query", "").strip()
    mode = request.form.get("mode", "detailed")
    showcase_id = str(request.form.get("showcase_id") or "1")
    
    print(f"Generate request - Query: '{query}', Mode: {mode}, Showcase: {showcase_id}")

    if not query:
        return jsonify({"error": "No query provided for generation."})

    docs_dir = SHOWCASE_DOCS_DIR.get(showcase_id, "docs")
    
    # Ensure directory exists
    if not os.path.exists(docs_dir):
        print(f"WARNING: Directory {docs_dir} does not exist!")
        return jsonify({"error": f"Document directory '{docs_dir}' not found"})
    
    # Check directory contents
    files = os.listdir(docs_dir)
    print(f"Directory {docs_dir} contains {len(files)} files")
    
    print(f"Running query with mode={mode}, docs_dir={docs_dir}")
    result = run_query_json(query=query, mode=mode, docs_dir=docs_dir)
    
    # Log the result for debugging
    if "error" in result:
        print(f"Error returned: {result['error']}")
    elif "llm_answer" in result:
        answer_preview = result.get("llm_answer", "")[:50].replace("\n", " ")
        print(f"Answer preview: {answer_preview}...")
    else:
        print(f"No answer or error in result. Keys: {list(result.keys())}")
    
    return jsonify(result)

if __name__ == "__main__":
    # Ensure docs directories exist before starting
    for dir_key, dir_path in SHOWCASE_DOCS_DIR.items():
        if not os.path.exists(dir_path):
            print(f"WARNING: Creating missing directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            files = os.listdir(dir_path)
            print(f"Directory {dir_path} contains {len(files)} files")
            
    print("Starting Flask app...")
    app.run(host="127.0.0.1", port=5000, debug=True)