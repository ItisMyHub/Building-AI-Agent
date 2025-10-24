# This is the main indexing script for my project. Its job is to create the vector database (the FAISS index) that the Showcase 1 RAG agent needs to answer questions.
# When this script is run, it processes the .md files in the 'docs' folder and builds the 'index_s1' directory, which the main application uses.

import os
import subprocess
import sys

# This dictionary holds the configuration for the RAG agent's index.
SHOWCASE_1_CONFIG = {
    "name": "Showcase 1 (RAG Corpus)",
    "docs_dir": "docs",
    "index_dir": "index_s1"
}

def build_index(docs_dir: str, index_dir: str):
    """
    This function calls the 'mini_rag_index.py' script
    to perform the actual indexing of the documents.
    """
    print(f"--- Building the vector index from '{docs_dir}'... ---")
    
    # First, I'm making sure the documents folder actually exists and has files.
    if not os.path.isdir(docs_dir) or not os.listdir(docs_dir):
        print(f"❌ FATAL ERROR: The document source '{docs_dir}' is missing or empty.")
        print("   Please make sure the 'docs' folder is present and contains the .md files.")
        sys.exit(1)

    # Setting up environment variables so the other script knows which folders to read from and write to.
    env = os.environ.copy()
    env["RAG_DOCS_DIR"] = docs_dir
    env["RAG_INDEX_DIR"] = index_dir
    
    # This is the command that runs the main indexing logic.
    cmd = [sys.executable, "mini_rag_index.py"]
    
    try:
        # Here is running the command and checking for any errors.
        subprocess.run(cmd, env=env, check=True, text=True)
        print(f"✅ Success! The index has been built in the '{index_dir}' directory.\n")
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: 'mini_rag_index.py' was not found.")
        print("   This build script depends on it, so please make sure it's in the same folder.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ FATAL ERROR: The indexing process failed.")
        print(f"   The error was: {e}")
        sys.exit(1)

def main():
    """The main entry point of the script."""
    print("Starting the build process for the Showcase 1 vector index.")
    
    # Calling the main function with the config for Showcase 1.
    build_index(SHOWCASE_1_CONFIG["docs_dir"], SHOWCASE_1_CONFIG["index_dir"])
    
    print("Build process finished.")

if __name__ == "__main__":
    # A quick check to make sure the required script exists before starting.
    if not os.path.exists("mini_rag_index.py"):
        print("CRITICAL ERROR: 'mini_rag_index.py' is required to build the index.")
        print("Please ensure that file is in the project root directory.")
        sys.exit(1)
    
    main()