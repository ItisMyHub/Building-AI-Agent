# The core indexing script for my RAG agent, and its main job is to take all the documents from the source folder, clean them,
# chop itno smaller chunks, and turm them into vector embeddings. 
import os
import re
import json
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    print("Warning: `faiss-cpu` not installed. FAISS index will not be created.")
    print("Install with: pip install faiss-cpu")
    faiss = None

# The script reads its configuration from environment variables,allowing it to be controlled by the build_all_indexes.py script.
DOCS_DIR = os.environ.get("RAG_DOCS_DIR", "docs")
INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "index")
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 60))
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Defining paths based on the dynamic INDEX_DIR
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

SUPPORTED_EXTS = {".txt", ".md", ".markdown"}

def ensure_dirs():
    """Ensures both source and destination directories exist."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

def list_documents() -> List[str]:
    """Lists supported documents from the configured DOCS_DIR."""
    paths = []
    for name in sorted(os.listdir(DOCS_DIR)):
        p = os.path.join(DOCS_DIR, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTS:
            paths.append(p)
    return paths

# For cleaning and chuncking of the documents.
def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_markdown(md_text: str) -> str:
    text = md_text
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def load_and_clean(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    raw = read_text_file(path)
    if ext in {".md", ".markdown"}:
        return clean_markdown(raw)
    return raw

def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> List[Tuple[str, int]]:
    words = text.split()
    chunks = []
    if not words:
        return chunks
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, start))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks

def build_index(embeddings: np.ndarray):
    """This function takes the numpy array of embeddings and builds the FAISS index."""
    if faiss is None:
        print("Error: faiss-cpu is not installed. Cannot build FAISS index.")
        raise RuntimeError("faiss-cpu not installed.")
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # IP for Inner Product, equivalent to Cosine with normalized vectors
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

def main():
    """Main execution function."""
    ensure_dirs()
    files = list_documents()
    
    print(f"--- Starting Index Build ---")
    print(f"Document Source: '{DOCS_DIR}'")
    print(f"Output Index Dir:  '{INDEX_DIR}'")
    
    if not files:
        print(f"\nNo supported files (.txt, .md) found in '{DOCS_DIR}'. Add files and rerun.")
        return

    print(f"\nFound {len(files)} documents to process.")
    
    all_chunks = []
    metadata = []
    for path in files:
        src_name = os.path.basename(path)
        text = load_and_clean(path)
        chunks = chunk_text(text, chunk_size_words=CHUNK_SIZE, overlap_words=CHUNK_OVERLAP)
        print(f"  - Chunking {src_name} -> {len(chunks)} chunk(s)")
        for ci, (c_text, start_word) in enumerate(chunks):
            all_chunks.append(c_text)
            metadata.append({
                "source": src_name,
                "chunk_index": ci,
                "start_word": start_word,
                "text": c_text,
            })

    if not all_chunks:
        print("\nError: No chunks were created. Check document content.")
        return

    print(f"\nTotal chunks created: {len(all_chunks)}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Encoding chunks to embeddings (this may take a while)...")
    embeddings = embedder.encode(
        all_chunks,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    if faiss:
        print(f"Writing FAISS index to '{INDEX_PATH}'")
        build_index(embeddings)
    else:
        print("Skipping FAISS index creation as 'faiss' is not installed.")
    
    # All the chunk text and metadata is saved to a JSON file
    meta = {"count": len(metadata), "metadata": metadata}
    print(f"Writing metadata to '{META_PATH}'")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Indexing complete.")

if __name__ == "__main__":
    main()