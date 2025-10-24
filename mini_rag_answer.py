# This is the main retrieval-and-answer script for Showcase 1 (the RAG agent).
# Its job is to load the FAISS index and metadata, encode the user's query, retrieve relevant document chunks (using FAISS with a numpy fallback),
# synthesize a final answer using the Ollama model (or return an extractive fallback),return a structured JSON result.

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Determinig that which index to use.
DOCS_DIR = os.environ.get("RAG_DOCS_DIR", "docs")
INDEX_DIR = "index_s2" if DOCS_DIR == "docs_db" else "index_s1"

import sys, re, json, time, traceback
from typing import List, Dict, Any, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
from sentence_transformers import SentenceTransformer

# Paths for the chosen index.
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")

# Here, embedding and LLM configuration.
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.environ.get("RAG_TOP_K", "12"))
AS_JSON = os.environ.get("RAG_JSON", "1") == "1"
OUTPUT_MODE = os.environ.get("RAG_OUTPUT_MODE", "none").lower()
ENABLE_ABSTAIN = os.environ.get("RAG_ENABLE_ABSTAIN", "1") == "1"

# Simple thresholds to control relevance filtering.
MIN_TOP_SIM = float(os.environ.get("RAG_MIN_TOP_SIM", "0.30"))
MIN_CAND_SIM = float(os.environ.get("RAG_MIN_CAND_SIM", "0.25"))
REL_KEEP_FRACTION = float(os.environ.get("RAG_REL_KEEP_FRACTION", "0.75"))

# Ollama / LLM settings.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
RERANK_MODEL = os.environ.get("RAG_LLM_MODEL", "llama3.2")
SYNTH_MODEL = os.environ.get("RAG_SYNTH_MODEL", RERANK_MODEL)
NUM_PREDICT_DETAILED = int(os.environ.get("RAG_SYNTH_NUM_PREDICT", "512"))
NUM_PREDICT_BULLETS = int(os.environ.get("RAG_SYNTH_NUM_PREDICT_BULLETS", "256"))
VERSION = "6.2.1-multi-index-final"

def load_meta() -> dict:
    if not os.path.exists(META_PATH):
        # Provide a specific, helpful error message
        sys.exit(json.dumps({"error": f"Index metadata not found at '{META_PATH}'. Please run the indexing script for the '{DOCS_DIR}' document source."}))
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "metadata" not in meta or not isinstance(meta["metadata"], list) or not meta["metadata"]:
        sys.exit(json.dumps({"error": f"Malformed or empty meta file: {META_PATH}."}))
    return meta

def ensure_chunk_embeddings(embedder: SentenceTransformer, meta: dict) -> np.ndarray:
    if os.path.exists(EMB_PATH):
        try:
            emb = np.load(EMB_PATH)
            if emb.ndim == 2 and emb.shape[0] == len(meta["metadata"]):
                return emb
        except Exception:
            pass # Fall through to re-generate if loading fails
    print(f"Embeddings file '{EMB_PATH}' not found or invalid. Generating new embeddings...", file=sys.stderr)
    chunks = [md.get("text", md.get("chunk_text", "")) for md in meta["metadata"]]
    emb = embedder.encode(
        chunks, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(EMB_PATH, emb)
    return emb

def faiss_search(q_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    #Query FAISS index on disk. This is the preferred fast retrieval path.
    import faiss
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file not found at '{INDEX_PATH}'.")
    index = faiss.read_index(INDEX_PATH)
    return index.search(q_vec, k)

def cosine_search(q_vec: np.ndarray, chunk_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # A simple numpy-based cosine similarity fallback when FAISS is unavailable. This is slower but reliable and ensures the script works in constrained environments.
    sims = (chunk_emb @ q_vec.T).reshape(-1)
    idxs = np.argsort(-sims)[:k]
    scores = sims[idxs]
    return scores[None, :], idxs[None, :]

def meta_fields(md: dict, default_chunk_id: int) -> Tuple[str, int, str]:
    source = md.get("source", "unknown")
    chunk_id = md.get("chunk_id", md.get("chunk_index", default_chunk_id))
    text = md.get("text", md.get("chunk_text", ""))
    return source, int(chunk_id), text

def aggregate_by_document(indices: List[int], scores: List[float], meta_list: List[dict]) -> Dict[str, Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = {}
    for idx, sc in zip(indices, scores):
        md = meta_list[idx]
        src, cid, text = meta_fields(md, 0)
        entry = agg.setdefault(src, {"best_score": sc, "chunks": []})
        if sc > entry["best_score"]:
            entry["best_score"] = sc
        entry["chunks"].append({"chunk": cid, "score": sc, "text": text})
    for src in agg:
        agg[src]["chunks"].sort(key=lambda x: -x["score"])
    return agg

def simple_relevance_filter(agg_docs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not agg_docs:
        return []
    top_score = max((v["best_score"] for v in agg_docs.values()), default=0)
    if top_score < MIN_TOP_SIM:
        return []
    keep_cut = max(MIN_CAND_SIM, top_score * REL_KEEP_FRACTION)
    kept = []
    for name, info in agg_docs.items():
        if info["best_score"] >= keep_cut:
            kept.append({
                "name": name,
                "score": float(info["best_score"]),
                "top_chunks": info["chunks"]
            })
    kept.sort(key=lambda d: -d["score"])
    return kept

_token_re = re.compile(r"[a-z0-9]{3,}")

def _query_tokens(q: str) -> List[str]:
    return list({t for t in _token_re.findall((q or "").lower())})

def _passes_lexical_sanity(query: str, doc_name: str, doc_text: str) -> bool:
    toks = _query_tokens(query)
    if not toks:
        return True
    hay = f"{doc_name} {doc_text[:600]}".lower()
    return any(t in hay for t in toks)

def _ollama_generate(model: str, prompt: str, num_predict: int) -> Optional[str]:
    candidates = [model]
    if ":" in model:
        base = model.split(":", 1)[0]
        if base != model:
            candidates.append(base)
    if "llama3.2" not in candidates:
        candidates.append("llama3.2")

    for m in candidates:
        payload = {"model": m, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": num_predict}}
        data = json.dumps(payload).encode("utf-8")
        req = Request(OLLAMA_HOST + "/api/generate", data=data, headers={"Content-Type": "application/json"}, method="POST")
        for attempt in range(1, 4):
            try:
                with urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8")
                    obj = json.loads(raw) if raw.strip().startswith("{") else {"response": raw}
                    out = (obj.get("response") or "").strip()
                    if out:
                        return out
            except Exception:
                time.sleep(0.5 * attempt)
    return None

def build_prompt(query: str, contexts: List[Dict[str, Any]], mode: str) -> str:
    # Construct the LLM prompt that instructs the model to answer ONLY using the provided context. This reduces hallucination and keeps answers grounded in the retrieved text.
    lines = []
    lines.append("You are a helpful assistant.")
    lines.append("Answer ONLY using the CONTEXT below. If the context lacks the answer, reply exactly: Sorry, the provided documents do not contain information on this topic.")
    if mode == "bulleted":
        lines.append("Return 5-7 concise bullet points. No preamble, no concluding line.")
    else:
        lines.append("Return a single, well-structured paragraph. No preamble, no concluding line.")
    lines.append("\nCONTEXT:")
    for i, d in enumerate(contexts, 1):
        lines.append(f"{i}) [{d['name']} | score={d['score']:.3f}] {d['snippet']}")
    lines.append(f"\nQUESTION: {query.strip()}")
    lines.append("\nANSWER:")
    return "\n".join(lines)

def extractive_fallback(context_text: str, mode: str) -> str:
    text = re.sub(r"\s+", " ", (context_text or "").strip())
    if not text:
        return ""
    if mode == "bulleted":
        sents = re.split(r'(?<=[.!?])\s+', text)
        bullets = [f"- {s.strip()}" for s in sents if len(s.strip()) > 20][:6]
        return "\n".join(bullets) if bullets else text[:500]
    else:
        return text[:700]

def main():
    # This is the entry point for the script. This Parses the user's query, runs retrieval, relevance filtering, and (optionally) LLM synthesis, then prints a JSON result if AS_JSON is enabled.
    t0 = time.time()
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        if AS_JSON:
            print(json.dumps({"error": "No query provided."}))
        return
    
    # Helpful for reviewing index file    
    print(f"Querying with DOCS_DIR='{DOCS_DIR}', using INDEX_DIR='{INDEX_DIR}'", file=sys.stderr)

    meta = load_meta()
    meta_list = meta["metadata"]
    embedder = SentenceTransformer(EMBED_MODEL)

    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    k = min(TOP_K, len(meta_list))
    used = "faiss"
    try:
        scores, idxs = faiss_search(q_vec, k)
    except Exception:
        print("FAISS search failed, falling back to numpy cosine search.", file=sys.stderr)
        used = "numpy"
        chunk_emb = ensure_chunk_embeddings(embedder, meta)
        scores, idxs = cosine_search(q_vec, chunk_emb, k)

    idxs_row = list(map(int, idxs[0]))
    sims_row = [float(s) for s in scores[0]]

    agg = aggregate_by_document(idxs_row, sims_row, meta_list)
    kept_docs = simple_relevance_filter(agg)

    if kept_docs:
        top_doc = kept_docs[0]
        top_text = top_doc["top_chunks"][0]["text"] if top_doc["top_chunks"] else ""
        if not _passes_lexical_sanity(query, top_doc["name"], top_text):
            kept_docs = []

    abstained = ENABLE_ABSTAIN and (len(kept_docs) == 0)
    llm_answer = ""
    contexts_used: List[Dict[str, Any]] = []

    if not abstained and OUTPUT_MODE in ("bulleted", "detailed"):
        contexts_used = []
        max_docs_for_prompt = 3
        max_snippet_chars = 600
        for doc in kept_docs[:max_docs_for_prompt]:
            top_chunk = doc["top_chunks"][0] if doc["top_chunks"] else {"text": ""}
            snippet = re.sub(r"\s+", " ", (top_chunk.get("text") or "").strip())
            if len(snippet) > max_snippet_chars:
                cut = snippet.rfind(". ", 0, max_snippet_chars)
                snippet = snippet[:cut + 1] if cut > 80 else snippet[:max_snippet_chars]
            contexts_used.append({"name": doc["name"], "score": doc["score"], "snippet": snippet})

        if contexts_used:
            prompt = build_prompt(query, contexts_used, OUTPUT_MODE)
            llm_answer = _ollama_generate(SYNTH_MODEL, prompt, num_predict=NUM_PREDICT_DETAILED if OUTPUT_MODE == "detailed" else NUM_PREDICT_BULLETS) or ""

            if not llm_answer.strip():
                top_text = contexts_used[0]["snippet"]
                llm_answer = extractive_fallback(top_text, OUTPUT_MODE)

            if llm_answer.strip() and contexts_used:
                # Create a citation string from the top 3 sources used for the answer
                top_sources_for_citation = contexts_used[:3]
                citations = ", ".join([f"({s['name']}, {s['score']:.3f})" for s in top_sources_for_citation])
                llm_answer = f"{llm_answer}\n\nSources: {citations}"

    # Build the structured result object for downstream code
    sources = [{"name": d["name"], "score": d["score"]} for d in kept_docs[:3]]
    result = {
        "version": VERSION, "query": query, "retriever": used, "abstained": abstained,
        "abstain_message": "Sorry, the provided documents do not contain information on this topic.",
        "relevant_docs": [{"name": d["name"], "chunk": d["top_chunks"][0]["chunk"] if d["top_chunks"] else 0, "score": d["score"]} for d in kept_docs],
        "llm_answer": llm_answer, "sources": sources, "output_mode": OUTPUT_MODE,
        "timing_stats": {"total_sec": round(time.time() - t0, 2)}
    }

    if AS_JSON:
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        if AS_JSON:
            print(json.dumps({"error": str(ex), "trace": traceback.format_exc()}))
        else:
            print("Fatal error:", ex, file=sys.stderr)
            traceback.print_exc()
        sys.exit(1)