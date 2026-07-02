# retriever.py

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load same embedding model as in embedder.py
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index & metadata
def load_vector_store(vector_dir="vector_store"):
    index = faiss.read_index(os.path.join(vector_dir, "index.faiss"))
    with open(os.path.join(vector_dir, "index.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# Search top-k similar code chunks for a question.
# vector_dir lets the caller point at a session-specific index (see app.py),
# so two users / two parsed repos in the same deployment don't clobber each
# other's vector store.
def retrieve_similar_code(question, k=5, vector_dir="vector_store"):
    index, metadata = load_vector_store(vector_dir)
    question_embedding = model.encode([question])

    # Over-fetch so that after dedup we still have room to return up to k
    # distinct chunks. The brace-matching extractors in embedder.py intentionally
    # produce overlapping chunks (e.g. a class body AND its inner method as two
    # separate entries), so without this the LLM can receive several
    # near-identical copies of the same function and mistakenly report them as
    # "duplicate definitions."
    fetch_k = max(k * 4, 20)
    D, I = index.search(question_embedding, min(fetch_k, len(metadata)))

    results = []
    seen_normalized = set()
    seen_exact_file_code = set()

    for distance, i in zip(D[0], I[0]):
        if i < 0 or i >= len(metadata):
            continue
        entry = metadata[i]
        code = entry.get("code", "")
        file = entry.get("file", "")

        # Exact duplicate (same file, same chunk text) — always skip.
        key_exact = (file, code)
        if key_exact in seen_exact_file_code:
            continue

        # Near-duplicate: one chunk is fully contained in another (this is the
        # class-vs-inner-method overlap case). Normalize whitespace so we can
        # compare reliably, then skip if this chunk's code is a substring of a
        # chunk we've already kept, or vice versa.
        normalized = " ".join(code.split())
        is_redundant = False
        for kept_normalized in seen_normalized:
            if normalized in kept_normalized or kept_normalized in normalized:
                is_redundant = True
                break

        if is_redundant:
            continue

        seen_exact_file_code.add(key_exact)
        seen_normalized.add(normalized)

        # Attach the raw L2 distance so the UI can surface retrieval confidence.
        # Smaller distance = closer match. We don't reinterpret this as a
        # percentage/score here since that depends on embedding scale; the UI
        # can display it as-is or rank-order it.
        result_entry = dict(entry)
        result_entry["distance"] = float(distance)
        results.append(result_entry)

        if len(results) >= k:
            break

    return results