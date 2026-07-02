# embedder.py

import os
import ast
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load embedding model (you can change this)
model = SentenceTransformer("all-MiniLM-L6-v2")  # or use a code-specific model

# Embedding dimension for all-MiniLM-L6-v2. Update this if you change the model.
EMBED_DIM = model.get_sentence_embedding_dimension()

# Extensions we know how to handle, grouped by extractor type
PYTHON_EXTS = (".py",)
JS_TS_EXTS = (".js", ".jsx", ".ts", ".tsx")
JAVA_EXTS = (".java",)
SUPPORTED_EXTS = PYTHON_EXTS + JS_TS_EXTS + JAVA_EXTS


def extract_functions_from_python(source):
    """Extract function/class definitions from Python source using ast."""
    tree = ast.parse(source)
    functions = []
    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno - 1
            end_line = max(getattr(node, "end_lineno", None) or start_line + 1, start_line + 1)
            code_chunk = "\n".join(lines[start_line:end_line])
            functions.append(code_chunk)

    return functions


def extract_functions_from_js_ts(source):
    """
    Extract function/class-like blocks from JS/TS source using brace matching.
    This is a lightweight, dependency-free approach (no AST parser required).
    Covers: function decls, arrow functions assigned to const/let/var,
    class declarations, and object/class methods.
    """
    # Optional TypeScript bits that can appear between a parameter list and the
    # opening brace: generic type params on the name (e.g. "foo<T>(...)") and a
    # return-type annotation (e.g. "(...): string {"). Without accounting for
    # these, typed TS functions/methods/arrow-functions never match.
    generics = r"(?:<[^>]*>)?"
    return_type = r"(?:\s*:\s*[^{=]+?)?"  # ": string", ": Promise<void>", etc. (non-greedy, stops before { or =>)

    patterns = [
        rf"function\s*{generics}\s+\w+\s*{generics}\s*\([^)]*\)\s*{return_type}\s*{{",  # function foo<T>(...): T {
        rf"(?:const|let|var)\s+\w+\s*(?:\s*:\s*[^=]+?)?=\s*(?:async\s*)?{generics}\s*\([^)]*\)\s*{return_type}\s*=>\s*{{",  # const foo = (...): T => {
        rf"(?:const|let|var)\s+\w+\s*=\s*function\s*\([^)]*\)\s*{return_type}\s*{{",  # const foo = function(...): T {
        r"class\s+\w+(?:<[^>]*>)?(?:\s+extends\s+\w+(?:<[^>]*>)?)?(?:\s+implements\s+[\w,\s<>]+)?\s*{",  # class Foo<T> {
        rf"(?:(?:public|private|protected|static|readonly|async)\s+)?\b(?!if\b|for\b|while\b|switch\b|catch\b|function\b)\w+\s*{generics}\s*\([^)]*\)\s*{return_type}\s*{{",  # method(...): T { (inside classes/objects), excluding control-flow keywords
    ]
    combined = re.compile("|".join(patterns))

    functions = []
    for match in combined.finditer(source):
        start = match.start()
        brace_start = source.index("{", match.end() - 1)

        # Walk forward to find the matching closing brace
        depth = 0
        i = brace_start
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        chunk = source[start:i + 1].strip()
        if chunk:
            functions.append(chunk)

    return functions


def extract_functions_from_java(source):
    """
    Extract method/class/interface/enum blocks from Java source using brace matching.
    This is a lightweight, dependency-free approach (no AST parser required).
    Covers: class/interface/enum declarations and method definitions
    (including annotated, generic, and modifier-decorated signatures).
    """
    # Strip block/line comments, replacing them with same-length whitespace so
    # character offsets stay identical to the original source. This lets us match
    # against the comment-free text (avoiding false brace matches inside comments)
    # while still slicing chunks out of `cleaned` at the correct positions.
    def strip_comments(code):
        code = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), code, flags=re.DOTALL)
        code = re.sub(r"//.*", lambda m: " " * len(m.group(0)), code)
        return code

    cleaned = strip_comments(source)

    patterns = [
        # class/interface/enum/record Foo (extends/implements ...) {
        r"(?:public|private|protected|static|final|abstract)?\s*"
        r"(?:class|interface|enum|record)\s+\w+"
        r"(?:<[^>]*>)?"
        r"(?:\s+extends\s+[\w<>,\s.]+)?"
        r"(?:\s+implements\s+[\w<>,\s.]+)?\s*{",

        # method declarations:
        # (modifiers) (generics) returnType name(args) (throws ...) {
        r"(?:@\w+(?:\([^)]*\))?\s*)*"                     # optional annotations
        r"(?:public|private|protected|static|final|synchronized|abstract|native)\s+"
        r"(?:[\w<>,\[\]\s.]+?)\s+"                         # return type
        r"\w+\s*\([^)]*\)\s*"                              # method name + args
        r"(?:throws\s+[\w,\s.]+)?\s*{",
    ]
    combined = re.compile("|".join(patterns))

    functions = []
    for match in combined.finditer(cleaned):
        start = match.start()
        brace_start = cleaned.index("{", match.end() - 1)

        depth = 0
        i = brace_start
        while i < len(cleaned):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        # Pull the chunk from the ORIGINAL source (not the comment-stripped copy)
        # so the stored code chunk keeps original formatting/comments.
        chunk = source[start:i + 1].strip()
        if chunk:
            functions.append(chunk)

    return functions


def extract_functions_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    ext = os.path.splitext(file_path)[1].lower()

    if ext in PYTHON_EXTS:
        return extract_functions_from_python(source)
    elif ext in JS_TS_EXTS:
        return extract_functions_from_js_ts(source)
    elif ext in JAVA_EXTS:
        return extract_functions_from_java(source)
    else:
        return []


def embed_code_chunks(repo_path, repo_name, vector_dir="vector_store"):
    texts = []
    metadata = []
    matched_file_count = 0

    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTS:
                matched_file_count += 1
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                try:
                    chunks = extract_functions_from_file(full_path)
                    for chunk in chunks:
                        texts.append(chunk)
                        metadata.append({"file": rel_path, "code": chunk})
                except Exception as e:
                    print(f"⚠️ Skipping {rel_path}: {e}")

    # --- Guard: no chunks found ---
    if not texts:
        if matched_file_count == 0:
            raise ValueError(
                f"No supported source files (.py, .js, .jsx, .ts, .tsx, .java) were found "
                f"in '{repo_name}'."
            )
        else:
            raise ValueError(
                f"Found {matched_file_count} source file(s) in '{repo_name}', but no "
                "functions or classes could be extracted from them (files may be empty, "
                "syntax-invalid, or contain only top-level/script code)."
            )

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Defensive reshape in case a single chunk collapses to 1D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and metadata
    os.makedirs(vector_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(vector_dir, "index.faiss"))

    with open(os.path.join(vector_dir, "index.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Embedded {len(texts)} code chunks from {matched_file_count} source file(s).")