import streamlit as st
from chatbot import answer_question
from retriever import retrieve_similar_code
from embedder import embed_code_chunks
from git import Repo
import os
import shutil
import tempfile
import stat
import uuid


st.title("🧠 AI Codebase Navigator")
st.caption("Portfolio project - built to demonstrate RAG + LLM integration. May be rate-limited on the free API tier.")

st.markdown("""
Analyze source code using AI.

Upload a repository or paste code, then ask questions such as:
- Explain this code
- Find bugs
- Suggest improvements
- Generate documentation
""")

# --- Persistent state across reruns ---
if "parsed" not in st.session_state:
    st.session_state.parsed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question": ..., "answer": ..., "chunks": [...]}
if "session_id" not in st.session_state:
    # Unique per browser session, so two users (or two repos parsed back-to-back
    # in the same session) each get their own FAISS index instead of silently
    # overwriting a shared "vector_store" folder.
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "question" not in st.session_state:
    st.session_state.question = ""

VECTOR_DIR = os.path.join(tempfile.gettempdir(), f"vector_store_{st.session_state.session_id}")


# --- Helper: fix Windows "Access is denied" errors when deleting .git folders ---
def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# --- Map dropdown language -> correct file extension for pasted-code mode ---
LANGUAGE_TO_EXT = {
    "Python": ".py",
    "JavaScript": ".js",
    "TypeScript": ".ts",
    "Java": ".java",
}


# --- Sidebar: show chat history ---
with st.sidebar:
    st.subheader("Chat History")
    if not st.session_state.chat_history:
        st.caption("No questions asked yet.")
    for entry in st.session_state.chat_history:
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")
        st.markdown("---")


# --- 1. Let user choose: GitHub repo OR pasted code ---
input_mode = st.radio("How do you want to provide code?", ["GitHub Repo URL", "Paste Code"])

# --- 2. Let user choose the language ---
language = st.selectbox("Select language", ["Python", "JavaScript", "TypeScript", "Java"])

if input_mode == "GitHub Repo URL":
    repo_url = st.text_input("Paste GitHub Repository URL")
    source_code = None
else:
    repo_url = None
    source_code = st.text_area("Paste your code here", height=300)

if st.button("Parse"):
    if input_mode == "GitHub Repo URL" and repo_url:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        clone_path = os.path.join(tempfile.gettempdir(), repo_name)
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path, onexc=remove_readonly)
        try:
            with st.spinner("Cloning and parsing repo..."):
                Repo.clone_from(repo_url, clone_path)
                embed_code_chunks(clone_path, repo_name, vector_dir=VECTOR_DIR)
            st.session_state.parsed = True
            st.success("Repo cloned and parsed as: " + language)
        except Exception as e:
            st.error(f"Failed to parse repo: {e}")

    elif input_mode == "Paste Code" and source_code:
        temp_dir = tempfile.mkdtemp()
        # Use the extension that matches the selected language, so the embedder
        # routes the file to the correct extractor (Python/JS-TS/Java) instead
        # of always treating it as Python.
        file_ext = LANGUAGE_TO_EXT.get(language, ".py")
        pasted_file_path = os.path.join(temp_dir, f"pasted_code{file_ext}")
        with open(pasted_file_path, "w", encoding="utf-8") as f:
            f.write(source_code)
        try:
            embed_code_chunks(temp_dir, "pasted_code", vector_dir=VECTOR_DIR)
            st.session_state.parsed = True
            st.success("Parsed pasted code as: " + language)
        except Exception as e:
            st.error(f"Failed to parse code: {e}")
    else:
        st.warning("Please provide a repo URL or paste some code first.")


# --- Chat section: ALWAYS visible, but warns if nothing parsed yet ---
st.markdown("---")
st.subheader("Ask a question about the code")

if not st.session_state.parsed:
    st.caption("Parse a repo or paste some code above first, then ask your question here.")

# --- Example question quick-buttons ---
st.subheader("Example Questions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Explain this code"):
        st.session_state.question = "Explain this code."
    if st.button("Find Bugs"):
        st.session_state.question = "Find possible bugs."

with col2:
    if st.button("Suggest Improvements"):
        st.session_state.question = "Suggest improvements."
    if st.button("Generate Documentation"):
        st.session_state.question = "Generate documentation."

question = st.text_input(
    "Ask something about this repo or code...",
    key="question",
)

if st.button("Ask") and question:
    if not st.session_state.parsed:
        st.warning("Please parse a repo or paste code first.")
    else:
        with st.spinner("Retrieving relevant code and generating answer..."):
            similar_chunks = retrieve_similar_code(question, k=5, vector_dir=VECTOR_DIR)
            answer = answer_question(question, similar_chunks)
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "chunks": similar_chunks,
        })
        st.write(f"**A:** {answer}")

        # --- Retrieval visibility: show exactly what was retrieved and used ---
        # Sorted by distance (ascending = closer match = more relevant).
        with st.expander(f"🔍 Show {len(similar_chunks)} retrieved chunk(s) used for this answer"):
            if not similar_chunks:
                st.caption("No chunks were retrieved.")
            for idx, chunk in enumerate(similar_chunks, start=1):
                file = chunk.get("file", "unknown file")
                distance = chunk.get("distance")
                dist_label = (
                    f" — distance: {distance:.3f} (lower = closer match)"
                    if distance is not None
                    else ""
                )
                st.markdown(f"**{idx}. `{file}`**{dist_label}")
                code = chunk.get("code", "")
                lang_hint = (
                    "java" if file.endswith(".java")
                    else "typescript" if file.endswith((".ts", ".tsx"))
                    else "javascript" if file.endswith((".js", ".jsx"))
                    else "python"
                )
                st.code(code, language=lang_hint)