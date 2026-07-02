from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
import os
import time

system_prompt = """
You are an AI code navigator and programming assistant.
Your goal is to answer with **precise details** based ONLY on the provided code context.
If answering about code:
- Always mention **file names** and **line numbers** where possible.
- Refer to **function names, variables, and classes** exactly as they appear.
- Use **step-by-step explanations** for complex code.
- If something is unclear or outside the given files, say exactly: "I don't know based on the provided context."

Do NOT make up file names, functions, or features that are not in the given code.
Your tone should be clear, concise, and technical.
"""

llm = ChatOpenAI(
    model = "llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)
chat_history = []

# --- Retry settings for the free/rate-limited endpoint ---
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 2  # backoff: 2s, 4s, 8s


def get_chat_history() -> str:
    return "\n".join(reversed([
        f"^Q: {q}\n^A: {a}" for q, a in chat_history
    ]))


def _invoke_with_retry(messages):
    """
    Calls the LLM with retry + exponential backoff, specifically to handle
    the rate limits on Together AI's free-tier model. Raises the last
    exception if all retries are exhausted, so the caller decides what
    message to show the user.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY_SECONDS * (2 ** attempt))
                continue
        except (APIConnectionError, APITimeoutError) as e:
            # Transient network issues are also worth a retry
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY_SECONDS * (2 ** attempt))
                continue
    raise last_error


def answer_question(question: str, similar_chunks: list) -> str:
    # Format code context, including the source file for each chunk so the
    # model can actually cite file names (the system prompt requires this,
    # but the file name was previously dropped here, so the model never had
    # access to it).
    context_blocks = []
    for chunk in similar_chunks:
        code = chunk.get("code", "")
        if not code:
            continue
        file = chunk.get("file", "unknown file")
        context_blocks.append(f"File: {file}\n```\n{code}\n```")

    context = "\n\n---\n\n".join(context_blocks)

    if not context.strip():
        return "No relevant code context found to answer your question."

    prompt = f"""Code Context:
{context}

---

Question: {question}

Note: some of the code blocks above may come from the same function or
overlap with each other (e.g. a class shown alongside one of its own
methods) — this is a retrieval artifact, not evidence that the codebase
contains duplicate definitions. Only report duplication if the same
function name appears in DIFFERENT files, or with a different signature in
the same file.

Provide a clear and helpful answer based on the code above.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    try:
        response = _invoke_with_retry(messages)
        answer = response.content.strip()
        chat_history.append((question, answer))
        return answer
    except RateLimitError:
        return (
            "⚠️ The free model is currently rate-limited (too many requests "
            "in a short time). Please wait a few seconds and try again."
        )
    except (APIConnectionError, APITimeoutError):
        return (
            "⚠️ Couldn't reach the model right now (connection/timeout issue). "
            "Please check your connection and try again."
        )
    except APIError as e:
        return f"⚠️ The model API returned an error: {str(e)}"
    except Exception as e:
        return f"LLM invocation failed: {str(e)}"