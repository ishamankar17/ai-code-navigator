# 🧠 AI Codebase Navigator

An AI-powered app that helps you understand any codebase by asking questions in plain English. Upload a GitHub repository or paste code, and the AI answers using the actual code with a Retrieval-Augmented Generation (RAG) pipeline.

---
 
## 📸 Screenshots
 
> *(Add your screenshots here)*
 
<!-- Example:
![Parsing a repo](<img width="1522" height="916" alt="image" src="https://github.com/user-attachments/assets/53b9a133-2da0-4e91-a929-5949e2654909" />
)
![Asking a question](screenshots/ask.png)
![Retrieved chunks panel](screenshots/chunks.png)
-->
 
---

## ✨ Features

- Upload a GitHub repository or paste code
- Supports Python, JavaScript, TypeScript, and Java
- Finds relevant code using semantic search with FAISS
- Answers questions based only on the uploaded code
- Shows the code snippets used to generate each answer
- Keeps chat history during the session

---

## 🛠️ How It Works

```
Repository / Code
        ↓
Extract functions and classes
        ↓
Create embeddings and store in FAISS
        ↓
Find relevant code for the question
        ↓
Generate an answer using GPT-4o-mini
```

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/ishamankar17/ai-code-navigator
cd ai-code-navigator
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add your API key

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

### Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📂 Project Structure

```
ai-code-navigator/
├── app.py
├── embedder.py
├── retriever.py
├── chatbot.py
├── requirements.txt
├── tests/
└── .gitignore
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 💻 Tech Stack

- Python
- Streamlit
- OpenAI GPT-4o-mini
- FAISS
- sentence-transformers
- Python AST & Regex

---

## ⚠️ Limitations

- JavaScript, TypeScript, and Java parsing uses regex, so very complex code may not be parsed perfectly.
- Large repositories may take longer to process.
- Answers depend on the most relevant retrieved code snippets.

---

