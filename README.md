# 🧠 AI Codebase Navigator

An AI-powered app that helps you understand any codebase by asking questions in plain English. Upload a GitHub repository or paste code, and the AI answers using the actual code with a Retrieval-Augmented Generation (RAG) pipeline.

---
 
## 📸 Screenshots
 
### Parsing a Repository

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/884d902a-7c62-43e7-916d-7397b02323bc" />

### Asking a Question

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/40c3590b-4b65-4ce1-a25c-84296d4f990a" />


### Retrieved Chunks Panel

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/9f02b009-3400-4ba3-adba-2b7866bd6929" />


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

