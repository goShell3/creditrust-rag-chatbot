# 🧠 CrediTrust RAG System

A Retrieval-Augmented Generation (RAG) system designed to help analysts and customers query consumer financial complaints from the CFPB dataset using natural language. Built with `LangChain`, `ChromaDB`, `Sentence Transformers`, and `Streamlit`.

---

## 🚀 Features

* End-to-end pipeline from raw complaint data to interactive semantic search.
* Data cleaning, narrative filtering, and chunking for optimized embeddings.
* Vector search with `ChromaDB` using metadata for traceability.
* Question-answering using `MiniLM` for embedding and `Mistral-7B-Instruct` for LLM generation.
* Clean, interactive chat interface built with `Streamlit`.
* Modular design with reproducible scripts and Docker support.

---

## 📂 Project Structure

```
rag_project/
├── app.py                         # Streamlit interface (Task 4)
├── Dockerfile                     # Docker container setup
├── requirements.txt               # Python dependencies
├── data/
│   └── filtered_complaints.csv    # Cleaned data ready for chunking
├── vector_store/
│   └── chroma_db/                 # Persistent ChromaDB store
├── notebooks/
│   └── eda_preprocessing.ipynb    # Task 1: EDA + cleaning notebook
├── src/
│   ├── chunking_embedding_indexing.py  # Task 2: chunk + embed + index
│   ├── rag_pipeline.py            # Task 3: retrieval + generation logic
│   └── utils.py                   # Optional helpers
└── .dockerignore
```

---

## 🛠️ Setup Instructions

### 1. 📦 Install Requirements

```bash
pip install -r requirements.txt
```

### 2. 🧹 Task 1: Clean & Filter the Dataset

```bash
jupyter notebook notebooks/eda_preprocessing.ipynb
```

### 3. 🔎 Task 2: Chunk + Embed + Index

```bash
python src/chunking_embedding_indexing.py
```

### 4. 🧠 Task 3: RAG Pipeline (retrieval + generation)

```bash
python src/rag_pipeline.py
```

### 5. 💬 Task 4: Launch Streamlit Chat UI

```bash
streamlit run app.py
```

---

## 🐳 Docker Usage

### Build the container:

```bash
docker build -t creditrust-rag .
```

### Run the app:

```bash
docker run -p 8501:8501 creditrust-rag
```

> The app will be available at `http://localhost:8501`

---

## 🧪 Evaluation

Check `evaluation.md` or the final report for:

* A table of 5–10 test questions
* Quality scores from 1–5
* Example retrieved chunks
* Generation analysis

---

## 🤝 Credits

* CFPB Public Complaint Dataset
* Hugging Face Transformers
* LangChain
* ChromaDB
* Streamlit

---

## 📌 Future Improvements

* Add support for feedback-driven retraining
* Swap in larger LLMs via LangChain integration (e.g., Llama 3)
* Add support for citation highlighting
* Deploy as a cloud API
