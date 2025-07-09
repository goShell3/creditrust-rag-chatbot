import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from transformers import pipeline

# === Load Embedding Model ===
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Load Vector Store ===
persist_path = "vector_store/chroma_db"
client = chromadb.PersistentClient(path=persist_path)
collection = client.get_collection("complaint_chunks")

# === Load LLM ===
generator = pipeline("text-generation", 
                    model="mistralai/Mistral-7B-Instruct-v0.1", 
                    use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"))  

# Prompt Template 
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""

# === RAG Function ===
def answer_question(question: str, k: int = 5) -> str:
    # Step 1: Embed the question
    question_embedding = embedding_model.encode([question])[0]

    # Step 2: Retrieve top-k relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    retrieved_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Step 3: Construct context
    context = "\n\n".join(retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Step 4: Generate response
    response = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]['generated_text']
    return {
        "question": question,
        "response": response,
        "sources": retrieved_chunks[:2],  # first 2 chunks
        "metadata": metadatas[:2]
    }

# === Example Usage ===
if __name__ == "__main__":
    sample_question = "What issues do customers face with money transfers?"
    result = answer_question(sample_question)
    print("Question:", result["question"])
    print("\nAnswer:\n", result["response"])
    print("\nSources:\n", result["sources"])
    print("\nMetadata:\n", result["metadata"])
