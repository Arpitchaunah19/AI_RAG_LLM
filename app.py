from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# -----------------------------
# STEP 1: Load your text file
# -----------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# STEP 2: Better chunking
# -----------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

documents = chunk_text(text)

# -----------------------------
# STEP 3: Load embedding model
# -----------------------------
model = SentenceTransformer("BAAI/bge-small-en")

# -----------------------------
# STEP 4: Create embeddings (IMPORTANT FIX)
# -----------------------------
doc_embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

# -----------------------------
# STEP 5: FAISS index
# -----------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity
index.add(np.array(doc_embeddings))

# -----------------------------
# STEP 6: Chat loop
# -----------------------------
while True:
    query = input("\nAsk your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Query embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # -----------------------------
    # STEP 7: Search
    # -----------------------------
    k = 3
    distances, indices = index.search(np.array(query_embedding), k)

    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n".join(retrieved_docs)

    # -----------------------------
    # STEP 8: Strong prompt
    # -----------------------------
    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer.
If the answer is not in the context, say:
"I don't know based on the provided data."

Context:
{context}

Question:
{query}

Answer:
"""

    # -----------------------------
    # STEP 9: Ollama
    # -----------------------------
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:\n", response['message']['content'])