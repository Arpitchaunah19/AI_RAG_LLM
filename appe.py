import os
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import ollama
import pickle



#-----------------------------
# CHECK VERSIONS
#-----------------------------
print("numpy version:", np.__version__)
print("faiss version:", faiss.__version__)

print("streamlit version:", st.__version__)
print("pickle version:", pickle.format_version)



# -----------------------------
# SETTINGS
# -----------------------------
DATA_FILE = "data.txt"
INDEX_PATH = "faiss_index/index.faiss"
DOC_PATH = "faiss_index/docs.pkl"

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en")

model = load_model()

# -----------------------------
# LOAD TEXT FILE
# -----------------------------
def load_documents():
    if not os.path.exists(DATA_FILE):
        st.error("❌ data.txt file not found!")
        st.stop()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return [f.read()]

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# -----------------------------
# BUILD & SAVE INDEX
# -----------------------------
def build_index():
    documents = load_documents()
    chunks = []

    for doc in documents:
        chunks.extend(chunk_text(doc))

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOC_PATH, "wb") as f:
        pickle.dump(chunks, f)

# -----------------------------
# LOAD INDEX
# -----------------------------
def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(DOC_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# -----------------------------
# SEARCH
# -----------------------------
def search(query, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖")

st.title("💬 Local RAG Chatbot")
st.write("Ask questions from your own data.txt file")

# -----------------------------
# 🧠 MEMORY INIT
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = {}

# Optional user name input
name = st.text_input("👤 Enter your name (optional)")
if name:
    st.session_state.memory["name"] = name

user_name = st.session_state.memory.get("name", "User")

# -----------------------------
# Rebuild button
# -----------------------------
if st.button("🔄 Rebuild Index"):
    build_index()
    st.success("✅ Index rebuilt successfully!")

# Load index
if os.path.exists(INDEX_PATH):
    index, chunks = load_index()
else:
    st.warning("⚠️ No index found. Click 'Rebuild Index' first.")
    st.stop()

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("🔍 Ask your question:")

if query:
    results = search(query, index, chunks)
    context = "\n".join(results)

    # -----------------------------
    # 🔥 UPDATED PROMPT (WITH MEMORY)
    # -----------------------------
    prompt = f"""
You are a helpful AI assistant.

User name: {user_name}

Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response['message']['content']

        # -----------------------------
        # SAVE CHAT HISTORY
        # -----------------------------
        st.session_state.chat_history.append((query, answer))

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.write("### 🤖 Answer")
        st.write(answer)

        st.write("### 📚 Retrieved Context")
        for r in results:
            st.write("-", r)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# -----------------------------
# 💬 CHAT HISTORY DISPLAY
# -----------------------------
st.write("---")
st.write("## 🧠 Chat History")

for q, a in reversed(st.session_state.chat_history):
    st.write(f"🧑 {user_name}: {q}")
    st.write(f"🤖 Bot: {a}")
    st.write("---")
    