import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import json

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "mkdocs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
GEMINI_API_KEY = "AIzaSyAbVkgr0qaPV2RISNOoKszcTta_dMI-SHs"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

client = PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

def retrieve_relevant_chunks(query: str, top_k: int = TOP_K):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return "\n\n".join(results['documents'][0])

def ask_gemini(query: str, context: str):
    prompt_text = (
        f"Answer the question based on the following MkDocs documentation:\n\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    data = response.json()
    try:
        return data['candidates'][0]['content']['parts'][0]['text']
    except KeyError:
        return "API returned empty response or unexpected format"

def query_rag(query: str):
    context = retrieve_relevant_chunks(query)
    return ask_gemini(query, context)

st.title("MkDocs RAG Assistant")
user_question = st.text_input("Enter your MkDocs question:")

if st.button("Ask"):
    if user_question.strip():
        with st.spinner("Fetching answer from Gemini..."):
            answer = query_rag(user_question)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question.")

