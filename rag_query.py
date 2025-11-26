import streamlit as st
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import requests
import json

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "mkdocs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

GEMINI_API_KEY = "API-KEY"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

client = PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)
embed_model = SentenceTransformer(EMBEDDING_MODEL)

def retrieve_relevant_chunks(query: str):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )
    return "\n\n".join(results["documents"][0])

def ask_gemini(query: str, context: str):
    system_instructions = (
        "You are an MkDocs documentation assistant.\n"
        "You MUST answer strictly based on the provided documentation context.\n"
        "If the user asks anything outside the MkDocs documentation, reply ONLY with:\n"
        "'I cannot answer this question because it is outside the MkDocs documentation.'\n"
        "Do NOT use external knowledge. Do NOT guess or hallucinate.\n"
        "If the answer cannot be found in the context, say:\n"
        "'The documentation does not contain the answer.'\n"
        "ALWAYS stay within the scope of MkDocs.\n\n"
    )

    full_prompt = (
        f"{system_instructions}"
        f"Context:\n{context}\n\n"
        f"User question:\n{query}\n\n"
        "Answer using ONLY the context above."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": full_prompt}]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return "Unexpected API response format."

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
