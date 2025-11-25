# RAG-System
A Retrieval-Augmented Generation (RAG) system for answering questions about MkDocs documentation using embeddings and Google Gemini API, with an interactive Streamlit interface.
__________________________________________________________________________________________________________________________________________________________________________________
## Features
- Uses vector embeddings to retrieve relevant chunks from MkDocs documentation.
- Answers user queries using Google Gemini API (generative model).
- Fully running **Streamlit** interface for interactive Q&A.
__________________________________________________________________________________________________________________________________________________________________________________
## Deliverables

1. **Chunking Method**
   - Split MkDocs documentation into sections per Markdown headings and paragraphs.
   - Reason: preserves semantic context while keeping embeddings efficient.

2. **Chunks Cleaning Method**
   - Remove special characters, excessive whitespace, and frontmatter.
   
3. **Embedding Model**
   - `all-MiniLM-L6-v2` from Sentence Transformers.
   - Lightweight and fast for semantic similarity search.

4. **Vector Database**
   - ChromaDB used for storing embeddings and retrieving relevant chunks.

5. **Sample Questions**
   - *How do I add a new page to my MkDocs documentation?*
     - **Context retrieved**: Markdown snippets from `docs/` folder.
   - *How can I change the theme in MkDocs?*
     - **Context retrieved**: Theme configuration sections in `mkdocs.yml`.
__________________________________________________________________________________________________________________________________________________________________________________
## Bonus
1. Streamlit app allows users to interactively ask questions.
__________________________________________________________________________________________________________________________________________________________________________________
