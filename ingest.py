import argparse
from pathlib import Path
import subprocess
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Clone repo
def clone_repo(repo_url: str, dest: Path):
    if dest.exists():
        return dest
    print(f"Cloning {repo_url} -> {dest}")
    subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dest)])
    return dest

# Find Markdown files
def find_markdown_files(root: Path, docs_path: str):
    docs_dir = root / docs_path
    if not docs_dir.exists():
        raise FileNotFoundError(f"{docs_dir} not found")
    return list(docs_dir.rglob("*.md"))

# Clean markdown
def clean_markdown_text(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"[*_~]", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()

# Split into chunks
def markdown_to_chunks(text: str, source_file: str, chunk_size: int = 500, chunk_overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        heading = chunk_text.split("\n")[0] if "\n" in chunk_text else source_file
        chunks.append({
            "text": chunk_text,
            "file": source_file,
            "heading": heading
        })
        start += chunk_size - chunk_overlap
    return chunks

def main(args):
    # Use absolute path for Chroma
    persist_path = Path(args.persist_dir).absolute()

    if args.local_path:
        repo_root = Path(args.local_path)
    else:
        repo_root = clone_repo(args.repo, Path("./mkdocs_repo"))

    md_files = find_markdown_files(repo_root, args.docs_path)
    print(f"Found {len(md_files)} markdown files under {args.docs_path}")

    # Use PersistentClient with absolute path
    client = PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(args.collection_name)

    model = SentenceTransformer(args.embedding_model)

    all_texts, metadatas, ids = [], [], []
    idx = 0
    for md_path in tqdm(md_files, desc="Processing Markdown files"):
        text = md_path.read_text(encoding="utf-8")
        cleaned = clean_markdown_text(text)
        chunks = markdown_to_chunks(cleaned, str(md_path))
        for c in chunks:
            all_texts.append(c['text'])
            metadatas.append({'file': c['file'], 'heading': c['heading']})
            ids.append(f"{md_path.name}::chunk_{idx}")
            idx += 1

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

    print("Upserting into Chroma...")
    collection.add(
        ids=ids,
        documents=all_texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print("Done. Chroma DB persisted at:", persist_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="https://github.com/mkdocs/mkdocs.git")
    parser.add_argument("--local-path", type=str, default=None)
    parser.add_argument("--docs-path", type=str, default="docs")
    parser.add_argument("--persist-dir", type=str, default="./chroma_db")
    parser.add_argument("--collection-name", type=str, default="mkdocs")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(args)
