import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

DOCUMENTS_FOLDER = "documents"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "TOC_Info"

CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks


def load_documents(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found")

    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append((filename, text))
    return documents


def chunk_text(text, filename):
    # Split by paragraph first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge small paragraphs / split large ones into size-limited chunks
    pieces = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                pieces.append(current)
            # paragraph itself too long -> slice with overlap
            if len(para) > CHUNK_SIZE:
                start = 0
                while start < len(para):
                    pieces.append(para[start:start + CHUNK_SIZE])
                    start += CHUNK_SIZE - CHUNK_OVERLAP
                current = ""
            else:
                current = para
    if current:
        pieces.append(current)

    return [
        {"id": f"{filename}_chunk_{i}", "text": piece, "source": filename}
        for i, piece in enumerate(pieces)
    ]


def build_vector_db():
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    if COLLECTION_NAME in {collection.name for collection in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    documents = load_documents(DOCUMENTS_FOLDER)
    all_chunks = []
    for filename, text in documents:
        all_chunks.extend(chunk_text(text, filename))

    if not all_chunks:
        print("No .txt content found. Nothing to index.")
        return

    print(f"Loaded {len(documents)} documents -> split into {len(all_chunks)} chunks")

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )

    print(f"Vector database built successfully at {DB_PATH}")
    print(f"Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    build_vector_db()

    