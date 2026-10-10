import sys
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

COLLECTION_NAME = "TOC_Info"


def retrieve_relevant_chunks(query, n_results=3):
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]

    return list(zip(chunks, sources, distances))


def build_augmented_prompt(query, retrieved_chunks):
    context_text = "\n\n".join(
        f"[source: {source}]\n{chunk}" for chunk, source, _ in retrieved_chunks
    )

    prompt = f"""You are a helpful assistant. Answer the user's question
using ONLY the context provided below. If the answer is not contained
in the context, say "I don't have enough information to answer that."

Context:
{context_text}

Question: {query}

Answer:"""
    return prompt


def generate_answer(prompt, model="llama3.2"):
    try:
        import ollama
    except ImportError:
        print("ollama package not installed (pip install ollama).")
        return None

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 300},
        )
    except Exception as e:
        print(f"Ollama error: {e}\nIs the server running and the model pulled?")
        return None

    return response["message"]["content"]


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask a question: ").strip()

    if not query:
        print("No query provided.")
        return

    retrieved_chunks = retrieve_relevant_chunks(query, n_results=3)

    print("\nRetrieved chunks:")
    for i, (chunk, source, distance) in enumerate(retrieved_chunks, 1):
        print(f"\n[{i}] (source: {source}, distance: {distance:.4f})")
        print(chunk)
    print("\n" + "-" * 60)

    prompt = build_augmented_prompt(query, retrieved_chunks)
    answer = generate_answer(prompt)

    if answer:
        print("\nAnswer:\n" + answer)


if __name__ == "__main__":
    main()