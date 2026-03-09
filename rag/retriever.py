"""
RAG Retriever

Loads ChromaDB and exposes a single function:
    retrieve(query, k=5) -> str

Called by the Agent and LLM layer at runtime to get research context
before generating forensic or virality reports.
"""

from pathlib import Path

## Same embedding model used during build — must match so vectors are comparable
from langchain_community.embeddings import HuggingFaceEmbeddings
## Chroma loads the pre-built index from disk instead of re-building it
from langchain_community.vectorstores import Chroma

## Path to the ChromaDB folder that build_vectorstore.py created
CHROMA_DIR  = Path(__file__).parent / "chroma_db"
## Must be the exact same model used when we built the vectorstore
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

## Singleton — loaded once into memory on first call, reused for every retrieve() after that
## Avoids reloading the ~80MB model on every single agent tool call
_vectorstore = None


def _load():
    global _vectorstore
    ## If already loaded, return it immediately — no work needed
    if _vectorstore is not None:
        return _vectorstore

    ## If ChromaDB folder does not exist, the user forgot to run build_vectorstore.py first
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}.\n"
            "Run this first:  python rag/build_vectorstore.py"
        )

    ## Load the same embedding model used at build time
    ## This converts the incoming query into a vector so ChromaDB can compare it
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ## Connect to the existing ChromaDB on disk — no re-embedding, just loading the index
    _vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    return _vectorstore


def retrieve(query: str, k: int = 5) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.

    Returns a single formatted string ready to drop into an LLM prompt
    as {rag_context}.

    Example:
        context = retrieve("deepfake frequency artifacts")
        # -> "[1] (Source: deepfake_detection/2020_deepfake.txt)\n..."
    """
    ## Load (or reuse) the vectorstore
    vs = _load()
    ## Embed the query and find the k most semantically similar chunks using cosine similarity
    results = vs.similarity_search(query, k=k)

    ## If nothing useful was found, return a safe fallback string for the LLM
    if not results:
        return "No relevant context found in knowledge base."

    ## Format each retrieved chunk with its index number and source file path
    ## This lets the LLM know which document the information came from
    parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (Source: {source})\n{doc.page_content.strip()}")

    ## Join all chunks with a separator so the LLM can clearly see where one chunk ends and another begins
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    ## Quick manual test — run with: python rag/retriever.py
    test_queries = [
        "deepfake detection frequency artifacts DCT SRM",
        "virality prediction features engagement",
        "AI generated video diffusion model detection",
        "cascaded two-stage synthetic media pipeline",
        "fair features virality model subscriber count",
    ]

    print(" RAG Retriever Test \n")
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 60)
        result = retrieve(query, k=2)
        ## Print first 400 chars of result to keep output readable
        preview = result[:400] + "..." if len(result) > 400 else result
        print(preview)
        print()
