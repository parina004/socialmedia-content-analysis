"""
RAG Knowledge Base Builder

Reads every .txt file under rag/docs/ (all subfolders),
chunks them into ~300-word pieces with ~50-word overlap,
embeds with sentence-transformers/all-MiniLM-L6-v2,
and stores the vectors in ChromaDB at rag/chroma_db/.

Run once before using the system:
    python rag/build_vectorstore.py
"""

import shutil
from pathlib import Path

## LangChain tools: TextLoader reads .txt files, HuggingFaceEmbeddings converts text to vectors,
## Chroma is the local vector database, RecursiveCharacterTextSplitter breaks documents into chunks
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Where our knowledge base documents live (all the .txt research files)
DOCS_DIR    = Path(__file__).parent / "docs"
## Where ChromaDB will save the vector index on disk
CHROMA_DIR  = Path(__file__).parent / "chroma_db"
## The embedding model — free, runs on CPU, produces 384-dimensional semantic vectors
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

## ~300 words * 6 chars/word = 1800 chars | ~50 words overlap = 300 chars
## Overlap ensures a concept that spans a chunk boundary is not cut in half
CHUNK_SIZE    = 1800
CHUNK_OVERLAP = 300


def load_all_docs():
    ## Recursively find every .txt file inside rag/docs/ and all its subfolders
    txt_files = sorted(DOCS_DIR.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {DOCS_DIR}")

    print(f"Found {len(txt_files)} documents across {DOCS_DIR}")
    docs = []
    for path in txt_files:
        ## Read the file as a LangChain Document object
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            ## Tag each document with its relative path and parent folder name
            ## so the retriever can later tell the LLM which source a chunk came from
            doc.metadata["source"]   = str(path.relative_to(DOCS_DIR))
            doc.metadata["category"] = path.parent.name
        docs.extend(loaded)
        print(f"  Loaded -> {path.relative_to(DOCS_DIR)}")
    return docs


def build():
    print("\n RAG Knowledge Base Builder \n")

    ## Step 1: Load all .txt documents from rag/docs/
    docs = load_all_docs()
    print(f"\nTotal documents loaded: {len(docs)}")

    ## Step 2: Split each document into overlapping ~300-word chunks
    ## RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    ## then line breaks, then sentences — to keep chunks semantically coherent
    print(f"\nSplitting into ~300-word chunks (overlap ~50 words)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    ## Step 3: Load the embedding model that will convert text chunks into vectors
    ## First run downloads ~80MB from HuggingFace — after that it is cached locally
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("(First run downloads ~80MB — subsequent runs are instant)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},        ## run on CPU, no GPU needed
        encode_kwargs={"normalize_embeddings": True},  ## normalise so cosine similarity works correctly
    )

    ## Step 4: Wipe any existing ChromaDB so we start fresh
    ## This prevents duplicate entries if you re-run after adding new documents
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"\nCleared existing ChromaDB at {CHROMA_DIR}")

    ## Step 5: Embed all chunks and write them into ChromaDB on disk
    ## After this step, rag/chroma_db/ contains the full searchable vector index
    print(f"\nEmbedding and storing {len(chunks)} chunks in ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"\ Done ")
    print(f"{len(chunks)} chunks from {len(docs)} documents indexed.")
    print(f"ChromaDB saved to: {CHROMA_DIR}")
    print("\nRun a quick test:")
    print("  python rag/retriever.py")


if __name__ == "__main__":
    build()
