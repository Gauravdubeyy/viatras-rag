import chromadb
from typing import List
from rag.embedder import embed_texts, embed_query

COLLECTION_NAME = "humanic_manual"
CHROMA_PATH = "./chroma_store"

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def index_chunks(chunks: List[dict]) -> None:
    collection = get_collection()
    if collection.count() > 0:
        print(f"✅ ChromaDB already has {collection.count()} chunks — skipping re-indexing")
        return

    print(f"⏳ Indexing {len(chunks)} chunks into ChromaDB...")
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embed_texts([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[{"section": c["section"], "subsection": c.get("subsection", "")} for c in chunks]
    )
    print(f"✅ Indexed {len(chunks)} chunks successfully")

def search_chunks(query: str, top_k: int = 3) -> List[str]:
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, collection.count())
    )
    return results["documents"][0] if results["documents"] else []
