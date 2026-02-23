from fastembed import TextEmbedding
from typing import List

_model = None

def get_embedding_model() -> TextEmbedding:
    global _model
    if _model is None:
        print("⏳ Loading embedding model (first time only)...")
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("✅ Embedding model loaded")
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]

def embed_query(query: str) -> List[float]:
    model = get_embedding_model()
    embeddings = list(model.embed([query]))
    return embeddings[0].tolist()
