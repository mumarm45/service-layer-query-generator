from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS


def create_hybrid_retriever(chunks, embedding_model, bm25_weight=0.5, faiss_weight=0.5, k=7):
    """
    Create a hybrid retriever combining BM25 (keyword) and FAISS (semantic) search.
    
    Args:
        chunks: List of text chunks
        embedding_model: Embedding model for FAISS
        bm25_weight: Weight for BM25 results (0-1)
        faiss_weight: Weight for FAISS results (0-1)
        k: Number of results to return
    
    Returns:
        EnsembleRetriever combining both methods
    """
    # Create BM25 retriever for keyword matching (exact entity names, field names)
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = k
    
    # Create FAISS retriever for semantic search
    faiss_index = FAISS.from_texts(chunks, embedding_model)
    faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": k})
    
    # Combine both retrievers
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[bm25_weight, faiss_weight]
    )
    
    return hybrid_retriever


def hybrid_retrieve(query, hybrid_retriever, k=7):
    """
    Retrieve documents using hybrid search.
    
    Args:
        query: Search query
        hybrid_retriever: EnsembleRetriever instance
        k: Number of results
    
    Returns:
        List of relevant documents
    """
    results = hybrid_retriever.invoke(query)
    return results[:k]
