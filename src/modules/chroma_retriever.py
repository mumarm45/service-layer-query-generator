from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import os

# Default persist directory for ChromaDB
DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "chroma_db"
)


def create_chroma_vectorstore(chunks, embedding_model, persist_directory=None, collection_name="sap_service_layer"):
    """
    Create or load a ChromaDB vector store with persistent storage.
    
    Args:
        chunks: List of text chunks to embed (only used if collection doesn't exist)
        embedding_model: Embedding model for vectorization
        persist_directory: Directory to persist the ChromaDB data
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        Chroma: ChromaDB vector store instance
    """
    if persist_directory is None:
        persist_directory = DEFAULT_PERSIST_DIR
    
    # Check if collection already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # Load existing collection
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
        print(f"Loaded existing ChromaDB collection from {persist_directory}")
    else:
        # Create new collection with embeddings
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        print(f"Created new ChromaDB collection at {persist_directory}")
    
    return vectorstore


def create_chroma_hybrid_retriever(chunks, embedding_model, persist_directory=None, 
                                    collection_name="sap_service_layer",
                                    bm25_weight=0.5, chroma_weight=0.5, k=7):
    """
    Create a hybrid retriever combining BM25 (keyword) and ChromaDB (semantic) search.
    
    Args:
        chunks: List of text chunks
        embedding_model: Embedding model for ChromaDB
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the ChromaDB collection
        bm25_weight: Weight for BM25 results (0-1)
        chroma_weight: Weight for ChromaDB results (0-1)
        k: Number of results to return
    
    Returns:
        EnsembleRetriever combining both methods
    """
    # Create BM25 retriever for keyword matching
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = k
    
    # Create or load ChromaDB vector store
    chroma_vectorstore = create_chroma_vectorstore(
        chunks, embedding_model, persist_directory, collection_name
    )
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": k})
    
    # Combine both retrievers
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[bm25_weight, chroma_weight]
    )
    
    return hybrid_retriever


def chroma_retrieve(query, hybrid_retriever, k=7):
    """
    Retrieve documents using hybrid search with ChromaDB.
    
    Args:
        query: Search query
        hybrid_retriever: EnsembleRetriever instance
        k: Number of results
    
    Returns:
        List of relevant documents
    """
    results = hybrid_retriever.invoke(query)
    return results[:k]


def get_chroma_vectorstore(embedding_model, persist_directory=None, collection_name="sap_service_layer"):
    """
    Get an existing ChromaDB vector store (for querying only).
    
    Args:
        embedding_model: Embedding model for vectorization
        persist_directory: Directory where ChromaDB data is persisted
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        Chroma: ChromaDB vector store instance or None if not found
    """
    if persist_directory is None:
        persist_directory = DEFAULT_PERSIST_DIR
    
    if not os.path.exists(persist_directory):
        return None
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )


def rebuild_chroma_collection(chunks, embedding_model, persist_directory=None, collection_name="sap_service_layer"):
    """
    Rebuild the ChromaDB collection from scratch (useful when source data changes).
    
    Args:
        chunks: List of text chunks to embed
        embedding_model: Embedding model for vectorization
        persist_directory: Directory to persist the ChromaDB data
        collection_name: Name of the collection in ChromaDB
    
    Returns:
        Chroma: New ChromaDB vector store instance
    """
    import shutil
    
    if persist_directory is None:
        persist_directory = DEFAULT_PERSIST_DIR
    
    # Remove existing collection
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Removed existing ChromaDB collection at {persist_directory}")
    
    # Create new collection
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    print(f"Rebuilt ChromaDB collection at {persist_directory}")
    
    return vectorstore
