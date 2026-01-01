"""
OData Query Generator using ChromaDB for persistent vector storage.

This is an alternative implementation that uses ChromaDB instead of FAISS.
The embeddings are persisted to disk, so they don't need to be regenerated on each run.
"""
from modules.data_extract import text_to_vector
from modules.langchain_data import chunk_transcript
from modules.prompts import create_summary_prompt_odata, create_chain
from modules.llm_model import create_anthropic_llm
from modules.embedding_model import setup_embedding_model
from modules.chroma_retriever import create_chroma_hybrid_retriever, chroma_retrieve
from datetime import date
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cache for expensive objects (built once, reused)
_cache = {}


def _get_cached_resources():
    """Load and cache hybrid retriever with ChromaDB, LLM, and chain (only built once)."""
    if not _cache:
        fetched_transcript = text_to_vector(os.path.join(root_path, "SAP_SL_Rule_Book_2.txt"))
        chunks = chunk_transcript(fetched_transcript, chunk_size=1000, chunk_overlap=100)
        embedding_model = setup_embedding_model()
        
        # Use ChromaDB-based hybrid retriever (BM25 + ChromaDB) for persistent storage
        _cache['hybrid_retriever'] = create_chroma_hybrid_retriever(
            chunks, 
            embedding_model,
            persist_directory=os.path.join(root_path, "chroma_db"),
            collection_name="sap_service_layer"
        )
        _cache['llm'] = create_anthropic_llm()
        _cache['summary_prompt'] = create_summary_prompt_odata()
        _cache['summary_chain'] = create_chain(_cache['llm'], _cache['summary_prompt'], verbose=False)
    return _cache


def perform_odata_query(query):
    """
    Generate OData query from natural language using ChromaDB for retrieval.
    
    Args:
        query: Natural language question about SAP Service Layer
    
    Returns:
        JSON string with entity, query parameters, and full URL
    """
    resources = _get_cached_resources()
    if resources['hybrid_retriever']:
        # Retrieve using hybrid search with ChromaDB
        relevant_docs = chroma_retrieve(query, resources['hybrid_retriever'])
        context = relevant_docs
        
        # Generate answer with today's date
        today = date.today().isoformat()
        answer = resources['summary_chain'].predict(
            context=context, 
            question=query, 
            today=today
        )
        return answer
    else:
        return "No transcript available. Please fetch the transcript first."


if __name__ == "__main__":
    print(perform_odata_query("Find Business Partner with name 'John Doe'"))
