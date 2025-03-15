"""
This module handles the semantic search and document reconstruction functionality for the QA system.
It provides capabilities to:
- Retrieve relevant document chunks based on semantic similarity to a query
- Convert queries into embeddings using llama-text-embed-v2 model
- Search for similar content in the Pinecone vector database
- Reconstruct coherent document segments from retrieved chunks

The module uses:
- Pinecone for vector similarity search
- llama-text-embed-v2 for generating embeddings
- Custom document reconstruction logic to maintain context coherence

Environment variables required:
- PINECONE_API_KEY: API key for Pinecone service
- PINECONE_INDEX_NAME: Name of the Pinecone index to use
"""

from ingestion import DocumentChunkWithMetadata
from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

def retieve_and_reconstruct(query, k = 8):
    """
    Retrieves relevant document chunks based on a query and reconstructs them into a single document.
    
    Args:
        query (str): The search query to find relevant document chunks.
        k (int, optional): The number of top chunks to retrieve. Defaults to 8.
        
    Returns:
        str: The reconstructed document containing concatenated relevant chunks.
        
    Note:
        The function uses the llama-text-embed-v2 model to create query embeddings
        and retrieves chunks from a Pinecone vector database.
    """
    query_embedding = pc.inference.embed(
        model = "llama-text-embed-v2",
        inputs = [query],
        parameters = {
            'input_type': 'query'
        }
    )
    results = index.query(
        vector=query_embedding[0].values,
        top_k=k,
        include_values=False,
        include_metadata=True
    )
    chunks_with_metadata = []
    for result in results['matches']:
        metadata = result['metadata']
        chunk = DocumentChunkWithMetadata(
            text = metadata['text'],
            document_id = metadata['document_id'],
            chunk_id = metadata['chunk_id'],
            start = metadata['start'],
            end = metadata['end']
        )
        chunks_with_metadata.append(chunk)
    reconstructed_document = reconstruct_document(chunks_with_metadata)
    return reconstructed_document

def reconstruct_document(chunks_with_metadata):
    """
    Reconstructs a document from chunks by sorting them based on their start positions
    and concatenating their text.
    
    Args:
        chunks_with_metadata (list[DocumentChunkWithMetadata]): List of document chunks
            with their associated metadata including position information.
            
    Returns:
        str: The reconstructed document text with chunks arranged in proper order.
    """
    chunks_with_metadata = sorted(chunks_with_metadata, key=lambda x: x.start)
    reconstructed_document = "".join([chunk.text for chunk in chunks_with_metadata])
    return reconstructed_document
