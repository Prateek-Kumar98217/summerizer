from ingestion import DocumentChunkWithMetadata
from embedding import get_embeddings
from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

def retieve_and_reconstruct(query, k = 8):
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
    chunks_with_metadata = sorted(chunks_with_metadata, key=lambda x: x.start)
    reconstructed_document = "".join([chunk.text for chunk in chunks_with_metadata])
    return reconstructed_document
