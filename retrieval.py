from .ingestion import DocumentChunkWithMetadata
from .main import index
from .embedding import get_embeddings

def retieve_and_reconstruct(query, index, k = 10):
    query_embeddings = get_embeddings(query).cpu().numpy().tolist()
    results = index.query(queries=query_embeddings, top_k=k)
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
