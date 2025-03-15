"""
This module handles the document ingestion pipeline for the document QA system.
It performs the following key operations:
- Loading PDF documents using LangChain's PyPDFLoader
- Splitting documents into manageable chunks with metadata
- Computing embeddings using llama-text-embed-v2 model
- Storing document chunks and their embeddings in Pinecone vector database
- Tracking processed documents to avoid duplicate processing

Environment variables required:
- PINECONE_API_KEY: API key for Pinecone service
- PINECONE_INDEX_NAME: Name of the Pinecone index to use

The module supports both single document processing and batch processing of
all PDF documents in a specified directory.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from math import ceil
from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

class DocumentChunkWithMetadata:
    """
    A class to represent a chunk of a document with associated metadata.
    
    Attributes:
        text (str): The text content of the chunk, encoded in UTF-8
        document_id (str): Unique identifier for the source document
        chunk_id (int): Identifier for this specific chunk within the document
        start (int): Starting position of this chunk in the original document
        end (int): Ending position of this chunk in the original document
    """
    def __init__(self, text, document_id, chunk_id, start, end):
        self.text = text.encode('utf-8', 'ignore').decode('utf-8')
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.start = start
        self.end = end

def split_document_with_metadata(documents, document_id, chunk_size = 500, chunk_overlap = 50):
    """
    Splits a document into chunks with associated metadata.
    
    Args:
        documents (list): List of document objects to be split
        document_id (str): Unique identifier for the document
        chunk_size (int, optional): Size of each chunk in characters. Defaults to 500.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Defaults to 50.
    
    Returns:
        list[DocumentChunkWithMetadata]: List of document chunks with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks")
    chunk_with_metadata =[]
    for chunk_id, chunk in enumerate(chunks):
        start = chunk_id * (chunk_size - chunk_overlap)
        end = start + len(chunk.page_content)
        chunk_with_metadata.append(DocumentChunkWithMetadata(
            text = chunk.page_content,
            document_id = document_id,
            chunk_id = chunk_id,
            start = start,
            end = end
        ))
    return chunk_with_metadata

def load_processed_document_record():
    """
    Loads the record of previously processed documents from a file.
    
    Returns:
        set: Set of document IDs that have already been processed
    """
    if os.path.exists("processed_document_record.txt"):
        with open("processed_document_record.txt", "r") as file:
            processed_document_record = set(file.read().split("\n"))
            return processed_document_record
    return set()

def save_processed_document_record(document_id):
    """
    Saves a document ID to the processed document record file.
    
    Args:
        document_id (str): ID of the document that has been processed
    """
    with open("processed_document_record.txt", "a") as file:
        file.write(f"{document_id}\n")

def store_embeddings(chunks_with_metadata):
    """
    Stores document chunk embeddings in Pinecone vector database.
    
    Args:
        chunks_with_metadata (list[DocumentChunkWithMetadata]): List of document chunks to embed and store
        
    Note:
        Processes chunks in batches of 96 using the llama-text-embed-v2 model
    """
    i = 0
    while i < len(chunks_with_metadata):
        data = []
        for chunk in chunks_with_metadata[i:i+96]:
            data.append({
                'id': f"{chunk.document_id}_{chunk.chunk_id}",
                'metadata': {
                    'document_id': chunk.document_id,
                    'chunk_id': chunk.chunk_id,
                    'start': chunk.start,
                    'end': chunk.end,
                    'text': chunk.text
                }
            })
        embeddings = pc.inference.embed(
            model = "llama-text-embed-v2",
            inputs = [chunk.text for chunk in chunks_with_metadata[i:i+96]],
            parameters = {
                "input_type": "passage"
            }
        )
        to_index = []
        for d, e in zip(data, embeddings):
            to_index.append({
                'id': d['id'],
                'values': e['values'],
                'metadata': d['metadata']
            })
        index.upsert(vectors=to_index)
        i = i + 96

def process_document(document_path, document_id):
    """
    Processes a single PDF document: loads, splits, embeds, and stores it.
    
    Args:
        document_path (str): Path to the PDF document
        document_id (str): Unique identifier for the document
        
    Note:
        Skips processing if the document has already been processed
    """
    processed_document_record = load_processed_document_record()
    if document_id not in processed_document_record:
        loader = PyPDFLoader(document_path)
        document = loader.load()  
        print(f"Document {document_id} loaded successfully")
        chunks_with_metadata = split_document_with_metadata(document, document_id)
        print(f"Document {document_id} split into {len(chunks_with_metadata)} chunks")
        store_embeddings(chunks_with_metadata)
        save_processed_document_record(document_id)
        print(f"Document {document_id} processed and stored successfully")

def process_all_documents_in_directory(directory_path):
    """
    Processes all PDF documents in a specified directory.
    
    Args:
        directory_path (str): Path to the directory containing PDF documents
    """
    for filename in os.listdir(directory_path):
        document_path = os.path.join(directory_path, filename)
        if document_path.endswith('.pdf'):
            document_id = os.path.splitext(filename)[0]
            print(f"Processing document {document_id}")
            process_document(document_path, document_id)

if __name__ == "__main__":
    data_directory = os.getenv('DOCUMENT_DIRECTORY_PATH')
    process_all_documents_in_directory(data_directory)
    print("All documents processed and stored successfully")