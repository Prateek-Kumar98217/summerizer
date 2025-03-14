from langchain.documnet_loader import PyPDFDirectoryLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from .embedding import get_embeddings

from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.index(pinecone_index_name)

#class to define the structure of document chunks
class DocumentChunkWithMetadata:
    #constructor
    def __init__(self, text, document_id, chunk_id, start, end):
        self.text = text
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.start = start
        self.end = end

#function to split the document into chunks
def split_document_with_metadata(document, document_id, chunk_size = 1000, chunk_overlap = 100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_document([document])

    chunk_with_metadata =[]
    for chunk_id, chunk in enumerate(chunks):
        start = chunk_id * (chunk_size - chunk_overlap)
        end = start + len(chunk.page_content)
        chunk_with_metadata.append(DocumentWithMetadata(
            text = chunk.page_content,
            document_id = document_id,
            chunk_id = chunk_id,
            start = start,
            end = end
        ))
    return chunk_with_metadata

def load_processed_document_record():
    if os.path.exists("processed_document_record.txt"):
        with open("processed_document_record.txt", "r") as file:
            processed_document_record = set(file.read().split("\n"))
            return processed_document_record
    return set()

#funtion to save the processed document record
def save_processed_document_record(document_id):
    eith open("processed_document_record.txt", "a") as file:
        file.write(f"{document_id}\n")

def store_embeddings(chunks_with_metadata):
    to_index = []
    for chunk in chunks_with_metadata:
        embeddings = get_embeddings(chunk.text).cpu().numpy().tolist()
        to_index.append(
            f"{chunk.document_id}_{chunk.chunk_id}",
            embeddings,
            {
                document_id: chunk.document_id,
                chunk_id: chunk.chunk_id,
                start: chunk.start,
                end: chunk.end,
                text: chunk.text
            }
        )
    index.upsert(vectors=to_index)

def process_document(document_path, documnet_id):
    processed_document_record = load_processed_document_record()
    if document_id not in processed_document_record:
        loader = PyPDFDirectoryLoader(document_path)
        document = loader.load()[0]      
        chunks_with_metadata = split_document_with_metadata(document, document_id)
        store_embeddings(chunks_with_metadata, tokenizer, model)
        save_processed_document_record(document_id)
        print(f"Document {document_id} processed and stored successfully")

def process_all_documents_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        document_path = os.path.join(directory_path, filename)
        if document_path.endswith('.pdf'):
            document_id = os.path.splitext(filename)[0]
            process_document(document_path, document_id)