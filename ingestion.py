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

#class to define the structure of document chunks
class DocumentChunkWithMetadata:
    #constructor
    def __init__(self, text, document_id, chunk_id, start, end):
        self.text = text.encode('utf-8', 'ignore').decode('utf-8')
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.start = start
        self.end = end

#function to split the document into chunks
def split_document_with_metadata(documents, document_id, chunk_size = 500, chunk_overlap = 50):
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
    if os.path.exists("processed_document_record.txt"):
        with open("processed_document_record.txt", "r") as file:
            processed_document_record = set(file.read().split("\n"))
            return processed_document_record
    return set()

#funtion to save the processed document record
def save_processed_document_record(document_id):
    with open("processed_document_record.txt", "a") as file:
        file.write(f"{document_id}\n")

def store_embeddings(chunks_with_metadata):

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