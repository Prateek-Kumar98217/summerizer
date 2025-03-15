# Document QA System

A powerful document question-answering system that allows users to query PDF documents using natural language. The system uses advanced NLP models and vector similarity search to provide accurate answers and summaries from the document collection.

## Features

- PDF document processing and ingestion
- Semantic search capabilities using document embeddings
- Question answering based on document context
- Text summarization of answers and context
- Automatic document chunking and metadata tracking
- Batch processing of multiple PDF documents
- Vector similarity search using Pinecone

## Architecture

The system consists of several key components:

1. **Document Ingestion** (`ingestion.py`)
   - Loads and processes PDF documents
   - Splits documents into manageable chunks
   - Computes embeddings using llama-text-embed-v2
   - Stores chunks and embeddings in Pinecone

2. **Document Retrieval** (`retrieval.py`)
   - Handles semantic search functionality
   - Converts queries to embeddings
   - Retrieves relevant document chunks
   - Reconstructs coherent document segments

3. **Question Answering** (`question_answer.py`)
   - Provides QA capabilities using DistilBERT
   - Generates text summaries using BART
   - Processes user queries and generates responses

4. **Document Indexing** (`document_indexing.py`)
   - Manages document naming and organization
   - Standardizes PDF filenames for consistency

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
DOCUMENT_DIRECTORY_PATH=path_to_pdf_documents
DOCUMENT_BASE_INDEX=0
```

4. Prepare your documents:
   - Place PDF documents in the specified directory
   - Run document indexing to standardize filenames:
   ```bash
   python document_indexing.py
   ```

5. Process the documents:
   ```bash
   python ingestion.py
   ```

## Usage

1. Start the interactive QA system:
```bash
python main.py
```

2. Enter your questions when prompted. The system will:
   - Search for relevant document segments
   - Generate an answer based on the context
   - Provide a summarized response

## Models Used

- **Text Embeddings**: llama-text-embed-v2
- **Question Answering**: distilbert-base-cased-distilled-squad
- **Summarization**: facebook/bart-large-cnn

## Dependencies

Key dependencies include:
- langchain
- transformers
- pinecone-client
- python-dotenv
- PyPDF2
- torch

For a complete list of dependencies, see `requirements.txt`.

## File Structure

```
.
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── main.py
├── ingestion.py
├── retrieval.py
├── question_answer.py
└── document_indexing.py
```

## Environment Variables

- `PINECONE_API_KEY`: API key for Pinecone service
- `PINECONE_INDEX_NAME`: Name of the Pinecone index to use
- `DOCUMENT_DIRECTORY_PATH`: Path to the directory containing PDF documents
- `DOCUMENT_BASE_INDEX`: Starting index for document numbering

## Notes

- The system processes documents only once and maintains a record of processed documents
- Documents are split into chunks with configurable size and overlap
- Vector embeddings are processed in batches of 96 for efficiency
- All text is encoded in UTF-8 to handle special characters

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Specify your license here] 