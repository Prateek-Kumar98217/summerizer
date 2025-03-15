"""
This module handles the standardization of PDF document filenames in a directory.
It implements a systematic renaming scheme for PDF documents to ensure consistent
naming across the document collection.

Key features:
- Renames PDF files to a standard format: 'document_N.pdf'
- Uses a base index from environment variables to start numbering
- Preserves original files by creating new names with sequential numbering
- Only processes files with .pdf extension

Environment variables required:
- DOCUMENT_DIRECTORY_PATH: Path to the directory containing PDF files
- DOCUMENT_BASE_INDEX: Starting index for document numbering
"""

import os
from dotenv import load_dotenv

load_dotenv()
directory_path = os.getenv('DOCUMENT_DIRECTORY_PATH')

for index, filename in enumerate(os.listdir(directory_path)):
    base = int(os.getenv('DOCUMENT_BASE_INDEX'))
    old_name = os.path.join(directory_path, filename)
    if filename.endswith('.pdf'):
        new_name = os.path.join(directory_path, f'document_{base+index}.pdf')
        os.rename(old_name, new_name) 
