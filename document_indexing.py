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
