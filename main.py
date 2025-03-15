"""
This is the main entry point for the document QA system.
It provides an interactive command-line interface where users can:
- Enter questions about the documents in the system
- Receive relevant answers extracted from the document context
- Get summarized responses combining the answer and context

The module uses:
- retrieval module for semantic search and document reconstruction
- question_answer module for processing queries and generating responses

The system runs in an infinite loop, continuously accepting user queries
until manually terminated.
"""

from retrieval import retieve_and_reconstruct
from question_answer import QA

while True:
    query = input("Enter a query: ")
    context = retieve_and_reconstruct(query)
    model = QA()
    answer = model.answer(context , query)
    answer = model.summarize(answer + context)
    print(answer)
