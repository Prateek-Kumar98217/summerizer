from retrieval import retieve_and_reconstruct
from question_answer import QA

while True:
    query = input("Enter a query: ")
    context = retieve_and_reconstruct(query)
    model = QA()
    answer = model.answer(context , query)
    answer = model.summarize(answer + context)
    print(answer)
