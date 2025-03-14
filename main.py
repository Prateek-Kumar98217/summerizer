from retrieval import retieve_and_reconstruct
from summerizer import Summerizer

while True:
    query = input("Enter a query for summerization: ")
    reconstructed_document = retieve_and_reconstruct(query)
    summerizer = Summerizer()
    summerized_text = summerizer.summerize(reconstructed_document)
    print(summerized_text)
