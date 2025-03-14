from transformers import pipeline

class Summerizer():
    def summerize(self,reconstructed_document, max_length = 1000):
        summerizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summerized_text = summerizer(reconstructed_document, max_length=max_length, min_length=100)[0]['summary_text']
        return summerized_text