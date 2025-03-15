from transformers import pipeline

class QA():
    def answer(self,reconstructed_document, query, max_length = 1000):
        qa_model = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
        answer_text = qa_model(context = reconstructed_document, question = query, max_length = max_length)['answer']
        return answer_text

    def summarize(self, text, max_length = 1000):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary_text = summarizer(text, max_length = max_length, min_length = 100)[0]['summary_text']
        return summary_text