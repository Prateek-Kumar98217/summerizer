"""
This module provides question answering and text summarization capabilities for documents.
It uses transformer-based models from Hugging Face:
- DistilBERT for question answering (distilbert-base-cased-distilled-squad)
- BART for text summarization (facebook/bart-large-cnn)

The module contains a QA class that handles both question answering on document contexts
and generating concise summaries of text content.
"""

from transformers import pipeline

class QA():
    """
    A class that provides question answering and text summarization capabilities using transformer models.
    """
    
    def answer(self, reconstructed_document, query, max_length = 1000):
        """
        Answers a question based on the provided document context.
        
        Args:
            reconstructed_document (str): The document text to use as context
            query (str): The question to answer
            max_length (int, optional): Maximum length of the answer. Defaults to 1000.
            
        Returns:
            str: The answer extracted from the context
        """
        qa_model = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
        answer_text = qa_model(context = reconstructed_document, question = query, max_length = max_length, min_length = max_length//2)['answer']
        return answer_text

    def summarize(self, text, max_length = 1000):
        """
        Generates a summary of the provided text.
        
        Args:
            text (str): The text to summarize
            max_length (int, optional): Maximum length of the summary. Defaults to 1000.
            
        Returns:
            str: The generated summary
        """
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary_text = summarizer(text, max_length = max_length, min_length = 100)[0]['summary_text']
        return summary_text