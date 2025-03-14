import torch
from transformers import DistilBertTokenizer, DistilBertModel

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim = 1).squeeze()
    embeddings = torch.round(embeddings * 10**14) / 10**14
    return embeddings