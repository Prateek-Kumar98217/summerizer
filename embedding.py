import torch
from .main import tokenizer, model

def get_embeddings(text):
    imputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim = 1).squeeze()
    return embeddings