from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import  numpy as np
load_dotenv()
# Connect to Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(
    api_key = PINECONE_API_KEY,
)
#load the model and the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
#function to get the embeddings
def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze() #average pooling of the last layer for embeddings
    embeddings = embeddings.numpy()
    return embeddings

#test
input = 'I am a data scientist'
print(get_embeddings(input)[:100]) #print the first 100 elements of the embeddings
print(len(get_embeddings(input))) #print the length of the embeddings
index_name = "rag-trial1"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 768,
        metric = 'cosine',
        spec = ServerlessSpec(
            cloud = 'aws',
            region = 'us-east-1'
        )
)

pc_index = pc.Index(index_name)

#add the embeddings to the index
sentences = ['I am a data scientist', 'I love to code', 'I am a software engineer']
embeddings = [get_embeddings(sentence) for sentence in sentences]
ids = [str(i) for i in range(len(sentences))]
pc_index.upsert(zip(ids, embeddings))

#check the stats
stats = pc_index.describe_index_stats()
print(stats)
#search
query = 'who am I?'
query_embedding = np.array(get_embeddings(query), dtype=np.float32).tolist()
results = pc_index.query(queries=[query_embedding], top_k=3)
print(results)