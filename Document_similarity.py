from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

load_dotenv()
Embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Tokyo is the capital of Japan",
    "London is the capital of UnitedKingdom"
]

query = "Tell me about London"

doc_embedding = Embedding.embed_documents(documents)
query_embedding = Embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embedding)[0]

index,score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity search :",score)