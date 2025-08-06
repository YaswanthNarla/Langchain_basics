from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()   

Embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=16)

documents = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Tokyo is the capital of Japan"
]
result = Embedding.embed_documents("Delhi is the capital of India")

print(str(result))