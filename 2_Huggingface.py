from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation")
Model = ChatHuggingFace(llm = llm)

result = Model.invoke("What is the capital of France?")

print(result.content)

