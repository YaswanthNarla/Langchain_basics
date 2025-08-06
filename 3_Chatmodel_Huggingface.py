from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                                        task="text-generation",
                                        pipeline_kwargs=dict(temperature = 0.5,max_new_tokens = 100))

Model = ChatHuggingFace(llm = llm)

result = Model.invoke("What is the capital of France?")

print(result.content)