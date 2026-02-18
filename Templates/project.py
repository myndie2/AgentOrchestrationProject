import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

question = "What is the most famous cooker in France?"

template = "Answer the following question clearly:\n{question}"

prompt = template.format(question=question)

response = client.responses.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    input=prompt
)

print(response.output_text)
