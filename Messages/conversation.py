import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

conversation = [
    {"role": "user", "content": "What can I cook with milk and beans?"},
    {"role": "assistant", "content": "Bean and Rice Bowl with Milk Gravy"},
    {"role": "user", "content": "I don't have rice, can you give me another recipe?"}
]

response = client.responses.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    input=conversation,
)

print(response.output_text)
