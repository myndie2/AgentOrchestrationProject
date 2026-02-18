import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

messages = [
    {
        "role": "system",
        "content": "You are a head chef."
    },
    {
        "role": "user",
        "content": "Give a receipe with something everybody have in their fridge."
    }
]


response = client.responses.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    input=messages
)

print(response.output_text)
