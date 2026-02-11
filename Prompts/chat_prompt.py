import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain prompt engineering in 5 sentences."}
    ],
    max_tokens=150
)

print(response.choices[0].message.content)