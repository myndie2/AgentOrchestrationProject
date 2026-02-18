import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

response = client.responses.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    input="Give me a receipe with oil, garlic, mozarella"
)

print(response.output_text)
