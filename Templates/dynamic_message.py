import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def explain(topic: str) -> str:
    prompt = f"Give a receipe with: {topic}."
    resp = client.responses.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        input=[{"role": "user", "content": prompt}],
    )
    return resp.output_text

print(explain("tomato"))
