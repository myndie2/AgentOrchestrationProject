import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def explain(topic: str) -> str:
    prompt = f"Explain {topic} in simple terms."
    resp = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5
    )
    return resp.choices[0].message.content

print(explain("vector databases"))
