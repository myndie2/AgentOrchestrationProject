import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def ask_fixed_prompt(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5
    )
    return resp.choices[0].message.content

prompt = "Give me a receipe with oil, garlic, mozarella"
print(ask_fixed_prompt(prompt))
