# Propose une amélioration

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def improvement_agent(recipe):

    messages = [
        {
            "role": "system",
            "content": "You are a professional chef improving recipes."
        },
        {
            "role": "user",
            "content": f"""
Improve this recipe:

{recipe}

Suggest:
- better ingredients
- flavor improvements
- presentation tips
"""
        }
    ]

    response = client.responses.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        input=messages
    )

    return response.output_text