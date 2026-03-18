# Vérifie le type de régime

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def diet_agent(recipe):

    messages = [
        {
            "role": "system",
            "content": "You are a diet expert."
        },
        {
            "role": "user",
            "content": f"""
Analyze the diet compatibility of this recipe:

{recipe}

Is it:
- vegetarian
- vegan
- high protein
- healthy

Explain briefly.
"""
        }
    ]

    response = client.responses.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        input=messages
    )

    return response.output_text