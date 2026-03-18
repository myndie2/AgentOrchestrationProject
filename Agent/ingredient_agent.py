# Agent qui analyse les ingrédients

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def ingredient_agent(ingredients):

    messages = [
        {
            "role": "system",
            "content": "You are a professional chef analyzing ingredients."
        },
        {
            "role": "user",
            "content": f"""
Analyze these ingredients:

{ingredients}

Return:
Cuisine type
Dish idea
Possible cooking style
"""
        }
    ]

    response = client.responses.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        input=messages
    )

    return response.output_text