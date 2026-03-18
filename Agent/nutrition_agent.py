# Agent qui estime les valeurs nutritionnelles

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def nutrition_agent(recipe):

    messages = [
        {
            "role": "system",
            "content": "You are a nutrition expert."
        },
        {
            "role": "user",
            "content": f"""
Estimate the nutrition values of this recipe:

{recipe}

Return:

Calories
Protein
Carbs
Fat
"""
        }
    ]

    response = client.responses.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        input=messages
    )

    return response.output_text