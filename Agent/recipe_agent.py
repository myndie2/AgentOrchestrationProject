# Agent qui génère la recette

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

def recipe_agent(ingredients, analysis):

    messages = [
        {
            "role": "system",
            "content": "You are a head chef creating recipes."
        },
        {
            "role": "user",
            "content": f"""
Ingredients:
{ingredients}

Analysis:
{analysis}

Create a complete recipe with:

Title
Ingredients
Steps
Cooking time
Difficulty
"""
        }
    ]

    response = client.responses.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        input=messages
    )

    return response.output_text