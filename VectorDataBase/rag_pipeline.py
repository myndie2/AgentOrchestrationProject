import os
from openai import OpenAI
from dotenv import load_dotenv
from retrieval import similarity_search, vector_db
from embedding import embed_query

load_dotenv()

# 1. Initialisation du LLM
client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

# 2. Récupération du contexte (Retrieval)
query = "I have eggs, flour, and honey. What can I make?"
query_vec = embed_query(query)
docs = similarity_search(query_vec, vector_db, k=2)

# 3. Préparation du contexte pour le prompt
context = "\n".join([d["page_content"] for d in docs])

# 4. Appel au LLM avec le contexte
messages = [
    {
        "role": "system", 
        "content": "You are a head chef. You ONLY suggest recipes using the provided context. Do not invent new ingredients."
    },
    {
        "role": "user", 
        "content": f"Context (available recipes):\n{context}\n\nQuestion: {query}"
    }
]

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=messages
)
print("\n*********  PROPOSITION DU CHEF ********")
# 5. Affichage de la réponse finale
print(response.choices[0].message.content)