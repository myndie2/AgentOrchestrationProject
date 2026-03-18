# embedding.py
import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def embed_query(text: str) -> list:
    """
    Génère un vrai vecteur sémantique via HuggingFace Router.
    Equivalent de OpenAIEmbeddings().embed_query() du cours.
    """
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text}
    )
    return response.json()

#TEST DE VECTEUR EMBEDDING

# if __name__ == "__main__":
#     vector = embed_query("What is a vector database?")
#     print(f"Taille du vecteur : {len(vector)}")
#     print(f"Premiers éléments : {vector[:5]}")

#     # Test sémantique : preuve que les vecteurs ont du sens
#     v1 = np.array(embed_query("I have eggs and flour"))
#     v2 = np.array(embed_query("Pancakes recipe with eggs"))
#     v3 = np.array(embed_query("How to fix a car engine"))

#     print("\n--- Test sémantique ---")
#     print(f"Similarité cuisine/cuisine : {np.dot(v1, v2):.3f}") 
#     print(f"Similarité cuisine/voiture : {np.dot(v1, v3):.3f}")