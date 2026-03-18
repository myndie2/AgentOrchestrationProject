import os
from openai import OpenAI
from dotenv import load_dotenv

# On importe la fonction de génération que nous avons validée
from embedding import embed_query

load_dotenv()

# 1. Tes données (Chunked documents)
texts = ["Pancakes: flour, eggs, milk, sugar. Mix and fry.",
    "Honey Cookies: flour, honey, eggs. Bake at 180°C.",
    "Omelette: eggs, milk, salt. Fry in a pan."]

# 2. Initialisation de la base locale (équivalent à 'db' dans ton guide)
vector_db = []

print("Stockage des embeddings + métadonnées...")

# 3. Processus 'from_texts' manuel
for t in texts:
    vector = embed_query(t)
    
    # On respecte la consigne : Store embeddings + metadata
    vector_db.append({
        "page_content": t,  # Le texte (métadonnée)
        "vector": vector    # L'embedding
    })

print(f"Base de données vectorielle locale créée avec {len(vector_db)} documents.")

# Vérification du stockage
for entry in vector_db:
    print(f"- Stocké: {entry['page_content']} (Vecteur de taille {len(entry['vector'])})")