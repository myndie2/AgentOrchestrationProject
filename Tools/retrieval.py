# Permet à un modèle de langage de chercher dans des documents locaux par sens grâce aux embeddings et à FAISS.

import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

# création du modèle d'embedding qui transforme une phrase en vecteur numérique
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#construction du vecteux store qui sert à indexer les documents
def build_vector_store(folder: str = "./docs"):
    chunks, metadata = [], []
    for filename in os.listdir(folder): # lecture des fichiers
        if filename.endswith((".txt", ".md")): # on parcours les fichiers de docs
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                content = f.read() # on lit le texte en entier
            # On coupe le texte en morceaux de 300 caractères car les modèles travaillent mieux avec des petits blocs
            for i in range(0, len(content), 300):
                chunk = content[i:i+300].strip()
                if chunk:
                    chunks.append(chunk) # chunk: texte brut
                    metadata.append({"filename": filename, "chunk": chunk})  # informations utiles

    embeddings = embedder.encode(chunks, convert_to_numpy=True) # création des embeddings, chaque chunk devient un vecteur
    index = faiss.IndexFlatL2(embeddings.shape[1]) #FAISS est une vector database spécialisée dans la recherche de similarité. Elle permet de trouver rapidement les vecteurs les plus proches dans un espace de grande dimension.
    index.add(embeddings) # ajout des embeddinfs dans FAISS
    return index, metadata



# fonction de recherche sémantique
def semantic_search(query: str, index, metadata, top_k: int = 3) -> str:
    query_vec = embedder.encode([query], convert_to_numpy=True) # embedding de la qustion
    distances, indices = index.search(query_vec, top_k) # FAISS cherche les vecteurs les plsu proche
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "filename": metadata[idx]["filename"],
            "score": float(1 / (1 + distances[0][i])),  # scrore de similarité(+ distance petite, plus texte est similaire)
            "content": metadata[idx]["chunk"]
        })
    return json.dumps(results, ensure_ascii=False)



tools = [{
    "type": "function",
    "function": {
        "name": "semantic_search",
        "description": "Recherche sémantique dans les documents locaux.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La question en langage naturel."}
            },
            "required": ["query"]
        }
    }
}]


def ask_with_retrieval(question, index, metadata):
    messages = [
        {"role": "system", "content": "Tu es un assistant. Utilise semantic_search pour chercher dans les documents avant de répondre."},
        {"role": "user", "content": question}
    ]

    # ÉTAPE A : Premier appel (L'IA analyse la question
    # Cas 1: le modèle répond directement
    # Cas 2: le modèle demande un appel d'outil
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        tools=tools
    )

    # récupération du message du modèle (soit le premier message généré)
    msg = response.choices[0].message

    # ÉTAPE B : Exécution de l'outil
    if msg.tool_calls: 
        messages.append(msg)
        for tool_call in msg.tool_calls: 
            # On décode la recherche suggérée par l'IA
            args = json.loads(tool_call.function.arguments) # on renvoie les arguments sous forme de chaine json
            print(f"Recherche sémantique : {args['query']}")

            # On lance la recherche réelle
            result = semantic_search(args["query"], index, metadata)

            # On injecte les résultats du web dans la conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,  # id de l'appel
                "content": result
            })

        # ÉTAPE C : Réponse finale
        final_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages
        )
        return final_response.choices[0].message.content

    return msg.content

if __name__ == "__main__":
    print(" Construction du vector store...")
    index, metadata = build_vector_store("./docs")
    print(f"{len(metadata)} chunks indexés\n")

    print("--- RÉPONSE ---\n")
    print(ask_with_retrieval("Quelle est la recette de la carbonara ?", index, metadata))




# Étape 0
# Le programme prépare la base de connaissances.
# Les documents du dossier ./docs sont chargés.
# Ils sont découpés en petits morceaux (chunks).
# Chaque chunk est transformé en embedding (vecteur).
# Tous les vecteurs sont stockés dans FAISS (vector database).

# Étape 1
# L’utilisateur pose une question.

# Étape 2
# Le modèle reçoit :
# la question
# la consigne système
# la liste des outils disponibles (semantic_search)

# Étape 3
# Le modèle choisit :
# soit répondre directement
# soit demander l’appel de l’outil semantic_search

# Étape 4
# Le code Python exécute vraiment l’outil demandé.
# la question est transformée en embedding
# FAISS cherche les chunks les plus similaires

# Étape 5
# Le code renvoie les documents trouvés au modèle dans l’historique.

# Étape 6
# Le modèle lit ces documents et rédige la réponse finale.
