# Permettre à un modèle de langage d’utiliser une recherche web réelle (DuckDuckGo) pour récupérer des informations à jour.

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from ddgs import DDGS

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)

# Effectue une recherche réelle sur DuckDuckGo
def web_search(query):
    with DDGS() as ddgs:
        # ouvre DuckDuckGo et fait une recherche texte avec query puis récupère jusqu'à 5 résultats
        results = list(ddgs.text(query, max_results=5))
        # On renvoie uniquement le titre, le résumé et l'URL 
        return [
            {
                "title": r['title'], 
                "snippet": r['body'], 
                "url": r['href']
            } 
            for r in results
        ]

tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Rechercher des recettes et des tendances culinaires sur internet",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}]

# Le modèle ne peut pas exécuter la fonction lui-même
# Il peut seulement demander à l’appeler
# C’est le code Python qui exécute réellement l’outil

def ask_AI(question): # gère la conversation

    messages = [
        {"role": "system", "content": "Tu es un chef expert. Utilise l'outil web_search pour vérifier les tendances de 2026 avant de répondre."},
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
    if msg.tool_calls: # le modèle ne veut pas encore répondre directement ; il veut d’abord utiliser un outil

        messages.append(msg) # On ajoute la décision du modèle à l’historique, important car pour le second appel, le modèle doit “se souvenir” qu’il a demandé un outil
        
        for tool_call in msg.tool_calls:
            # On décode la recherche suggérée par l'IA
            args = json.loads(tool_call.function.arguments) # on renvoie les arguments sous forme de chaine json
            query = args["query"]
            print(f"Recherche web en cours : {query}")
            
            # On lance la recherche réelle
            search_results = web_search(query)
            
            # On injecte les résultats du web dans la conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id, # id de l'appel
                "content": json.dumps(search_results) # contenu retouné par l'outil
            })

        # ÉTAPE C : Réponse finale (L'IA synthétise tout)
        final_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return msg.content


if __name__ == "__main__":
    prompt = "Donne moi 3 recettes de pâtes ultra tendances en 2026 avec leurs sources."
    print("\n---RÉPONSE---\n")
    print(ask_AI(prompt))





# Étape 1
# L’utilisateur pose une question.

# Étape 2
# Le modèle reçoit :
# la question
# la consigne
# la liste des outils disponibles

# Étape 3
# Le modèle choisit :
# soit répondre directement
# soit demander l’appel d’un outil

# Étape 4
# Le code Python exécute vraiment l’outil demandé.

# Étape 5
# Le code renvoie le résultat de l’outil au modèle dans l’historique.

# Étape 6
# Le modèle lit ces résultats et rédige la réponse finale.