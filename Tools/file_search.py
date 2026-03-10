# Permet à un modèle de langage de chercher des informations dans des fichiers locaux en faisant une recherche simple par mots-clés.

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)



# cherche un fichier dans un texte 
def search_in_files(query: str, folder: str = "./docs") -> str:
    
    results = []  # Liste qui contiendra les résultats trouvés 

    if not os.path.exists(folder):
        return "Dossier introuvable."


    for filename in os.listdir(folder): # Parcourt tous les fichiers présents dans le dossier

        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(folder, filename) # Construit le chemin complet vers le fichier
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read() # Ouvre le fichier et lit tout son contenu dans la variable "content"

                # Recherche simple par mots-clés
                if any(word.lower() in content.lower() for word in query.split()): # On découpe la question en mots puis on vérifie si au moins un mot apparaît dans le contenu du fichier

                    idx = content.lower().find(query.split()[0].lower())  # Cherche la position du premier mot de la requête dans le texte

                    excerpt = content[max(0, idx-100):idx+300]
                    # On extrait un morceau du texte autour du mot trouvé
                    # -100 caractères avant
                    # +300 caractères après
                    # max(0, ...) évite d'aller avant le début du fichier

                    results.append({"filename": filename, "excerpt": excerpt})

            except Exception as e:
                results.append({"filename": filename, "error": str(e)}) # On ajoute l'erreur dans les résultats pour savoir quel fichier pose problème

    return json.dumps(results, ensure_ascii=False) if results else "Aucun résultat trouvé."




tools = [{
    "type": "function",
    "function": {
        "name": "search_in_files",
        "description": "Recherche des informations dans des fichiers locaux.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Les mots-clés à rechercher."},
                "folder": {"type": "string", "description": "Dossier où chercher (défaut: ./docs)"}
            },
            "required": ["query"]
        }
    }
}]



def ask_with_files(question):
    messages = [
        {"role": "system", "content": "Tu es un assistant. Utilise search_in_files pour chercher dans les documents locaux avant de répondre."},
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
            args = json.loads(tool_call.function.arguments)  # on renvoie les arguments sous forme de chaine json
            print(f"Recherche dans les fichiers : {args['query']}")

            # On lance la recherche réelle
            result = search_in_files(args["query"], args.get("folder", "./docs"))

            # On injecte les résultats du web dans la conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id, # id de l'appel
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
    print("\n--- RÉPONSE ---\n")
    print(ask_with_files("Que dit le document sur les pâtes ?"))




# Étape 0
# Le programme définit un outil search_in_files pour chercher des mots dans les fichiers du dossier ./docs.

# Étape 1
# L’utilisateur pose une question.

# Étape 2
# Le modèle reçoit :
# la question
# la consigne système
# la liste des outils disponibles (search_in_files)

# Étape 3
# Le modèle choisit :
# soit répondre directement
# soit demander l’appel de l’outil search_in_files

# Étape 4
# Le code Python exécute l’outil demandé.
# La fonction :
# search_in_files(query)
# ouvre les fichiers du dossier ./docs
# lit leur contenu
# cherche les mots de la requête

# Étape 5
# Si un mot est trouvé dans un fichier.
# Le code :
# récupère la position du mot
# extrait un morceau du texte autour du mot
# ajoute l’extrait dans la liste des résultats

# Étape 6
# Les résultats sont convertis en JSON et renvoyés au modèle.

# Étape 7
# Le code ajoute ces résultats dans l’historique de conversation.

# Étape 8
# Le modèle lit les extraits trouvés et rédige la réponse finale.