# Ce fichier permet au modèle de langage d’utiliser le terminal pour exécuter des commandes sur les fichiers de recettes stockés localement.

import os
import json
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)


# exécute une commande shell dans le dossier ./docs
def execute_shell(command: str) -> str:
    try:
        result = subprocess.run(
            command,            # commande envoyée par le modèle 
            shell=True,         # exécute la commande dans le terminal
            capture_output=True,# capture la sortie du terminal (stdout et stderr)
            text=True,          # retourne la sortie sous forme de texte (string)
            timeout=10,         # limite d'exécution de la commande à 10 secondes
            cwd="./docs"        # force l'exécution de la commande dans le dossier ./docs
        )

        # retourne la sortie standard de la commande
        return result.stdout.strip() or result.stderr.strip() or "Commande exécutée."

    except Exception as e:
        return f"Erreur : {e}"


tools = [{
    "type": "function",
    "function": {
        "name": "execute_shell",
        "description": "Exécuter une commande dans le terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    }
}]

def ask_shell(question):
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant culinaire technique. "
                "Utilise execute_shell quand il faut lister, rechercher ou consulter des fichiers de recettes dans le terminal. "
                "Quand un outil est utilisé, base ta réponse uniquement sur son résultat exact. "
                "N'invente aucun contenu qui n'apparaît pas dans la sortie de l'outil."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # ÉTAPE A : Premier appel (L'IA analyse la question
    # Cas 1 : le modèle répond directement
    # Cas 2 : le modèle demande un appel d'outil
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        tools=tools
    )

    # récupération du message du modèle (soit le premier message généré)
    msg = response.choices[0].message


    # ÉTAPE B : Exécution de l'outil
    # le modèle ne veut pas encore répondre directement ; il veut d’abord utiliser l’outil execute_shell
    if msg.tool_calls:

        # On ajoute la décision du modèle à l’historique c’est important pour que le modèle se souvienne qu’il a demandé l’appel de l’outil
        messages.append(msg)

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)# les arguments sont envoyés sous forme JSON
            print(f"Commande exécutée : {args['command']}")

            # On lance réellement la commande dans le terminal
            result = execute_shell(args["command"])

            # On injecte la sortie du terminal dans la conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })


        # ÉTAPE C : Réponse finale
        # Le modèle lit la sortie du terminal et rédige la réponse finale pour l'utilisateur
        final_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages
        )

        return final_response.choices[0].message.content

    return msg.content


if __name__ == "__main__":
    print("\n--- RÉPONSE ---\n")
    print(ask_shell("Affiche le contenu de carbonara.txt"))


# Étape 0
# Le programme définit un outil execute_shell qui permet d’exécuter des commandes terminal dans le dossier ./docs contenant les recettes.

# Étape 1
# L’utilisateur pose une question.

# Étape 2
# Le modèle reçoit :
# la question
# la consigne système
# la liste des outils disponibles (execute_shell)

# Étape 3
# Le modèle choisit :
# soit répondre directement
# soit demander l’appel de l’outil execute_shell

# Étape 4
# Le code Python exécute réellement la commande demandée dans le terminal avec subprocess.

# Étape 5
# La sortie de la commande (résultat du terminal) est renvoyée au modèle dans l’historique de conversation.

# Étape 6
# Le modèle lit cette sortie et rédige la réponse finale pour l’utilisateur.