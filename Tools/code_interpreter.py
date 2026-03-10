# Permet à un modèle de langage de déléguer des calculs à Python quand une question demande un résultat exact.

import os
import json
import sys
import io
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url="https://router.huggingface.co/v1"
)


# Fonction qui exécute du code Python envoyé par le modèle
def execute_python(code: str) -> str:
    buf = io.StringIO()   # mémoire tampon pour capturer ce qui est affiché avec print()
    local_env = {}        # dictionnaire pour stocker les variables créées pendant exec()

    try:
        sys.stdout = buf
        exec(code, {}, local_env)# Exécute le code Python

        # Si le code a fait des print(), on les renvoie
        # Sinon, on essaie de renvoyer la variable 'result'
        return buf.getvalue().strip() or str(local_env.get("result", "Code exécuté (pas de sortie)."))

    except Exception as e:
        return f"Erreur : {e}"

    finally:
        # Remet stdout à son état normal
        sys.stdout = sys.__stdout__



tools = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Exécuter du code Python pour faire des calculs.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"]
        }
    }
}]


def ask_AI(question: str):
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant technique. "
                "Quand un calcul est nécessaire, appelle execute_python. "
                "Mets le résultat dans une variable 'result'."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # ÉTAPE A : Premier appel (L'IA analyse la question
    # Cas 1: le modèle répond directement
    # Cas 2: le modèle demande un appel d'outil
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # récupération du message du modèle (soit le premier message généré)
    msg = response.choices[0].message

    # ÉTAPE B : Exécution de l'outil
    if msg.tool_calls: # le modèle ne veut pas encore répondre directement ; il veut d’abord utiliser un outil
        messages.append(msg) # On ajoute la décision du modèle à l’historique, important car pour le second appel, le modèle doit “se souvenir” qu’il a demandé un outil

        for tool_call in msg.tool_calls:
            # On décode le code Python suggéré par l'IA
            args = json.loads(tool_call.function.arguments)  # on renvoie les arguments sous forme de chaine json
            print(f"Exécution du code Python : {args['code']}")

            # On lance l'exécution réelle du code Python
            result = execute_python(args["code"])

            # On injecte le résultat dans la conversation
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
    prompt = (
        "J'ai une recette pour 4 personnes qui demande 3 oeufs, 125g de beurre et 200ml de lait. "
        "Calcule-moi les doses exactes pour 13 personnes et donne-moi le total de calories "
        "si 100g de beurre = 717 kcal, 1 oeuf = 70 kcal et 100ml de lait = 42 kcal."
    )
    print("\n--- RÉPONSE ---")
    print(ask_AI(prompt))



# Étape 0
# Le programme définit un outil execute_python pour exécuter du code Python.

# Étape 1
# L’utilisateur pose une question.

# Étape 2
# Le modèle reçoit :
# la question
# la consigne système
# la liste des outils disponibles (execute_python)

# Étape 3
# Le modèle choisit :
# soit répondre directement
# soit demander l’appel de l’outil execute_python

# Étape 4
# Le code Python exécute vraiment l’outil demandé.
# Le code généré par le modèle est lancé avec exec().

# Étape 5
# Le résultat du calcul est récupéré.

# Étape 6
# Le code renvoie ce résultat au modèle dans l’historique.

# Étape 7
# Le modèle lit le résultat du calcul et rédige la réponse finale.